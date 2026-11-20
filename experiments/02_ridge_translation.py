# experiments/02_ridge_translation.py
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.cv import (
    CORE_LINES,
    get_core_eval_mask,
    make_gene_group_folds,
    rowwise_pearson_scores,
)
from src.data import load_genes, load_sample_submission, load_test, load_train

ROOT = Path(__file__).resolve().parents[1]
SUB_DIR = ROOT / "submissions"
ART_DIR = ROOT / "artifacts"

# Hyperparamètres globaux (propres, calibrés sur la géométrie de l'EDA)
EPS_X = 0.05          # Adoucissement z-score des 6 546 gènes en entrée
EPS_G = 0.02          # Adoucissement ratio std des 500 gènes (prior)
SCORED_BOOST = 2.0    # Poids relatif des 500 gènes scorés dans le noyau 6 546 gènes
TAU_X = 0.50          # Soft-normalisation L2 des entrées
TAU_Y = 0.35          # Soft-normalisation L2 des cibles (alignement Pearson)
C_SAMPLE_W = 30.0     # Pondération qualité n_cells des exemples d'entraînement
C_CONSENSUS = 40.0    # Pondération n_cells pour construire le profil consensus 2B
C0_SQ = 1.0           # Terme de biais (intercept exact dans le noyau)

GRID_ETAS = (0.0, 0.30, 0.60)
GRID_ALPHAS = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
GRID_BETAS = (0.0, 0.35, 0.70, 1.0)


# ---------------------------------------------------------------------------
# 1. Statistiques & transformations sans fuite (fit strictement sur tr_df)
# ---------------------------------------------------------------------------

def fit_full_line_stats(
    tr_df: pd.DataFrame,
    all_genes: list[str],
    scored_idx: np.ndarray,
) -> dict[str, dict]:
    """Calcule les statistiques par lignée sur les 6 546 gènes et 500 scorés."""
    stats: dict[str, dict] = {}
    boost = np.ones(len(all_genes), dtype=np.float32)
    boost[scored_idx] = SCORED_BOOST

    for line, grp in tr_df.groupby("line"):
        M = grp[all_genes].to_numpy(dtype=np.float32)
        valid_col = np.isfinite(M).any(axis=0)

        mean_all = np.zeros(len(all_genes), dtype=np.float32)
        std_all = np.ones(len(all_genes), dtype=np.float32)
        if valid_col.any():
            M_v = M[:, valid_col]
            mean_all[valid_col] = np.nanmean(M_v, axis=0)
            std_all[valid_col] = np.nanstd(M_v, axis=0)

        # Norme médiane des features normalisées (pour soft-L2)
        Z = np.where(np.isfinite(M), M - mean_all[None, :], 0.0)
        Z = Z / (std_all[None, :] + EPS_X)
        Z[:, ~valid_col] = 0.0
        Z *= boost[None, :]
        med_feat_l2 = float(np.median(np.linalg.norm(Z, axis=1)))

        # Statistiques sur les 500 gènes scorés
        Y = M[:, scored_idx].astype(np.float64)
        valid_y = np.isfinite(Y)
        Y_clean = np.where(valid_y, Y, 0.0)
        cnt = np.maximum(valid_y.sum(axis=1, keepdims=True), 1)
        Y_c = np.where(valid_y, Y_clean - Y_clean.sum(axis=1, keepdims=True) / cnt, 0.0)
        med_target_l2 = float(np.median(np.linalg.norm(Y_c, axis=1)))

        stats[line] = {
            "valid_col": valid_col,
            "mean_all": mean_all,
            "std_all": std_all,
            "med_feat_l2": max(med_feat_l2, 1e-6),
            "scored_std": std_all[scored_idx].astype(np.float64),
            "med_target_l2": max(med_target_l2, 1e-6),
            "boost": boost,
        }
    return stats


def transform_features(X_raw: np.ndarray, line: str, stats: dict) -> np.ndarray:
    """Transforme (N, 6546) brut d'une lignée en features normalisées float32."""
    st = stats[line]
    Z = np.where(np.isfinite(X_raw), X_raw - st["mean_all"][None, :], 0.0).astype(np.float32)
    Z /= (st["std_all"][None, :] + EPS_X)
    Z[:, ~st["valid_col"]] = 0.0
    Z *= st["boost"][None, :]
    l2 = np.linalg.norm(Z, axis=1, keepdims=True)
    denom = np.sqrt(l2 ** 2 + (TAU_X * st["med_feat_l2"]) ** 2)
    return (Z / np.maximum(denom, 1e-8)).astype(np.float32)


def transform_prior_500(
    X_scored_raw: np.ndarray,
    s_line: str,
    t_line: str,
    stats: dict,
) -> np.ndarray:
    """Prior biologique aligné par variance de gène, centré et soft-normalisé."""
    scale = (stats[t_line]["scored_std"] + EPS_G) / (stats[s_line]["scored_std"] + EPS_G)
    valid = np.isfinite(X_scored_raw)
    P = np.where(valid, X_scored_raw, 0.0).astype(np.float64) * scale[None, :]
    cnt = np.maximum(valid.sum(axis=1, keepdims=True), 1)
    P_c = np.where(valid, P - P.sum(axis=1, keepdims=True) / cnt, 0.0)
    l2 = np.linalg.norm(P_c, axis=1, keepdims=True)
    denom = np.sqrt(l2 ** 2 + (TAU_Y * stats[t_line]["med_target_l2"]) ** 2)
    return P_c / np.maximum(denom, 1e-8)


def prepare_target_arrays(
    Y_raw: np.ndarray,
    target_genes: np.ndarray,
    scored_arr: np.ndarray,
    t_line: str,
    stats: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prépare la cible soft-normalisée (pour Ridge) et les tenseurs Pearson exacts."""
    masks = (scored_arr[None, :] != target_genes[:, None]) & np.isfinite(Y_raw)
    Y_clean = np.where(masks, Y_raw, 0.0).astype(np.float64)
    counts = np.maximum(masks.sum(axis=1, keepdims=True), 1)
    Y_c = np.where(masks, Y_clean - Y_clean.sum(axis=1, keepdims=True) / counts, 0.0)
    y_norms = np.linalg.norm(Y_c, axis=1)

    denom = np.sqrt(y_norms[:, None] ** 2 + (TAU_Y * stats[t_line]["med_target_l2"]) ** 2)
    Y_check = np.where(masks, Y_c / np.maximum(denom, 1e-8), 0.0)
    return Y_check, masks, Y_c, y_norms, counts


def vectorized_pearson_mean(
    P: np.ndarray,
    Y_c: np.ndarray,
    y_norms: np.ndarray,
    masks: np.ndarray,
    counts: np.ndarray,
) -> float:
    P_clean = np.where(masks, P, 0.0)
    P_c = np.where(masks, P_clean - P_clean.sum(axis=1, keepdims=True) / counts, 0.0)
    denom = np.linalg.norm(P_c, axis=1) * y_norms
    corrs = np.divide(
        np.sum(P_c * Y_c, axis=1),
        denom,
        out=np.zeros(len(Y_c), dtype=np.float64),
        where=denom > 0,
    )
    return float(corrs.mean())


def row_standardize_for_blend(
    P: np.ndarray,
    target_genes: np.ndarray,
    scored_arr: np.ndarray,
) -> np.ndarray:
    """Centre et normalise chaque prédiction sur les gènes scorés hors target_gene."""
    masks = scored_arr[None, :] != target_genes[:, None]
    P_clean = np.where(masks, np.nan_to_num(P, nan=0.0), 0.0)
    counts = np.maximum(masks.sum(axis=1, keepdims=True), 1)
    means = P_clean.sum(axis=1, keepdims=True) / counts
    P_c = np.where(masks, P_clean - means, 0.0)
    norms = np.linalg.norm(P_c, axis=1, keepdims=True)
    return np.divide(P_c, norms, out=np.zeros_like(P_c), where=norms > 1e-12)


# ---------------------------------------------------------------------------
# 2. Dual Kernel Ridge résiduel avec sélection exacte par LOO Pearson
# ---------------------------------------------------------------------------

class ResidualDualKernelRidge:
    """
    Résout min_W sum_i d_i ||phi(x_i)^T W - (y_i - beta * p_i)||^2 + alpha ||W||^2
    dans l'espace dual (N x N) avec noyau hybride (1-eta)*Lin + eta*RBF + biais.
    Sélectionne (eta, alpha, beta) par LOO analytique exact sur la métrique Pearson.
    """

    def __init__(
        self,
        etas: tuple[float, ...] = GRID_ETAS,
        alphas: tuple[float, ...] = GRID_ALPHAS,
        betas: tuple[float, ...] = GRID_BETAS,
        c0_sq: float = C0_SQ,
    ) -> None:
        self.etas = etas
        self.alphas = alphas
        self.betas = betas
        self.c0_sq = c0_sq

    def fit(
        self,
        X_tr: np.ndarray,
        Y_check: np.ndarray,
        P_check: np.ndarray,
        masks: np.ndarray,
        Y_c: np.ndarray,
        y_norms: np.ndarray,
        counts: np.ndarray,
        sample_weights: np.ndarray,
    ) -> np.ndarray:
        """Fit le modèle et retourne les prédictions LOO exactes (N, 500)."""
        X32 = np.ascontiguousarray(X_tr, dtype=np.float32)
        d = sample_weights.astype(np.float64)
        d = d / max(float(d.mean()), 1e-12)
        s_d = np.sqrt(d)
        W_mat = np.outer(s_d, s_d)

        # Noyaux de base (N x N)
        K_lin = (X32 @ X32.T).astype(np.float64)
        x_sq = np.sum(X32.astype(np.float64) ** 2, axis=1)
        D2 = np.maximum(x_sq[:, None] + x_sq[None, :] - 2.0 * K_lin, 0.0)
        pos_d2 = D2[D2 > 1e-12]
        gamma = 1.0 / float(np.median(pos_d2)) if len(pos_d2) > 0 else 1.0
        K_rbf = np.exp(-gamma * D2)

        Y_masked = np.where(masks, Y_check, 0.0)
        P_masked = np.where(masks, P_check, 0.0)
        Y_tilde = s_d[:, None] * Y_masked
        P_tilde = s_d[:, None] * P_masked

        best_score = -2.0
        best_params = None
        best_eig = None
        best_loo = None

        for eta in self.etas:
            K_base = (1.0 - eta) * K_lin + eta * K_rbf + self.c0_sq
            K_tilde = W_mat * K_base
            K_tilde = 0.5 * (K_tilde + K_tilde.T)

            eigvals, Q = np.linalg.eigh(K_tilde)
            eigvals = np.maximum(eigvals, 0.0)
            Q2 = Q ** 2

            Qt_Y = Q.T @ Y_tilde
            Qt_P = Q.T @ P_tilde

            for alpha in self.alphas:
                f = eigvals / (eigvals + alpha)
                h_diag = Q2 @ f
                denom_loo = np.maximum(1.0 - h_diag, 1e-6)

                Y_hat_t = Q @ (f[:, None] * Qt_Y)
                P_hat_t = Q @ (f[:, None] * Qt_P)

                Y_loo = ((Y_hat_t - h_diag[:, None] * Y_tilde) / denom_loo[:, None]) / s_d[:, None]
                P_loo = ((P_hat_t - h_diag[:, None] * P_tilde) / denom_loo[:, None]) / s_d[:, None]
                diff_P = P_check - P_loo

                for beta in self.betas:
                    pred_loo = Y_loo + beta * diff_P
                    sc = vectorized_pearson_mean(pred_loo, Y_c, y_norms, masks, counts)
                    if sc > best_score:
                        best_score = sc
                        best_params = (eta, alpha, beta)
                        best_eig = (eigvals, Q)
                        best_loo = pred_loo

        best_eta, best_alpha, best_beta = best_params
        eigvals, Q = best_eig

        # Calcul des coefficients duaux pour l'inférence
        R_tilde = s_d[:, None] * (Y_masked - best_beta * P_masked)
        inv_f = 1.0 / (eigvals + best_alpha)
        A_tilde = Q @ (inv_f[:, None] * (Q.T @ R_tilde))

        self.best_eta_ = float(best_eta)
        self.best_alpha_ = float(best_alpha)
        self.best_beta_ = float(best_beta)
        self.best_loo_score_ = float(best_score)
        self.gamma_ = float(gamma)
        self.X_tr_ = X32
        self.X_tr_sq_ = x_sq
        self.dual_coef_ = s_d[:, None] * A_tilde
        return best_loo

    def predict(self, X_new: np.ndarray, P_new: np.ndarray) -> np.ndarray:
        if len(X_new) == 0:
            return np.zeros((0, self.dual_coef_.shape[1]), dtype=np.float64)
        X32 = np.ascontiguousarray(X_new, dtype=np.float32)
        K_lin = (X32 @ self.X_tr_.T).astype(np.float64)
        if self.best_eta_ > 0.0:
            x_new_sq = np.sum(X32.astype(np.float64) ** 2, axis=1)
            D2 = np.maximum(x_new_sq[:, None] + self.X_tr_sq_[None, :] - 2.0 * K_lin, 0.0)
            K = (1.0 - self.best_eta_) * K_lin + self.best_eta_ * np.exp(-self.gamma_ * D2) + self.c0_sq
        else:
            K = K_lin + self.c0_sq
        return self.best_beta_ * P_new + K @ self.dual_coef_


# ---------------------------------------------------------------------------
# 3. Modèle complet 2A (Pairwise) + 2B (Consensus) + Blend Pearson-optimal
# ---------------------------------------------------------------------------

def build_full_source_index(
    src_df: pd.DataFrame,
    all_genes: list[str],
) -> tuple[np.ndarray, dict[str, list[tuple[str, float, int]]]]:
    """Indexe rapidement les profils sources : target_gene -> [(line, n_cells, row_pos)]."""
    X_all = src_df[all_genes].to_numpy(dtype=np.float32)
    lookup: dict[str, list[tuple[str, float, int]]] = {}
    for pos, (tg, line, nc) in enumerate(
        zip(src_df["target_gene"], src_df["line"], src_df["n_cells"])
    ):
        lookup.setdefault(tg, []).append((line, float(max(nc, 1.0)), pos))
    return X_all, lookup


class CrossLineRidgeEnsemble:
    """
    Encapsule :
      - 2A : 20 traducteurs dirigés Pairwise Residual Kernel Ridge (s -> t)
             + agrégation pondérée (w(s->t) * n_cells) apprise sur LOO exactes.
      - 2B : 4 modèles Consensus-to-Target Residual Kernel Ridge (toutes sources -> t).
      - Blend 2A + 2B avec standardisation par ligne et poids lambda_t appris sur LOO.
    """

    def __init__(
        self,
        all_genes: list[str],
        scored_genes: list[str],
        scored_idx: np.ndarray,
        verbose: bool = False,
    ) -> None:
        self.all_genes = all_genes
        self.scored_genes = scored_genes
        self.scored_arr = np.asarray(scored_genes)
        self.scored_idx = scored_idx
        self.verbose = verbose

    def _build_consensus_features(
        self,
        eval_df: pd.DataFrame,
        t_line: str,
        X_src_all: np.ndarray,
        lookup: dict[str, list[tuple[str, float, int]]],
        fit_median_l2: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Construit (X_cons_6546, P_cons_500, has_src_mask) pour une lignée cible."""
        n_rows = len(eval_df)
        n_all = len(self.all_genes)
        n_sc = len(self.scored_genes)

        X_cons = np.zeros((n_rows, n_all), dtype=np.float32)
        P_cons = np.zeros((n_rows, n_sc), dtype=np.float64)
        has_src = np.zeros(n_rows, dtype=bool)

        # Pré-calcule les features par lignée source à la volée via cache par position
        feat_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

        def get_src_transformed(s_line: str, pos: int):
            if pos not in feat_cache:
                raw_row = X_src_all[pos : pos + 1]
                z = transform_features(raw_row, s_line, self.stats_)[0]
                valid_x = self.stats_[s_line]["valid_col"] & np.isfinite(raw_row[0])
                p = transform_prior_500(raw_row[:, self.scored_idx], s_line, t_line, self.stats_)[0]
                valid_p = np.isfinite(raw_row[0, self.scored_idx])
                feat_cache[pos] = (z, valid_x, p, valid_p)
            return feat_cache[pos]

        for i, tg in enumerate(eval_df["target_gene"].to_numpy()):
            sources = [s for s in lookup.get(tg, []) if s[0] != t_line]
            if not sources:
                continue
            has_src[i] = True
            num_x = np.zeros(n_all, dtype=np.float32)
            den_x = np.zeros(n_all, dtype=np.float32)
            num_p = np.zeros(n_sc, dtype=np.float64)
            den_p = np.zeros(n_sc, dtype=np.float64)

            for s_line, nc, pos in sources:
                w = nc / (nc + C_CONSENSUS)
                z, valid_x, p, valid_p = get_src_transformed(s_line, pos)
                num_x += np.float32(w) * z
                den_x += np.float32(w) * valid_x
                num_p += w * p
                den_p += w * valid_p

            X_cons[i] = np.divide(num_x, den_x, out=np.zeros(n_all, dtype=np.float32), where=den_x > 0)
            p_avg = np.divide(num_p, den_p, out=np.zeros(n_sc, dtype=np.float64), where=den_p > 0)
            p_c = p_avg - p_avg.mean()
            p_norm = np.linalg.norm(p_c)
            p_denom = np.sqrt(p_norm ** 2 + (TAU_Y * self.stats_[t_line]["med_target_l2"]) ** 2)
            P_cons[i] = p_c / max(p_denom, 1e-8)

        # Soft-L2 normalisation globale du consensus 6 546 gènes
        l2_x = np.linalg.norm(X_cons[has_src], axis=1)
        if fit_median_l2:
            self.cons_med_l2_[t_line] = max(float(np.median(l2_x)), 1e-6) if len(l2_x) > 0 else 1.0
        med_l2 = self.cons_med_l2_[t_line]
        all_l2 = np.linalg.norm(X_cons, axis=1, keepdims=True)
        denom_x = np.sqrt(all_l2 ** 2 + (TAU_X * med_l2) ** 2)
        X_cons = (X_cons / np.maximum(denom_x, 1e-8)).astype(np.float32)
        return X_cons, P_cons, has_src

    def fit(self, tr_df: pd.DataFrame) -> dict:
        self.stats_ = fit_full_line_stats(tr_df, self.all_genes, self.scored_idx)
        self.all_lines_ = sorted(tr_df["line"].unique().tolist())
        self.pair_models_: dict[tuple[str, str], ResidualDualKernelRidge] = {}
        self.pair_weights_: dict[str, dict[str, float]] = {}
        self.pair_c_cells_: dict[str, float] = {}
        self.cons_models_: dict[str, ResidualDualKernelRidge] = {}
        self.cons_med_l2_: dict[str, float] = {}
        self.blend_lambda_: dict[str, float] = {}

        # Indexation par lignée dans tr_df
        by_line: dict[str, tuple[pd.DataFrame, np.ndarray, dict[str, int]]] = {}
        for line in self.all_lines_:
            sub = tr_df.loc[tr_df["line"].eq(line)].reset_index(drop=True)
            X_l = sub[self.all_genes].to_numpy(dtype=np.float32)
            g2i = {g: i for i, g in enumerate(sub["target_gene"].to_numpy())}
            by_line[line] = (sub, X_l, g2i)

        X_tr_all, tr_lookup = build_full_source_index(tr_df, self.all_genes)
        diagnostics = {"pair_summary": [], "cons_summary": [], "blend_summary": []}

        for t_line in CORE_LINES:
            t_sub, X_t_all, t_g2i = by_line[t_line]
            N_t = len(t_sub)
            n_sc = len(self.scored_genes)
            src_lines = [l for l in self.all_lines_ if l != t_line]
            l2j = {l: j for j, l in enumerate(src_lines)}

            # Tenseur des prédictions LOO de chaque traducteur (s -> t)
            S_loo = np.zeros((N_t, len(src_lines), n_sc), dtype=np.float64)
            NC_mat = np.zeros((N_t, len(src_lines)), dtype=np.float64)
            has_pair = np.zeros((N_t, len(src_lines)), dtype=bool)

            # ---- 2A : Entraînement des 5 traducteurs dirigés (s -> t) ----
            for s_line in src_lines:
                s_sub, X_s_all, s_g2i = by_line[s_line]
                common_genes = [g for g in t_sub["target_gene"].to_numpy() if g in s_g2i]
                t_idx = np.array([t_g2i[g] for g in common_genes], dtype=np.int64)
                s_idx = np.array([s_g2i[g] for g in common_genes], dtype=np.int64)

                X_s_feat = transform_features(X_s_all[s_idx], s_line, self.stats_)
                P_prior = transform_prior_500(
                    X_s_all[s_idx][:, self.scored_idx], s_line, t_line, self.stats_
                )
                Y_raw = X_t_all[t_idx][:, self.scored_idx].astype(np.float64)
                tg_arr = np.asarray(common_genes)

                Y_check, masks, Y_c, y_norms, counts = prepare_target_arrays(
                    Y_raw, tg_arr, self.scored_arr, t_line, self.stats_
                )
                nc_s = np.maximum(s_sub["n_cells"].to_numpy(dtype=np.float64)[s_idx], 1.0)
                nc_t = np.maximum(t_sub["n_cells"].to_numpy(dtype=np.float64)[t_idx], 1.0)
                sample_w = (nc_s / (nc_s + C_SAMPLE_W)) * (nc_t / (nc_t + C_SAMPLE_W))

                model = ResidualDualKernelRidge()
                loo_preds = model.fit(
                    X_s_feat, Y_check, P_prior, masks, Y_c, y_norms, counts, sample_w
                )
                self.pair_models_[(s_line, t_line)] = model

                j = l2j[s_line]
                S_loo[t_idx, j] = loo_preds
                NC_mat[t_idx, j] = nc_s
                has_pair[t_idx, j] = True

                diagnostics["pair_summary"].append(
                    {
                        "pair": f"{s_line}->{t_line}",
                        "n": len(common_genes),
                        "eta": model.best_eta_,
                        "alpha": model.best_alpha_,
                        "beta": model.best_beta_,
                        "loo_pearson": model.best_loo_score_,
                    }
                )

            # Optimisation des poids d'agrégation w(s->t) et c_cells sur les LOO de 2A
            valid_rows = has_pair.any(axis=1)
            S_v = S_loo[valid_rows]
            NC_v = NC_mat[valid_rows]
            HP_v = has_pair[valid_rows]
            Y_t_raw = X_t_all[valid_rows][:, self.scored_idx].astype(np.float64)
            tg_v = t_sub["target_gene"].to_numpy()[valid_rows]
            _, masks_v, Y_c_v, y_norms_v, counts_v = prepare_target_arrays(
                Y_t_raw, tg_v, self.scored_arr, t_line, self.stats_
            )

            best_agg_sc = -2.0
            best_w = None
            best_c = 0.0
            best_oof_2a = None

            for c_cells in (0.0, 30.0, 80.0):
                W_nc = np.where(
                    HP_v,
                    NC_v / (NC_v + c_cells) if c_cells > 0.0 else 1.0,
                    0.0,
                )
                S_weighted = S_v * W_nc[:, :, None]

                def loss_fn(log_w: np.ndarray) -> float:
                    w = np.exp(np.clip(log_w, -4.0, 4.0))
                    P = np.tensordot(S_weighted, w, axes=([1], [0]))
                    sc = vectorized_pearson_mean(P, Y_c_v, y_norms_v, masks_v, counts_v)
                    return -sc + 2e-4 * float(np.sum(log_w ** 2))

                res = minimize(
                    loss_fn,
                    np.zeros(len(src_lines), dtype=np.float64),
                    method="Nelder-Mead",
                    options={"maxiter": 600},
                )
                w_opt = np.exp(np.clip(res.x, -4.0, 4.0))
                w_opt /= w_opt.mean()
                oof_2a_cand = np.tensordot(S_weighted, w_opt, axes=([1], [0]))
                sc_cand = vectorized_pearson_mean(oof_2a_cand, Y_c_v, y_norms_v, masks_v, counts_v)
                if sc_cand > best_agg_sc:
                    best_agg_sc = sc_cand
                    best_w = w_opt
                    best_c = c_cells
                    best_oof_2a = oof_2a_cand

            self.pair_weights_[t_line] = {l: float(w) for l, w in zip(src_lines, best_w)}
            self.pair_c_cells_[t_line] = float(best_c)

            # ---- 2B : Consensus-to-Target Residual Kernel Ridge ----
            X_cons, P_cons, has_cons = self._build_consensus_features(
                t_sub, t_line, X_tr_all, tr_lookup, fit_median_l2=True
            )
            # On aligne sur valid_rows (identique à has_cons)
            Y_c_raw = X_t_all[has_cons][:, self.scored_idx].astype(np.float64)
            tg_c = t_sub["target_gene"].to_numpy()[has_cons]
            Y_check_c, masks_c, Y_c_c, y_norms_c, counts_c = prepare_target_arrays(
                Y_c_raw, tg_c, self.scored_arr, t_line, self.stats_
            )
            nc_t_c = np.maximum(t_sub["n_cells"].to_numpy(dtype=np.float64)[has_cons], 1.0)
            sample_w_c = nc_t_c / (nc_t_c + C_SAMPLE_W)

            cons_model = ResidualDualKernelRidge()
            oof_2b = cons_model.fit(
                X_cons[has_cons],
                Y_check_c,
                P_cons[has_cons],
                masks_c,
                Y_c_c,
                y_norms_c,
                counts_c,
                sample_w_c,
            )
            self.cons_models_[t_line] = cons_model
            diagnostics["cons_summary"].append(
                {
                    "target": t_line,
                    "n": int(has_cons.sum()),
                    "eta": cons_model.best_eta_,
                    "alpha": cons_model.best_alpha_,
                    "beta": cons_model.best_beta_,
                    "loo_pearson": cons_model.best_loo_score_,
                }
            )

            # ---- Blend 2A + 2B appris strictement sur LOO exactes ----
            # (valid_rows et has_cons sont identiques : mêmes lignes ayant >=1 source)
            assert np.array_equal(valid_rows, has_cons)
            A_std = row_standardize_for_blend(best_oof_2a, tg_v, self.scored_arr)
            B_std = row_standardize_for_blend(oof_2b, tg_v, self.scored_arr)

            best_lam = 0.5
            best_blend_sc = -2.0
            for lam in np.linspace(0.0, 1.0, 41):
                P_bl = (1.0 - lam) * A_std + lam * B_std
                sc_bl = vectorized_pearson_mean(P_bl, Y_c_v, y_norms_v, masks_v, counts_v)
                if sc_bl > best_blend_sc:
                    best_blend_sc = sc_bl
                    best_lam = float(lam)

            self.blend_lambda_[t_line] = best_lam
            diagnostics["blend_summary"].append(
                {
                    "target": t_line,
                    "loo_2A_pair": round(best_agg_sc, 5),
                    "loo_2B_cons": round(cons_model.best_loo_score_, 5),
                    "loo_blend": round(best_blend_sc, 5),
                    "lambda_2B": round(best_lam, 3),
                    "c_cells_2A": best_c,
                }
            )

            if self.verbose:
                print(
                    f"  [{t_line:<6}] LOO 2A={best_agg_sc:.5f} | LOO 2B={cons_model.best_loo_score_:.5f} "
                    f"| Blend={best_blend_sc:.5f} (lam_2B={best_lam:.2f}, c={best_c:.0f})"
                )

        return diagnostics

    def predict_components(
        self,
        eval_df: pd.DataFrame,
        src_df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Retourne (preds_2A_pairwise, preds_2B_consensus, preds_blend)."""
        X_src_all, lookup = build_full_source_index(src_df, self.all_genes)
        n_rows = len(eval_df)
        n_sc = len(self.scored_genes)

        preds_2a = np.zeros((n_rows, n_sc), dtype=np.float64)
        preds_2b = np.zeros((n_rows, n_sc), dtype=np.float64)
        preds_blend = np.zeros((n_rows, n_sc), dtype=np.float64)

        eval_lines = eval_df["line"].to_numpy()
        eval_tgs = eval_df["target_gene"].to_numpy()

        for t_line in CORE_LINES:
            row_idx = np.where(eval_lines == t_line)[0]
            if len(row_idx) == 0:
                continue

            sub_eval = eval_df.iloc[row_idx]
            sub_tgs = eval_tgs[row_idx]

            # ---- Prédictions 2A (batchées par lignée source pour la vitesse) ----
            acc_2a = np.zeros((len(row_idx), n_sc), dtype=np.float64)
            w_sum = np.zeros(len(row_idx), dtype=np.float64)
            c_cells = self.pair_c_cells_[t_line]
            w_map = self.pair_weights_[t_line]

            # Regroupe les requêtes par lignée source disponible
            by_src_req: dict[str, list[tuple[int, int, float]]] = {}
            for local_i, tg in enumerate(sub_tgs):
                for s_line, nc, src_pos in lookup.get(tg, []):
                    if s_line != t_line and (s_line, t_line) in self.pair_models_:
                        by_src_req.setdefault(s_line, []).append((local_i, src_pos, nc))

            for s_line, reqs in by_src_req.items():
                loc_arr = np.array([r[0] for r in reqs], dtype=np.int64)
                pos_arr = np.array([r[1] for r in reqs], dtype=np.int64)
                nc_arr = np.array([r[2] for r in reqs], dtype=np.float64)

                X_s_feat = transform_features(X_src_all[pos_arr], s_line, self.stats_)
                P_prior = transform_prior_500(
                    X_src_all[pos_arr][:, self.scored_idx], s_line, t_line, self.stats_
                )
                pred_pair = self.pair_models_[(s_line, t_line)].predict(X_s_feat, P_prior)

                w_nc = (nc_arr / (nc_arr + c_cells)) if c_cells > 0.0 else np.ones_like(nc_arr)
                w_tot = w_map[s_line] * w_nc
                acc_2a[loc_arr] += w_tot[:, None] * pred_pair
                w_sum[loc_arr] += w_tot

            p2a_line = np.divide(
                acc_2a,
                w_sum[:, None],
                out=np.zeros_like(acc_2a),
                where=w_sum[:, None] > 0,
            )
            preds_2a[row_idx] = p2a_line

            # ---- Prédictions 2B (Consensus-to-Target) ----
            X_cons, P_cons, has_cons = self._build_consensus_features(
                sub_eval, t_line, X_src_all, lookup, fit_median_l2=False
            )
            p2b_line = np.zeros((len(row_idx), n_sc), dtype=np.float64)
            if has_cons.any():
                p2b_line[has_cons] = self.cons_models_[t_line].predict(
                    X_cons[has_cons], P_cons[has_cons]
                )
            preds_2b[row_idx] = p2b_line

            # ---- Blend standardisé par ligne ----
            lam = self.blend_lambda_[t_line]
            a_std = row_standardize_for_blend(p2a_line, sub_tgs, self.scored_arr)
            b_std = row_standardize_for_blend(p2b_line, sub_tgs, self.scored_arr)
            preds_blend[row_idx] = (1.0 - lam) * a_std + lam * b_std

        preds_2a = np.nan_to_num(preds_2a, nan=0.0, posinf=0.0, neginf=0.0)
        preds_2b = np.nan_to_num(preds_2b, nan=0.0, posinf=0.0, neginf=0.0)
        preds_blend = np.nan_to_num(preds_blend, nan=0.0, posinf=0.0, neginf=0.0)
        return preds_2a, preds_2b, preds_blend


# ---------------------------------------------------------------------------
# 4. Pipeline principal : Val -> 5-Fold GroupKFold -> Test Submission
# ---------------------------------------------------------------------------

def eval_5fold_cv(
    train_df: pd.DataFrame,
    all_genes: list[str],
    scored_genes: list[str],
    scored_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Index]:
    """5-fold GroupKFold strict sur target_gene (zéro fuite)."""
    folds = make_gene_group_folds(train_df, n_splits=5)
    eval_mask = get_core_eval_mask(train_df)
    eval_indices = train_df.index[eval_mask]
    idx_to_pos = {idx: pos for pos, idx in enumerate(eval_indices)}

    n_eval = len(eval_indices)
    n_sc = len(scored_genes)
    oof_2a = np.zeros((n_eval, n_sc), dtype=np.float64)
    oof_2b = np.zeros((n_eval, n_sc), dtype=np.float64)
    oof_bl = np.zeros((n_eval, n_sc), dtype=np.float64)

    for k in range(5):
        t0 = time.perf_counter()
        tr_k = train_df.loc[folds != k]
        va_k_all = train_df.loc[folds == k]
        va_k_eval = va_k_all.loc[eval_mask.loc[va_k_all.index]]

        model_k = CrossLineRidgeEnsemble(all_genes, scored_genes, scored_idx, verbose=False)
        model_k.fit(tr_k)
        p2a_k, p2b_k, pbl_k = model_k.predict_components(va_k_eval, va_k_all)

        for row_i, orig_idx in enumerate(va_k_eval.index):
            pos = idx_to_pos[orig_idx]
            oof_2a[pos] = p2a_k[row_i]
            oof_2b[pos] = p2b_k[row_i]
            oof_bl[pos] = pbl_k[row_i]

        y_va = va_k_eval[scored_genes].to_numpy(dtype=np.float64)
        tg_va = va_k_eval["target_gene"].to_numpy()
        s2a = rowwise_pearson_scores(y_va, p2a_k, tg_va, scored_genes).mean()
        s2b = rowwise_pearson_scores(y_va, p2b_k, tg_va, scored_genes).mean()
        sbl = rowwise_pearson_scores(y_va, pbl_k, tg_va, scored_genes).mean()
        dt = time.perf_counter() - t0
        print(
            f"  Fold {k} ({len(va_k_eval):>4d} lignes) | "
            f"2A={s2a:.5f} | 2B={s2b:.5f} | Blend={sbl:.5f}  ({dt:.1f}s)"
        )

    return oof_2a, oof_2b, oof_bl, eval_indices


def main() -> None:
    t_start = time.perf_counter()
    train_df = load_train()
    test_df = load_test()
    _, all_genes, scored_genes = load_genes()
    sample_sub = load_sample_submission()

    gene_to_idx = {g: i for i, g in enumerate(all_genes)}
    scored_idx = np.array([gene_to_idx[g] for g in scored_genes], dtype=np.int64)

    # ------------------------------------------------------------------
    # 1. Évaluation sur fold == 'val' (fit strictement sur fold == 'train')
    # ------------------------------------------------------------------
    print("=== 1. Fit sur fold == 'train' et évaluation sur fold == 'val' ===")
    tr_sub = train_df.loc[train_df["fold"].eq("train")].reset_index(drop=True)
    va_sub = train_df.loc[train_df["fold"].eq("val")].reset_index(drop=True)
    Y_val = va_sub[scored_genes].to_numpy(dtype=np.float64)
    tg_val = va_sub["target_gene"].to_numpy()

    model_val = CrossLineRidgeEnsemble(all_genes, scored_genes, scored_idx, verbose=True)
    diag_val = model_val.fit(tr_sub)
    val_2a, val_2b, val_bl = model_val.predict_components(va_sub, tr_sub)

    sc_2a = rowwise_pearson_scores(Y_val, val_2a, tg_val, scored_genes)
    sc_2b = rowwise_pearson_scores(Y_val, val_2b, tg_val, scored_genes)
    sc_bl = rowwise_pearson_scores(Y_val, val_bl, tg_val, scored_genes)

    print(f"\n>>> Val Pearson — 2A (Pairwise Kernel Ridge)  : {sc_2a.mean():.5f}")
    print(f">>> Val Pearson — 2B (Consensus Kernel Ridge) : {sc_2b.mean():.5f}")
    print(f">>> Val Pearson — 02 Blend (2A + 2B)          : {sc_bl.mean():.5f}")

    val_table = pd.DataFrame(
        {"line": va_sub["line"].values, "2A_pair": sc_2a, "2B_cons": sc_2b, "02_blend": sc_bl}
    )
    print("\nDétail par lignée sur fold == 'val' :")
    print(val_table.groupby("line").mean().round(5))

    print("\nHyperparamètres LOO sélectionnés (2B Consensus) :")
    print(pd.DataFrame(diag_val["cons_summary"]).to_string(index=False))

    print("\nPoids d'agrégation 2A appris w(source -> target) :")
    print(pd.DataFrame(model_val.pair_weights_).T.round(3))

    # ------------------------------------------------------------------
    # 2. Validation croisée 5-fold GroupKFold (zéro fuite de target_gene)
    # ------------------------------------------------------------------
    print("\n=== 2. Validation croisée 5-fold GroupKFold ===")
    oof_2a, oof_2b, oof_bl, eval_indices = eval_5fold_cv(
        train_df, all_genes, scored_genes, scored_idx
    )
    eval_df = train_df.loc[eval_indices]
    Y_oof = eval_df[scored_genes].to_numpy(dtype=np.float64)
    tg_oof = eval_df["target_gene"].to_numpy()

    cv_2a = rowwise_pearson_scores(Y_oof, oof_2a, tg_oof, scored_genes).mean()
    cv_2b = rowwise_pearson_scores(Y_oof, oof_2b, tg_oof, scored_genes).mean()
    cv_bl = rowwise_pearson_scores(Y_oof, oof_bl, tg_oof, scored_genes).mean()

    print(f"\n>>> 5-Fold OOF ({len(eval_indices):,} entrées core) :")
    print(f"    2A Pairwise Kernel Ridge  : {cv_2a:.5f}")
    print(f"    2B Consensus Kernel Ridge : {cv_2b:.5f}")
    print(f"    02 Blend (2A + 2B)        : {cv_bl:.5f}")

    # Sauvegarde OOF alignés sur artifacts/oof_index.csv pour le futur stacking
    ART_DIR.mkdir(parents=True, exist_ok=True)
    np.save(ART_DIR / "oof_02_ridge_pairwise.npy", oof_2a.astype(np.float32))
    np.save(ART_DIR / "oof_02_ridge_consensus.npy", oof_2b.astype(np.float32))
    np.save(ART_DIR / "oof_02_ridge_translation.npy", oof_bl.astype(np.float32))

    oof_index_path = ART_DIR / "oof_index.csv"
    if not oof_index_path.exists():
        eval_df[["id", "line", "target_gene"]].to_csv(oof_index_path, index=False)

    # Comparaison / sanity check si oof_01_weighted_norm existe déjà
    oof_01_path = ART_DIR / "oof_01_weighted_norm.npy"
    if oof_01_path.exists():
        oof_01 = np.load(oof_01_path).astype(np.float64)
        if oof_01.shape == oof_bl.shape:
            cv_01 = rowwise_pearson_scores(Y_oof, oof_01, tg_oof, scored_genes).mean()
            print(f"    Comparaison vs 01_weighted_norm OOF : {cv_01:.5f} (gain: {cv_bl - cv_01:+.5f})")

    # ------------------------------------------------------------------
    # 3. Ré-entraînement sur 100% de train.parquet et soumission test
    # ------------------------------------------------------------------
    print("\n=== 3. Fit final sur 100% de train.parquet et génération soumission ===")
    model_full = CrossLineRidgeEnsemble(all_genes, scored_genes, scored_idx, verbose=True)
    model_full.fit(train_df)
    _, _, test_preds = model_full.predict_components(test_df, train_df)

    sub = pd.DataFrame(test_preds, columns=scored_genes)
    sub.insert(0, "id", test_df["id"].values)
    sub = sub[sample_sub.columns]

    assert len(sub) == len(sample_sub), f"Attendu {len(sample_sub)} lignes, reçu {len(sub)}"
    assert list(sub.columns) == list(sample_sub.columns), "Colonnes non alignées"
    assert (sub["id"].values == sample_sub["id"].values).all(), "IDs non alignés"
    assert np.isfinite(sub.iloc[:, 1:].to_numpy(dtype=np.float64)).all(), "NaN/Inf détectés !"

    SUB_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SUB_DIR / "02_ridge_translation.csv"
    sub.to_csv(out_path, index=False)
    print(
        f"\nSoumission vérifiée et sauvegardée : {out_path} {sub.shape} "
        f"(temps total : {time.perf_counter() - t_start:.1f}s)"
    )


if __name__ == "__main__":
    main()