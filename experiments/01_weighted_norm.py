from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.cv import CORE_LINES, get_core_eval_mask, make_gene_group_folds, rowwise_pearson_scores
from src.data import load_genes, load_sample_submission, load_test, load_train

ROOT = Path(__file__).resolve().parents[1]
SUB_DIR = ROOT / "submissions"
ART_DIR = ROOT / "artifacts"


def fit_line_stats(src_df: pd.DataFrame, scored_genes: list[str]) -> dict:
    stats = {}
    for line, grp in src_df.groupby("line"):
        M = grp[scored_genes].to_numpy(dtype=np.float64)
        M_zero = np.nan_to_num(M, nan=0.0)
        stats[line] = {
            "global_std": float(np.nanstd(M)),
            "gene_std": np.nanstd(M, axis=0),
            "median_row_l2": float(np.median(np.linalg.norm(M_zero, axis=1))),
        }
    return stats


def build_source_lookup(src_df: pd.DataFrame, scored_genes: list[str]) -> dict:
    """target_gene -> list[(line, n_cells, vec_500)]"""
    Y = src_df[scored_genes].to_numpy(dtype=np.float64)
    lookup = {}
    for idx, (tg, line, nc) in enumerate(
        zip(src_df["target_gene"], src_df["line"], src_df["n_cells"])
    ):
        lookup.setdefault(tg, []).append((line, float(nc), Y[idx]))
    return lookup


def transform_source_vec(
    v: np.ndarray,
    s_line: str,
    t_line: str,
    stats: dict,
    mode: str,
    gene_eps: float = 0.02,
    row_tau: float = 0.5,
) -> np.ndarray:
    if mode == "raw":
        return v
    if mode == "line_std":
        return v / max(stats[s_line]["global_std"], 1e-8)
    if mode == "row_norm":
        norm = np.sqrt(np.sum(v ** 2) + (row_tau * stats[s_line]["median_row_l2"]) ** 2)
        return v / max(norm, 1e-8)
    if mode == "gene_std":
        t_gstd = stats[t_line]["gene_std"]
        s_gstd = stats[s_line]["gene_std"]
        return v * ((t_gstd + gene_eps) / (s_gstd + gene_eps))
    if mode == "gene_and_row":
        t_gstd = stats[t_line]["gene_std"]
        s_gstd = stats[s_line]["gene_std"]
        v_g = v * ((t_gstd + gene_eps) / (s_gstd + gene_eps))
        norm = np.sqrt(np.sum(v ** 2) + (row_tau * stats[s_line]["median_row_l2"]) ** 2)
        return v_g / max(norm, 1e-8)
    raise ValueError(f"Mode inconnu : {mode}")


def predict_weighted(
    eval_df: pd.DataFrame,
    lookup: dict,
    stats: dict,
    n_scored: int,
    mode: str = "raw",
    weights_by_target: dict | None = None,
    c_cells: float = 0.0,
    gene_eps: float = 0.02,
    row_tau: float = 0.5,
) -> np.ndarray:
    preds = np.zeros((len(eval_df), n_scored), dtype=np.float64)
    for i, (t_line, tg) in enumerate(zip(eval_df["line"], eval_df["target_gene"])):
        sources = [s for s in lookup.get(tg, []) if s[0] != t_line]
        if not sources:
            continue
        num = np.zeros(n_scored, dtype=np.float64)
        den = np.zeros(n_scored, dtype=np.float64)
        w_map = weights_by_target.get(t_line, {}) if weights_by_target else {}

        for s_line, nc, vec in sources:
            w_pair = w_map.get(s_line, 1.0)
            w_nc = (nc / (nc + c_cells)) if c_cells > 0.0 else 1.0
            w = w_pair * w_nc

            valid = np.isfinite(vec)
            v = np.where(valid, vec, 0.0)
            v_tr = transform_source_vec(v, s_line, t_line, stats, mode, gene_eps, row_tau)
            num += w * v_tr
            den += w * valid

        preds[i] = np.divide(num, den, out=np.zeros(n_scored, dtype=np.float64), where=den > 0)
    return preds


def fit_pairwise_weights(
    tr_df: pd.DataFrame,
    scored_genes: list[str],
    stats: dict,
    mode: str = "gene_std",
    c_cells: float = 50.0,
    gene_eps: float = 0.02,
    row_tau: float = 0.5,
) -> dict:
    """
    Apprend 5 poids positifs w(source -> target) par lignée cible en maximisant
    directement le Pearson en Leave-One-Line-Out sur tr_df uniquement.
    """
    all_lines = sorted(tr_df["line"].unique().tolist())
    lookup = build_source_lookup(tr_df, scored_genes)
    scored_arr = np.asarray(scored_genes)
    n_scored = len(scored_genes)
    weights_by_target = {}

    for t_line in CORE_LINES:
        sub = tr_df.loc[tr_df["line"].eq(t_line)].reset_index(drop=True)
        src_lines = [l for l in all_lines if l != t_line]
        l2idx = {l: j for j, l in enumerate(src_lines)}

        S_mat = np.zeros((len(sub), len(src_lines), n_scored), dtype=np.float64)
        Y_true = sub[scored_genes].to_numpy(dtype=np.float64)
        t_genes = sub["target_gene"].to_numpy()

        valid_rows = []
        for i, tg in enumerate(t_genes):
            sources = [s for s in lookup.get(tg, []) if s[0] != t_line]
            if not sources:
                continue
            valid_rows.append(i)
            for s_line, nc, vec in sources:
                j = l2idx[s_line]
                w_nc = (nc / (nc + c_cells)) if c_cells > 0.0 else 1.0
                v = np.nan_to_num(vec, nan=0.0)
                v_tr = transform_source_vec(v, s_line, t_line, stats, mode, gene_eps, row_tau)
                S_mat[i, j] = w_nc * v_tr

        S_mat = S_mat[valid_rows]
        Y_true = Y_true[valid_rows]
        t_genes = t_genes[valid_rows]

        masks = (scored_arr[None, :] != t_genes[:, None]) & np.isfinite(Y_true)
        Y_clean = np.where(masks, Y_true, 0.0)
        counts = masks.sum(axis=1, keepdims=True)
        Y_c = np.where(masks, Y_clean - Y_clean.sum(axis=1, keepdims=True) / counts, 0.0)
        y_norms = np.linalg.norm(Y_c, axis=1)

        def loss_fn(log_w: np.ndarray) -> float:
            w = np.exp(np.clip(log_w, -5.0, 5.0))
            P = np.tensordot(S_mat, w, axes=([1], [0]))
            P_clean = np.where(masks, P, 0.0)
            P_c = np.where(masks, P_clean - P_clean.sum(axis=1, keepdims=True) / counts, 0.0)
            denom = np.linalg.norm(P_c, axis=1) * y_norms
            corrs = np.divide(
                np.sum(P_c * Y_c, axis=1),
                denom,
                out=np.zeros(len(Y_c), dtype=np.float64),
                where=denom > 0,
            )
            return -float(corrs.mean()) + 1e-4 * float(np.sum(log_w ** 2))

        res = minimize(loss_fn, np.zeros(len(src_lines)), method="Nelder-Mead", options={"maxiter": 800})
        best_w = np.exp(np.clip(res.x, -5.0, 5.0))
        best_w = best_w / best_w.mean()
        weights_by_target[t_line] = {l: float(w) for l, w in zip(src_lines, best_w)}

    return weights_by_target


def eval_5fold_cv(
    train_df: pd.DataFrame,
    scored_genes: list[str],
    mode: str,
    c_cells: float,
    use_opt_weights: bool = True,
    gene_eps: float = 0.02,
    row_tau: float = 0.5,
) -> tuple[float, np.ndarray, pd.Index]:
    """5-fold GroupKFold sur target_gene : zéro fuite de gènes pour les stats et poids."""
    folds = make_gene_group_folds(train_df, n_splits=5)
    eval_mask = get_core_eval_mask(train_df)
    eval_indices = train_df.index[eval_mask]

    oof_preds = np.zeros((len(eval_indices), len(scored_genes)), dtype=np.float64)
    idx_to_pos = {idx: pos for pos, idx in enumerate(eval_indices)}

    for k in range(5):
        tr_k = train_df.loc[folds != k]
        va_k_all = train_df.loc[folds == k]
        va_k_eval = va_k_all.loc[eval_mask.loc[va_k_all.index]]

        stats_k = fit_line_stats(tr_k, scored_genes)
        w_k = (
            fit_pairwise_weights(tr_k, scored_genes, stats_k, mode, c_cells, gene_eps, row_tau)
            if use_opt_weights
            else None
        )
        # Les sources des gènes de va_k sont leurs autres lignées (excluant la lignée cible)
        lookup_va = build_source_lookup(va_k_all, scored_genes)
        preds_k = predict_weighted(
            va_k_eval, lookup_va, stats_k, len(scored_genes), mode, w_k, c_cells, gene_eps, row_tau
        )
        for row_i, orig_idx in enumerate(va_k_eval.index):
            oof_preds[idx_to_pos[orig_idx]] = preds_k[row_i]

    eval_df = train_df.loc[eval_indices]
    scores = rowwise_pearson_scores(
        eval_df[scored_genes].to_numpy(dtype=np.float64),
        oof_preds,
        eval_df["target_gene"].to_numpy(),
        scored_genes,
    )
    return float(scores.mean()), oof_preds, eval_indices


def main() -> None:
    train_df = load_train()
    test_df = load_test()
    _, _, scored_genes = load_genes()
    sample_sub = load_sample_submission()

    tr_sub = train_df.loc[train_df["fold"].eq("train")].reset_index(drop=True)
    va_sub = train_df.loc[train_df["fold"].eq("val")].reset_index(drop=True)
    Y_val = va_sub[scored_genes].to_numpy(dtype=np.float64)
    tg_val = va_sub["target_gene"].to_numpy()

    stats_tr = fit_line_stats(tr_sub, scored_genes)
    lookup_tr = build_source_lookup(tr_sub, scored_genes)

    print("=== 1. Ablation rapide sur fold == 'val' (poids égaux, c_cells=0) ===")
    for mode in ["raw", "line_std", "row_norm", "gene_std", "gene_and_row"]:
        preds = predict_weighted(va_sub, lookup_tr, stats_tr, len(scored_genes), mode=mode, c_cells=0.0)
        sc = rowwise_pearson_scores(Y_val, preds, tg_val, scored_genes).mean()
        print(f"  mode={mode:<14} | val Pearson = {sc:.5f}")

    print("\n=== 2. Recherche (mode, c_cells) avec poids appris 6x4 sur fold == 'train' ===")
    best_cfg = None
    best_val = -1.0
    for mode in ["line_std", "row_norm", "gene_std", "gene_and_row"]:
        for c_cells in [0.0, 25.0, 60.0, 120.0]:
            w_tr = fit_pairwise_weights(tr_sub, scored_genes, stats_tr, mode=mode, c_cells=c_cells)
            preds = predict_weighted(
                va_sub, lookup_tr, stats_tr, len(scored_genes),
                mode=mode, weights_by_target=w_tr, c_cells=c_cells,
            )
            sc = rowwise_pearson_scores(Y_val, preds, tg_val, scored_genes).mean()
            print(f"  mode={mode:<14} | c={c_cells:>5.1f} | val Pearson = {sc:.5f}")
            if sc > best_val:
                best_val = sc
                best_cfg = (mode, c_cells, w_tr, preds)

    best_mode, best_c, best_w_tr, best_val_preds = best_cfg
    print(f"\n>>> Meilleure config sur val : mode={best_mode}, c_cells={best_c} -> {best_val:.5f} <<<")

    # Détail par lignée sur val
    val_scores = rowwise_pearson_scores(Y_val, best_val_preds, tg_val, scored_genes)
    res_df = pd.DataFrame({"line": va_sub["line"].values, "pearson": val_scores})
    print("\nDétail par lignée sur fold == 'val' :")
    print(res_df.groupby("line")["pearson"].agg(["count", "mean", "std"]).round(5))

    print("\nPoids appris w(source -> target) sur fold == 'train' :")
    print(pd.DataFrame(best_w_tr).T.round(3))

    # Validation croisée 5-fold GroupKFold (zéro fuite de gènes)
    cv_raw, _, _ = eval_5fold_cv(train_df, scored_genes, mode="raw", c_cells=0.0, use_opt_weights=False)
    cv_best, oof_preds, oof_idx = eval_5fold_cv(
        train_df, scored_genes, mode=best_mode, c_cells=best_c, use_opt_weights=True
    )
    print(f"\n=== 3. Validation croisée 5-fold GroupKFold ({len(oof_idx):,} entrées core) ===")
    print(f"  00_baseline (raw)        : {cv_raw:.5f}")
    print(f"  01_weighted_norm (best)  : {cv_best:.5f}  (gain: {cv_best - cv_raw:+.5f})")

    # Sauvegarde OOF pour le futur stacking
    ART_DIR.mkdir(parents=True, exist_ok=True)
    np.save(ART_DIR / "oof_01_weighted_norm.npy", oof_preds.astype(np.float32))
    train_df.loc[oof_idx, ["id", "line", "target_gene"]].to_csv(
        ART_DIR / "oof_index.csv", index=False
    )

    # Ré-entraînement sur 100% de train.parquet et génération soumission test
    stats_full = fit_line_stats(train_df, scored_genes)
    w_full = fit_pairwise_weights(train_df, scored_genes, stats_full, mode=best_mode, c_cells=best_c)
    lookup_full = build_source_lookup(train_df, scored_genes)
    test_preds = predict_weighted(
        test_df, lookup_full, stats_full, len(scored_genes),
        mode=best_mode, weights_by_target=w_full, c_cells=best_c,
    )

    sub = pd.DataFrame(test_preds, columns=scored_genes)
    sub.insert(0, "id", test_df["id"].values)
    sub = sub[sample_sub.columns]
    assert len(sub) == len(sample_sub)
    assert (sub["id"].values == sample_sub["id"].values).all()
    assert np.isfinite(sub.iloc[:, 1:].to_numpy(dtype=np.float64)).all()

    SUB_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SUB_DIR / "01_weighted_norm.csv"
    sub.to_csv(out_path, index=False)
    print(f"\nSoumission sauvegardée et vérifiée : {out_path} {sub.shape}")


if __name__ == "__main__":
    main()