import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

CORE_LINES = ["K562", "RPE1", "Jurkat", "HepG2"]


def make_gene_group_folds(train_df: pd.DataFrame, n_splits: int = 5) -> pd.Series:
    """
    Assigne un numéro de fold (0..n_splits-1) à chaque ligne de train_df,
    groupé strictement par target_gene (aucun gène ne traverse deux folds).
    """
    gkf = GroupKFold(n_splits=n_splits)
    folds = pd.Series(-1, index=train_df.index, dtype=int)
    for k, (_, val_idx) in enumerate(gkf.split(train_df, groups=train_df["target_gene"])):
        folds.iloc[val_idx] = k
    return folds


def get_core_eval_mask(train_df: pd.DataFrame) -> pd.Series:
    """
    Masque des entrées évaluables en Leave-One-Line-Out :
    lignées core (K562, RPE1, Jurkat, HepG2) dont le target_gene est mesuré
    dans au moins 2 lignées core (pour qu'il reste >=1 lignée core source, comme val/test).
    """
    core_counts = (
        train_df.loc[~train_df["extra_line"]]
        .groupby("target_gene")["line"]
        .nunique()
    )
    has_core_src = train_df["target_gene"].map(core_counts).fillna(0) >= 2
    return (~train_df["extra_line"]) & has_core_src


def rowwise_pearson_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_genes: np.ndarray,
    scored_genes: list[str],
) -> np.ndarray:
    """Calcul exact de la métrique officielle par ligne (en sautant target_gene)."""
    scored_arr = np.asarray(scored_genes)
    scores = np.zeros(len(y_true), dtype=np.float64)
    for i in range(len(y_true)):
        mask = (scored_arr != target_genes[i]) & np.isfinite(y_true[i])
        t = y_true[i, mask]
        p = y_pred[i, mask]
        tc = t - t.mean()
        pc = p - p.mean()
        denom = np.linalg.norm(tc) * np.linalg.norm(pc)
        scores[i] = 0.0 if denom == 0.0 else float(np.dot(tc, pc) / denom)
    return scores