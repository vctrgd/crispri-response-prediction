from __future__ import annotations

import numpy as np
import pandas as pd


def pearson_per_row(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    target_genes: pd.Series,
) -> np.ndarray:
    """
    Calcule la corrélation de Pearson ligne par ligne sur les 500 gènes scorés.
    - Ignore le gène knock-downé (target_gene) s'il est dans les colonnes.
    - Une ligne prédite constante obtient un score de 0.0.
    - Lève une erreur si y_pred contient des NaN ou Inf.
    """
    if not y_true.index.equals(y_pred.index):
        raise ValueError("Les index de y_true et y_pred doivent être identiques.")
    if list(y_true.columns) != list(y_pred.columns):
        raise ValueError("Les colonnes de y_true et y_pred doivent être identiques et dans le même ordre.")
    if len(target_genes) != len(y_true):
        raise ValueError("target_genes doit avoir la même longueur que y_true.")

    cols = np.asarray(y_true.columns)
    T = y_true.to_numpy(dtype=np.float64)
    P = y_pred.to_numpy(dtype=np.float64)
    targets = np.asarray(target_genes)

    if not np.isfinite(P).all():
        raise ValueError("y_pred contient des valeurs NaN ou infinies.")

    scores = np.zeros(len(T), dtype=np.float64)

    for i in range(len(T)):
        mask = (cols != targets[i]) & np.isfinite(T[i])
        t = T[i, mask]
        p = P[i, mask]

        p_c = p - p.mean()
        t_c = t - t.mean()
        denom = np.linalg.norm(p_c) * np.linalg.norm(t_c)

        scores[i] = 0.0 if denom == 0.0 else float(np.dot(p_c, t_c) / denom)

    return scores


def pearson_competition_score(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    target_genes: pd.Series,
) -> float:
    """Score officiel : moyenne des corrélations de Pearson par entrée."""
    return float(pearson_per_row(y_true, y_pred, target_genes).mean())


def evaluate_by_line(
    meta_df: pd.DataFrame,
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
) -> tuple[float, pd.DataFrame]:
    """Renvoie le score global et le détail par lignée cellulaire."""
    row_scores = pearson_per_row(y_true, y_pred, meta_df["target_gene"])
    summary = (
        pd.DataFrame({"line": meta_df["line"].to_numpy(), "pearson": row_scores})
        .groupby("line", as_index=True)["pearson"]
        .agg(["count", "mean", "std"])
        .sort_index()
    )
    return float(row_scores.mean()), summary


if __name__ == "__main__":
    # Tests rapides de non-régression sur la métrique
    cols = ["G1", "G2", "G3", "G4"]
    yt = pd.DataFrame(
        [
            [np.nan, 1.0, 2.0, 3.0],  # G1 est knock-downé -> ignoré
            [1.0, 2.0, 3.0, 4.0],     # prédiction constante -> 0.0
            [1.0, -1.0, 2.0, -2.0],   # prédiction inversée -> -1.0
        ],
        columns=cols,
    )
    yp = pd.DataFrame(
        [
            [999.0, 2.0, 4.0, 6.0],   # 999.0 doit être ignoré -> +1.0
            [5.0, 5.0, 5.0, 5.0],     # constante -> 0.0
            [-1.0, 1.0, -2.0, 2.0],   # opposée -> -1.0
        ],
        columns=cols,
    )
    tg = pd.Series(["G1", "G99", "G99"])
    s = pearson_per_row(yt, yp, tg)
    assert np.allclose(s, [1.0, 0.0, -1.0]), f"Échec du test métrique : {s}"
    print("Tests unitaires src/metrics.py OK :", s)