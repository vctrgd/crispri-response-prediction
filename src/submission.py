from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.config import DATA_DIR, SUBMISSIONS_DIR


def build_submission_df(
    ids: pd.Series | np.ndarray,
    pred_matrix: np.ndarray | pd.DataFrame,
    scored_genes: list[str],
) -> pd.DataFrame:
    """Construit un DataFrame au format exact attendu par Kaggle."""
    if isinstance(pred_matrix, pd.DataFrame):
        values = pred_matrix[scored_genes].to_numpy(dtype=np.float64)
    else:
        values = np.asarray(pred_matrix, dtype=np.float64)

    if values.shape != (len(ids), len(scored_genes)):
        raise ValueError(
            f"Shape invalide : attendu {(len(ids), len(scored_genes))}, reçu {values.shape}"
        )

    sub = pd.DataFrame(values, columns=scored_genes)
    sub.insert(0, "id", np.asarray(ids))
    return sub


def save_and_verify_submission(
    preds: pd.DataFrame,
    filename: str,
    sample_sub_path: Path | None = None,
    out_dir: Path | None = None,
) -> Path:
    """Vérifie strictement le format contre sample_submission.csv puis exporte."""
    sample_path = sample_sub_path or (DATA_DIR / "sample_submission.csv")
    output_dir = out_dir or SUBMISSIONS_DIR
    out_path = output_dir / filename

    sample = pd.read_csv(sample_path)

    assert list(preds.columns) == list(sample.columns), "Colonnes différentes de sample_submission.csv"
    assert len(preds) == len(sample), f"Attendu {len(sample)} lignes, reçu {len(preds)}"
    assert preds["id"].is_unique, "IDs dupliqués détectés !"
    assert (preds["id"].to_numpy() == sample["id"].to_numpy()).all(), "Ordre ou valeurs d'IDs différents"

    values = preds.iloc[:, 1:].to_numpy(dtype=np.float64)
    assert np.isfinite(values).all(), "La soumission contient des NaN ou Inf !"

    zero_var_rows = int((np.ptp(values, axis=1) == 0.0).sum())
    if zero_var_rows > 0:
        print(f"[ATTENTION] {zero_var_rows} ligne(s) constante(s) -> score 0.0 sur ces entrées.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out_path, index=False)
    print(f"Soumission vérifiée et sauvegardée : {out_path} {preds.shape}")
    return out_path
