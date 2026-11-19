from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import load_genes, load_sample_submission, load_test, load_train
from src.metrics import evaluate_by_line
from src.submission import build_submission_df, save_and_verify_submission


def predict_mean_other_lines(
    source_df: pd.DataFrame,
    query_df: pd.DataFrame,
    scored_genes: list[str],
) -> tuple[pd.DataFrame, int]:
    """
    Prédit, pour chaque ligne de query_df, la moyenne des réponses du même
    target_gene observées dans source_df (dans d'autres lignées).
    """
    mean_by_gene = source_df.groupby("target_gene")[scored_genes].mean()
    preds = mean_by_gene.reindex(query_df["target_gene"].to_numpy())
    preds.index = query_df.index

    # Nombre d'entrées de query_df sans aucune lignée source disponible
    n_missing = int(preds.isna().all(axis=1).sum())

    # Remplace les NaN (notamment la colonne du target_gene lui-même) par 0.0
    preds = preds.fillna(0.0)
    return preds, n_missing


def main() -> None:
    _, _, scored_genes = load_genes()
    train_all = load_train(only_scored_genes=True, include_extra_lines=True)
    test_df = load_test()
    sample_sub = load_sample_submission()

    tr_mask = train_all["fold"].eq("train")
    va_mask = train_all["fold"].eq("val")

    val_df = train_all.loc[va_mask].reset_index(drop=True)
    y_val = val_df[scored_genes]

    print(f"Train total : {len(train_all):,} lignes | fold=='train' : {tr_mask.sum():,} | fold=='val' : {va_mask.sum():,}")
    print(f"Test        : {len(test_df):,} lignes")
    print()

    # 1. Évaluation sur val SANS extra_line vs AVEC extra_line
    for include_extra in (False, True):
        src = train_all.loc[tr_mask & (include_extra | ~train_all["extra_line"].astype(bool))]
        preds_val, n_missing = predict_mean_other_lines(src, val_df, scored_genes)
        global_score, by_line = evaluate_by_line(val_df, y_val, preds_val)

        label = "AVEC extra_line (6 lignées)" if include_extra else "SANS extra_line (4 lignées)"
        print(f"=== Baseline Val — {label} ===")
        print(f"Entrées sans source : {n_missing} | Score Pearson global : {global_score:.5f}")
        print(by_line.round(5))
        print()

    # 2. Comparaison avec sample_submission.csv sur test
    for include_extra in (False, True):
        src = train_all if include_extra else train_all.loc[~train_all["extra_line"].astype(bool)]
        preds_test, n_missing = predict_mean_other_lines(src, test_df, scored_genes)
        sub_df = build_submission_df(test_df["id"], preds_test, scored_genes)

        # Compare uniquement les colonnes numériques alignées sur sample_submission
        sub_aligned = sub_df.set_index("id").loc[sample_sub["id"], scored_genes].to_numpy()
        ref_vals = sample_sub[scored_genes].to_numpy()
        max_diff = float(np.max(np.abs(sub_aligned - ref_vals)))
        mean_diff = float(np.mean(np.abs(sub_aligned - ref_vals)))

        label = "6 lignées" if include_extra else "4 lignées"
        print(f"[Sanity check vs sample_submission — {label}] missing={n_missing}, mean_abs_diff={mean_diff:.6f}, max_abs_diff={max_diff:.6f}")

    # 3. Export d'une première soumission vérifiée (ici avec toutes les données train+val)
    preds_test_full, _ = predict_mean_other_lines(train_all, test_df, scored_genes)
    sub_full = build_submission_df(test_df["id"], preds_test_full, scored_genes)
    # Aligne explicitement sur l'ordre de sample_submission.csv
    sub_full = sub_full.set_index("id").loc[sample_sub["id"]].reset_index()
    save_and_verify_submission(sub_full, filename="00_baseline.csv")


if __name__ == "__main__":
    main()
