from __future__ import annotations

import pandas as pd

from src.config import DATA_DIR, META_COLS


def load_genes() -> tuple[pd.DataFrame, list[str], list[str]]:
    """Charge genes.csv et renvoie (df, all_gene_ids, scored_gene_ids)."""
    genes = pd.read_csv(DATA_DIR / "genes.csv")
    all_genes = genes["gene_id"].tolist()
    scored_genes = genes.loc[genes["scored"].astype(bool), "gene_id"].tolist()
    assert len(scored_genes) == 500, f"Attendu 500 gènes scorés, trouvé {len(scored_genes)}"
    return genes, all_genes, scored_genes


def load_train(
    only_scored_genes: bool = False,
    include_extra_lines: bool = True,
) -> pd.DataFrame:
    """Charge train.parquet en limitant la RAM si only_scored_genes=True."""
    columns = None
    if only_scored_genes:
        _, _, scored_genes = load_genes()
        columns = list(META_COLS) + scored_genes

    df = pd.read_parquet(DATA_DIR / "train.parquet", columns=columns)

    if not include_extra_lines:
        df = df.loc[~df["extra_line"].astype(bool)].copy()

    return df.reset_index(drop=True)


def load_test() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "test.csv")


def load_baseline_expression() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "baseline_expression.csv")


def load_line_mean_response() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "line_mean_response.csv")


def load_sample_submission() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "sample_submission.csv")
