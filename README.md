# INF8245AE — Kaggle Perturb-seq (Automne 2026)

Compétition Kaggle du cours **INF8245AE (Machine Learning, Polytechnique Montréal)**.

## Objectif

Prédire la variation d'expression de **500 gènes scorés** (`scored = True` dans `genes.csv`) lorsqu'un gène (`target_gene`) est inhibé par CRISPRi dans une lignée cellulaire cible (`K562`, `RPE1`, `Jurkat` ou `HepG2`), à partir des mesures de ce même knock-down dans d'autres lignées cellulaires et de l'expression basale des lignées.

* **Métrique officielle :** corrélation de Pearson moyenne par entrée sur les 500 gènes scorés (en excluant `target_gene` s'il fait partie des 500 ; une ligne constante vaut `0`).
* **Split Leaderboard :** public/privé séparés sur des `target_gene` différents ($\approx 50\,\% / 50\,\%$).
* **Règles strictes :** utiliser **uniquement** les fichiers fournis dans `data/`. Aucune donnée biologique externe, aucune annotation externe et aucun modèle/embedding biologique pré-entraîné.

---

## Arborescence du dépôt

```text
.
├── .gitignore
├── README.md
├── requirements.txt
├── data/                     
├── submissions/              # CSV soumis sur Kaggle
├── notebooks/
│   └── 01_eda.ipynb          # Exploration et graphiques du rapport
├── src/
│   ├── __init__.py
│   ├── config.py             # Chemins, constantes, seeds
│   ├── data.py               # Chargement optimisé des données
│   ├── metrics.py            # Métrique Pearson officielle + tests
│   └── submission.py         # Vérification stricte et export CSV
└── experiments/
    └── 00_baseline.py        # Baseline moyenne inter-lignées + vérif sample_submission
```


---

## Historique

| ID | Modèle / Idée | Features / Extra lines | Val Pearson (global) | K562 | RPE1 | Jurkat | HepG2 | Public LB | Auteur |
|---|---|---|---|---|---|---|---|---|---|
| `00a` | Mean other lines | 4 lignées (`extra_line=False`) | _à remplir_ | - | - | - | - | - | - |
| `00b` | Mean other lines | 6 lignées (`extra_line=True`) | _à remplir_ | - | - | - | - | - | - |