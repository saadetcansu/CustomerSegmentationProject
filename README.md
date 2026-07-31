# Predicting Marketing Campaign Response
### Supervised Data Mining Pipeline using CRISP-DM

> **Course:** BIL 476 Data Mining  
> **Project Type:** Individual Assignment  

---

## Project Overview

This project applies supervised machine learning techniques to predict whether a customer will respond positively to a marketing campaign based on demographics, purchase history, and past campaign interactions. The project strictly follows the CRISP-DM methodology.

### Methodology and Algorithms

The project compares a variety of classification algorithms, satisfying BIL 476 Topic 7 requirements:

| Algorithm Family | Model Implemented | Key Characteristics |
|---|---|---|
| Probabilistic | Naive Bayes (GaussianNB) | Assumes conditional independence; fast baseline. |
| Distance-based | k-Nearest Neighbors (k-NN) | `k` optimized via Cross-Validation (`GridSearchCV`). |
| Tree-based | Decision Tree | Interpretable, non-linear splits. |
| Ensembles | Random Forest, Gradient Boosting, XGBoost, LightGBM | High predictive power, handles class imbalance well. |

### Evaluation Metrics

Given the class imbalance in the dataset, models are rigorously evaluated using:
- Accuracy & Balanced Accuracy
- Precision, Recall, and F1-Score
- ROC-AUC and Precision-Recall AUC (PR-AUC)
- Confusion Matrices

---

## Repository Structure

```
CustomerSegmentationProject/
├── data/                  # Raw and engineered datasets (not committed to git)
├── BIL476_Project.ipynb   # Final Master Notebook
├── notebooks/             # Additional modular Jupyter notebooks
├── src/                   # Python generator scripts for notebooks
├── outputs/               # Generated figures and tables
│   ├── bil476_final/      # Outputs from the final master notebook
│   └── crisp_dm_ablation/ # Ablation study outputs
├── README.md              # This file
└── requirements.txt       # Python dependencies
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd CustomerSegmentationProject
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Execute the Final Pipeline

The entire CRISP-DM pipeline is consolidated into a single master notebook.

```bash
jupyter lab
```

Open `BIL476_Project.ipynb` and select **Run All**. The notebook will execute the Exploratory Data Mining, Feature Engineering, Feature Selection, Modeling (including k-NN Cross-Validation), SHAP Interpretation, and final Business Deployment phases sequentially.

---

## Reproducibility

- Global random seed: **42** (set via `random_state=42`)
- Class imbalance handled via `class_weight='balanced'` and `scale_pos_weight`.
- Data leakage strictly prevented using `sklearn.pipeline.Pipeline` and `StandardScaler` inside cross-validation splits.
- All figures and tables are programmatically generated.

---

## License

For academic use only.
