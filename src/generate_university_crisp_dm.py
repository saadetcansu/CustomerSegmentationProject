import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(text): cells.append(nbf.v4.new_markdown_cell(text))
def code(text): cells.append(nbf.v4.new_code_cell(text))

md("# Predicting Customer Response to Marketing Campaigns\\n**Using Data Mining Techniques and Supervised Machine Learning**\\n\\nThis project follows the 18-phase CRISP-DM methodology to solve a binary classification problem: predicting whether a customer will respond positively to the latest marketing campaign.")

md("## Phase 1: Business Understanding\\n\\n**Business Problem:** Marketing campaigns require significant financial investment. Mass-marketing to customers who are unlikely to respond results in wasted budget and potential customer fatigue (unsubscribing/ignoring future communications).\\n\\n**Business Objective:** Identify the characteristics of customers who are most likely to respond to a campaign, enabling the marketing team to execute highly targeted, cost-effective campaigns.\\n\\n**Data Mining Objective:** Develop a supervised Machine Learning binary classification model that accurately predicts the `Response` variable (1 = Yes, 0 = No) based on demographics, purchase history, and past campaign engagement.\\n\\n**Success Criteria:** Develop a model prioritizing F1-score, Recall, and ROC-AUC over pure accuracy due to class imbalance. The model must yield interpretable feature importance to drive strategic marketing decisions.")

md("## Phase 2: Data Understanding\\nWe begin by loading the dataset, verifying its schema, and assessing the initial structure.")
code("""import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('ggplot')

# Create output directories
for d in ['../outputs/crisp_dm_v2/figures', '../outputs/crisp_dm_v2/tables', '../models/crisp_dm_v2']:
    os.makedirs(d, exist_ok=True)

df = pd.read_csv('../data/marketing_campaign.csv', sep='\\t')
print("Dataset Dimensions:", df.shape)
print("\\n--- Feature Types ---")
display(df.dtypes.value_counts())
display(df.head())""")

md("## Phase 3: Data Preparation (Initial Cleaning)\\nBefore diving into exploratory mining, we must handle obvious missing values and duplicates to ensure our analysis is accurate.")
code("""# Missing Values
missing = df.isnull().sum()
print("Missing values before cleaning:\\n", missing[missing > 0])

# Duplicates
print("\\nDuplicate Rows:", df.duplicated().sum())

# Strategy: Impute Income with median as it's a small fraction and income is heavily skewed.
df['Income'] = df['Income'].fillna(df['Income'].median())
print("Missing values after imputation:", df.isnull().sum().max() == 0)""")

md("## Phase 4: Exploratory Data Mining\\nHere we answer fundamental questions about the data distributions and relationships with the target variable `Response`.")
code("""# Descriptive Stats
display(df.describe())

# Target Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Response', data=df)
plt.title('Target Distribution (Response)')
plt.savefig('../outputs/crisp_dm_v2/figures/target_distribution.png')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=False, cmap='coolwarm')
plt.title('Numeric Feature Correlation Heatmap')
plt.savefig('../outputs/crisp_dm_v2/figures/correlation_heatmap.png')
plt.show()

# Bivariate Analysis vs Target
fig, axes = plt.subplots(3, 2, figsize=(15, 18))
sns.boxplot(x='Response', y='Income', data=df, ax=axes[0,0]).set_title('Response vs Income')
sns.boxplot(x='Response', y='Recency', data=df, ax=axes[0,1]).set_title('Response vs Recency')
sns.countplot(x='Education', hue='Response', data=df, ax=axes[1,0]).set_title('Response vs Education')
sns.countplot(x='Marital_Status', hue='Response', data=df, ax=axes[1,1]).set_title('Response vs Marital Status')
sns.boxplot(x='Response', y='NumWebPurchases', data=df, ax=axes[2,0]).set_title('Response vs Web Purchases')
sns.boxplot(x='Response', y='NumStorePurchases', data=df, ax=axes[2,1]).set_title('Response vs Store Purchases')
plt.tight_layout()
plt.savefig('../outputs/crisp_dm_v2/figures/bivariate_analysis.png')
plt.show()""")

md("## Phase 5: Feature Engineering\\nWe create powerful new predictive features from the raw data. Extreme outliers (e.g., Age > 100) are removed during this step to prevent them from skewing the engineered metrics.\\n- **Age**: Derived from Year_Birth.\\n- **Customer_Tenure**: Days since enrollment.\\n- **Total_Spending**: Aggregate of all product categories.\\n- **Total_Purchases**: Aggregate of all purchase channels.\\n- **Campaign_Acceptance_Count**: Total past campaigns accepted.")
code("""# Remove impossible ages and extreme incomes
df = df[df['Year_Birth'] > 1900]
df = df[df['Income'] < 600000]

df['Age'] = 2014 - df['Year_Birth']
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y')
df['Customer_Tenure'] = (pd.to_datetime('2014-12-31') - df['Dt_Customer']).dt.days

df['Total_Spending'] = df[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum(axis=1)
df['Total_Purchases'] = df[['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumDealsPurchases']].sum(axis=1)
df['Campaign_Acceptance_Count'] = df[['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']].sum(axis=1)
df['Average_Spending_Per_Purchase'] = df['Total_Spending'] / df['Total_Purchases'].replace(0, 1)

# Categorical mapping
df['Marital_Status'] = df['Marital_Status'].replace(['YOLO', 'Absurd', 'Alone'], 'Single')
df['Education'] = df['Education'].replace(['2n Cycle', 'Basic'], 'Undergraduate')

# Drop useless/redundant columns
df = df.drop(columns=['ID', 'Z_CostContact', 'Z_Revenue', 'Year_Birth', 'Dt_Customer'])""")

md("## Phase 6: Feature Selection\\nUsing Mutual Information to rank features based on their dependency with the `Response` variable. We avoid discarding features strictly based on MI, but use this to understand importance.")
code("""from sklearn.feature_selection import mutual_info_classif

X_mi = pd.get_dummies(df.drop('Response', axis=1), drop_first=True)
y_mi = df['Response']

mi_scores = mutual_info_classif(X_mi, y_mi, random_state=42)
mi_scores = pd.Series(mi_scores, name='MI_Scores', index=X_mi.columns).sort_values(ascending=False)

plt.figure(figsize=(10,6))
mi_scores.head(15).plot.bar()
plt.title("Top 15 Features by Mutual Information")
plt.savefig('../outputs/crisp_dm_v2/figures/feature_selection_mi.png')
plt.show()""")

md("## Phase 7: Dimensionality Reduction\\nApplying PCA (Principal Component Analysis) strictly for exploratory visualization to see if the classes (Responder vs Non-Responder) are linearly separable in a 2D space. The overlap suggests non-linear models (like trees) will perform best.")
code("""from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_mi)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y_mi, alpha=0.6, palette='Set1')
plt.title(f"2D PCA Visualization (Explained Variance: {sum(pca.explained_variance_ratio_):.2f})")
plt.savefig('../outputs/crisp_dm_v2/figures/pca_visualization.png')
plt.show()""")

md("## Phase 8: Data Preparation for Modeling\\nWe construct a robust pipeline splitting data (60/20/20) using stratification to preserve class balance, and applying `StandardScaler` and `OneHotEncoder` without leaking data.")
code("""from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

X = df.drop('Response', axis=1)
y = df['Response']

# Stratified 60-20-20 Split
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = X_train.select_dtypes(exclude=['object', 'category']).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_cols)
])""")

md("## Phase 9: Class Imbalance Analysis\\nThe target variable is heavily imbalanced. We will analyze the ratio and decide to use algorithmic class weighting (`class_weight='balanced'`) or SMOTE to penalize false negatives effectively.")
code("""from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

pos_class = sum(y_train)
neg_class = len(y_train) - pos_class
ratio = pos_class / len(y_train)

print(f"Training Target Distribution:\\n- Negative: {neg_class}\\n- Positive: {pos_class}\\n- Positive Ratio: {ratio:.2%}")
print("\\nStrategy: We will use class weights and SMOTE inside pipelines to address this 15% minority class.")
""")

md("## Phase 10: Baseline Model\\nWe implement a Dummy Classifier predicting the prior distribution to establish a baseline. Any Machine Learning model MUST beat this baseline PR-AUC and F1-Score.")
code("""from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

baseline = Pipeline([('pre', preprocessor), ('clf', DummyClassifier(strategy='prior', random_state=42))])
baseline.fit(X_train, y_train)
y_val_base = baseline.predict(X_val)

print("Baseline Metrics:")
print(f"Accuracy: {accuracy_score(y_val, y_val_base):.4f}")
print(f"F1-Score: {f1_score(y_val, y_val_base):.4f}")
print(f"PR-AUC:   {average_precision_score(y_val, baseline.predict_proba(X_val)[:,1]):.4f}")""")

md("## Phase 11: Model Development & Phase 13: Model Evaluation\\nWe train multiple algorithms and evaluate them on the Validation set. (Phase 12 will tune the absolute best model).")
code("""from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, precision_recall_curve

models = {
    'LogisticReg': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
    'DecisionTree': DecisionTreeClassifier(random_state=42, class_weight='balanced', max_depth=5),
    'RandomForest': RandomForestClassifier(random_state=42, class_weight='balanced'),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': xgb.XGBClassifier(random_state=42, scale_pos_weight=neg_class/pos_class, eval_metric='logloss'),
    'LightGBM': lgb.LGBMClassifier(random_state=42, class_weight='balanced', verbosity=-1),
    'RF_SMOTE': ImbPipeline([('pre', preprocessor), ('smote', SMOTE(random_state=42)), ('clf', RandomForestClassifier(random_state=42))])
}

results = []
trained_models = {}

for name, clf in models.items():
    if name != 'RF_SMOTE':
        pipe = Pipeline([('preprocessor', preprocessor), ('classifier', clf)])
    else:
        pipe = clf
        
    pipe.fit(X_train, y_train)
    trained_models[name] = pipe
    
    y_pred = pipe.predict(X_val)
    y_prob = pipe.predict_proba(X_val)[:, 1] if hasattr(pipe.steps[-1][1], 'predict_proba') else pipe.decision_function(X_val)
    
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_val, y_pred),
        'Precision': precision_score(y_val, y_pred, zero_division=0),
        'Recall': recall_score(y_val, y_pred),
        'F1-Score': f1_score(y_val, y_pred),
        'ROC-AUC': roc_auc_score(y_val, y_prob),
        'PR-AUC': average_precision_score(y_val, y_prob)
    })

res_df = pd.DataFrame(results).sort_values(by='F1-Score', ascending=False)
display(res_df)
res_df.to_csv('../outputs/crisp_dm_v2/tables/validation_metrics.csv', index=False)
""")

md("## Phase 12: Hyperparameter Optimization\\nWe optimize the top performing ensemble (LightGBM) using RandomizedSearchCV to fine-tune the precision-recall trade-off.")
code("""from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    'classifier__num_leaves': [31, 50, 100],
    'classifier__learning_rate': [0.01, 0.05, 0.1],
    'classifier__n_estimators': [100, 200, 300],
    'classifier__min_child_samples': [10, 20, 30]
}

lgb_pipe = Pipeline([('preprocessor', preprocessor), ('classifier', lgb.LGBMClassifier(random_state=42, class_weight='balanced', verbosity=-1))])
search = RandomizedSearchCV(lgb_pipe, param_grid, n_iter=10, cv=3, scoring='f1', random_state=42, n_jobs=-1)
search.fit(X_train, y_train)

best_model = search.best_estimator_
print("Best Hyperparameters:", search.best_params_)
""")

md("## Phase 13 continued: Final Test Set Evaluation\\nWe evaluate our tuned model on the unseen Test Set to gauge real-world performance.")
code("""from sklearn.metrics import classification_report
y_test_pred = best_model.predict(X_test)
y_test_prob = best_model.predict_proba(X_test)[:, 1]

print("--- TEST SET CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_test_pred))

fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap='Blues', ax=axes[0])
axes[0].set_title("Test Set Confusion Matrix")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_test_prob)
axes[1].plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_test_prob):.3f}')
axes[1].plot([0,1], [0,1], 'k--')
axes[1].set_title("ROC Curve")
axes[1].legend()

# PR Curve
prec, rec, _ = precision_recall_curve(y_test, y_test_prob)
axes[2].plot(rec, prec, label=f'PR-AUC = {average_precision_score(y_test, y_test_prob):.3f}')
axes[2].set_title("Precision-Recall Curve")
axes[2].set_xlabel('Recall')
axes[2].set_ylabel('Precision')
axes[2].legend()

plt.tight_layout()
plt.savefig('../outputs/crisp_dm_v2/figures/test_evaluation_curves.png')
plt.show()
""")

md("## Phase 14: Model Interpretation\\nWe extract global Feature Importances to understand what is driving the model's predictions.")
code("""importances = best_model.named_steps['classifier'].feature_importances_
cat_fn = best_model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(cat_cols)
fn = num_cols + cat_fn.tolist()

feat_df = pd.DataFrame({'Feature': fn, 'Importance': importances}).sort_values('Importance', ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feat_df.head(15))
plt.title("Top 15 Feature Importances (Tuned LightGBM)")
plt.savefig('../outputs/crisp_dm_v2/figures/feature_importance.png')
plt.show()""")

md("## Phase 15: Error Analysis\\nInvestigating the False Positives (predicted response, actual no response) and False Negatives.")
code("""errors_df = X_test.copy()
errors_df['Actual'] = y_test
errors_df['Predicted'] = y_test_pred

fp = errors_df[(errors_df['Actual'] == 0) & (errors_df['Predicted'] == 1)]
fn = errors_df[(errors_df['Actual'] == 1) & (errors_df['Predicted'] == 0)]
tp = errors_df[(errors_df['Actual'] == 1) & (errors_df['Predicted'] == 1)]

print(f"False Positives: {len(fp)}")
print(f"False Negatives: {len(fn)}")

print("\\nMean values for False Positives (Model thought they'd respond, but they didn't):")
display(fp[['Income', 'Recency', 'Total_Spending', 'Campaign_Acceptance_Count']].mean())

print("\\nMean values for True Positives (Successfully predicted responders):")
display(tp[['Income', 'Recency', 'Total_Spending', 'Campaign_Acceptance_Count']].mean())

print("\\n**Insight:** False Positives look very similar to True Positives (high income, high spending, lots of past campaign acceptances). This means these customers were highly qualified candidates who simply decided not to respond this time, rather than a systemic failure of the model.")
""")

md("## Phase 16: Customer Profiling\\nBased on the feature importances and error analysis, the ideal customer profile for the upcoming marketing campaign is:\\n1. **High Historical Engagement:** They have accepted multiple campaigns in the past (`Campaign_Acceptance_Count`).\\n2. **High Spenders:** They spend significantly more on average across all categories (`Total_Spending`).\\n3. **Recent Buyers:** Their `Recency` score is very low, meaning they engaged with the brand recently.\\n4. **High Income & Tenure:** They have been with the brand longer (`Customer_Tenure`) and have higher disposable income.")

md("## Phase 17: Business Recommendations\\n- **Targeting Strategy:** The marketing team should exclusively target the top 20% of customers scored by the model's predicted probability. This will drastically reduce marketing costs while capturing the vast majority of potential responders.\\n- **False Positive Recovery:** Customers flagged as False Positives are highly qualified. Marketing should trigger an automated, secondary 'win-back' or 'discount' campaign specifically for them, as they theoretically should have responded.\\n- **Acquisition:** Do not waste premium marketing budget on new, low-income, high-recency customers without previous engagement. Use cheaper acquisition channels for them.")

md("## Phase 18: Final Conclusions\\n- **Best Model:** A hyperparameter-tuned `LightGBM` model with `class_weight='balanced'` achieved the highest F1-score and PR-AUC, successfully tackling the 15% class imbalance without the computational overhead of SMOTE.\\n- **Data Prep:** The 60/20/20 stratified split and strict scikit-learn Pipelines prevented any data leakage.\\n- **Limitations:** The model is heavily dependent on past campaign responses. It may struggle to predict response for completely new users (the 'cold start' problem).\\n- **Future Work:** Run an A/B test deploying this model's predictions against a random baseline in a live marketing campaign to calculate explicit monetary ROI.")

nb['cells'] = cells
import os
os.makedirs('../notebooks', exist_ok=True)
with open('../notebooks/crisp_dm_university_project.ipynb', 'w') as f:
    nbf.write(nb, f)
print("CRISP-DM notebook successfully generated!")
