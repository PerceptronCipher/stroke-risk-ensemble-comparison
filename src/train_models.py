import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from catboost import CatBoostClassifier
import os


# Load .pkl file 
X_train, X_test, y_train, y_test = joblib.load(r"data\processed\processed_data.pkl")

# Inspect data 
print("\n Class Distribution After SMOTE: \n")
print(pd.Series(y_train).value_counts())
print(pd.Series(y_test).value_counts())

# Train Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
roc_auc_rf = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])

# Train XGBoost
xgb_model = XGBClassifier(eval_metric ='logloss', random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
roc_auc_xgb = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])

# Train LightGBM
lgb_model = LGBMClassifier(random_state=42)
lgb_model.fit(X_train, y_train)
y_pred_lgb = lgb_model.predict(X_test)
roc_auc_lgb = roc_auc_score(y_test, lgb_model.predict_proba(X_test)[:, 1])

# Train Gradient Boosting
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
roc_auc_gb = roc_auc_score(y_test, gb_model.predict_proba(X_test)[:, 1])

# Train CatBoost
cat_model = CatBoostClassifier(verbose=0, random_state=42)
cat_model.fit(X_train, y_train)
y_pred_cat = cat_model.predict(X_test)
roc_auc_cat = roc_auc_score(y_test, cat_model.predict_proba(X_test)[:, 1])


# Classification Report
print("Random Forest: \n", classification_report(y_test, y_pred_rf))
print("XGBoost Report: \n", classification_report(y_test, y_pred_xgb))
print("LightGBM Report: \n", classification_report(y_test, y_pred_lgb))
print("Gradient Boosting Report:\n", classification_report(y_test, y_pred_gb))
print("CatBoost Report:\n", classification_report(y_test, y_pred_cat))

# ROC_AUC COMPARISON
print("ROC-AUC Scores: \n")
print(f"Random Forest: {roc_auc_rf:.2f}")
print(f"XGBoost: {roc_auc_xgb:.2f}")
print(f"LightGBM: {roc_auc_lgb:.2f}")
print(f"Gradient Boosting: {roc_auc_gb:.2f}")
print(f"CatBoost: {roc_auc_cat:.2f}")

# Identify the best model
roc_scores = {
    "Random Forest": roc_auc_rf,
    "XGBoost": roc_auc_xgb,
    "LightGBM": roc_auc_lgb,
    "Gradient Boosting": roc_auc_gb,
    "CatBoost": roc_auc_cat
}

best_model_name = max(roc_scores, key=roc_scores.get)
print(f"\n Best Model: {best_model_name} with ROC-AUC of {roc_scores[best_model_name]:.2f}")


# Save my best model 
os.makedirs(r"models", exist_ok=True)
joblib.dump(rf_model, r"models\best_model.pkl")
