import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import os


# Load .pkl file 
X_train, X_test, y_train, y_test = joblib.load(r"C:\Users\USER\My notebook\DataSciencePro\stroke-risk-ensemble-comparison\data\processed\processed_data.pkl")

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

# Classification Report
print("Random Forest: \n", classification_report(y_test, y_pred_rf))
print("XGBoost Report: \n", classification_report(y_test, y_pred_xgb))
print("LightGBM Report: \n", classification_report(y_test, y_pred_lgb))

# ROC_AUC COMPARISON
print("ROC-AUC Scores: \n")
print(f"Random Forest: {roc_auc_rf:.2f}")
print(f"XGBoost: {roc_auc_xgb:.2f}")
print(f"LightGBM: {roc_auc_lgb:.2f}")


# Save my best model 
os.makedirs(r"C:\Users\USER\My notebook\DataSciencePro\stroke-risk-ensemble-comparison\models", exist_ok=True)
joblib.dump(rf_model, r"C:\Users\USER\My notebook\DataSciencePro\stroke-risk-ensemble-comparison\models\best_model.pkl")
