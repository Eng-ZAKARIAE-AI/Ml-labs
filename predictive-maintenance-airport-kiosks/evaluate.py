import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

model = joblib.load('model_artifacts/predictive_maintenance_rf.pkl')
features = joblib.load('model_artifacts/model_features.pkl')

df_test = pd.read_csv('data/test/test_data_2023_2024.csv')

X_test = df_test[features]
y_test = df_test['Target_Failure']

y_proba = model.predict_proba(X_test)[:, 1]

# (Optimal for Predictive Maintenance) sensitivity of model
custom_threshold = 0.25
y_pred_custom = (y_proba >= custom_threshold).astype(int)

print("==========================================")
print(f"📊 Performance Report (Threshold = {custom_threshold})")
print("==========================================")
print(classification_report(y_test, y_pred_custom))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
