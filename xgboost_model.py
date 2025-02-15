from utils import load_and_preprocess_data, train_model
from xgboost import XGBClassifier

# Load data
X_train, X_test, y_train, y_test, appointments_merged, patients_df = load_and_preprocess_data()

# Initialize and train XGBoost model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
print("\nTraining XGBoost...")
train_model(xgb_model, X_train, X_test, y_train, y_test)
