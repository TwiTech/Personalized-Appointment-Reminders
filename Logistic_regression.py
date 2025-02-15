from utils import load_and_preprocess_data, train_model
from sklearn.linear_model import LogisticRegression

# Load data
X_train, X_test, y_train, y_test, appointments_merged, patients_df = load_and_preprocess_data()

# Initialize and train Logistic Regression model
log_reg = LogisticRegression(max_iter=1000, random_state=42)
print("\nTraining Logistic Regression...")
train_model(log_reg, X_train, X_test, y_train, y_test)
