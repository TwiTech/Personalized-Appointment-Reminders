from utils import load_and_preprocess_data, train_model
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
# Load data
X_train, X_test, y_train, y_test, appointments_merged, patients_df = load_and_preprocess_data()

# Initialize and train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
y_pred = train_model(model, X_train, X_test, y_train, y_test)

# Plot feature importance
importances = model.feature_importances_
feature_names = X_train.columns
plt.barh(feature_names, importances)
plt.xlabel("Feature Importance")
plt.ylabel("Feature Name")
plt.title("Random Forest Feature Importance")
plt.show()
