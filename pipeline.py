from utils import load_and_preprocess_data, train_model, generate_reminders
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load and preprocess data
X_train, X_test, y_train, y_test, appointments_merged, patients_df = load_and_preprocess_data()

# Check class distribution
print("Class Distribution in Training Set:", y_train.value_counts(normalize=True))
print("Class Distribution in Test Set:", y_test.value_counts(normalize=True))

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=50,  # Reduced number of iterations
        learning_rate=0.1,
        max_depth=2,  # Reduced depth
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=10,  # Reduced number of trees
        max_depth=2,       # Reduced depth
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    )
}

# Train and evaluate each model using Stratified Cross-Validation
results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"{name} Cross-Validation Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    # Train the model on the full training set and evaluate on the test set
    y_pred = train_model(model, X_train, X_test, y_train, y_test)
    results[name] = y_pred

# Plot feature importance for HistGradientBoosting and Random Forest
for model_name in ["HistGradientBoosting", "Random Forest"]:
    model = models[model_name]
    model.fit(X_train, y_train)
    
    if model_name == "HistGradientBoosting":
        importances = model.feature_importances()  # Correct way to get feature importance
    else:
        importances = model.feature_importances_
    
    feature_names = X_train.columns
    plt.figure(figsize=(8, 5))
    plt.barh(feature_names, importances)
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature Name")
    plt.title(f"{model_name} Feature Importance")
    plt.show()

# Generate reminders for predicted no-shows
best_model = models["HistGradientBoosting"]
best_model.fit(X_train, y_train)
no_show_indices = y_test[y_test == 1].index
reminders_df = generate_reminders(no_show_indices, appointments_merged, patients_df)

# Save or display reminders
print("\nGenerated Reminders:")
print(reminders_df.head())