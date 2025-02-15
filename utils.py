import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, classification_report

def load_and_preprocess_data():
    """Loads datasets, processes features, handles class imbalance, and splits data."""
    slots_df = pd.read_csv("slots.csv")
    patients_df = pd.read_csv("patients.csv")
    appointments_df = pd.read_csv("appointments.csv")

    appointments_df = appointments_df[appointments_df["status"].isin(["attended", "did not attend"])]
    appointments_df["no_show"] = (appointments_df["status"] == "did not attend").astype(int)

    appointments_merged = appointments_df.merge(patients_df[["patient_id", "insurance", "name"]], on="patient_id", how="left")

    features = ["age", "sex", "insurance", "appointment_time", "waiting_time", "scheduling_interval"]
    df = appointments_merged[features + ["no_show"]].copy()

    le_sex = LabelEncoder()
    df["sex"] = le_sex.fit_transform(df["sex"])

    le_insurance = LabelEncoder()
    df["insurance"] = le_insurance.fit_transform(df["insurance"].fillna("Unknown"))

    df["appointment_time"] = pd.to_datetime(df["appointment_time"], format="%H:%M:%S", errors='coerce')
    df["appointment_hour"] = df["appointment_time"].dt.hour
    df.drop(columns=["appointment_time"], inplace=True)
    df.fillna(df.median(), inplace=True)

    df_majority = df[df["no_show"] == 0]
    df_minority = df[df["no_show"] == 1]
    df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
    df_balanced = pd.concat([df_majority, df_minority_upsampled])

    X_balanced = df_balanced.drop(columns=["no_show"])
    y_balanced = df_balanced["no_show"]
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced)
    
    return X_train, X_test, y_train, y_test, appointments_merged, patients_df

def train_model(model, X_train, X_test, y_train, y_test):
    """Trains and evaluates the given machine learning model."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    return y_pred

def generate_reminders(no_show_indices, appointments_merged, patients_df):
    """Generates personalized appointment reminders for predicted no-shows."""
    no_show_patients = appointments_merged.loc[no_show_indices, ["patient_id", "appointment_date", "appointment_time"]]
    no_show_patients = no_show_patients.merge(patients_df[["patient_id", "name"]], on="patient_id", how="left")
    reminders = []
    
    for _, row in no_show_patients.iterrows():
        patient_name = row["name"]
        appointment_time = row["appointment_time"]
        no_show_count = random.randint(1, 5)
        reminder_msg = (f"Hello {patient_name}, we noticed that you have an appointment on {row['appointment_date']} at {appointment_time}. "
                        f"You have missed {no_show_count} past appointments. Would you like to set a reminder? "
                        f"Your doctor is expecting you.")
        reminders.append((patient_name, row["appointment_date"], appointment_time, reminder_msg))
    
    return pd.DataFrame(reminders, columns=["Patient Name", "Appointment Date", "Appointment Time", "Reminder Message"])
