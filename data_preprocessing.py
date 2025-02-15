import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

def load_and_preprocess_data():
    # Load datasets
    slots_df = pd.read_csv("slots.csv")
    patients_df = pd.read_csv("patients.csv")
    appointments_df = pd.read_csv("appointments.csv")

    # Filter relevant data (remove 'unknown' and 'scheduled' statuses)
    appointments_df = appointments_df[appointments_df["status"].isin(["attended", "did not attend"])]

    # Encode target variable: 1 for "did not attend", 0 for "attended"
    appointments_df["no_show"] = (appointments_df["status"] == "did not attend").astype(int)

    # Merge appointments with patients to get insurance information
    appointments_merged = appointments_df.merge(patients_df[["patient_id", "insurance", "name"]], on="patient_id", how="left")

    # Select features
    features = ["age", "sex", "insurance", "appointment_time", "waiting_time", "scheduling_interval"]
    df = appointments_merged[features + ["no_show"]].copy()

    # Encode categorical variables
    le_sex = LabelEncoder()
    df["sex"] = le_sex.fit_transform(df["sex"])

    le_insurance = LabelEncoder()
    df["insurance"] = le_insurance.fit_transform(df["insurance"].fillna("Unknown"))

    # Convert appointment time to numerical (hour) with explicit format
    df["appointment_time"] = pd.to_datetime(df["appointment_time"], format="%H:%M:%S", errors='coerce')
    df["appointment_hour"] = df["appointment_time"].dt.hour
    df.drop(columns=["appointment_time"], inplace=True)
    df.fillna(df.median(), inplace=True)

    # Handle class imbalance with oversampling
    df_majority = df[df["no_show"] == 0]
    df_minority = df[df["no_show"] == 1]
    df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
    df_balanced = pd.concat([df_majority, df_minority_upsampled])

    # Split data
    X_balanced = df_balanced.drop(columns=["no_show"])
    y_balanced = df_balanced["no_show"]
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced)
    
    return X_train, X_test, y_train, y_test, appointments_merged, patients_df

def generate_reminders(no_show_indices, appointments_merged, patients_df):
    no_show_patients = appointments_merged.loc[no_show_indices, ["patient_id", "appointment_date", "appointment_time"]]
    no_show_patients = no_show_patients.merge(patients_df[["patient_id", "name"]], on="patient_id", how="left")
    reminders = []
    
    for _, row in no_show_patients.iterrows():
        patient_name = row["name"]
        appointment_time = row["appointment_time"]
        no_show_count = random.randint(1, 5)  # Simulate past missed appointments
        reminder_msg = (f"Hello {patient_name}, we noticed that you have an appointment on {row['appointment_date']} at {appointment_time}. "
                        f"You have missed {no_show_count} past appointments. Would you like to set a reminder? "
                        f"Your doctor is expecting you.")
        reminders.append((patient_name, row["appointment_date"], appointment_time, reminder_msg))
    
    return pd.DataFrame(reminders, columns=["Patient Name", "Appointment Date", "Appointment Time", "Reminder Message"])
