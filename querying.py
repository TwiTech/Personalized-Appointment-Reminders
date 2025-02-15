import os
import faiss
import numpy as np
import jsonlines
import logging
import smtplib
import requests
from openai import OpenAI
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FAISS_INDEX_FILE = "reminder_faiss.index"
EMBEDDINGS_FILE = "reminder_embeddings.ndjson"
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
FREE_SMS_API_KEY = os.getenv("FREE_SMS_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def load_embeddings():
    if not os.path.exists(EMBEDDINGS_FILE):
        raise FileNotFoundError("Embedding file not found. Ensure you have run the embedding script.")
    
    embeddings = []
    texts = []
    with jsonlines.open(EMBEDDINGS_FILE, mode='r') as reader:
        for obj in reader:
            texts.append(obj["text"])
            embeddings.append(np.array(obj["embedding"], dtype="float32"))

    return texts, np.array(embeddings, dtype="float32")

def load_faiss_index():
    if not os.path.exists(FAISS_INDEX_FILE):
        raise FileNotFoundError("FAISS index file not found. Ensure you have built the index.")
    
    return faiss.read_index(FAISS_INDEX_FILE)

def search_with_gpt(query_text, k=3):
    texts, embeddings = load_embeddings()
    faiss_index = load_faiss_index()
    
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=query_text
    )
    query_embedding = np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)
    
    distances, indices = faiss_index.search(query_embedding, k)
    similar_texts = [texts[i] for i in indices[0]]
    appointment_details = similar_texts[0]  # Assume first retrieved reminder is the best match
    
    messages = [
        {"role": "system", "content": "You are an AI that generates personalized appointment reminders."},
        {"role": "user", "content": f"Patient has missed similar appointments. "
                                        f"Their upcoming appointment is on {appointment_details.split('|')[1].strip()} at "
                                        f"{appointment_details.split('|')[2].strip()}. Generate a friendly reminder."}
    ]
    
    gpt_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
        max_tokens=150
    )
    
    return gpt_response.choices[0].message.content.strip()

def send_sms_reminder(phone_number, reminder_message):
    url = f"https://www.freesmsapi.com/send?apikey={FREE_SMS_API_KEY}&to={phone_number}&message={reminder_message}"
    response = requests.get(url)
    logger.info(f"SMS Sent to {phone_number}: {response.json()}")

def send_email_reminder(recipient_email, subject, body):
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        message = f"Subject: {subject}\n\n{body}"
        server.sendmail(EMAIL_ADDRESS, recipient_email, message)
        logger.info(f"Email Sent to {recipient_email}")

if __name__ == "__main__":
    query = "I tend to forget morning appointments."
    gpt_summary = search_with_gpt(query)
    logger.info("\nGPT-Generated Reminder Summary:")
    logger.info(gpt_summary)
    
    # Example: Sending an SMS and Email reminder
    test_phone_number = "+447442831735"  # Replace with actual number
    test_email = "olamidetawakalit@gmail.com.com"  # Replace with actual email
    
    send_sms_reminder(test_phone_number, gpt_summary)
    send_email_reminder(test_email, "Upcoming Appointment Reminder", gpt_summary)
