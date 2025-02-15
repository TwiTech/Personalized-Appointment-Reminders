import os
import jsonlines
import faiss
import numpy as np
import time
import logging
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from utils import load_and_preprocess_data, generate_reminders
from embeddings_utils import OpenAIEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Ensure API key is loaded
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set. Check your .env file.")

# Initialize OpenAI embedding model
embedding_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# File paths for embeddings and FAISS index
EMBEDDINGS_FILE = "reminder_embeddings.ndjson"
FAISS_INDEX_FILE = "reminder_faiss.index"

# Function to enhance reminder texts with metadata
def enhance_reminder_texts(reminders_df):
    """Adds more contextual metadata to reminder messages before embedding."""
    enhanced_texts = []
    for _, row in reminders_df.iterrows():
        past_no_shows = row.get('Past No-Shows', 0)  # Use 0 if column is missing
        enhanced_text = (
            f"Patient: {row['Patient Name']} | Date: {row['Appointment Date']} | "
            f"Time: {row['Appointment Time']} | Past No-Shows: {past_no_shows} | "
            f"Reminder: {row['Reminder Message']}"
        )
        enhanced_texts.append(enhanced_text)
    return enhanced_texts

# Function to load stored embeddings
def load_embeddings():
    if not os.path.exists(EMBEDDINGS_FILE):
        return [], None

    embeddings = []
    texts = []
    with jsonlines.open(EMBEDDINGS_FILE, mode='r') as reader:
        for obj in reader:
            texts.append(obj["text"])
            embeddings.append(np.array(obj["embedding"], dtype="float32"))

    return texts, np.array(embeddings, dtype="float32")

# Function to save embeddings
def save_embeddings(texts, embeddings):
    with jsonlines.open(EMBEDDINGS_FILE, mode='w') as writer:
        for text, embedding in zip(texts, embeddings):
            writer.write({"text": text, "embedding": embedding.tolist()})
    logger.info(f"Stored {len(texts)} embeddings.")

# Function to build FAISS index using cosine similarity
def build_faiss_index(embeddings):
    embeddings = np.asarray(embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings)  # Normalize for cosine similarity
    
    quantizer = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index = faiss.IndexIVFFlat(quantizer, embeddings.shape[1], 100)  # Inverted file index
    faiss_index.train(embeddings)
    faiss_index.add(embeddings)
    faiss.write_index(faiss_index, FAISS_INDEX_FILE)
    logger.info(f"FAISS index saved successfully with {faiss_index.ntotal} entries.")

# Function to retrieve similar reminders using FAISS
def search_similar_reminders(query_text, k=3):
    if not os.path.exists(FAISS_INDEX_FILE):
        raise FileNotFoundError("FAISS index not found. Run the embedding script first.")
    
    faiss_index = faiss.read_index(FAISS_INDEX_FILE)
    _, all_embeddings = load_embeddings()
    
    if query_text in all_texts:
        query_index = all_texts.index(query_text)
        query_embedding = all_embeddings[query_index].reshape(1, -1)
    else:
        raise ValueError("Query text not found in FAISS index. Ensure it exists in embeddings.")
    
    distances, indices = faiss_index.search(query_embedding, k)
    return [all_texts[i] for i in indices[0]]

# Separate processing and querying logic
if __name__ == "__main__":
    process_embeddings = True  # Change to False if you only want to query
    
    if process_embeddings:
        # Load data and predicted no-shows
        X_train, X_test, y_train, y_test, appointments_merged, patients_df = load_and_preprocess_data()
        no_show_indices = y_test[y_test == 1].index
        reminders_df = generate_reminders(no_show_indices, appointments_merged, patients_df)
        
        # Enhance reminder texts with metadata
        reminder_texts = enhance_reminder_texts(reminders_df)
        
        # Load previous embeddings or generate missing ones
        existing_texts, existing_embeddings = load_embeddings()
        
        # Identify texts that need embedding
        new_texts = [text for text in reminder_texts if text not in existing_texts]
        new_embeddings = []
        
        if new_texts:
            logger.info(f"Generating embeddings for {len(new_texts)} new reminders...")
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=5) as executor:
                batch_size = 10
                future_batches = [
                    executor.submit(embedding_model.embed_batch, new_texts[i:i + batch_size])
                    for i in range(0, len(new_texts), batch_size)
                ]
                
                for future in future_batches:
                    new_embeddings.extend(future.result())
            
            logger.info(f"Embedding generation time: {(time.time() - start_time):.2f} seconds")
        
        # Combine old and new embeddings
        all_texts = existing_texts + new_texts
        all_embeddings = np.vstack([existing_embeddings, new_embeddings]) if existing_embeddings is not None else np.array(new_embeddings)
        
        # Save embeddings and build FAISS index
        save_embeddings(all_texts, all_embeddings)
        build_faiss_index(all_embeddings)
    
    else:
        # Querying example
        query = "I tend to forget morning appointments."
        similar_reminders = search_similar_reminders(query)
        logger.info("\nTop Similar Reminders:")
        for reminder in similar_reminders:
            logger.info(f"- {reminder}")
