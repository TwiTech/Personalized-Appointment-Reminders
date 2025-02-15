from openai import OpenAI
import time
import logging

logger = logging.getLogger(__name__)

class OpenAIEmbeddings:
    """
    A utility class for interacting with OpenAI's embedding API using the updated API structure.
    """

    def __init__(self, api_key, model="text-embedding-ada-002"):
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def embed_text(self, text):
        """Generate an embedding for a single text."""
        if not text or not isinstance(text, str):
            return [0] * 1536  # Return zero vector if text is empty

        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text[:8192]  # Truncate text to avoid exceeding token limits
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            time.sleep(2)  # Retry delay in case of rate limits
            return [0] * 1536  

    def embed_batch(self, texts):
        """Generate embeddings for a batch of texts."""
        texts = [text[:8192] if isinstance(text, str) else "" for text in texts]
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            time.sleep(2)
            return [[0] * 1536] * len(texts)
