'''

import os
import json
import uuid
import time
import random
import logging
import vertexai
from vertexai.language_models import TextEmbeddingModel
from langchain_chroma import Chroma
from langchain.embeddings.base import Embeddings
from google.api_core.exceptions import ResourceExhausted

# === Configuration ===
PROJECT_ID = "935917861333"
REGION = "us-west1"
MODEL_ID = "text-multilingual-embedding-002"
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "my_collection"
INPUT_FOLDER = r"D:\pdf_chtbot\ai-assistant\women_data_pdf"
LOG_FILE_PATH = "embedding_process.log"
ERROR_LOG_PATH = "embedding_errors.json"

# === Setup Logging ===
logging.basicConfig(filename=LOG_FILE_PATH, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
error_files = []

# === Custom Embedding Wrapper ===
class PrecomputedEmbeddings(Embeddings):
    def __init__(self, precomputed_vectors):
        self.precomputed_vectors = precomputed_vectors
        self.index = 0

    def embed_documents(self, texts):
        start = self.index
        end = start + len(texts)
        self.index = end
        return self.precomputed_vectors[start:end]

    def embed_query(self, text):
        raise NotImplementedError("Query embedding not supported.")

# === Extract text from Document AI JSON ===
def extract_text_by_page_from_dict(data_dict):
    result = []
    for i, page in enumerate(data_dict.get("document", {}).get("pages", []), start=1):
        text = page.get("text", "").strip()
        if text:
            result.append({"text": text, "page_number": i})
    return result

# === Get Embeddings with Retry and Batching ===
def get_embeddings_with_retry(texts, model, batch_size=1, max_retries=5, delay=500):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        for attempt in range(max_retries):
            try:
                responses = model.get_embeddings(batch)
                embeddings.extend([res.values for res in responses])
                time.sleep(1.0)
                break
            except ResourceExhausted as e:
                if "Quota exceeded" in str(e):
                    wait_time = delay * (2 ** attempt) + random.uniform(0, 5)
                    logging.warning(f"Quota exceeded. Retry {attempt + 1}/{max_retries} in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                else:
                    raise e
        else:
            logging.error(f"Failed to get embeddings after {max_retries} retries. Using dummy vector.")
            dummy = [0.0] * 768
            embeddings.extend([dummy for _ in batch])
    return embeddings

# === Main Process ===
def process_json_file(json_path, model):
    filename = os.path.basename(json_path)
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        page_data = extract_text_by_page_from_dict(data)
        if not page_data:
            logging.error(f"No valid text found in file: {filename}")
            error_files.append(filename)
            return False

        texts = [entry["text"] for entry in page_data]

        logging.info(f"Generating embeddings for: {filename}")
        embeddings = get_embeddings_with_retry(texts, model)

        enriched_data = []
        for entry, embedding in zip(page_data, embeddings):
            enriched_data.append({
                "text": entry["text"],
                "page_number": entry["page_number"],
                "embedding": embedding
            })

        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(os.path.dirname(json_path), base_name + "_embedding.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(enriched_data, f, indent=2, ensure_ascii=False)

        logging.info(f"Embeddings saved to {output_path}")

        doc_ids = [str(uuid.uuid4()) for _ in enriched_data]
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=PrecomputedEmbeddings(embeddings),
            persist_directory=CHROMA_PERSIST_DIR
        )
        vectorstore.add_texts(texts=texts, ids=doc_ids)

        logging.info(f"✅ Added {len(texts)} chunks to ChromaDB")
        return True

    except Exception as e:
        logging.error(f"Exception while processing {filename}: {e}")
        error_files.append(filename)
        return False

# === Entry Point ===
if __name__ == "__main__":
    logging.info("🚀 Starting embedding process...")
    vertexai.init(project=PROJECT_ID, location=REGION)
    model = TextEmbeddingModel.from_pretrained(MODEL_ID)

    all_json_files = [os.path.join(INPUT_FOLDER, f) for f in os.listdir(INPUT_FOLDER)
                      if f.endswith(".json") and not f.endswith("_embedding.json")]

    total_files = len(all_json_files)
    success_count = 0

    for json_file in all_json_files:
        if process_json_file(json_file, model):
            success_count += 1

    # Save list of failed files
    with open(ERROR_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(error_files, f, indent=2)

    logging.info(f"🔚 All processing completed.")
    logging.info(f"Total Files Found: {total_files}")
    logging.info(f"Files Succeeded: {success_count}")
    logging.info(f"Files Failed: {total_files - success_count}")
    print("✅ Done. Check logs and error file for details.")
'''


import os
import json
import uuid
import time
import random
import logging
import vertexai
from vertexai.language_models import TextEmbeddingModel
from langchain_chroma import Chroma  # ✅ Use updated Chroma import
from langchain.embeddings.base import Embeddings
from google.api_core.exceptions import ResourceExhausted

# === Configuration ===
PROJECT_ID = "935917861333"
REGION = "us-west1"
MODEL_ID = "text-multilingual-embedding-002"
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "my_collection"
INPUT_FOLDER = r"D:\pdf_chtbot\ai-assistant\Women_of_Color_Magazine"
LOG_FILE_PATH = "embedding_process.log"
ERROR_LOG_PATH = "embedding_errors.json"

# === Setup Logging ===
logging.basicConfig(filename=LOG_FILE_PATH, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
error_files = []

# === Custom Embedding Wrapper ===
class PrecomputedEmbeddings(Embeddings):
    def __init__(self, precomputed_vectors):
        self.precomputed_vectors = precomputed_vectors
        self.index = 0

    def embed_documents(self, texts):
        start = self.index
        end = start + len(texts)
        self.index = end
        return self.precomputed_vectors[start:end]

    def embed_query(self, text):
        raise NotImplementedError("Query embedding not supported.")

# === Extract text from Document AI JSON ===
def extract_text_by_page_from_dict(data_dict):
    result = []
    for i, page in enumerate(data_dict.get("document", {}).get("pages", []), start=1):
        text = page.get("text", "").strip()
        if text:
            result.append({"text": text, "page_number": i})
    return result

# === Get Embeddings with Retry and Batching ===
def get_embeddings_with_retry(texts, model, batch_size=1, max_retries=5, delay=500):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        for attempt in range(max_retries):
            try:
                responses = model.get_embeddings(batch)
                embeddings.extend([res.values for res in responses])
                time.sleep(1.0)
                break
            except ResourceExhausted as e:
                if "Quota exceeded" in str(e):
                    wait_time = delay * (2 ** attempt) / 1000 + random.uniform(0, 2)
                    msg = f"Quota exceeded. Retry {attempt + 1}/{max_retries} in {wait_time:.2f} seconds..."
                    logging.warning(msg)
                    print("⚠️ " + msg)
                    time.sleep(wait_time)
                else:
                    raise e
        else:
            msg = f"❌ Failed to get embeddings after {max_retries} retries. Using dummy vector."
            logging.error(msg)
            print(msg)
            dummy = [0.0] * 768
            embeddings.extend([dummy for _ in batch])
    return embeddings

# === Main Process ===
def process_json_file(json_path, model):
    filename = os.path.basename(json_path)
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        page_data = extract_text_by_page_from_dict(data)
        if not page_data:
            msg = f"❌ No valid text found in file: {filename}"
            logging.error(msg)
            print(msg)
            error_files.append(filename)
            return False

        texts = [entry["text"] for entry in page_data]

        msg = f"🔍 Generating embeddings for: {filename}"
        logging.info(msg)
        print(msg)

        embeddings = get_embeddings_with_retry(texts, model)

        enriched_data = []
        for entry, embedding in zip(page_data, embeddings):
            enriched_data.append({
                "text": entry["text"],
                "page_number": entry["page_number"],
                "embedding": embedding
            })

        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(os.path.dirname(json_path), base_name + "_embedding.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(enriched_data, f, indent=2, ensure_ascii=False)

        msg = f"✅ Embeddings saved to {output_path}"
        logging.info(msg)
        print(msg)

        doc_ids = [str(uuid.uuid4()) for _ in enriched_data]
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=PrecomputedEmbeddings(embeddings),
            persist_directory=CHROMA_PERSIST_DIR
        )
        vectorstore.add_texts(texts=texts, ids=doc_ids)

        msg = f"✅ Added {len(texts)} chunks to ChromaDB"
        logging.info(msg)
        print(msg)
        return True

    except Exception as e:
        msg = f"❌ Exception while processing {filename}: {e}"
        logging.error(msg)
        print(msg)
        error_files.append(filename)
        return False

# === Entry Point ===
if __name__ == "__main__":
    start_msg = "🚀 Starting embedding process..."
    logging.info(start_msg)
    print(start_msg)

    vertexai.init(project=PROJECT_ID, location=REGION)
    model = TextEmbeddingModel.from_pretrained(MODEL_ID)

    all_json_files = [os.path.join(INPUT_FOLDER, f) for f in os.listdir(INPUT_FOLDER)
                      if f.endswith(".json") and not f.endswith("_embedding.json")]

    total_files = len(all_json_files)
    success_count = 0

    print(f"📂 Found {total_files} files to process.")

    for json_file in all_json_files:
        if process_json_file(json_file, model):
            success_count += 1

    with open(ERROR_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(error_files, f, indent=2)

    done_msg = "🔚 All processing completed."
    stats_msg = f"📊 Total: {total_files} | ✅ Success: {success_count} | ❌ Failed: {total_files - success_count}"
    logging.info(done_msg)
    logging.info(stats_msg)
    print(done_msg)
    print(stats_msg)
