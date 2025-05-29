
'''

import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
import numpy as np
import vertexai
from vertexai.language_models import TextEmbeddingModel
import google.generativeai as genai

# ========== Configuration ==========
PROJECT_ID = "935917861333"
REGION = "us-west1"
EMBEDDING_MODEL_ID = "text-multilingual-embedding-002"
GENAI_API_KEY = "AIzaSyDoQKApggoaG_UNtt0A5NkpePpPYdEg64k"
PERSIST_DIR = "./chroma_db"

# ========== Dummy Embedding Wrapper ==========
class PrecomputedEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [np.zeros(768).tolist() for _ in texts]
    def embed_query(self, text):
        return np.zeros(768).tolist()

# ========== Vertex AI Embeddings ==========
vertexai.init(project=PROJECT_ID, location=REGION)
embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_ID)

def embed_text(text):
    return embedding_model.get_embeddings([text])[0].values

# ========== Chroma Vectorstore ==========
vectorstore = Chroma(
    collection_name="my_collection",
    embedding_function=PrecomputedEmbeddings(),
    persist_directory=PERSIST_DIR
)

# ========== Load Sample Documents ==========
# def initialize_docs():
#     if not vectorstore._collection.count():
#         docs = [
#             Document(page_content="Page 1 is about artificial intelligence and its applications."),
#             Document(page_content="Page 2 explains health and wellness practices."),
#             Document(page_content="Page 5 discusses the impact of climate change on coastal regions."),
#         ]
#         embeddings = [embed_text(doc.page_content) for doc in docs]
#         vectorstore.add_documents(documents=docs, embeddings=embeddings)
#         vectorstore.persist()

# ========== Retrieve Most Relevant Document ==========
def retrieve_relevant_doc(query_text):
    query_embedding = embed_text(query_text)
    results = vectorstore.similarity_search_by_vector(query_embedding, k=3)
    return results[0] if results else None

# ========== Gemini Setup ==========
genai.configure(api_key=GENAI_API_KEY)
# genai_model = genai.GenerativeModel("gemini-1.5-flash")
genai_model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")

def generate_answer(question, document):
    context = document.page_content
    # prompt = f"Use the following context to answer the question:\n\n{context}\n\nQuestion: {question}\nAnswer:"
    prompt = f"""
    You are an AI assistant. Use ONLY the following context to answer the question below.  
    Do NOT refer to any external websites or sources.

    Context:  
    {context}

    Question:  
    {question}

    Answer:
    """
    print("context" , context)
    response = genai_model.generate_content(prompt)
    return response.text

# ========== Streamlit UI ==========
st.set_page_config(page_title="Vertex + Gemini QA", layout="centered")
st.title("   StemCity ")

# st.markdown("This app uses **Vertex AI** for embedding-based search and **Gemini** for answering questions.")


user_question = st.text_input("Enter your question:", placeholder="e.g., What is the main topic of page 5?")

if st.button("Get Answer") and user_question.strip():
    with st.spinner("Searching and generating answer..."):
        doc = retrieve_relevant_doc(user_question)
        if not doc:
            st.warning("No relevant content found.")
        else:
            answer = generate_answer(user_question, doc)
            st.success("✅ Answer:")
            st.markdown(f"**{answer}**")


# streamlit run streamlit_app.py


'''
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
import numpy as np
import vertexai
from vertexai.language_models import TextEmbeddingModel
import google.generativeai as genai
import datetime

# ========== Configuration ==========
PROJECT_ID = "935917861333"
REGION = "us-west1"
EMBEDDING_MODEL_ID = "text-multilingual-embedding-002"
GENAI_API_KEY = "AIzaSyDoQKApggoaG_UNtt0A5NkpePpPYdEg64k"
PERSIST_DIR = "./chroma_db"

# ========== Dummy Embedding Wrapper ==========
class PrecomputedEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [np.zeros(768).tolist() for _ in texts]
    def embed_query(self, text):
        return np.zeros(768).tolist()

# ========== Vertex AI Embeddings ==========
vertexai.init(project=PROJECT_ID, location=REGION)
embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_ID)

def embed_text(text):
    return embedding_model.get_embeddings([text])[0].values

# ========== Chroma Vectorstore ==========
vectorstore = Chroma(
    collection_name="my_collection",
    embedding_function=PrecomputedEmbeddings(),
    persist_directory=PERSIST_DIR
)

# ========== Retrieve Most Relevant Document ==========
def retrieve_relevant_doc(query_text):
    query_embedding = embed_text(query_text)
    results = vectorstore.similarity_search_by_vector(query_embedding, k=1)
    return results[0] if results else None

# ========== Gemini Setup ==========
genai.configure(api_key=GENAI_API_KEY)
# genai_model = genai.GenerativeModel("gemini-1.5-flash")
genai_model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")

def generate_answer(question, document):
    context = document.page_content
    prompt =  f"""You are a helpful and knowledgeable assistant.
    Use ONLY the following context to answer the question. Do NOT use external sources or general knowledge.
    Respond in a professional, informative tone. Provide a structured and readable response, using bullet points or paragraphs as needed.
    
    Context:
    {context}

    Question:
    {question}

    Answer:"""
    
    print("context", context)
    response = genai_model.generate_content(prompt)
    return response.text

# ========== Logging Function ==========
def log_qa(question, context, answer, log_file="qa_log.txt"):
    timestamp = datetime.datetime.now().isoformat()
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Question: {question}\n")
        f.write(f"Context: {context}\n")
        f.write(f"Answer: {answer}\n")
        f.write("="*80 + "\n")

# ========== Streamlit UI ==========
st.set_page_config(page_title="Vertex + Gemini QA", layout="centered")
st.title("   StemCity ")

user_question = st.text_input("Enter your question:", placeholder="e.g., What is the main topic of page 5?")

if st.button("Get Answer") and user_question.strip():
    with st.spinner("Searching and generating answer..."):
        doc = retrieve_relevant_doc(user_question)
        if not doc:
            st.warning("No relevant content found.")
        else:
            answer = generate_answer(user_question, doc)
            log_qa(user_question, doc.page_content, answer)  # Log to file
            st.success("✅ Answer:")
            st.markdown(f"**{answer}**")
