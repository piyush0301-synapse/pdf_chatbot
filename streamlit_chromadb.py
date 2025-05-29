

import streamlit as st
import chromadb

# Initialize the client for your persisted ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

# Get your collection by name
collection = client.get_collection("my_collection")

# Retrieve all documents, metadatas, and embeddings (ids are always included)
data = collection.get(include=["documents", "metadatas", "embeddings"])

st.title("ChromaDB Collection Viewer")
st.write(f"Total documents in collection: {len(data['ids'])}")

# Dropdown to select document by ID
selected_id = st.selectbox("Select Document ID to View", data["ids"])

# Find index of selected document
index = data["ids"].index(selected_id)

# Show selected document details
st.markdown(f"### Document ID: {selected_id}")
st.write(f"**Text Preview:** {data['documents'][index][:200]}...")
st.write(f"**Metadata:** {data['metadatas'][index]}")

embedding = data['embeddings'][index]

# Show embedding length
st.write(f"**Embedding Length:** {len(embedding)}")

# Show first 100 values as a list preview
st.write("**Embedding Vector (first 100 values):**")
st.write(embedding[:100])

# # Show a bar chart of the first 50 embedding values
# st.bar_chart(embedding[:50])


# Run code through this 
# streamlit run streamlit_chromadb.py

'''
import streamlit as st
import chromadb
from chromadb.config import Settings

# Initialize client with persistent storage settings
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",  # local persistent DB
    persist_directory="./chroma_db"
))

# Try to get existing collection or create new one
try:
    collection = client.get_collection("my_collection")
except Exception:
    # If collection does not exist, create it and add a dummy document
    collection = client.create_collection("my_collection")
    collection.add(
        documents=["This is a sample document."],
        metadatas=[{"source": "generated"}],
        ids=["doc_1"]
    )
    st.info("Created new collection 'my_collection' with sample document.")

# Fetch all data from the collection
data = collection.get(include=["documents", "metadatas", "embeddings"])

st.title("ChromaDB Collection Viewer")
st.write(f"Total documents in collection: {len(data['ids'])}")

# Select document by ID dropdown
selected_id = st.selectbox("Select Document ID to View", data["ids"])

# Find index of selected document
index = data["ids"].index(selected_id)

# Display document info
st.markdown(f"### Document ID: {selected_id}")
st.write(f"**Text Preview:** {data['documents'][index][:200]}...")
st.write(f"**Metadata:** {data['metadatas'][index]}")

embedding = data["embeddings"][index]
st.write(f"**Embedding Length:** {len(embedding)}")
st.write("**Embedding Vector (first 100 values):**")
st.write(embedding[:100])
'''