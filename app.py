import streamlit as st
from backend.upload import extract_text_from_pdf, extract_text_from_txt, save_uploaded_file
from backend.embedding import generate_embeddings
from backend.database import initialize_faiss, add_to_faiss
from backend.qa_logic import find_relevant_chunks


st.set_page_config(page_title="Document Q&A System")
st.header("Document Question-Answer System")

# Sidebar: Upload and List Documents
uploaded_files = st.sidebar.file_uploader("Upload Documents", type=["pdf", "txt"], accept_multiple_files=True)
uploaded_docs = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = save_uploaded_file(uploaded_file)
        uploaded_docs.append(file_path)

# Sidebar: Select Document
selected_doc = st.sidebar.selectbox("Select a Document", uploaded_docs)

if selected_doc:
    # Display Document Content
    st.sidebar.write(f"Selected: {selected_doc}")
    if selected_doc.endswith(".pdf"):
        doc_text = extract_text_from_pdf(selected_doc)
    elif selected_doc.endswith(".txt"):
        doc_text = extract_text_from_txt(selected_doc)
    
    st.write("Document Content Preview:")
    st.text(doc_text[:-1])  # Show first 1000 characters for preview

    # Process Document
    sentences, embeddings = generate_embeddings(doc_text)
    index = initialize_faiss(dimensions=len(embeddings[0]))
    add_to_faiss(index, embeddings)

    # Question-Answering
    st.write("Ask a Question:")
    query = st.text_input("Your Question:")
    if query:
        results = find_relevant_chunks(query, index, sentences)
        st.write("Answer :")
        for result in results:
            st.write(result)
