import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import fitz  # PyMuPDF
import base64

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Store uploaded PDFs
PDF_FOLDER = "uploaded_pdfs"
if not os.path.exists(PDF_FOLDER):
    os.makedirs(PDF_FOLDER)

# ----------------------
# Save Uploaded Files
# ----------------------
def save_uploaded_files(uploaded_files):
    saved_files = []
    for file in uploaded_files:
        file_path = os.path.join(PDF_FOLDER, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        saved_files.append(file_path)
    return saved_files

# ----------------------
# Extract Text with Positions
# ----------------------
def extract_text_with_positions(pdf_path):
    """Extract text with page numbers and coordinates using PyMuPDF."""
    doc = fitz.open(pdf_path)
    text_blocks = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("blocks")  # Get text blocks with coordinates
        for block in blocks:
            x0, y0, x1, y1, text, block_no, block_type = block
            if block_type == 0:  # Only process text blocks
                text_blocks.append({
                    "text": text.strip(),
                    "page": page_num,
                    "coordinates": (x0, y0, x1, y1)
                })
    doc.close()
    return text_blocks

# ----------------------
# Split Text into Chunks with Metadata
# ----------------------
def get_text_chunks_with_metadata(pdf_paths):
    """Split text into chunks with metadata (page, coordinates)."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    all_chunks = []
    for pdf_path in pdf_paths:
        text_blocks = extract_text_with_positions(pdf_path)
        for block in text_blocks:
            chunks = text_splitter.split_text(block["text"])
            for chunk in chunks:
                all_chunks.append({
                    "text": chunk,
                    "metadata": {
                        "source": pdf_path,
                        "page": block["page"],
                        "coordinates": block["coordinates"]
                    }
                })
    return all_chunks

# ----------------------
# Create Vector Store
# ----------------------
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    texts = [chunk["text"] for chunk in text_chunks]
    metadatas = [chunk["metadata"] for chunk in text_chunks]
    vector_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
    vector_store.save_local("faiss_index")

# ----------------------
# Highlight PDF Region
# ----------------------
def highlight_pdf_region(pdf_path, page_num, coordinates):
    """Highlight a specific region in the PDF using coordinates."""
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    x0, y0, x1, y1 = coordinates
    rect = fitz.Rect(x0, y0, x1, y1)
    highlight = page.add_highlight_annot(rect)
    highlight.set_colors(stroke=(1, 1, 0))  # Yellow highlight
    highlight.update()
    highlighted_pdf_path = "highlighted_pdf.pdf"
    doc.save(highlighted_pdf_path)
    doc.close()
    return highlighted_pdf_path

# ----------------------
# Conversational Chain
# ----------------------
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not available in the context, 
    just say: "Answer is not available in the context." 

    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.8)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# ----------------------
# Display PDF in Streamlit
# ----------------------
def display_pdf(pdf_path):
    with open(pdf_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="800"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# ----------------------
# User Input Handling
# ----------------------
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question, k=5)
    
    if not docs:
        st.write("No relevant document found.")
        return

    # Get metadata from the top matching chunk
    source = docs[0].metadata["source"]
    page = docs[0].metadata["page"]
    coordinates = docs[0].metadata["coordinates"]
    
    # Highlight the region in the PDF
    highlighted_pdf = highlight_pdf_region(source, page, coordinates)
    
    # Display answer and PDF
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Answer:", response["output_text"])
    st.write(f"**Source:** `{source}` (Page {page + 1})")
    
    display_pdf(highlighted_pdf)
    with open(highlighted_pdf, "rb") as f:
        st.download_button("Download Highlighted PDF", f.read(), file_name="highlighted.pdf")

# ----------------------
# Streamlit UI
# ----------------------
def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini")
    user_question = st.text_input("Ask a Question")
    
    with st.sidebar:
        st.title("Upload PDFs")
        uploaded_files = st.file_uploader("Choose PDFs", accept_multiple_files=True, type=["pdf"])
        if uploaded_files:
            saved_files = save_uploaded_files(uploaded_files)
            text_chunks = get_text_chunks_with_metadata(saved_files)
            get_vector_store(text_chunks)
            st.success("PDFs processed!")

    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()

    