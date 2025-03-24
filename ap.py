# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv
# import fitz  # PyMuPDF
# import base64
# import uuid
# from collections import defaultdict

# load_dotenv()

# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # Configuration
# PDF_FOLDER = "uploaded_pdfs"
# HIGHLIGHT_FOLDER = "highlighted_pdfs"
# os.makedirs(PDF_FOLDER, exist_ok=True)
# os.makedirs(HIGHLIGHT_FOLDER, exist_ok=True)

# # ----------------------
# # Enhanced PDF Processing
# # ----------------------
# def save_uploaded_files(uploaded_files):
#     return [os.path.join(PDF_FOLDER, f.name) for f in uploaded_files]

# def extract_text_with_positions(pdf_path):
#     doc = fitz.open(pdf_path)
#     text_blocks = []
#     for page_num in range(len(doc)):
#         page = doc.load_page(page_num)
#         blocks = page.get_text("blocks")
#         for block in blocks:
#             x0, y0, x1, y1, text, _, block_type = block
#             if block_type == 0 and text.strip():
#                 text_blocks.append({
#                     "text": text.strip(),
#                     "page": page_num,
#                     "coordinates": (x0, y0, x1, y1)
#                 })
#     doc.close()
#     return text_blocks

# def get_text_chunks_with_metadata(pdf_paths):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000, chunk_overlap=200, length_function=len
#     )
#     all_chunks = []
#     for pdf_path in pdf_paths:
#         text_blocks = extract_text_with_positions(pdf_path)
#         for block in text_blocks:
#             chunks = text_splitter.split_text(block["text"])
#             for chunk in chunks:
#                 all_chunks.append({
#                     "text": chunk,
#                     "metadata": {
#                         "source": pdf_path,
#                         "page": block["page"],
#                         "coordinates": block["coordinates"]
#                     }
#                 })
#     return all_chunks

# # ----------------------
# # Enhanced Highlighting
# # ----------------------
# def highlight_pdf_regions(pdf_path, regions):
#     """Highlight multiple regions in a PDF"""
#     doc = fitz.open(pdf_path)
#     for page_num, coordinates in regions:
#         page = doc.load_page(page_num)
#         x0, y0, x1, y1 = coordinates
#         rect = fitz.Rect(x0, y0, x1, y1)
#         highlight = page.add_highlight_annot(rect)
#         highlight.set_colors(stroke=(1, 1, 0))
    
#     filename = f"highlighted_{uuid.uuid4().hex}_{os.path.basename(pdf_path)}"
#     highlighted_path = os.path.join(HIGHLIGHT_FOLDER, filename)
#     doc.save(highlighted_path)
#     doc.close()
#     return highlighted_path

# # ----------------------
# # Enhanced QA System
# # ----------------------
# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     texts = [chunk["text"] for chunk in text_chunks]
#     metadatas = [chunk["metadata"] for chunk in text_chunks]
#     vector_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
#     vector_store.save_local("faiss_index")

# def get_conversational_chain():
#     prompt_template = """
#     Analyze and synthesize information from the following contexts to answer the question.
#     Consider all relevant sections and provide a comprehensive answer. If the answer isn't found,
#     state "Answer is not available in the context."

#     Contexts:
#     {context}

#     Question: {question}

#     Provide a detailed answer with references to document sections:
#     """
#     model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# # ----------------------
# # Enhanced UI Handling
# # ----------------------
# def display_pdf(pdf_path):
#     with open(pdf_path, "rb") as f:
#         base64_pdf = base64.b64encode(f.read()).decode("utf-8")
#     pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="800"></iframe>'
#     st.markdown(pdf_display, unsafe_allow_html=True)

# def user_input(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     docs = vector_store.similarity_search(user_question, k=1)  # Increased to 5 relevant chunks
    
#     if not docs:
#         st.warning("No relevant information found in documents.")
#         return

#     # Process multiple sources and pages
#     source_regions = defaultdict(list)
#     sources_info = set()
    
#     for doc in docs:
#         source = doc.metadata["source"]
#         page = doc.metadata["page"]
#         coords = doc.metadata["coordinates"]
#         source_regions[source].append((page, coords))
#         sources_info.add((source, page))

#     # Generate highlighted PDFs
#     highlighted_files = []
#     for source, regions in source_regions.items():
#         try:
#             highlighted = highlight_pdf_regions(source, regions)
#             highlighted_files.append((source, highlighted))
#         except Exception as e:
#             st.error(f"Error processing {source}: {str(e)}")

#     # Generate answer
#     chain = get_conversational_chain()
#     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
#     # Display results
#     st.subheader("Answer")
#     st.markdown(f"**{response['output_text']}**")

#     st.subheader("Source References")
#     for source, page in sources_info:
#         st.markdown(f"- `{os.path.basename(source)}` (Page {page + 1})")

#     st.subheader("Relevant Document Sections")
#     for source_path, highlighted_path in highlighted_files:
#         st.markdown(f"**{os.path.basename(source_path)}**")
#         display_pdf(highlighted_path)
#         with open(highlighted_path, "rb") as f:
#             st.download_button(
#                 f"Download highlighted {os.path.basename(source_path)}",
#                 f.read(),
#                 file_name=f"highlighted_{os.path.basename(source_path)}"
#             )

# # ----------------------
# # Main App
# # ----------------------
# def main():
#     st.set_page_config("Smart PDF Analyzer", page_icon=":books:")
#     st.header("Advanced PDF Analysis with Gemini")
    
#     with st.sidebar:
#         st.title("Document Management")
#         uploaded_files = st.file_uploader(
#             "Upload research papers/documentation",
#             type=["pdf"],
#             accept_multiple_files=True
#         )
#         if uploaded_files:
#             saved_files = save_uploaded_files(uploaded_files)
#             with st.spinner("Processing documents..."):
#                 text_chunks = get_text_chunks_with_metadata(saved_files)
#                 get_vector_store(text_chunks)
#             st.success("Documents indexed successfully!")

#     user_question = st.text_input("Ask complex questions (e.g., 'Compare and contrast X and Y in the documents'):")
#     if user_question:
#         with st.spinner("Analyzing documents..."):
#             user_input(user_question)

# if __name__ == "__main__":
#     main()