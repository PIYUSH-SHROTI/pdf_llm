# import streamlit as st
# import os
# import base64
# import uuid
# from collections import defaultdict
# from dotenv import load_dotenv
# import google.generativeai as genai
# import fitz  # PyMuPDF
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate

# # Load environment variables
# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# print(555)




# # Configuration
# PDF_FOLDER = "uploaded_pdfs"
# HIGHLIGHT_FOLDER = "highlighted_pdfs"
# os.makedirs(PDF_FOLDER, exist_ok=True)
# os.makedirs(HIGHLIGHT_FOLDER, exist_ok=True)

# # ----------------------
# # Line-by-Line Processing (Keep this)
# # ----------------------
# def extract_lines_with_positions(pdf_path):
#     """Extract individual text lines with coordinates"""
#     doc = fitz.open(pdf_path)
#     text_lines = []
    
#     for page_num in range(len(doc)):
#         page = doc.load_page(page_num)
#         blocks = page.get_text("dict")["blocks"]
        
#         for block in blocks:
#             if block['type'] == 0:  # Text block
#                 for line in block["lines"]:
#                     line_text = " ".join(span["text"] for span in line["spans"])
#                     if line_text.strip():
#                         text_lines.append({
#                             "text": line_text,
#                             "page": page_num,
#                             "coordinates": line["bbox"]
#                         })
#     doc.close()
#     return text_lines

# def get_line_chunks(pdf_paths):
#     """Treat each line as an individual chunk"""
#     all_chunks = []
#     for pdf_path in pdf_paths:
#         lines = extract_lines_with_positions(pdf_path)
#         for line in lines:
#             all_chunks.append({
#                 "text": line["text"],
#                 "metadata": {
#                     "source": pdf_path,
#                     "page": line["page"],
#                     "coordinates": line["coordinates"]
#                 }
#             })
#     return all_chunks

# # ----------------------
# # Line Highlighting (Keep this)
# # ----------------------
# def highlight_text_lines(pdf_path, line_regions):
#     """Highlight individual text lines"""
#     doc = fitz.open(pdf_path)
#     for page_num, coords in line_regions:
#         page = doc.load_page(page_num)
#         rect = fitz.Rect(coords)
        
#         # Create line highlight
#         annot = page.add_highlight_annot(rect)
#         annot.set_colors({"stroke": (1, 1, 0), "fill": (1, 1, 0.3)})
#         annot.set_opacity(0.4)
#         annot.update()
    
#     filename = f"line_hl_{uuid.uuid4().hex[:6]}_{os.path.basename(pdf_path)}"
#     save_path = os.path.join(HIGHLIGHT_FOLDER, filename)
#     doc.save(save_path)
#     doc.close()
#     return save_path

# # ----------------------
# # Previous Answer Generation System (Restore this)
# # ----------------------
# def create_vector_store(chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     texts = [c["text"] for c in chunks]
#     metadatas = [c["metadata"] for c in chunks]
#     vector_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
#     vector_store.save_local("faiss_index")
#     print(55555)

# def get_qa_pipeline():
#     prompt_template = """
#     Analyze the context from technical documents and provide detailed answers.
#     Format your response with clear section references. If unsure, state:
#     "Information not found in documents."
    
#     Context:
#     {context}
    
#     Question: {question}
    
#     Structured Answer:
#     """
#     model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.2)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# # ----------------------
# # Integrated UI Components
# # ----------------------
# def display_pdf(pdf_path):
#     with open(pdf_path, "rb") as f:
#         base64_pdf = base64.b64encode(f.read()).decode("utf-8")
#     return f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="900"></iframe>'

# def process_query(question):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     docs = vector_store.similarity_search(question, k=10)
    
#     if not docs:
#         st.warning("No matching content found in documents.")
#         return

#     # Organize highlights by document
#     doc_highlights = defaultdict(lambda: {"pages": set(), "lines": []})
#     for doc in docs:
#         source = doc.metadata["source"]
#         doc_highlights[source]["pages"].add(doc.metadata["page"] + 1)
#         doc_highlights[source]["lines"].append((
#             doc.metadata["page"],
#             doc.metadata["coordinates"]
#         ))
    
#     # Generate highlighted versions
#     hl_files = []
#     for source, data in doc_highlights.items():
#         try:
#             hl_path = highlight_text_lines(source, data["lines"])
#             hl_files.append((source, hl_path))
#         except Exception as e:
#             st.error(f"Error processing {os.path.basename(source)}: {str(e)}")

#     # Generate answer using previous system
#     chain = get_qa_pipeline()
#     response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    
#     # Display results
#     st.subheader("Technical Answer")
#     st.markdown(f"```\n{response['output_text']}\n```")
    
#     st.subheader("Document References")
#     for source, data in doc_highlights.items():
#         pages = ", ".join(map(str, sorted(data["pages"])))
#         st.markdown(f"🔖 **{os.path.basename(source)}** - Pages: {pages}")
    
#     st.subheader("Highlighted Lines Preview")
#     for src_path, hl_path in hl_files:
#         with st.expander(f"📑 {os.path.basename(src_path)}"):
#             st.markdown(display_pdf(hl_path), unsafe_allow_html=True)
#             with open(hl_path, "rb") as f:
#                 st.download_button(
#                     "Download Highlighted Version",
#                     f.read(),
#                     file_name=f"hl_{os.path.basename(src_path)}"
#                 )

# # ----------------------
# # Main Application
# # ----------------------
# def main():
#     st.set_page_config("Technical Document Analyzer", page_icon=":books:")
#     st.title("Integrated PDF Analysis System")
#     st.markdown("Line-level highlighting with comprehensive answers")
    
#     with st.sidebar:
#         st.header("Document Management")
#         uploaded_files = st.file_uploader(
#             "Upload Technical PDFs",
#             type=["pdf"],
#             accept_multiple_files=True
#         )
        
#         if uploaded_files:
#             with st.spinner("Analyzing document structure..."):
#                 saved_files = [os.path.join(PDF_FOLDER, f.name) for f in uploaded_files]
#                 chunks = get_line_chunks(saved_files)
#                 create_vector_store(chunks)
#             st.success(f"Processed {len(uploaded_files)} documents!")

#     query = st.text_input("Enter technical query:")
#     if query:
#         with st.spinner("Searching documents..."):
#             process_query(query)

# if __name__ == "__main__":
#     main()





























# # import streamlit as st
# # import os
# # import base64
# # import uuid
# # from collections import defaultdict
# # from dotenv import load_dotenv
# # import google.generativeai as genai
# # import fitz  # PyMuPDF
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# # from langchain.vectorstores import FAISS
# # from langchain.chains.question_answering import load_qa_chain
# # from langchain.prompts import PromptTemplate

# # # Load environment variables
# # load_dotenv()
# # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # # Configuration
# # PDF_FOLDER = "uploaded_pdfs"
# # HIGHLIGHT_FOLDER = "highlighted_pdfs"
# # os.makedirs(PDF_FOLDER, exist_ok=True)
# # os.makedirs(HIGHLIGHT_FOLDER, exist_ok=True)

# # # ----------------------
# # # Line-by-Line Processing
# # # ----------------------
# # def extract_lines_with_positions(pdf_path):
# #     """Extract individual text lines with coordinates"""
# #     doc = fitz.open(pdf_path)
# #     text_lines = []
    
# #     for page_num in range(len(doc)):
# #         page = doc.load_page(page_num)
# #         blocks = page.get_text("dict")["blocks"]
        
# #         for block in blocks:
# #             if block['type'] == 0:  # Text block
# #                 for line in block["lines"]:
# #                     line_text = " ".join(span["text"] for span in line["spans"])
# #                     if line_text.strip():
# #                         text_lines.append({
# #                             "text": line_text,
# #                             "page": page_num,
# #                             "coordinates": line["bbox"]
# #                         })
# #     doc.close()
# #     return text_lines

# # def get_line_chunks(pdf_paths):
# #     """Treat each line as an individual chunk"""
# #     all_chunks = []
# #     for pdf_path in pdf_paths:
# #         lines = extract_lines_with_positions(pdf_path)
# #         for line in lines:
# #             all_chunks.append({
# #                 "text": line["text"],
# #                 "metadata": {
# #                     "source": pdf_path,
# #                     "page": line["page"],
# #                     "coordinates": line["coordinates"]
# #                 }
# #             })
# #     return all_chunks

# # # ----------------------
# # # Line Highlighting
# # # ----------------------
# # def highlight_text_lines(pdf_path, line_regions):
# #     """Highlight individual text lines"""
# #     doc = fitz.open(pdf_path)
# #     for page_num, coords in line_regions:
# #         page = doc.load_page(page_num)
# #         rect = fitz.Rect(coords)
        
# #         # Create line highlight
# #         annot = page.add_highlight_annot(rect)
# #         annot.set_colors({"stroke": (1, 1, 0), "fill": (1, 1, 0.3)})
# #         annot.set_opacity(0.4)
# #         annot.update()
    
# #     filename = f"line_hl_{uuid.uuid4().hex[:6]}_{os.path.basename(pdf_path)}"
# #     save_path = os.path.join(HIGHLIGHT_FOLDER, filename)
# #     doc.save(save_path)
# #     doc.close()
# #     return save_path

# # # ----------------------
# # # Modified QA System
# # # ----------------------
# # def create_vector_store(chunks):
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# #     texts = [c["text"] for c in chunks]
# #     metadatas = [c["metadata"] for c in chunks]
# #     vector_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
# #     vector_store.save_local("faiss_index")

# # def process_query(question):
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# #     vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
# #     docs = vector_store.similarity_search(question, k=10)  # Get more lines for context
    
# #     if not docs:
# #         st.warning("No matching lines found in documents.")
# #         return

# #     # Organize highlights by document
# #     doc_highlights = defaultdict(lambda: {"pages": set(), "lines": []})
# #     for doc in docs:
# #         source = doc.metadata["source"]
# #         doc_highlights[source]["pages"].add(doc.metadata["page"] + 1)
# #         doc_highlights[source]["lines"].append((
# #             doc.metadata["page"],
# #             doc.metadata["coordinates"]
# #         ))
    
# #     # Generate highlighted versions
# #     hl_files = []
# #     for source, data in doc_highlights.items():
# #         try:
# #             hl_path = highlight_text_lines(source, data["lines"])
# #             hl_files.append((source, hl_path))
# #         except Exception as e:
# #             st.error(f"Error processing {os.path.basename(source)}: {str(e)}")

# #     # Generate answer using line-based context
# #     chain = get_qa_pipeline()
# #     response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    
# #     # Display results
# #     st.subheader("Answer from Document Lines")
# #     st.markdown(response["output_text"])
    
# #     st.subheader("Highlighted Lines Preview")
# #     for src_path, hl_path in hl_files:
# #         with st.expander(f"📄 {os.path.basename(src_path)}"):
# #             st.markdown(display_pdf(hl_path), unsafe_allow_html=True)
# #             with open(hl_path, "rb") as f:
# #                 st.download_button(
# #                     "Download Highlighted Lines",
# #                     f.read(),
# #                     file_name=f"line_hl_{os.path.basename(src_path)}"
# #                 )

# # # ----------------------
# # # Rest of the code remains similar to previous version
# # # (display_pdf, main function, etc.)
# # # ----------------------

# # def display_pdf(pdf_path):
# #     with open(pdf_path, "rb") as f:
# #         base64_pdf = base64.b64encode(f.read()).decode("utf-8")
# #     return f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="900"></iframe>'

# # def get_qa_pipeline():
# #     prompt_template = """
# #     Analyze these document lines and provide a precise answer. 
# #     Reference specific line numbers if possible.
    
# #     Context lines:
# #     {context}
    
# #     Question: {question}
    
# #     Line-based Answer:
# #     """
# #     model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
# #     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
# #     return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# # def main():
# #     st.set_page_config("Line-Based PDF Analyzer", page_icon=":page_facing_up:")
# #     st.title("Line-by-Line PDF Highlighter")
    
# #     with st.sidebar:
# #         st.header("Upload Documents")
# #         uploaded_files = st.file_uploader(
# #             "Upload PDFs for line analysis",
# #             type=["pdf"],
# #             accept_multiple_files=True
# #         )
        
# #         if uploaded_files:
# #             with st.spinner("Extracting lines..."):
# #                 saved_files = [os.path.join(PDF_FOLDER, f.name) for f in uploaded_files]
# #                 chunks = get_line_chunks(saved_files)
# #                 create_vector_store(chunks)
# #             st.success(f"Processed {len(uploaded_files)} documents with line extraction!")

# #     query = st.text_input("Search for specific content:")
# #     if query:
# #         with st.spinner("Finding matching lines..."):
# #             process_query(query)

# # if __name__ == "__main__":
# #     main()























# # import streamlit as st
# # import os
# # import base64
# # import uuid
# # from collections import defaultdict
# # from dotenv import load_dotenv
# # import google.generativeai as genai
# # import fitz  # PyMuPDF
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# # from langchain.vectorstores import FAISS
# # from langchain.chains.question_answering import load_qa_chain
# # from langchain.prompts import PromptTemplate

# # # Load environment variables
# # load_dotenv()
# # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # # Configuration
# # PDF_FOLDER = "uploaded_pdfs"
# # HIGHLIGHT_FOLDER = "highlighted_pdfs"
# # os.makedirs(PDF_FOLDER, exist_ok=True)
# # os.makedirs(HIGHLIGHT_FOLDER, exist_ok=True)

# # # ----------------------
# # # Enhanced PDF Processing
# # # ----------------------
# # def extract_paragraphs_with_style(pdf_path):
# #     """Extract paragraphs with styling information and exact coordinates"""
# #     doc = fitz.open(pdf_path)
# #     paragraphs = []
    
# #     for page_num in range(len(doc)):
# #         page = doc.load_page(page_num)
# #         blocks = page.get_text("dict")["blocks"]
        
# #         for block in blocks:
# #             if block['type'] == 0:  # Text block
# #                 text = []
# #                 for line in block["lines"]:
# #                     for span in line["spans"]:
# #                         text.append(span["text"])
# #                 full_text = " ".join(text).strip()
                
# #                 if full_text:
# #                     paragraphs.append({
# #                         "text": full_text,
# #                         "page": page_num,
# #                         "coordinates": block["bbox"],
# #                         "font_size": block["lines"][0]["spans"][0]["size"]
# #                     })
# #     doc.close()
# #     return paragraphs

# # def get_content_chunks(pdf_paths):
# #     """Preserve original document structure in chunks"""
# #     chunks = []
# #     for path in pdf_paths:
# #         paragraphs = extract_paragraphs_with_style(path)
# #         for para in paragraphs:
# #             chunks.append({
# #                 "text": para["text"],
# #                 "metadata": {
# #                     "source": path,
# #                     "page": para["page"],
# #                     "coordinates": para["coordinates"],
# #                     "font_size": para["font_size"]
# #                 }
# #             })
# #     return chunks

# # # ----------------------
# # # Precision Highlighting
# # # ----------------------
# # def highlight_document(pdf_path, regions):
# #     """Highlight sections with exact style matching the sample"""
# #     doc = fitz.open(pdf_path)
# #     for page_num, coords in regions:
# #         page = doc.load_page(page_num)
# #         rect = fitz.Rect(coords)
        
# #         # Create semi-transparent yellow highlight
# #         annot = page.add_highlight_annot(rect)
# #         annot.set_colors({"stroke": (1, 1, 0), "fill": (1, 1, 0.3)})
# #         annot.set_opacity(0.35)
# #         annot.update()
    
# #     filename = f"hl_{uuid.uuid4().hex[:6]}_{os.path.basename(pdf_path)}"
# #     save_path = os.path.join(HIGHLIGHT_FOLDER, filename)
# #     doc.save(save_path)
# #     doc.close()
# #     return save_path

# # # ----------------------
# # # Enhanced QA System
# # # ----------------------
# # def create_vector_store(chunks):
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# #     texts = [c["text"] for c in chunks]
# #     metadatas = [c["metadata"] for c in chunks]
# #     vector_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
# #     vector_store.save_local("faiss_index")

# # def get_qa_pipeline():
# #     prompt_template = """
# #     Analyze the context from technical documents and provide detailed answers.
# #     Format your response with clear section references. If unsure, state:
# #     "Information not found in documents."
    
# #     Context:
# #     {context}
    
# #     Question: {question}
    
# #     Structured Answer:
# #     """
# #     model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.2)
# #     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
# #     return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# # # ----------------------
# # # UI Components
# # # ----------------------
# # def display_pdf(pdf_path):
# #     with open(pdf_path, "rb") as f:
# #         base64_pdf = base64.b64encode(f.read()).decode("utf-8")
# #     return f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="900"></iframe>'

# # def process_query(question):
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# #     vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
# #     docs = vector_store.similarity_search(question, k=1)
    
# #     if not docs:
# #         st.warning("No matching content found in documents.")
# #         return

# #     # Organize highlights by document
# #     doc_highlights = defaultdict(lambda: {"pages": set(), "regions": []})
# #     for doc in docs:
# #         source = doc.metadata["source"]
# #         doc_highlights[source]["pages"].add(doc.metadata["page"] + 1)
# #         doc_highlights[source]["regions"].append((
# #             doc.metadata["page"],
# #             doc.metadata["coordinates"]
# #         ))
    
# #     # Generate highlighted versions
# #     hl_files = []
# #     for source, data in doc_highlights.items():
# #         try:
# #             hl_path = highlight_document(source, data["regions"])
# #             hl_files.append((source, hl_path))
# #         except Exception as e:
# #             st.error(f"Error processing {os.path.basename(source)}: {str(e)}")

# #     # Generate answer
# #     chain = get_qa_pipeline()
# #     response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    
# #     # Display results
# #     st.subheader("Technical Answer")
# #     st.markdown(f"```\n{response['output_text']}\n```")
    
# #     st.subheader("Reference Documents")
# #     for source, data in doc_highlights.items():
# #         pages = ", ".join(map(str, sorted(data["pages"])))
# #         st.markdown(f"🔖 **{os.path.basename(source)}** - Pages: {pages}")
    
# #     st.subheader("Highlighted Sections Preview")
# #     for src_path, hl_path in hl_files:
# #         with st.expander(f"📑 {os.path.basename(src_path)}"):
# #             st.markdown(display_pdf(hl_path), unsafe_allow_html=True)
# #             with open(hl_path, "rb") as f:
# #                 st.download_button(
# #                     "Download Highlighted Version",
# #                     f.read(),
# #                     file_name=f"hl_{os.path.basename(src_path)}"
# #                 )

# # # ----------------------
# # # Main Application
# # # ----------------------
# # def main():
# #     st.set_page_config("Technical Document Analyzer", page_icon=":books:")
# #     st.title("Advanced PDF Highlighter")
# #     st.markdown("Upload technical documents and get precise highlighted answers")
    
# #     with st.sidebar:
# #         st.header("Document Management")
# #         uploaded_files = st.file_uploader(
# #             "Upload Technical PDFs",
# #             type=["pdf"],
# #             accept_multiple_files=True
# #         )
        
# #         if uploaded_files:
# #             with st.spinner("Analyzing document structure..."):
# #                 saved_files = [os.path.join(PDF_FOLDER, f.name) for f in uploaded_files]
# #                 chunks = get_content_chunks(saved_files)
# #                 create_vector_store(chunks)
# #             st.success(f"Processed {len(uploaded_files)} documents!")

# #     query = st.text_input("Enter technical query:")
# #     if query:
# #         with st.spinner("Searching documents..."):
# #             process_query(query)

# # if __name__ == "__main__":
# #     main()























# # import streamlit as st
# # import os
# # import base64
# # import uuid
# # from collections import defaultdict
# # from dotenv import load_dotenv
# # import google.generativeai as genai
# # import fitz  # PyMuPDF
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# # from langchain.vectorstores import FAISS
# # from langchain.chains.question_answering import load_qa_chain
# # from langchain.prompts import PromptTemplate

# # # Load environment variables
# # load_dotenv()
# # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # # Configuration
# # PDF_FOLDER = "uploaded_pdfs"
# # HIGHLIGHT_FOLDER = "highlighted_pdfs"
# # os.makedirs(PDF_FOLDER, exist_ok=True)
# # os.makedirs(HIGHLIGHT_FOLDER, exist_ok=True)

# # # ----------------------
# # # File Handling
# # # ----------------------
# # def save_uploaded_files(uploaded_files):
# #     saved_files = []
# #     for file in uploaded_files:
# #         file_path = os.path.join(PDF_FOLDER, file.name)
# #         with open(file_path, "wb") as f:
# #             f.write(file.getbuffer())
# #         saved_files.append(file_path)
# #     return saved_files

# # # ----------------------
# # # PDF Processing
# # # ----------------------
# # def extract_text_with_positions(pdf_path):
# #     """Extract text with paragraph-level coordinates using PyMuPDF"""
# #     doc = fitz.open(pdf_path)
# #     text_blocks = []
    
# #     for page_num in range(len(doc)):
# #         page = doc.load_page(page_num)
# #         blocks = page.get_text("dict")["blocks"]
        
# #         for b in blocks:
# #             if b['type'] == 0:  # Text block
# #                 text = " ".join(
# #                     span["text"] 
# #                     for line in b["lines"] 
# #                     for span in line["spans"]
# #                 ).strip()
                
# #                 if text:
# #                     text_blocks.append({
# #                         "text": text,
# #                         "page": page_num,
# #                         "coordinates": (
# #                             b["bbox"][0],  # x0
# #                             b["bbox"][1],  # y0
# #                             b["bbox"][2],  # x1
# #                             b["bbox"][3]   # y1
# #                         )
# #                     })
# #     doc.close()
# #     return text_blocks

# # def get_text_chunks(pdf_paths):
# #     """Treat each paragraph as a single chunk"""
# #     all_chunks = []
# #     for pdf_path in pdf_paths:
# #         text_blocks = extract_text_with_positions(pdf_path)
# #         for block in text_blocks:
# #             all_chunks.append({
# #                 "text": block["text"],
# #                 "metadata": {
# #                     "source": pdf_path,
# #                     "page": block["page"],
# #                     "coordinates": block["coordinates"]
# #                 }
# #             })
# #     return all_chunks

# # # ----------------------
# # # Vector Store
# # # ----------------------
# # def create_vector_store(text_chunks):
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# #     texts = [chunk["text"] for chunk in text_chunks]
# #     metadatas = [chunk["metadata"] for chunk in text_chunks]
# #     vector_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
# #     vector_store.save_local("faiss_index")

# # # ----------------------
# # # PDF Highlighting
# # # ----------------------
# # def highlight_paragraphs(pdf_path, regions):
# #     """Highlight multiple paragraphs in a PDF with styling"""
# #     doc = fitz.open(pdf_path)
# #     for page_num, coordinates in regions:
# #         page = doc.load_page(page_num)
# #         rect = fitz.Rect(*coordinates)
        
# #         # Create highlight with yellow color and opacity
# #         highlight = page.add_highlight_annot(rect)
# #         highlight.set_colors({"stroke": (1, 1, 0), "fill": (1, 1, 0.3)})
# #         highlight.set_opacity(0.4)
# #         highlight.update()
    
# #     filename = f"highlighted_{uuid.uuid4().hex[:8]}_{os.path.basename(pdf_path)}"
# #     highlighted_path = os.path.join(HIGHLIGHT_FOLDER, filename)
# #     doc.save(highlighted_path)
# #     doc.close()
# #     return highlighted_path

# # # ----------------------
# # # QA System
# # # ----------------------
# # def get_qa_chain():
# #     prompt_template = """
# #     Analyze the context and provide a detailed answer. If the answer isn't in the context, 
# #     state "Answer not found in documents."

# #     Context:
# #     {context}

# #     Question: {question}

# #     Provide a comprehensive answer with paragraph references:
# #     """
# #     model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
# #     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
# #     return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# # # ----------------------
# # # UI Components
# # # ----------------------
# # def display_pdf(pdf_path):
# #     """Embed PDF in Streamlit UI"""
# #     with open(pdf_path, "rb") as f:
# #         base64_pdf = base64.b64encode(f.read()).decode("utf-8")
# #     pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="800"></iframe>'
# #     st.markdown(pdf_display, unsafe_allow_html=True)

# # def handle_query(user_question):
# #     """Process user question and display results"""
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# #     vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
# #     docs = vector_store.similarity_search(user_question, k=5)
    
# #     if not docs:
# #         st.warning("No relevant information found in documents.")
# #         return

# #     # Organize results by source document
# #     source_data = defaultdict(lambda: {"pages": set(), "regions": []})
# #     for doc in docs:
# #         source = doc.metadata["source"]
# #         source_data[source]["pages"].add(doc.metadata["page"] + 1)
# #         source_data[source]["regions"].append((
# #             doc.metadata["page"],
# #             doc.metadata["coordinates"]
# #         ))

# #     # Generate highlighted PDFs
# #     highlighted_files = []
# #     for source, data in source_data.items():
# #         try:
# #             highlighted_path = highlight_paragraphs(source, data["regions"])
# #             highlighted_files.append((source, highlighted_path))
# #         except Exception as e:
# #             st.error(f"Error processing {os.path.basename(source)}: {str(e)}")

# #     # Generate answer
# #     chain = get_qa_chain()
# #     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
# #     # Display results
# #     st.subheader("Answer")
# #     st.markdown(response["output_text"])

# #     st.subheader("Document References")
# #     for source, data in source_data.items():
# #         pages = ", ".join(map(str, sorted(data["pages"])))
# #         st.markdown(f"📄 **{os.path.basename(source)}** (Pages: {pages})")

# #     st.subheader("Highlighted Sections")
# #     for source_path, highlighted_path in highlighted_files:
# #         st.markdown(f"### {os.path.basename(source_path)}")
# #         display_pdf(highlighted_path)
# #         with open(highlighted_path, "rb") as f:
# #             st.download_button(
# #                 f"Download {os.path.basename(source_path)}",
# #                 f.read(),
# #                 file_name=f"highlighted_{os.path.basename(source_path)}"
# #             )

# # # ----------------------
# # # Main App
# # # ----------------------
# # def main():
# #     st.set_page_config("Document Analyzer", page_icon=":mag:")
# #     st.title("PDF Insight Explorer")
# #     st.markdown("Upload PDFs and ask questions about their content")
    
# #     # File upload sidebar
# #     with st.sidebar:
# #         st.header("Document Management")
# #         uploaded_files = st.file_uploader(
# #             "Choose PDF documents",
# #             type=["pdf"],
# #             accept_multiple_files=True
# #         )
        
# #         if uploaded_files:
# #             with st.spinner("Processing documents..."):
# #                 saved_files = save_uploaded_files(uploaded_files)
# #                 text_chunks = get_text_chunks(saved_files)
# #                 create_vector_store(text_chunks)
# #             st.success(f"Processed {len(uploaded_files)} documents!")

# #     # Main query interface
# #     user_question = st.text_input("Ask about the document content:")
# #     if user_question:
# #         with st.spinner("Analyzing documents..."):
# #             handle_query(user_question)

# # if __name__ == "__main__":
# #     main()



















# # import streamlit as st
# # from PyPDF2 import PdfReader
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # import os
# # from langchain_google_genai import GoogleGenerativeAIEmbeddings
# # import google.generativeai as genai
# # from langchain.vectorstores import FAISS
# # from langchain_google_genai import ChatGoogleGenerativeAI
# # from langchain.chains.question_answering import load_qa_chain
# # from langchain.prompts import PromptTemplate
# # from dotenv import load_dotenv
# # import fitz  # PyMuPDF
# # import base64
# # import uuid
# # from collections import defaultdict

# # load_dotenv()

# # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # # Configuration
# # PDF_FOLDER = "uploaded_pdfs"
# # HIGHLIGHT_FOLDER = "highlighted_pdfs"
# # os.makedirs(PDF_FOLDER, exist_ok=True)
# # os.makedirs(HIGHLIGHT_FOLDER, exist_ok=True)

# # # ----------------------
# # # Enhanced PDF Processing
# # # ----------------------
# # def save_uploaded_files(uploaded_files):
# #     return [os.path.join(PDF_FOLDER, f.name) for f in uploaded_files]

# # def extract_text_with_positions(pdf_path):
# #     doc = fitz.open(pdf_path)
# #     text_blocks = []
# #     for page_num in range(len(doc)):
# #         page = doc.load_page(page_num)
# #         blocks = page.get_text("blocks")
# #         for block in blocks:
# #             x0, y0, x1, y1, text, _, block_type = block
# #             if block_type == 0 and text.strip():
# #                 text_blocks.append({
# #                     "text": text.strip(),
# #                     "page": page_num,
# #                     "coordinates": (x0, y0, x1, y1)
# #                 })
# #     doc.close()
# #     return text_blocks

# # def get_text_chunks_with_metadata(pdf_paths):
# #     text_splitter = RecursiveCharacterTextSplitter(
# #         chunk_size=1000, chunk_overlap=200, length_function=len
# #     )
# #     all_chunks = []
# #     for pdf_path in pdf_paths:
# #         text_blocks = extract_text_with_positions(pdf_path)
# #         for block in text_blocks:
# #             chunks = text_splitter.split_text(block["text"])
# #             for chunk in chunks:
# #                 all_chunks.append({
# #                     "text": chunk,
# #                     "metadata": {
# #                         "source": pdf_path,
# #                         "page": block["page"],
# #                         "coordinates": block["coordinates"]
# #                     }
# #                 })
# #     return all_chunks

# # # ----------------------
# # # Enhanced Highlighting
# # # ----------------------
# # def highlight_pdf_regions(pdf_path, regions):
# #     """Highlight multiple regions in a PDF"""
# #     doc = fitz.open(pdf_path)
# #     for page_num, coordinates in regions:
# #         page = doc.load_page(page_num)
# #         x0, y0, x1, y1 = coordinates
# #         rect = fitz.Rect(x0, y0, x1, y1)
# #         highlight = page.add_highlight_annot(rect)
# #         highlight.set_colors(stroke=(1, 1, 0))
    
# #     filename = f"highlighted_{uuid.uuid4().hex}_{os.path.basename(pdf_path)}"
# #     highlighted_path = os.path.join(HIGHLIGHT_FOLDER, filename)
# #     doc.save(highlighted_path)
# #     doc.close()
# #     return highlighted_path

# # # ----------------------
# # # Enhanced QA System
# # # ----------------------
# # def get_vector_store(text_chunks):
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# #     texts = [chunk["text"] for chunk in text_chunks]
# #     metadatas = [chunk["metadata"] for chunk in text_chunks]
# #     vector_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
# #     vector_store.save_local("faiss_index")

# # def get_conversational_chain():
# #     prompt_template = """
# #     Analyze and synthesize information from the following contexts to answer the question.
# #     Consider all relevant sections and provide a comprehensive answer. If the answer isn't found,
# #     state "Answer is not available in the context."

# #     Contexts:
# #     {context}

# #     Question: {question}

# #     Provide a detailed answer with references to document sections:
# #     """
# #     model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
# #     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
# #     return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# # # ----------------------
# # # Enhanced UI Handling
# # # ----------------------
# # def display_pdf(pdf_path):
# #     with open(pdf_path, "rb") as f:
# #         base64_pdf = base64.b64encode(f.read()).decode("utf-8")
# #     pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="800"></iframe>'
# #     st.markdown(pdf_display, unsafe_allow_html=True)

# # def user_input(user_question):
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# #     vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
# #     docs = vector_store.similarity_search(user_question, k=1)  # Increased to 5 relevant chunks
    
# #     if not docs:
# #         st.warning("No relevant information found in documents.")
# #         return

# #     # Process multiple sources and pages
# #     source_regions = defaultdict(list)
# #     sources_info = set()
    
# #     for doc in docs:
# #         source = doc.metadata["source"]
# #         page = doc.metadata["page"]
# #         coords = doc.metadata["coordinates"]
# #         source_regions[source].append((page, coords))
# #         sources_info.add((source, page))

# #     # Generate highlighted PDFs
# #     highlighted_files = []
# #     for source, regions in source_regions.items():
# #         try:
# #             highlighted = highlight_pdf_regions(source, regions)
# #             highlighted_files.append((source, highlighted))
# #         except Exception as e:
# #             st.error(f"Error processing {source}: {str(e)}")

# #     # Generate answer
# #     chain = get_conversational_chain()
# #     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
# #     # Display results
# #     st.subheader("Answer")
# #     st.markdown(f"**{response['output_text']}**")

# #     st.subheader("Source References")
# #     for source, page in sources_info:
# #         st.markdown(f"- `{os.path.basename(source)}` (Page {page + 1})")

# #     st.subheader("Relevant Document Sections")
# #     for source_path, highlighted_path in highlighted_files:
# #         st.markdown(f"**{os.path.basename(source_path)}**")
# #         display_pdf(highlighted_path)
# #         with open(highlighted_path, "rb") as f:
# #             st.download_button(
# #                 f"Download highlighted {os.path.basename(source_path)}",
# #                 f.read(),
# #                 file_name=f"highlighted_{os.path.basename(source_path)}"
# #             )

# # # ----------------------
# # # Main App
# # # ----------------------
# # def main():
# #     st.set_page_config("Smart PDF Analyzer", page_icon=":books:")
# #     st.header("Advanced PDF Analysis with Gemini")
    
# #     with st.sidebar:
# #         st.title("Document Management")
# #         uploaded_files = st.file_uploader(
# #             "Upload research papers/documentation",
# #             type=["pdf"],
# #             accept_multiple_files=True
# #         )
# #         if uploaded_files:
# #             saved_files = save_uploaded_files(uploaded_files)
# #             with st.spinner("Processing documents..."):
# #                 text_chunks = get_text_chunks_with_metadata(saved_files)
# #                 get_vector_store(text_chunks)
# #             st.success("Documents indexed successfully!")

# #     user_question = st.text_input("Ask complex questions (e.g., 'Compare and contrast X and Y in the documents'):")
# #     if user_question:
# #         with st.spinner("Analyzing documents..."):
# #             user_input(user_question)

# # if __name__ == "__main__":
# #     main()


#     # pip install streamlit PyPDF2 langchain langchain-google-genai google-generativeai faiss-cpu python-dotenv pymupdf
