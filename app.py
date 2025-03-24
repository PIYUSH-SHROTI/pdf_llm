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

# load_dotenv()

# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# PDF_FOLDER = "uploaded_pdfs"
# if not os.path.exists(PDF_FOLDER):
#     os.makedirs(PDF_FOLDER)

# def save_uploaded_files(uploaded_files):
#     saved_files = []
#     for file in uploaded_files:
#         file_path = os.path.join(PDF_FOLDER, file.name)
#         with open(file_path, "wb") as f:
#             f.write(file.getbuffer())
#         saved_files.append(file_path)
#     return saved_files

# def extract_text_with_positions(pdf_path):
#     doc = fitz.open(pdf_path)
#     text_blocks = []
#     for page_num in range(len(doc)):
#         page = doc.load_page(page_num)
#         blocks = page.get_text("blocks")
#         for block in blocks:
#             x0, y0, x1, y1, text, block_no, block_type = block
#             if block_type == 0:
#                 text_blocks.append({
#                     "text": text.strip(),
#                     "page": page_num,
#                     "coordinates": (x0, y0, x1, y1)
#                 })
#     doc.close()
#     return text_blocks

# def get_text_chunks_with_metadata(pdf_paths):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=800,
#         chunk_overlap=100,
#         separators=["\n\n", "\n", ". ", "?", "!", ", "]
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

# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     texts = [chunk["text"] for chunk in text_chunks]
#     metadatas = [chunk["metadata"] for chunk in text_chunks]
#     vector_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
#     vector_store.save_local("faiss_index")

# def highlight_definitions(pdf_path, components):
#     doc = fitz.open(pdf_path)
#     highlighted = False
    
#     for page_num in range(len(doc)):
#         page = doc.load_page(page_num)
#         text = page.get_text("text").lower()
        
#         # Look for component definitions
#         for component in components:
#             if component in text:
#                 # Find exact matches
#                 instances = page.search_for(component)
#                 for inst in instances:
#                     highlight = page.add_highlight_annot(inst)
#                     highlight.set_colors(stroke=(1, 1, 0))  # Yellow
#                     highlighted = True
                
#     highlighted_pdf_path = "highlighted_pdf.pdf"
#     doc.save(highlighted_pdf_path)
#     doc.close()
#     return highlighted_pdf_path if highlighted else None

# def get_conversational_chain():
#     prompt_template = """Answer the question precisely using the context. 
#     List only the main components mentioned in the context.
#     Context:\n{context}\n
#     Question: {question}
#     Answer:"""
    
#     model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# def display_pdf(pdf_path):
#     with open(pdf_path, "rb") as f:
#         base64_pdf = base64.b64encode(f.read()).decode("utf-8")
#     pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="800"></iframe>'
#     st.markdown(pdf_display, unsafe_allow_html=True)

# def user_input(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
#     # Get answer
#     chain = get_conversational_chain()
#     docs = vector_store.similarity_search(user_question, k=1)
#     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
#     if not docs:
#         st.write("No relevant document found.")
#         return
    
#     # Extract components from answer
#     answer = response["output_text"].lower()
#     components = [word.strip() for word in answer.replace("and", ",").split(",") if word.strip()]
    
#     # Highlight matching terms in source PDF
#     source_pdf = docs[0].metadata["source"]
#     highlighted_pdf = highlight_definitions(source_pdf, components)
    
#     if highlighted_pdf:
#         st.write("**Answer:**", response["output_text"])
#         st.write(f"**Source:** `{source_pdf}`")
#         display_pdf(highlighted_pdf)
#         with open(highlighted_pdf, "rb") as f:
#             st.download_button("Download Highlighted PDF", f.read(), file_name="highlighted.pdf")
#     else:
#         st.write("Could not find component definitions in the document")

# def main():
#     st.set_page_config("Chat PDF")
#     st.header("Chat with PDF using Gemini")
#     user_question = st.text_input("Ask a Question")
    
#     with st.sidebar:
#         st.title("Upload PDFs")
#         uploaded_files = st.file_uploader("Choose PDFs", accept_multiple_files=True, type=["pdf"])
#         if uploaded_files:
#             saved_files = save_uploaded_files(uploaded_files)
#             text_chunks = get_text_chunks_with_metadata(saved_files)
#             get_vector_store(text_chunks)
#             st.success("PDFs processed!")

#     if user_question:
#         user_input(user_question)

# if __name__ == "__main__":
#     main()


















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

# # load_dotenv()

# # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # # Store uploaded PDFs
# # PDF_FOLDER = "uploaded_pdfs"
# # if not os.path.exists(PDF_FOLDER):
# #     os.makedirs(PDF_FOLDER)

# # def save_uploaded_files(uploaded_files):
# #     saved_files = []
# #     for file in uploaded_files:
# #         file_path = os.path.join(PDF_FOLDER, file.name)
# #         with open(file_path, "wb") as f:
# #             f.write(file.getbuffer())
# #         saved_files.append(file_path)
# #     return saved_files

# # def extract_text_with_positions(pdf_path):
# #     doc = fitz.open(pdf_path)
# #     text_blocks = []
# #     for page_num in range(len(doc)):
# #         page = doc.load_page(page_num)
# #         blocks = page.get_text("blocks")
# #         for block in blocks:
# #             x0, y0, x1, y1, text, block_no, block_type = block
# #             if block_type == 0:
# #                 text_blocks.append({
# #                     "text": text.strip(),
# #                     "page": page_num,
# #                     "coordinates": (x0, y0, x1, y1)
# #                 })
# #     doc.close()
# #     return text_blocks

# # def get_text_chunks_with_metadata(pdf_paths):
# #     text_splitter = RecursiveCharacterTextSplitter(
# #         chunk_size=500,  # Smaller chunks for precision
# #         chunk_overlap=50
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

# # def get_vector_store(text_chunks):
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# #     texts = [chunk["text"] for chunk in text_chunks]
# #     metadatas = [chunk["metadata"] for chunk in text_chunks]
# #     vector_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
# #     vector_store.save_local("faiss_index")

# # def highlight_specific_terms(pdf_path, terms, page_numbers):
# #     doc = fitz.open(pdf_path)
# #     for page_num in page_numbers:
# #         page = doc.load_page(page_num)
# #         for term in terms:
# #             text_instances = page.search_for(term)
# #             for inst in text_instances:
# #                 highlight = page.add_highlight_annot(inst)
# #                 highlight.set_colors(stroke=(1, 1, 0))  # Yellow
# #     highlighted_pdf_path = "highlighted_pdf.pdf"
# #     doc.save(highlighted_pdf_path)
# #     doc.close()
# #     return highlighted_pdf_path

# # def get_conversational_chain():
# #     prompt_template = """Extract ONLY the three main components from the context.
# #     Context: {context}
# #     Question: {question}
# #     Answer ONLY with the component names separated by commas:"""
    
# #     model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
# #     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
# #     return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# # def display_pdf(pdf_path):
# #     with open(pdf_path, "rb") as f:
# #         base64_pdf = base64.b64encode(f.read()).decode("utf-8")
# #     pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="800"></iframe>'
# #     st.markdown(pdf_display, unsafe_allow_html=True)

# # def user_input(user_question):
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# #     vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
# #     # Get precise answer first
# #     chain = get_conversational_chain()
# #     docs = vector_store.similarity_search(user_question, k=2)  # Fewer, more relevant chunks
# #     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
# #     # Extract components from answer
# #     components = [word.strip().lower() for word in response["output_text"].split(",")]
    
# #     # Find exact matches in PDF
# #     source_pdf = docs[0].metadata["source"]
# #     doc = fitz.open(source_pdf)
# #     pages_to_highlight = set()
    
# #     for page_num in range(len(doc)):
# #         page = doc.load_page(page_num)
# #         text = page.get_text().lower()
# #         if any(comp in text for comp in components):
# #             pages_to_highlight.add(page_num)
    
# #     # Highlight only component definitions
# #     highlighted_pdf = highlight_specific_terms(
# #         source_pdf,
# #         terms=["guest", "host", "virtualization layer"],
# #         page_numbers=pages_to_highlight
# #     )
    
# #     # Display results
# #     st.write("**Answer:**", ", ".join(components).title())
# #     st.write(f"**Source:** `{source_pdf}`")
# #     display_pdf(highlighted_pdf)
# #     with open(highlighted_pdf, "rb") as f:
# #         st.download_button("Download Highlighted PDF", f.read(), file_name="highlighted.pdf")

# # def main():
# #     st.set_page_config("Chat PDF")
# #     st.header("Chat with PDF using Gemini")
# #     user_question = st.text_input("Ask a Question")
    
# #     with st.sidebar:
# #         st.title("Upload PDFs")
# #         uploaded_files = st.file_uploader("Choose PDFs", accept_multiple_files=True, type=["pdf"])
# #         if uploaded_files:
# #             saved_files = save_uploaded_files(uploaded_files)
# #             text_chunks = get_text_chunks_with_metadata(saved_files)
# #             get_vector_store(text_chunks)
# #             st.success("PDFs processed!")

# #     if user_question:
# #         user_input(user_question)

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

# # load_dotenv()

# # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # # Store uploaded PDFs
# # PDF_FOLDER = "uploaded_pdfs"
# # if not os.path.exists(PDF_FOLDER):
# #     os.makedirs(PDF_FOLDER)

# # def save_uploaded_files(uploaded_files):
# #     saved_files = []
# #     for file in uploaded_files:
# #         file_path = os.path.join(PDF_FOLDER, file.name)
# #         with open(file_path, "wb") as f:
# #             f.write(file.getbuffer())
# #         saved_files.append(file_path)
# #     return saved_files

# # def extract_text_with_positions(pdf_path):
# #     doc = fitz.open(pdf_path)
# #     text_blocks = []
# #     for page_num in range(len(doc)):
# #         page = doc.load_page(page_num)
# #         blocks = page.get_text("blocks")
# #         for block in blocks:
# #             x0, y0, x1, y1, text, block_no, block_type = block
# #             if block_type == 0:
# #                 text_blocks.append({
# #                     "text": text.strip(),
# #                     "page": page_num,
# #                     "coordinates": (x0, y0, x1, y1)
# #                 })
# #     doc.close()
# #     return text_blocks

# # def get_text_chunks_with_metadata(pdf_paths):
# #     text_splitter = RecursiveCharacterTextSplitter(
# #         chunk_size=1000, chunk_overlap=200
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

# # def get_vector_store(text_chunks):
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# #     texts = [chunk["text"] for chunk in text_chunks]
# #     metadatas = [chunk["metadata"] for chunk in text_chunks]
# #     vector_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
# #     vector_store.save_local("faiss_index")

# # def highlight_multiple_pages(pdf_path, regions):
# #     doc = fitz.open(pdf_path)
# #     for region in regions:
# #         page_num = region["page"]
# #         coordinates = region["coordinates"]
# #         page = doc.load_page(page_num)
# #         rect = fitz.Rect(coordinates)
# #         highlight = page.add_highlight_annot(rect)
# #         highlight.set_colors(stroke=(1, 1, 0))  # Yellow color
# #     highlighted_pdf_path = "highlighted_pdf.pdf"
# #     doc.save(highlighted_pdf_path)
# #     doc.close()
# #     return highlighted_pdf_path

# # def get_conversational_chain():
# #     prompt_template = """
# #     Answer the question using the provided context. If the answer isn't in the context, say so.
# #     Context:\n{context}\n
# #     Question: \n{question}\n
# #     Answer:"""
# #     model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
# #     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
# #     return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# # def display_pdf(pdf_path):
# #     with open(pdf_path, "rb") as f:
# #         base64_pdf = base64.b64encode(f.read()).decode("utf-8")
# #     pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="800"></iframe>'
# #     st.markdown(pdf_display, unsafe_allow_html=True)

# # def user_input(user_question):
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# #     vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
# #     # Retrieve top 5 most relevant chunks
# #     docs = vector_store.similarity_search(user_question, k=1)
    
# #     if not docs:
# #         st.write("No relevant document found.")
# #         return

# #     # Collect unique regions across pages
# #     regions = []
# #     seen_pages = set()
# #     for doc in docs:
# #         metadata = doc.metadata
# #         region_key = (metadata["page"], metadata["coordinates"])
# #         if region_key not in seen_pages:
# #             regions.append({
# #                 "page": metadata["page"],
# #                 "coordinates": metadata["coordinates"]
# #             })
# #             seen_pages.add(region_key)

# #     # Highlight all relevant regions
# #     primary_source = docs[0].metadata["source"]
# #     highlighted_pdf = highlight_multiple_pages(primary_source, regions)
    
# #     # Generate answer
# #     chain = get_conversational_chain()
# #     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
# #     # Display results
# #     st.write("**Answer:**", response["output_text"])
# #     st.write(f"**Source:** `{primary_source}`")
# #     st.write(f"**Highlighted Pages:** {', '.join(str(region['page']+1) for region in regions)}")
    
# #     display_pdf(highlighted_pdf)
# #     with open(highlighted_pdf, "rb") as f:
# #         st.download_button("Download Highlighted PDF", f.read(), file_name="highlighted.pdf")

# # def main():
# #     st.set_page_config("Chat PDF")
# #     st.header("Chat with PDF using Gemini")
# #     user_question = st.text_input("Ask a Question")
    
# #     with st.sidebar:
# #         st.title("Upload PDFs")
# #         uploaded_files = st.file_uploader("Choose PDFs", accept_multiple_files=True, type=["pdf"])
# #         if uploaded_files:
# #             saved_files = save_uploaded_files(uploaded_files)
# #             text_chunks = get_text_chunks_with_metadata(saved_files)
# #             get_vector_store(text_chunks)
# #             st.success("PDFs processed!")

# #     if user_question:
# #         user_input(user_question)

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

# # load_dotenv()

# # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # # Store uploaded PDFs
# # PDF_FOLDER = "uploaded_pdfs"
# # if not os.path.exists(PDF_FOLDER):
# #     os.makedirs(PDF_FOLDER)

# # # ----------------------
# # # Save Uploaded Files
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
# # # Extract Text with Positions
# # # ----------------------
# # def extract_text_with_positions(pdf_path):
# #     """Extract text with page numbers and coordinates using PyMuPDF."""
# #     doc = fitz.open(pdf_path)
# #     text_blocks = []
# #     for page_num in range(len(doc)):
# #         page = doc.load_page(page_num)
# #         blocks = page.get_text("blocks")  # Get text blocks with coordinates
# #         for block in blocks:
# #             x0, y0, x1, y1, text, block_no, block_type = block
# #             if block_type == 0:  # Only process text blocks
# #                 text_blocks.append({
# #                     "text": text.strip(),
# #                     "page": page_num,
# #                     "coordinates": (x0, y0, x1, y1)
# #                 })
# #     doc.close()
# #     return text_blocks

# # # ----------------------
# # # Split Text into Chunks with Metadata
# # # ----------------------
# # def get_text_chunks_with_metadata(pdf_paths):
# #     """Split text into chunks with metadata (page, coordinates)."""
# #     text_splitter = RecursiveCharacterTextSplitter(
# #         chunk_size=1000, chunk_overlap=200
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
# # # Create Vector Store
# # # ----------------------
# # def get_vector_store(text_chunks):
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# #     texts = [chunk["text"] for chunk in text_chunks]
# #     metadatas = [chunk["metadata"] for chunk in text_chunks]
# #     vector_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
# #     vector_store.save_local("faiss_index")

# # # ----------------------
# # # Highlight Multiple Regions in PDF
# # # ----------------------
# # def highlight_multiple_regions(pdf_path, regions):
# #     """Highlight multiple regions in the PDF."""
# #     doc = fitz.open(pdf_path)
# #     for region in regions:
# #         page_num = region["page"]
# #         coordinates = region["coordinates"]
# #         page = doc.load_page(page_num)
# #         x0, y0, x1, y1 = coordinates
# #         rect = fitz.Rect(x0, y0, x1, y1)
# #         highlight = page.add_highlight_annot(rect)
# #         highlight.set_colors(stroke=(1, 1, 0))  # Yellow highlight
    
# #     highlighted_pdf_path = "highlighted_pdf.pdf"
# #     doc.save(highlighted_pdf_path)
# #     doc.close()
# #     return highlighted_pdf_path

# # # ----------------------
# # # Conversational Chain
# # # ----------------------
# # def get_conversational_chain():
# #     prompt_template = """
# #     Answer the question as detailed as possible from the provided context. If the answer is not available in the context, 
# #     just say: "Answer is not available in the context." 

# #     Context:\n {context}?\n
# #     Question: \n{question}\n

# #     Answer:
# #     """
# #     model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
# #     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
# #     return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# # # ----------------------
# # # Display PDF in Streamlit
# # # ----------------------
# # def display_pdf(pdf_path):
# #     with open(pdf_path, "rb") as f:
# #         base64_pdf = base64.b64encode(f.read()).decode("utf-8")
# #     pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="800"></iframe>'
# #     st.markdown(pdf_display, unsafe_allow_html=True)

# # # ----------------------
# # # User Input Handling
# # # ----------------------
# # def user_input(user_question):
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# #     vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
# #     # Get top 3 relevant chunks (adjust k as needed)
# #     docs = vector_store.similarity_search(user_question, k=3)
    
# #     if not docs:
# #         st.write("No relevant document found.")
# #         return

# #     # Collect all unique regions to highlight
# #     regions = []
# #     sources = set()
# #     pages = set()
    
# #     for doc in docs:
# #         source = doc.metadata["source"]
# #         page = doc.metadata["page"]
# #         coordinates = doc.metadata["coordinates"]
        
# #         regions.append({
# #             "source": source,
# #             "page": page,
# #             "coordinates": coordinates
# #         })
# #         sources.add(source)
# #         pages.add(page + 1)  # Convert to 1-based page numbering

# #     # Highlight all regions in the first source PDF (for multi-PDF support)
# #     primary_source = docs[0].metadata["source"]
# #     highlighted_pdf = highlight_multiple_regions(primary_source, regions)
    
# #     # Generate answer
# #     chain = get_conversational_chain()
# #     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
# #     # Display results
# #     st.write("Answer:", response["output_text"])
# #     st.write(f"**Source:** `{primary_source}`")
# #     st.write(f"**Relevant Pages:** {', '.join(map(str, sorted(pages)))}")
    
# #     display_pdf(highlighted_pdf)
# #     with open(highlighted_pdf, "rb") as f:
# #         st.download_button("Download Highlighted PDF", f.read(), file_name="highlighted.pdf")

# # # ----------------------
# # # Streamlit UI
# # # ----------------------
# # def main():
# #     st.set_page_config("Chat PDF")
# #     st.header("Chat with PDF using Gemini")
# #     user_question = st.text_input("Ask a Question")
    
# #     with st.sidebar:
# #         st.title("Upload PDFs")
# #         uploaded_files = st.file_uploader("Choose PDFs", accept_multiple_files=True, type=["pdf"])
# #         if uploaded_files:
# #             saved_files = save_uploaded_files(uploaded_files)
# #             text_chunks = get_text_chunks_with_metadata(saved_files)
# #             get_vector_store(text_chunks)
# #             st.success("PDFs processed!")

# #     if user_question:
# #         user_input(user_question)

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

# # load_dotenv()

# # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # # Store uploaded PDFs
# # PDF_FOLDER = "uploaded_pdfs"
# # if not os.path.exists(PDF_FOLDER):
# #     os.makedirs(PDF_FOLDER)

# # # ----------------------
# # # Save Uploaded Files
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
# # # Extract Text with Positions
# # # ----------------------
# # def extract_text_with_positions(pdf_path):
# #     """Extract text with page numbers and coordinates using PyMuPDF."""
# #     doc = fitz.open(pdf_path)
# #     text_blocks = []
# #     for page_num in range(len(doc)):
# #         page = doc.load_page(page_num)
# #         blocks = page.get_text("blocks")  # Get text blocks with coordinates
# #         for block in blocks:
# #             x0, y0, x1, y1, text, block_no, block_type = block
# #             if block_type == 0:  # Only process text blocks
# #                 text_blocks.append({
# #                     "text": text.strip(),
# #                     "page": page_num,
# #                     "coordinates": (x0, y0, x1, y1)
# #                 })
# #     doc.close()
# #     return text_blocks

# # # ----------------------
# # # Split Text into Chunks with Metadata
# # # ----------------------
# # def get_text_chunks_with_metadata(pdf_paths):
# #     """Split text into chunks with metadata (page, coordinates)."""
# #     text_splitter = RecursiveCharacterTextSplitter(
# #         chunk_size=1000, chunk_overlap=200
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
# # # Create Vector Store
# # # ----------------------
# # def get_vector_store(text_chunks):
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# #     texts = [chunk["text"] for chunk in text_chunks]
# #     metadatas = [chunk["metadata"] for chunk in text_chunks]
# #     vector_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
# #     vector_store.save_local("faiss_index")

# # # ----------------------
# # # Highlight PDF Region
# # # ----------------------
# # def highlight_pdf_region(pdf_path, page_num, coordinates):
# #     """Highlight a specific region in the PDF using coordinates."""
# #     doc = fitz.open(pdf_path)
# #     page = doc.load_page(page_num)
# #     x0, y0, x1, y1 = coordinates
# #     rect = fitz.Rect(x0, y0, x1, y1)
# #     highlight = page.add_highlight_annot(rect)
# #     highlight.set_colors(stroke=(1, 1, 0))  # Yellow highlight
# #     highlight.update()
# #     highlighted_pdf_path = "highlighted_pdf.pdf"
# #     doc.save(highlighted_pdf_path)
# #     doc.close()
# #     return highlighted_pdf_path

# # # ----------------------
# # # Conversational Chain
# # # ----------------------
# # def get_conversational_chain():
# #     prompt_template = """
# #     Answer the question as detailed as possible from the provided context. If the answer is not available in the context, 
# #     just say: "Answer is not available in the context." 

# #     Context:\n {context}?\n
# #     Question: \n{question}\n

# #     Answer:
# #     """
# #     model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
# #     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
# #     return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# # # ----------------------
# # # Display PDF in Streamlit
# # # ----------------------
# # def display_pdf(pdf_path):
# #     with open(pdf_path, "rb") as f:
# #         base64_pdf = base64.b64encode(f.read()).decode("utf-8")
# #     pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="800"></iframe>'
# #     st.markdown(pdf_display, unsafe_allow_html=True)

# # # ----------------------
# # # User Input Handling
# # # ----------------------
# # def user_input(user_question):
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# #     vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
# #     docs = vector_store.similarity_search(user_question, k=1)
    
# #     if not docs:
# #         st.write("No relevant document found.")
# #         return

# #     # Get metadata from the top matching chunk
# #     source = docs[0].metadata["source"]
# #     page = docs[0].metadata["page"]
# #     coordinates = docs[0].metadata["coordinates"]
    
# #     # Highlight the region in the PDF
# #     highlighted_pdf = highlight_pdf_region(source, page, coordinates)
    
# #     # Display answer and PDF
# #     chain = get_conversational_chain()
# #     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
# #     st.write("Answer:", response["output_text"])
# #     st.write(f"**Source:** `{source}` (Page {page + 1})")
    
# #     display_pdf(highlighted_pdf)
# #     with open(highlighted_pdf, "rb") as f:
# #         st.download_button("Download Highlighted PDF", f.read(), file_name="highlighted.pdf")

# # # ----------------------
# # # Streamlit UI
# # # ----------------------
# # def main():
# #     st.set_page_config("Chat PDF")
# #     st.header("Chat with PDF using Gemini")
# #     user_question = st.text_input("Ask a Question")
    
# #     with st.sidebar:
# #         st.title("Upload PDFs")
# #         uploaded_files = st.file_uploader("Choose PDFs", accept_multiple_files=True, type=["pdf"])
# #         if uploaded_files:
# #             saved_files = save_uploaded_files(uploaded_files)
# #             text_chunks = get_text_chunks_with_metadata(saved_files)
# #             get_vector_store(text_chunks)
# #             st.success("PDFs processed!")

# #     if user_question:
# #         user_input(user_question)

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

# # load_dotenv()

# # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # # Store uploaded PDFs
# # PDF_FOLDER = "uploaded_pdfs"
# # if not os.path.exists(PDF_FOLDER):
# #     os.makedirs(PDF_FOLDER)

# # def save_uploaded_files(uploaded_files):
# #     saved_files = []
# #     for file in uploaded_files:
# #         file_path = os.path.join(PDF_FOLDER, file.name)
# #         with open(file_path, "wb") as f:
# #             f.write(file.getbuffer())
# #         saved_files.append(file_path)
# #     return saved_files

# # def get_pdf_text(pdf_paths):
# #     text = ""
# #     for pdf_path in pdf_paths:
# #         pdf_reader = PdfReader(pdf_path)
# #         for page in pdf_reader.pages:
# #             text += page.extract_text()
# #     return text

# # def get_text_chunks(text):
# #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
# #     return text_splitter.split_text(text)

# # def get_vector_store(text_chunks, pdf_paths):
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# #     metadatas = [{"source": pdf_paths[i % len(pdf_paths)]} for i in range(len(text_chunks))]
# #     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings, metadatas=metadatas)
# #     vector_store.save_local("faiss_index")

# # def get_conversational_chain():
# #     prompt_template = """Answer the question as detailed as possible from the provided context. If the answer is not available in the context, just say: "Answer is not available in the context." Context:\n {context}? Question: \n{question}\n Answer:"""
# #     model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
# #     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
# #     return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# # def display_pdf(pdf_path):
# #     with open(pdf_path, "rb") as f:
# #         base64_pdf = base64.b64encode(f.read()).decode("utf-8")
# #     pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="800" type="application/pdf"></iframe>'
# #     st.markdown(pdf_display, unsafe_allow_html=True)

# # def highlight_text_in_pdf(pdf_path, text_to_highlight):
# #     # Open the PDF
# #     doc = fitz.open(pdf_path)
# #     for page_num in range(len(doc)):
# #         page = doc.load_page(page_num)
# #         text_instances = page.search_for(text_to_highlight)
# #         for inst in text_instances:
# #             # Highlight the text in yellow
# #             highlight = page.add_highlight_annot(inst)
# #             highlight.set_colors(stroke=(1, 1, 0))  # Yellow color (RGB: 1, 1, 0)
# #             highlight.update()
# #     highlighted_pdf_path = "highlighted_pdf.pdf"
# #     doc.save(highlighted_pdf_path)
# #     doc.close()
# #     return highlighted_pdf_path

# # def user_input(user_question):
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# #     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
# #     docs = new_db.similarity_search(user_question)
# #     chain = get_conversational_chain()
# #     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
# #     if docs:
# #         pdf_source = docs[0].metadata.get("source", "Unknown PDF")
# #         st.write("Reply:", response["output_text"])
# #         st.write(f"**Answer found in:** `{pdf_source}`")
        
# #         # Highlight the relevant text in the PDF
# #         relevant_text = docs[0].page_content  # Use the text chunk that was used for the answer
# #         highlighted_pdf_path = highlight_text_in_pdf(pdf_source, relevant_text)
        
# #         # Display the highlighted PDF
# #         with open(highlighted_pdf_path, "rb") as highlighted_pdf_file:
# #             highlighted_pdf_bytes = highlighted_pdf_file.read()
# #         st.download_button(label="Download Highlighted PDF", data=highlighted_pdf_bytes, file_name="highlighted_pdf.pdf")
# #         display_pdf(highlighted_pdf_path)
# #     else:
# #         st.write("No relevant document found.")

# # def main():
# #     st.set_page_config("Chat PDF")
# #     st.header("Chat with PDF using Gemini")
# #     user_question = st.text_input("Ask a Question from the PDF Files")
# #     if user_question:
# #         user_input(user_question)
# #     with st.sidebar:
# #         st.title("Menu:")
# #         uploaded_pdfs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=["pdf"])
# #         if uploaded_pdfs:
# #             saved_pdfs = save_uploaded_files(uploaded_pdfs)
# #             raw_text = get_pdf_text(saved_pdfs)
# #             text_chunks = get_text_chunks(raw_text)
# #             get_vector_store(text_chunks, saved_pdfs)
# #             st.success("PDFs uploaded and processed successfully!")

# # if __name__ == "__main__":
# #     main()





















# # import streamlit as st
# # import fitz  # PyMuPDF
# # import base64
# # import os

# # # Function to highlight the first line of the PDF
# # def highlight_first_line_in_pdf(pdf_path):
# #     # Open the PDF
# #     doc = fitz.open(pdf_path)
# #     page = doc.load_page(0)  # Load the first page
# #     text = page.get_text("text")  # Extract text from the first page
# #     lines = text.splitlines()  # Split text into lines
# #     if lines:
# #         first_line = lines[0]  # Get the first line
# #         # Search for the first line in the PDF
# #         text_instances = page.search_for(first_line)
# #         for inst in text_instances:
# #             # Highlight the first line in yellow
# #             highlight = page.add_highlight_annot(inst)
# #             highlight.set_colors(stroke=(1, 1, 0))  # Yellow color (RGB: 1, 1, 0)
# #             highlight.update()
# #     highlighted_pdf_path = "highlighted_pdf.pdf"
# #     doc.save(highlighted_pdf_path)
# #     doc.close()
# #     return highlighted_pdf_path

# # # Function to display the PDF in the UI
# # def display_pdf(pdf_path):
# #     with open(pdf_path, "rb") as f:
# #         base64_pdf = base64.b64encode(f.read()).decode("utf-8")
# #     pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="800" type="application/pdf"></iframe>'
# #     st.markdown(pdf_display, unsafe_allow_html=True)

# # # Main function
# # def main():
# #     st.set_page_config(page_title="PDF Highlighter", page_icon="📄")
# #     st.title("PDF First Line Highlighter")
# #     st.write("Upload a PDF, and the first line will be highlighted in yellow.")

# #     # Upload PDF
# #     uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
# #     if uploaded_file is not None:
# #         # Save the uploaded file
# #         with open("uploaded_file.pdf", "wb") as f:
# #             f.write(uploaded_file.getbuffer())
# #         st.success("PDF uploaded successfully!")

# #         # Highlight the first line
# #         highlighted_pdf_path = highlight_first_line_in_pdf("uploaded_file.pdf")
# #         st.success("First line highlighted in yellow!")

# #         # Display the highlighted PDF
# #         st.subheader("Highlighted PDF")
# #         display_pdf(highlighted_pdf_path)

# #         # Download the highlighted PDF
# #         with open(highlighted_pdf_path, "rb") as f:
# #             pdf_bytes = f.read()
# #         st.download_button(
# #             label="Download Highlighted PDF",
# #             data=pdf_bytes,
# #             file_name="highlighted_pdf.pdf",
# #             mime="application/pdf",
# #         )

# # # Run the app
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

# # load_dotenv()

# # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # # Store uploaded PDFs
# # PDF_FOLDER = "uploaded_pdfs"
# # if not os.path.exists(PDF_FOLDER):
# #     os.makedirs(PDF_FOLDER)

# # def save_uploaded_files(uploaded_files):
# #     saved_files = []
# #     for file in uploaded_files:
# #         file_path = os.path.join(PDF_FOLDER, file.name)
# #         with open(file_path, "wb") as f:
# #             f.write(file.getbuffer())
# #         saved_files.append(file_path)
# #     return saved_files

# # def get_pdf_text(pdf_paths):
# #     text = ""
# #     for pdf_path in pdf_paths:
# #         pdf_reader = PdfReader(pdf_path)
# #         for page in pdf_reader.pages:
# #             text += page.extract_text()
# #     return text

# # def get_text_chunks(text):
# #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
# #     return text_splitter.split_text(text)

# # def get_vector_store(text_chunks, pdf_paths):
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# #     metadatas = [{"source": pdf_paths[i % len(pdf_paths)]} for i in range(len(text_chunks))]
# #     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings, metadatas=metadatas)
# #     vector_store.save_local("faiss_index")

# # def get_conversational_chain():
# #     prompt_template = """Answer the question as detailed as possible from the provided context. If the answer is not available in the context, just say: "Answer is not available in the context." Context:\n {context}? Question: \n{question}\n Answer:"""
# #     model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
# #     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
# #     return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# # def display_pdf(pdf_path):
# #     with open(pdf_path, "rb") as f:
# #         base64_pdf = base64.b64encode(f.read()).decode("utf-8")
# #     pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="800" type="application/pdf"></iframe>'
# #     st.markdown(pdf_display, unsafe_allow_html=True)

# # def highlight_first_line_in_pdf(pdf_path):
# #     # Open the PDF
# #     doc = fitz.open(pdf_path)
# #     page = doc.load_page(0)  # Load the first page
# #     text = page.get_text("text")  # Extract text from the first page
# #     lines = text.splitlines()  # Split text into lines
# #     if lines:
# #         first_line = lines[0]  # Get the first line
# #         # Search for the first line in the PDF
# #         text_instances = page.search_for(first_line)
# #         for inst in text_instances:
# #             # Highlight the first line in yellow
# #             highlight = page.add_highlight_annot(inst)
# #             highlight.set_colors(stroke=(1, 1, 0))  # Yellow color (RGB: 1, 1, 0)
# #             highlight.update()
# #     highlighted_pdf_path = "highlighted_pdf.pdf"
# #     doc.save(highlighted_pdf_path)
# #     doc.close()
# #     return highlighted_pdf_path

# # def user_input(user_question):
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# #     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
# #     docs = new_db.similarity_search(user_question)
# #     chain = get_conversational_chain()
# #     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
# #     if docs:
# #         pdf_source = docs[0].metadata.get("source", "Unknown PDF")
# #         st.write("Reply:", response["output_text"])
# #         st.write(f"**Answer found in:** `{pdf_source}`")
        
# #         # Highlight the first line of the PDF in yellow
# #         highlighted_pdf_path = highlight_first_line_in_pdf(pdf_source)
        
# #         # Display the highlighted PDF
# #         with open(highlighted_pdf_path, "rb") as highlighted_pdf_file:
# #             highlighted_pdf_bytes = highlighted_pdf_file.read()
# #         st.download_button(label="Download Highlighted PDF", data=highlighted_pdf_bytes, file_name="highlighted_pdf.pdf")
# #         display_pdf(highlighted_pdf_path)
# #     else:
# #         st.write("No relevant document found.")

# # def main():
# #     st.set_page_config("Chat PDF")
# #     st.header("Chat with PDF using Gemini")
# #     user_question = st.text_input("Ask a Question from the PDF Files")
# #     if user_question:
# #         user_input(user_question)
# #     with st.sidebar:
# #         st.title("Menu:")
# #         uploaded_pdfs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=["pdf"])
# #         if uploaded_pdfs:
# #             saved_pdfs = save_uploaded_files(uploaded_pdfs)
# #             raw_text = get_pdf_text(saved_pdfs)
# #             text_chunks = get_text_chunks(raw_text)
# #             get_vector_store(text_chunks, saved_pdfs)
# #             st.success("PDFs uploaded and processed successfully!")

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

# # load_dotenv()

# # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # # Store uploaded PDFs
# # PDF_FOLDER = "uploaded_pdfs"
# # if not os.path.exists(PDF_FOLDER):
# #     os.makedirs(PDF_FOLDER)

# # def save_uploaded_files(uploaded_files):
# #     saved_files = []
# #     for file in uploaded_files:
# #         file_path = os.path.join(PDF_FOLDER, file.name)
# #         with open(file_path, "wb") as f:
# #             f.write(file.getbuffer())
# #         saved_files.append(file_path)
# #     return saved_files

# # def get_pdf_text(pdf_paths):
# #     text = ""
# #     for pdf_path in pdf_paths:
# #         pdf_reader = PdfReader(pdf_path)
# #         for page in pdf_reader.pages:
# #             text += page.extract_text()
# #     return text

# # def get_text_chunks(text):
# #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
# #     return text_splitter.split_text(text)

# # def get_vector_store(text_chunks, pdf_paths):
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# #     metadatas = [{"source": pdf_paths[i % len(pdf_paths)]} for i in range(len(text_chunks))]
# #     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings, metadatas=metadatas)
# #     vector_store.save_local("faiss_index")

# # def get_conversational_chain():
# #     prompt_template = """Answer the question as detailed as possible from the provided context. If the answer is not available in the context, just say: "Answer is not available in the context." Context:\n {context}? Question: \n{question}\n Answer:"""
# #     model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
# #     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
# #     return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# # def display_pdf(pdf_path):
# #     with open(pdf_path, "rb") as f:
# #         base64_pdf = base64.b64encode(f.read()).decode("utf-8")
# #     pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="800" type="application/pdf"></iframe>'
# #     st.markdown(pdf_display, unsafe_allow_html=True)

# # def highlight_first_line_in_pdf(pdf_path):
# #     # Open the PDF
# #     doc = fitz.open(pdf_path)
# #     page = doc.load_page(0)  # Load the first page
# #     text = page.get_text("text")  # Extract text from the first page
# #     lines = text.splitlines()  # Split text into lines
# #     if lines:
# #         first_line = lines[0]  # Get the first line
# #         # Search for the first line in the PDF
# #         text_instances = page.search_for(first_line)
# #         for inst in text_instances:
# #             highlight = page.add_highlight_annot(inst)  # Highlight the first line
# #             highlight.update()
# #     highlighted_pdf_path = "highlighted_pdf.pdf"
# #     doc.save(highlighted_pdf_path)
# #     doc.close()
# #     return highlighted_pdf_path

# # def user_input(user_question):
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# #     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
# #     docs = new_db.similarity_search(user_question)
# #     chain = get_conversational_chain()
# #     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
# #     if docs:
# #         pdf_source = docs[0].metadata.get("source", "Unknown PDF")
# #         st.write("Reply:", response["output_text"])
# #         st.write(f"**Answer found in:** `{pdf_source}`")
        
# #         # Highlight the first line of the PDF
# #         highlighted_pdf_path = highlight_first_line_in_pdf(pdf_source)
        
# #         # Display the highlighted PDF
# #         with open(highlighted_pdf_path, "rb") as highlighted_pdf_file:
# #             highlighted_pdf_bytes = highlighted_pdf_file.read()
# #         st.download_button(label="Download Highlighted PDF", data=highlighted_pdf_bytes, file_name="highlighted_pdf.pdf")
# #         display_pdf(highlighted_pdf_path)
# #     else:
# #         st.write("No relevant document found.")

# # def main():
# #     st.set_page_config("Chat PDF")
# #     st.header("Chat with PDF using Gemini")
# #     user_question = st.text_input("Ask a Question from the PDF Files")
# #     if user_question:
# #         user_input(user_question)
# #     with st.sidebar:
# #         st.title("Menu:")
# #         uploaded_pdfs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=["pdf"])
# #         if uploaded_pdfs:
# #             saved_pdfs = save_uploaded_files(uploaded_pdfs)
# #             raw_text = get_pdf_text(saved_pdfs)
# #             text_chunks = get_text_chunks(raw_text)
# #             get_vector_store(text_chunks, saved_pdfs)
# #             st.success("PDFs uploaded and processed successfully!")

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

# # load_dotenv()

# # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # # Store uploaded PDFs
# # PDF_FOLDER = "uploaded_pdfs"
# # if not os.path.exists(PDF_FOLDER):
# #     os.makedirs(PDF_FOLDER)

# # def save_uploaded_files(uploaded_files):
# #     saved_files = []
# #     for file in uploaded_files:
# #         file_path = os.path.join(PDF_FOLDER, file.name)
# #         with open(file_path, "wb") as f:
# #             f.write(file.getbuffer())
# #         saved_files.append(file_path)
# #     return saved_files

# # def get_pdf_text(pdf_paths):
# #     text = ""
# #     for pdf_path in pdf_paths:
# #         pdf_reader = PdfReader(pdf_path)
# #         for page in pdf_reader.pages:
# #             text += page.extract_text()
# #     return text

# # def get_text_chunks(text):
# #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
# #     return text_splitter.split_text(text)

# # def get_vector_store(text_chunks, pdf_paths):
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# #     metadatas = [{"source": pdf_paths[i % len(pdf_paths)]} for i in range(len(text_chunks))]
# #     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings, metadatas=metadatas)
# #     vector_store.save_local("faiss_index")

# # def get_conversational_chain():
# #     prompt_template = """Answer the question as detailed as possible from the provided context. If the answer is not available in the context, just say: "Answer is not available in the context." Context:\n {context}? Question: \n{question}\n Answer:"""
# #     model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
# #     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
# #     return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# # def display_pdf(pdf_path):
# #     with open(pdf_path, "rb") as f:
# #         base64_pdf = base64.b64encode(f.read()).decode("utf-8")
# #     pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="800" type="application/pdf"></iframe>'
# #     st.markdown(pdf_display, unsafe_allow_html=True)

# # def highlight_text_in_pdf(pdf_path, text_to_highlight):
# #     # Open the PDF
# #     doc = fitz.open(pdf_path)
# #     for page_num in range(len(doc)):
# #         page = doc.load_page(page_num)
# #         text_instances = page.search_for(text_to_highlight)
# #         for inst in text_instances:
# #             highlight = page.add_highlight_annot(inst)
# #             highlight.update()
# #     highlighted_pdf_path = "highlighted_pdf.pdf"
# #     doc.save(highlighted_pdf_path)
# #     doc.close()
# #     return highlighted_pdf_path

# # def user_input(user_question):
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# #     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
# #     docs = new_db.similarity_search(user_question)
# #     chain = get_conversational_chain()
# #     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
# #     if docs:
# #         pdf_source = docs[0].metadata.get("source", "Unknown PDF")
# #         st.write("Reply:", response["output_text"])
# #         st.write(f"**Answer found in:** `{pdf_source}`")
        
# #         # Highlight the relevant text in the PDF
# #         relevant_text = docs[0].page_content  # Use the text chunk that was used for the answer
# #         highlighted_pdf_path = highlight_text_in_pdf(pdf_source, relevant_text)
        
# #         # Display the highlighted PDF
# #         with open(highlighted_pdf_path, "rb") as highlighted_pdf_file:
# #             highlighted_pdf_bytes = highlighted_pdf_file.read()
# #         st.download_button(label="Download Highlighted PDF", data=highlighted_pdf_bytes, file_name="highlighted_pdf.pdf")
# #         display_pdf(highlighted_pdf_path)
# #     else:
# #         st.write("No relevant document found.")

# # def main():
# #     st.set_page_config("Chat PDF")
# #     st.header("Chat with PDF using Gemini")
# #     user_question = st.text_input("Ask a Question from the PDF Files")
# #     if user_question:
# #         user_input(user_question)
# #     with st.sidebar:
# #         st.title("Menu:")
# #         uploaded_pdfs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=["pdf"])
# #         if uploaded_pdfs:
# #             saved_pdfs = save_uploaded_files(uploaded_pdfs)
# #             raw_text = get_pdf_text(saved_pdfs)
# #             text_chunks = get_text_chunks(raw_text)
# #             get_vector_store(text_chunks, saved_pdfs)
# #             st.success("PDFs uploaded and processed successfully!")

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
# # import fitz
# # import base64

# # load_dotenv()

# # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # # Store uploaded PDFs
# # PDF_FOLDER = "uploaded_pdfs"
# # if not os.path.exists(PDF_FOLDER):
# #     os.makedirs(PDF_FOLDER)

# # def save_uploaded_files(uploaded_files):
# #     saved_files = []
# #     for file in uploaded_files:
# #         file_path = os.path.join(PDF_FOLDER, file.name)
# #         with open(file_path, "wb") as f:
# #             f.write(file.getbuffer())
# #         saved_files.append(file_path)
# #     return saved_files

# # def get_pdf_text(pdf_path):
# #     text = ""
# #     pdf_reader = PdfReader(pdf_path)
# #     for page in pdf_reader.pages:
# #         text += page.extract_text()
# #     return text

# # def get_text_chunks(text):
# #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
# #     return text_splitter.split_text(text)

# # def get_vector_store(text_chunks, pdf_path):
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# #     metadatas = [{"source": pdf_path}] * len(text_chunks)
# #     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings, metadatas=metadatas)
# #     vector_store.save_local("faiss_index")

# # def get_conversational_chain():
# #     prompt_template = """Answer the question as detailed as possible from the provided context. If the answer is not available in the context, just say: "Answer is not available in the context." Context:\n {context}? Question: \n{question}\n Answer:"""
# #     model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
# #     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
# #     return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# # def display_pdf(pdf_path):
# #     with open(pdf_path, "rb") as f:
# #         base64_pdf = base64.b64encode(f.read()).decode("utf-8")
# #     pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="800" type="application/pdf"></iframe>'
# #     st.markdown(pdf_display, unsafe_allow_html=True)

# # def user_input(user_question):
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# #     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
# #     docs = new_db.similarity_search(user_question)
# #     chain = get_conversational_chain()
# #     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
# #     if docs:
# #         pdf_source = docs[0].metadata.get("source", "Unknown PDF")
# #         st.write("Reply:", response["output_text"])
# #         st.write(f"**Answer found in:** `{pdf_source}`")
        
# #         # Highlight the first line of the PDF
# #         with open(pdf_source, "rb") as pdf_file:
# #             pdf_bytes = pdf_file.read()

# #         # Use PyMuPDF to highlight the first line
# #         doc = fitz.open(stream=pdf_bytes)
# #         page = doc.load_page(0)  # Load the first page
# #         text = page.get_text("text")
# #         lines = text.splitlines()
# #         if lines:
# #             first_line = lines[0]
# #             rect = page.search_for(first_line)
# #             if rect:
# #                 for r in rect:
# #                     page.add_highlight_annot(r, color=(1, 1, 0))  # Yellow color

# #         doc.save("highlighted_pdf.pdf")

# #         # Display the highlighted PDF
# #         with open("highlighted_pdf.pdf", "rb") as highlighted_pdf_file:
# #             highlighted_pdf_bytes = highlighted_pdf_file.read()
# #         st.download_button(label="Download Highlighted PDF", data=highlighted_pdf_bytes, file_name="highlighted_pdf.pdf")
# #         display_pdf("highlighted_pdf.pdf")
# #     else:
# #         st.write("No relevant document found.")

# # def main():
# #     st.set_page_config("Chat PDF")
# #     st.header("Chat with PDF using Gemini")
# #     user_question = st.text_input("Ask a Question from the PDF Files")
# #     if user_question:
# #         user_input(user_question)
# #     with st.sidebar:
# #         st.title("Menu:")
# #         uploaded_pdfs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=["pdf"])






















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
# # import fitz
# # import base64

# # load_dotenv()

# # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # # Store uploaded PDFs
# # PDF_FOLDER = "uploaded_pdfs"
# # if not os.path.exists(PDF_FOLDER):
# #     os.makedirs(PDF_FOLDER)

# # def save_uploaded_files(uploaded_files):
# #     saved_files = []
# #     for file in uploaded_files:
# #         file_path = os.path.join(PDF_FOLDER, file.name)
# #         with open(file_path, "wb") as f:
# #             f.write(file.getbuffer())
# #         saved_files.append(file_path)
# #     return saved_files

# # def get_pdf_text(pdf_path):
# #     text = ""
# #     pdf_reader = PdfReader(pdf_path)
# #     for page in pdf_reader.pages:
# #         text += page.extract_text()
# #     return text

# # def get_text_chunks(text):
# #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
# #     return text_splitter.split_text(text)

# # def get_vector_store(text_chunks, pdf_path):
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# #     metadatas = [{"source": pdf_path}] * len(text_chunks)
# #     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings, metadatas=metadatas)
# #     vector_store.save_local("faiss_index")

# # def get_conversational_chain():
# #     prompt_template = """Answer the question as detailed as possible from the provided context. If the answer is not available in the context, just say: "Answer is not available in the context." Context:\n {context}? Question: \n{question}\n Answer:"""
# #     model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
# #     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
# #     return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# # def display_pdf(pdf_path):
# #     with open(pdf_path, "rb") as f:
# #         base64_pdf = base64.b64encode(f.read()).decode("utf-8")
# #     pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="800" type="application/pdf"></iframe>'
# #     st.markdown(pdf_display, unsafe_allow_html=True)

# # def user_input(user_question):
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# #     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
# #     docs = new_db.similarity_search(user_question)
# #     chain = get_conversational_chain()
# #     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
# #     if docs:
# #         pdf_source = docs[0].metadata.get("source", "Unknown PDF")
# #         st.write("Reply:", response["output_text"])
# #         st.write(f"**Answer found in:** `{pdf_source}`")
        
# #         # Highlight the first line of the PDF
# #         with open(pdf_source, "rb") as pdf_file:
# #             pdf_bytes = pdf_file.read()

# #         # Use PyMuPDF to highlight the first line
# #         doc = fitz.open(stream=pdf_bytes)
# #         page = doc.load_page(0)  # Load the first page
# #         text = page.get_text("text")
# #         lines = text.splitlines()
# #         if lines:
# #             first_line = lines[0]
# #             rect = page.search_for(first_line)
# #             if rect:
# #                 page.add_highlight_annot(rect, color=(1, 1, 0))  # Yellow color

# #         doc.save("highlighted_pdf.pdf")

# #         # Display the highlighted PDF
# #         with open("highlighted_pdf.pdf", "rb") as highlighted_pdf_file:
# #             highlighted_pdf_bytes = highlighted_pdf_file.read()
# #         st.download_button(label="Download Highlighted PDF", data=highlighted_pdf_bytes, file_name="highlighted_pdf.pdf")
# #         display_pdf("highlighted_pdf.pdf")
# #     else:
# #         st.write("No relevant document found.")

# # def main():
# #     st.set_page_config("Chat PDF")
# #     st.header("Chat with PDF using Gemini")
# #     user_question = st.text_input("Ask a Question from the PDF Files")
# #     if user_question:
# #         user_input(user_question)
# #     with st.sidebar:
# #         st.title("Menu:")
# #         uploaded_pdfs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=["pdf"])

















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
# # import fitz
# # import base64
# # from fuzzywuzzy import fuzz

# # load_dotenv()

# # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # # Store uploaded PDFs
# # PDF_FOLDER = "uploaded_pdfs"
# # if not os.path.exists(PDF_FOLDER):
# #     os.makedirs(PDF_FOLDER)

# # def save_uploaded_files(uploaded_files):
# #     saved_files = []
# #     for file in uploaded_files:
# #         file_path = os.path.join(PDF_FOLDER, file.name)
# #         with open(file_path, "wb") as f:
# #             f.write(file.getbuffer())
# #         saved_files.append(file_path)
# #     return saved_files

# # def get_pdf_text(pdf_path):
# #     text = ""
# #     pdf_reader = PdfReader(pdf_path)
# #     for page in pdf_reader.pages:
# #         text += page.extract_text()
# #     return text

# # def get_text_chunks(text):
# #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
# #     return text_splitter.split_text(text)

# # def get_vector_store(text_chunks, pdf_path):
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# #     metadatas = [{"source": pdf_path}] * len(text_chunks)
# #     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings, metadatas=metadatas)
# #     vector_store.save_local("faiss_index")

# # def get_conversational_chain():
# #     prompt_template = """Answer the question as detailed as possible from the provided context. If the answer is not available in the context, just say: "Answer is not available in the context." Context:\n {context}? Question: \n{question}\n Answer:"""
# #     model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
# #     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
# #     return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# # def display_pdf(pdf_path):
# #     with open(pdf_path, "rb") as f:
# #         base64_pdf = base64.b64encode(f.read()).decode("utf-8")
# #     pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="800" type="application/pdf"></iframe>'
# #     st.markdown(pdf_display, unsafe_allow_html=True)

# # def user_input(user_question):
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# #     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
# #     docs = new_db.similarity_search(user_question)
# #     chain = get_conversational_chain()
# #     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
# #     if docs:
# #         pdf_source = docs[0].metadata.get("source", "Unknown PDF")
# #         st.write("Reply:", response["output_text"])
# #         st.write(f"**Answer found in:** `{pdf_source}`")
        
# #         # Highlight the answer in the PDF
# #         with open(pdf_source, "rb") as pdf_file:
# #             pdf_bytes = pdf_file.read()
        
# #         # Use PyMuPDF to highlight the answer
# #         doc = fitz.open(stream=pdf_bytes)
# #         page = doc.load_page(0)  # Load the first page
# #         text = page.get_text("text")
# #         answer = response["output_text"]
        
# #         # If exact answer is not found, highlight the relevant text
# #         if answer not in text:
# #             # Use a fuzzy search to find the most relevant text
# #             best_match = max(text.splitlines(), key=lambda x: fuzz.partial_ratio(x, answer))
# #             rect = page.search_for(best_match)
# #             if rect:
# #                 page.add_text_annot(rect, text="Relevant Text", fontsize=12, color=(1, 1, 0))
# #         else:
# #             rect = page.search_for(answer)
# #             if rect:
# #                 page.add_text_annot(rect, text="Answer", fontsize=12, color=(1, 1, 0))
        
# #         doc.save("highlighted_pdf.pdf")
        
# #         # Display the highlighted PDF
# #         with open("highlighted_pdf.pdf", "rb") as highlighted_pdf_file:
# #             highlighted_pdf_bytes = highlighted_pdf_file.read()
# #         # st.download_button(label="Download Highlighted PDF", data=highlighted_pdf_bytes, file)
# #         st.download_button(label="Download Highlighted PDF", data=highlighted_pdf_bytes, file_name="highlighted_pdf.pdf")






















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
# # import fitz
# # import base64

# # load_dotenv()

# # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # # Store uploaded PDFs
# # PDF_FOLDER = "uploaded_pdfs"
# # if not os.path.exists(PDF_FOLDER):
# #     os.makedirs(PDF_FOLDER)

# # def save_uploaded_files(uploaded_files):
# #     saved_files = []
# #     for file in uploaded_files:
# #         file_path = os.path.join(PDF_FOLDER, file.name)
# #         with open(file_path, "wb") as f:
# #             f.write(file.getbuffer())
# #         saved_files.append(file_path)
# #     return saved_files

# # def get_pdf_text(pdf_path):
# #     text = ""
# #     pdf_reader = PdfReader(pdf_path)
# #     for page in pdf_reader.pages:
# #         text += page.extract_text()
# #     return text

# # def get_text_chunks(text):
# #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
# #     return text_splitter.split_text(text)

# # def get_vector_store(text_chunks, pdf_path):
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# #     metadatas = [{"source": pdf_path}] * len(text_chunks)
# #     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings, metadatas=metadatas)
# #     vector_store.save_local("faiss_index")

# # def get_conversational_chain():
# #     prompt_template = """Answer the question as detailed as possible from the provided context. If the answer is not available in the context, just say: "Answer is not available in the context." Context:\n {context}? Question: \n{question}\n Answer:"""
# #     model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
# #     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
# #     return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# # def display_pdf(pdf_path):
# #     with open(pdf_path, "rb") as f:
# #         base64_pdf = base64.b64encode(f.read()).decode("utf-8")
# #     pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="800" type="application/pdf"></iframe>'
# #     st.markdown(pdf_display, unsafe_allow_html=True)

# # def user_input(user_question):
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# #     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
# #     docs = new_db.similarity_search(user_question)
# #     chain = get_conversational_chain()
# #     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
# #     if docs:
# #         pdf_source = docs[0].metadata.get("source", "Unknown PDF")
# #         st.write("Reply:", response["output_text"])
# #         st.write(f"**Answer found in:** `{pdf_source}`")
        
# #         # Highlight the answer in the PDF
# #         with open(pdf_source, "rb") as pdf_file:
# #             pdf_bytes = pdf_file.read()
        
# #         # Use PyMuPDF to highlight the answer
# #         doc = fitz.open(stream=pdf_bytes)
# #         page = doc.load_page(0)  # Load the first page
# #         text = page.get_text("text")
# #         answer = response["output_text"]
# #         rect = page.search_for(answer)
# #         if rect:
# #             page.add_highlight_annot(rect)
# #         doc.save("highlighted_pdf.pdf")
        
# #         # Display the highlighted PDF
# #         with open("highlighted_pdf.pdf", "rb") as highlighted_pdf_file:
# #             highlighted_pdf_bytes = highlighted_pdf_file.read()
# #         st.download_button(label="Download Highlighted PDF", data=highlighted_pdf_bytes)


























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

# # load_dotenv()
# # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # # Store uploaded PDFs
# # PDF_FOLDER = "uploaded_pdfs"
# # if not os.path.exists(PDF_FOLDER):
# #     os.makedirs(PDF_FOLDER)

# # def save_uploaded_files(uploaded_files):
# #     saved_files = []
# #     for file in uploaded_files:
# #         file_path = os.path.join(PDF_FOLDER, file.name)
# #         with open(file_path, "wb") as f:
# #             f.write(file.getbuffer())
# #         saved_files.append(file_path)
# #     return saved_files

# # def get_pdf_text(pdf_path):
# #     text = ""
# #     pdf_reader = PdfReader(pdf_path)
# #     for page in pdf_reader.pages:
# #         text += page.extract_text()
# #     return text

# # def get_text_chunks(text):
# #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
# #     return text_splitter.split_text(text)

# # def get_vector_store(text_chunks, pdf_path):
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# #     metadatas = [{"source": pdf_path}] * len(text_chunks)
# #     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings, metadatas=metadatas)
# #     vector_store.save_local("faiss_index")

# # def get_conversational_chain():
# #     prompt_template = """
# #     Answer the question as detailed as possible from the provided context.
# #     If the answer is not available in the context, just say: "Answer is not available in the context."
    
# #     Context:\n {context}?
# #     Question: \n{question}\n
# #     Answer:
# #     """
    
# #     model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
# #     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
# #     return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# # import streamlit as st
# # import base64

# # import streamlit as st
# # import base64

# # def display_pdf(pdf_path):
# #     """Embed a PDF inside Streamlit from a local file."""
# #     with open(pdf_path, "rb") as f:
# #         base64_pdf = base64.b64encode(f.read()).decode("utf-8")
# #         pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="800" type="application/pdf"></iframe>'
# #         st.markdown(pdf_display, unsafe_allow_html=True)

# # def user_input(user_question):
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# #     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
# #     docs = new_db.similarity_search(user_question)
# #     chain = get_conversational_chain()
# #     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

# #     if docs:
# #         pdf_source = docs[0].metadata.get("source", "Unknown PDF")
# #         st.write("Reply:", response["output_text"])
# #         st.write(f"**Answer found in:** `{pdf_source}`")

# #         # Show PDF in Streamlit
# #         if os.path.exists(pdf_source):
# #             with open(pdf_source, "rb") as pdf_file:
# #                 pdf_bytes = pdf_file.read()
# #                 st.download_button(label="Download PDF", data=pdf_bytes, file_name=os.path.basename(pdf_source), mime="application/pdf")

# #                 # Display PDF inside Streamlit
# #                 st.write("### 📄 Preview of the PDF:")
# #                 display_pdf(pdf_source)
# #         else:
# #             st.error("PDF file not found!")
# #     else:
# #         st.write("No relevant document found.")


# # def main():
# #     st.set_page_config("Chat PDF")
# #     st.header("Chat with PDF using Gemini💁")

# #     user_question = st.text_input("Ask a Question from the PDF Files")

# #     if user_question:
# #         user_input(user_question)

# #     with st.sidebar:
# #         st.title("Menu:")
# #         uploaded_pdfs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
# #         if st.button("Submit & Process"):
# #             with st.spinner("Processing..."):
# #                 pdf_paths = save_uploaded_files(uploaded_pdfs)
# #                 for pdf_path in pdf_paths:
# #                     raw_text = get_pdf_text(pdf_path)
# #                     text_chunks = get_text_chunks(raw_text)
# #                     get_vector_store(text_chunks, pdf_path)
# #                 st.success("Processing completed!")

# # if __name__ == "__main__":
# #     main()


















# # import streamlit as st
# # from PyPDF2 import PdfReader
# # import fitz  # PyMuPDF
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # import os
# # from langchain_google_genai import GoogleGenerativeAIEmbeddings
# # import google.generativeai as genai
# # from langchain.vectorstores import FAISS
# # from langchain_google_genai import ChatGoogleGenerativeAI
# # from langchain.chains.question_answering import load_qa_chain
# # from langchain.prompts import PromptTemplate
# # from dotenv import load_dotenv

# # # Load API Key
# # load_dotenv()
# # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # # Directory to store PDFs
# # PDF_STORAGE_PATH = "uploaded_pdfs"
# # os.makedirs(PDF_STORAGE_PATH, exist_ok=True)

# # def get_pdf_text(pdf_docs):
# #     """Extracts text from PDFs and stores them."""
# #     pdf_data = {}
    
# #     for pdf in pdf_docs:
# #         pdf_path = os.path.join(PDF_STORAGE_PATH, pdf.name)
# #         with open(pdf_path, "wb") as f:
# #             f.write(pdf.getbuffer())
        
# #         pdf_reader = PdfReader(pdf_path)
# #         pdf_text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        
# #         pdf_data[pdf_path] = pdf_text

# #     return pdf_data

# # def get_text_chunks(pdf_data):
# #     """Splits text into smaller chunks for processing."""
# #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
# #     chunks = []
# #     metadata = []

# #     for pdf_path, text in pdf_data.items():
# #         pdf_chunks = text_splitter.split_text(text)
# #         chunks.extend(pdf_chunks)
# #         metadata.extend([{"source": pdf_path}] * len(pdf_chunks))

# #     return chunks, metadata

# # def get_vector_store(text_chunks, metadata):
# #     """Creates vector embeddings and stores them in FAISS."""
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# #     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings, metadatas=metadata)
# #     vector_store.save_local("faiss_index")

# # def get_conversational_chain():
# #     """Loads LLM and sets up a prompt template for answering queries."""
# #     prompt_template = """
# #     Answer the question as detailed as possible from the provided context. If the answer is not found in the context, 
# #     just say: 'answer is not available in the context'. Do not provide a wrong answer.\n\n
# #     Context:\n {context}\n
# #     Question: {question}\n
# #     Answer:
# #     """
# #     model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
# #     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
# #     return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# # def highlight_text_in_pdf(pdf_path, answer_text):
# #     """Highlights the answer text in the given PDF and saves a new highlighted version."""
# #     highlighted_pdf_path = pdf_path.replace(".pdf", "_highlighted.pdf")

# #     doc = fitz.open(pdf_path)

# #     for page in doc:
# #         text_instances = page.search_for(answer_text)
# #         for inst in text_instances:
# #             page.add_highlight_annot(inst)  # Highlight in yellow

# #     doc.save(highlighted_pdf_path)
# #     return highlighted_pdf_path

# # def user_input(user_question):
# #     """Handles user queries by searching for answers in PDFs and displaying the correct file."""
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# #     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# #     docs = new_db.similarity_search(user_question)
# #     chain = get_conversational_chain()

# #     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
# #     answer_text = response["output_text"]

# #     # Find the PDF containing the answer
# #     pdf_found = None
# #     for doc in docs:
# #         source = doc.metadata.get("source")
# #         if source:
# #             pdf_found = source
# #             break

# #     st.write("📝 **Reply:**", answer_text)

# #     if pdf_found:
# #         st.write(f"📄 **Answer found in:** `{pdf_found}`")

# #         # Highlight text in the PDF
# #         highlighted_pdf = highlight_text_in_pdf(pdf_found, answer_text)

# #         with open(highlighted_pdf, "rb") as pdf_file:
# #             pdf_data = pdf_file.read()
# #             st.download_button(label="📥 Download Highlighted PDF", data=pdf_data, file_name="highlighted.pdf", mime="application/pdf")

# #         st.write("📄 **Preview of the PDF:**")
# #         st.write("⚠️ Streamlit currently does not support direct PDF display. Download the highlighted PDF above.")
# #     else:
# #         st.write("⚠️ **Answer found, but could not locate the exact PDF.**")

# # def main():
# #     """Main function to run the Streamlit app."""
# #     st.set_page_config(page_title="Chat PDF")
# #     st.header("Chat with PDF using Gemini💁")

# #     user_question = st.text_input("Ask a Question from the PDF Files")

# #     if user_question:
# #         user_input(user_question)

# #     with st.sidebar:
# #         st.title("Menu:")
# #         pdf_docs = st.file_uploader("Upload your PDF Files and Click Submit", accept_multiple_files=True)
# #         if st.button("Submit & Process"):
# #             with st.spinner("Processing..."):
# #                 pdf_data = get_pdf_text(pdf_docs)
# #                 text_chunks, metadata = get_text_chunks(pdf_data)
# #                 get_vector_store(text_chunks, metadata)
# #                 st.success("Processing Complete ✅")

# # if __name__ == "__main__":
# #     main()





























# # import streamlit as st
# # from PyPDF2 import PdfReader, PdfWriter
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # import os
# # from langchain_google_genai import GoogleGenerativeAIEmbeddings
# # import google.generativeai as genai
# # from langchain.vectorstores import FAISS
# # from langchain_google_genai import ChatGoogleGenerativeAI
# # from langchain.chains.question_answering import load_qa_chain
# # from langchain.prompts import PromptTemplate
# # from dotenv import load_dotenv

# # # Load API Key
# # load_dotenv()
# # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # # Directory to store PDFs
# # PDF_STORAGE_PATH = "uploaded_pdfs"
# # os.makedirs(PDF_STORAGE_PATH, exist_ok=True)

# # def get_pdf_text(pdf_docs):
# #     """Extracts text from PDFs and stores them."""
# #     pdf_data = {}
    
# #     for pdf in pdf_docs:
# #         pdf_path = os.path.join(PDF_STORAGE_PATH, pdf.name)
# #         with open(pdf_path, "wb") as f:
# #             f.write(pdf.getbuffer())
        
# #         pdf_reader = PdfReader(pdf_path)
# #         pdf_text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        
# #         pdf_data[pdf_path] = pdf_text

# #     return pdf_data

# # def get_text_chunks(pdf_data):
# #     """Splits text into smaller chunks for processing."""
# #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
# #     chunks = []
# #     metadata = []

# #     for pdf_path, text in pdf_data.items():
# #         pdf_chunks = text_splitter.split_text(text)
# #         chunks.extend(pdf_chunks)
# #         metadata.extend([{"source": pdf_path}] * len(pdf_chunks))

# #     return chunks, metadata

# # def get_vector_store(text_chunks, metadata):
# #     """Creates vector embeddings and stores them in FAISS."""
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# #     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings, metadatas=metadata)
# #     vector_store.save_local("faiss_index")

# # def get_conversational_chain():
# #     """Loads LLM and sets up a prompt template for answering queries."""
# #     prompt_template = """
# #     Answer the question as detailed as possible from the provided context. If the answer is not found in the context, 
# #     just say: 'answer is not available in the context'. Do not provide a wrong answer.\n\n
# #     Context:\n {context}\n
# #     Question: {question}\n
# #     Answer:
# #     """
# #     model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
# #     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
# #     return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# # def highlight_text_in_pdf(pdf_path, answer_text):
# #     """Highlights the answer text in the given PDF and saves a new highlighted version."""
# #     highlighted_pdf_path = pdf_path.replace(".pdf", "_highlighted.pdf")

# #     pdf_reader = PdfReader(pdf_path)
# #     pdf_writer = PdfWriter()

# #     for page in pdf_reader.pages:
# #         text = page.extract_text()
# #         if text and answer_text in text:
# #             # Add annotation for highlighting (Dummy Highlight, real highlight requires advanced libraries)
# #             page.add_highlight(0, 0, 0, 0)  # Dummy values
# #         pdf_writer.add_page(page)

# #     with open(highlighted_pdf_path, "wb") as f:
# #         pdf_writer.write(f)

# #     return highlighted_pdf_path

# # def user_input(user_question):
# #     """Handles user queries by searching for answers in PDFs and displaying the correct file."""
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# #     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# #     docs = new_db.similarity_search(user_question)
# #     chain = get_conversational_chain()

# #     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
# #     answer_text = response["output_text"]

# #     # Find the PDF containing the answer
# #     pdf_found = None
# #     for doc in docs:
# #         source = doc.metadata.get("source")
# #         if source:
# #             pdf_found = source
# #             break

# #     st.write("📝 **Reply:**", answer_text)

# #     if pdf_found:
# #         st.write(f"📄 **Answer found in:** `{pdf_found}`")

# #         # Highlight text in the PDF
# #         highlighted_pdf = highlight_text_in_pdf(pdf_found, answer_text)

# #         with open(highlighted_pdf, "rb") as pdf_file:
# #             pdf_data = pdf_file.read()
# #             st.download_button(label="📥 Download Highlighted PDF", data=pdf_data, file_name="highlighted.pdf", mime="application/pdf")

# #         st.write("📄 **Preview of the PDF:**")
# #         st.pdf(highlighted_pdf)
# #     else:
# #         st.write("⚠️ **Answer found, but could not locate the exact PDF.**")

# # def main():
# #     """Main function to run the Streamlit app."""
# #     st.set_page_config(page_title="Chat PDF")
# #     st.header("Chat with PDF using Gemini💁")

# #     user_question = st.text_input("Ask a Question from the PDF Files")

# #     if user_question:
# #         user_input(user_question)

# #     with st.sidebar:
# #         st.title("Menu:")
# #         pdf_docs = st.file_uploader("Upload your PDF Files and Click Submit", accept_multiple_files=True)
# #         if st.button("Submit & Process"):
# #             with st.spinner("Processing..."):
# #                 pdf_data = get_pdf_text(pdf_docs)
# #                 text_chunks, metadata = get_text_chunks(pdf_data)
# #                 get_vector_store(text_chunks, metadata)
# #                 st.success("Processing Complete ✅")

# # if __name__ == "__main__":
# #     main()


























# # import streamlit as st
# # from PyPDF2 import PdfReader, PdfWriter
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # import os
# # import shutil
# # from langchain_google_genai import GoogleGenerativeAIEmbeddings
# # import google.generativeai as genai
# # from langchain.vectorstores import FAISS
# # from langchain_google_genai import ChatGoogleGenerativeAI
# # from langchain.chains.question_answering import load_qa_chain
# # from langchain.prompts import PromptTemplate
# # from dotenv import load_dotenv

# # # Load environment variables
# # load_dotenv()
# # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # # Ensure directory for storing PDFs
# # PDF_STORAGE_PATH = "uploaded_pdfs"
# # os.makedirs(PDF_STORAGE_PATH, exist_ok=True)

# # def get_pdf_text(pdf_docs):
# #     """Extracts text from uploaded PDFs and saves them."""
# #     text = ""
# #     pdf_paths = []
    
# #     for pdf in pdf_docs:
# #         pdf_path = os.path.join(PDF_STORAGE_PATH, pdf.name)
# #         pdf_paths.append(pdf_path)
        
# #         # Save PDF to storage
# #         with open(pdf_path, "wb") as f:
# #             f.write(pdf.getbuffer())
        
# #         pdf_reader = PdfReader(pdf)
# #         for page in pdf_reader.pages:
# #             text += page.extract_text()
    
# #     return text, pdf_paths

# # def get_text_chunks(text):
# #     """Splits text into smaller chunks for processing."""
# #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
# #     return text_splitter.split_text(text)

# # def get_vector_store(text_chunks):
# #     """Creates vector embeddings and stores them in FAISS."""
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# #     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
# #     vector_store.save_local("faiss_index")

# # def get_conversational_chain():
# #     """Loads LLM and sets up a prompt template for answering queries."""
# #     prompt_template = """
# #     Answer the question as detailed as possible from the provided context. If the answer is not found in the context, 
# #     just say: 'answer is not available in the context'. Do not provide a wrong answer.\n\n
# #     Context:\n {context}\n
# #     Question: {question}\n
# #     Answer:
# #     """
# #     model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
# #     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
# #     return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# # def highlight_text_in_pdf(pdf_path, answer_text):
# #     """Highlights the answer text in the given PDF and saves a new highlighted version."""
# #     highlighted_pdf_path = pdf_path.replace(".pdf", "_highlighted.pdf")

# #     pdf_reader = PdfReader(pdf_path)
# #     pdf_writer = PdfWriter()

# #     for page in pdf_reader.pages:
# #         text = page.extract_text()
# #         if answer_text in text:
# #             # Add annotation for highlighting (basic implementation)
# #             page.add_highlight(0, 0, 0, 0)  # Dummy values, improve later
# #         pdf_writer.add_page(page)

# #     # Save highlighted PDF
# #     with open(highlighted_pdf_path, "wb") as f:
# #         pdf_writer.write(f)

# #     return highlighted_pdf_path

# # def user_input(user_question):
# #     """Handles user queries by searching for answers in PDFs and displaying the correct file."""
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# #     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# #     docs = new_db.similarity_search(user_question)
# #     chain = get_conversational_chain()

# #     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
# #     answer_text = response["output_text"]

# #     # Find the PDF containing the answer
# #     pdf_found = None
# #     for pdf_file in os.listdir(PDF_STORAGE_PATH):
# #         pdf_path = os.path.join(PDF_STORAGE_PATH, pdf_file)
# #         pdf_reader = PdfReader(pdf_path)
# #         for page in pdf_reader.pages:
# #             if answer_text in page.extract_text():
# #                 pdf_found = pdf_path
# #                 break
# #         if pdf_found:
# #             break

# #     # Display answer and PDF
# #     st.write("📝 **Reply:**", answer_text)
    
# #     if pdf_found:
# #         st.write(f"📄 **Answer found in:** `{pdf_found}`")
        
# #         # Highlight text in the PDF
# #         highlighted_pdf = highlight_text_in_pdf(pdf_found, answer_text)

# #         # Display PDF with highlights
# #         with open(highlighted_pdf, "rb") as pdf_file:
# #             pdf_data = pdf_file.read()
# #             st.download_button(label="📥 Download Highlighted PDF", data=pdf_data, file_name="highlighted.pdf", mime="application/pdf")

# #         st.write("📄 **Preview of the PDF:**")
# #         st.pdf(highlighted_pdf)
# #     else:
# #         st.write("⚠️ **Answer found, but could not locate the exact PDF.**")

# # def main():
# #     """Main function to run the Streamlit app."""
# #     st.set_page_config(page_title="Chat PDF")
# #     st.header("Chat with PDF using Gemini💁")

# #     user_question = st.text_input("Ask a Question from the PDF Files")

# #     if user_question:
# #         user_input(user_question)

# #     with st.sidebar:
# #         st.title("Menu:")
# #         pdf_docs = st.file_uploader("Upload your PDF Files and Click Submit", accept_multiple_files=True)
# #         if st.button("Submit & Process"):
# #             with st.spinner("Processing..."):
# #                 raw_text, pdf_paths = get_pdf_text(pdf_docs)
# #                 text_chunks = get_text_chunks(raw_text)
# #                 get_vector_store(text_chunks)
# #                 st.success("Processing Complete ✅")

# # if __name__ == "__main__":
# #     main()



























# # import streamlit as st
# # from PyPDF2 import PdfReader
# # import fitz  # PyMuPDF for highlighting text
# # import os
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # from langchain_google_genai import GoogleGenerativeAIEmbeddings
# # import google.generativeai as genai
# # from langchain.vectorstores import FAISS
# # from langchain_google_genai import ChatGoogleGenerativeAI
# # from langchain.chains.question_answering import load_qa_chain
# # from langchain.prompts import PromptTemplate
# # from dotenv import load_dotenv

# # # Load environment variables
# # load_dotenv()
# # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # # Ensure upload folder exists
# # UPLOAD_FOLDER = "uploaded_pdfs"
# # os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# # def save_uploaded_pdfs(uploaded_files):
# #     """Save uploaded PDFs to a directory"""
# #     pdf_paths = []
# #     for uploaded_file in uploaded_files:
# #         pdf_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
# #         with open(pdf_path, "wb") as f:
# #             f.write(uploaded_file.getbuffer())
# #         pdf_paths.append(pdf_path)
# #     return pdf_paths


# # def get_pdf_text(pdf_paths):
# #     """Extract text from PDFs"""
# #     text_data = {}
# #     for pdf_path in pdf_paths:
# #         text = ""
# #         pdf_reader = PdfReader(pdf_path)
# #         for page in pdf_reader.pages:
# #             text += page.extract_text() + "\n"
# #         text_data[pdf_path] = text
# #     return text_data


# # def get_text_chunks(text):
# #     """Split text into chunks"""
# #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
# #     return text_splitter.split_text(text)


# # def get_vector_store(text_chunks):
# #     """Store text embeddings in FAISS"""
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# #     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
# #     vector_store.save_local("faiss_index")


# # def highlight_text_in_pdf(pdf_path, answer_text):
# #     """Highlight the answer in the given PDF file and save it"""
# #     doc = fitz.open(pdf_path)

# #     for page in doc:
# #         text_instances = page.search_for(answer_text)
# #         for inst in text_instances:
# #             page.add_highlight_annot(inst)  # Add highlight

# #     highlighted_pdf_path = pdf_path.replace(".pdf", "_highlighted.pdf")
# #     doc.save(highlighted_pdf_path)
# #     return highlighted_pdf_path


# # def get_conversational_chain():
# #     """Define a prompt template and create a LangChain QA model"""
# #     prompt_template = """
# #     Answer the question as detailed as possible from the provided context. 
# #     If the answer is not in the provided context, just say, "answer is not available in the context".
    
# #     Context:\n {context}?\n
# #     Question: \n{question}\n

# #     Answer:
# #     """

# #     model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
# #     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
# #     return load_qa_chain(model, chain_type="stuff", prompt=prompt)


# # def user_input(user_question, pdf_texts):
# #     """Find the answer and highlight it in the PDF"""
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# #     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# #     docs = new_db.similarity_search(user_question)
# #     chain = get_conversational_chain()

# #     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
# #     answer_text = response["output_text"]

# #     # Find the PDF where the answer is found
# #     matching_pdf = None
# #     for pdf_path, text in pdf_texts.items():
# #         if answer_text in text:
# #             matching_pdf = pdf_path
# #             break

# #     if matching_pdf:
# #         highlighted_pdf_path = highlight_text_in_pdf(matching_pdf, answer_text)
# #         st.write(f"**Answer found in:** {os.path.basename(matching_pdf)}")
# #         st.write("📄 **Preview of the PDF with Highlighted Answer:**")
# #         st.pdf(highlighted_pdf_path)  # Display modified PDF
# #     else:
# #         st.write("Answer found, but could not locate the exact PDF.")

# #     st.write("📝 **Reply:**", answer_text)


# # def main():
# #     st.set_page_config("Chat PDF")
# #     st.header("Chat with PDF using Gemini💁")

# #     with st.sidebar:
# #         st.title("📂 Upload PDFs")
# #         uploaded_files = st.file_uploader("Upload PDF Files", accept_multiple_files=True)
# #         if st.button("Submit & Process"):
# #             if uploaded_files:
# #                 with st.spinner("Processing PDFs..."):
# #                     pdf_paths = save_uploaded_pdfs(uploaded_files)
# #                     pdf_texts = get_pdf_text(pdf_paths)
# #                     all_text_chunks = [chunk for text in pdf_texts.values() for chunk in get_text_chunks(text)]
# #                     get_vector_store(all_text_chunks)
# #                     st.session_state["pdf_texts"] = pdf_texts
# #                     st.success("PDFs Processed Successfully!")

# #     user_question = st.text_input("Ask a Question from the PDF Files")
# #     if user_question and "pdf_texts" in st.session_state:
# #         user_input(user_question, st.session_state["pdf_texts"])


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

# # load_dotenv()
# # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # # Store uploaded PDFs
# # PDF_FOLDER = "uploaded_pdfs"
# # if not os.path.exists(PDF_FOLDER):
# #     os.makedirs(PDF_FOLDER)

# # def save_uploaded_files(uploaded_files):
# #     saved_files = []
# #     for file in uploaded_files:
# #         file_path = os.path.join(PDF_FOLDER, file.name)
# #         with open(file_path, "wb") as f:
# #             f.write(file.getbuffer())
# #         saved_files.append(file_path)
# #     return saved_files

# # def get_pdf_text(pdf_path):
# #     text = ""
# #     pdf_reader = PdfReader(pdf_path)
# #     for page in pdf_reader.pages:
# #         text += page.extract_text()
# #     return text

# # def get_text_chunks(text):
# #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
# #     return text_splitter.split_text(text)

# # def get_vector_store(text_chunks, pdf_path):
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# #     metadatas = [{"source": pdf_path}] * len(text_chunks)
# #     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings, metadatas=metadatas)
# #     vector_store.save_local("faiss_index")

# # def get_conversational_chain():
# #     prompt_template = """
# #     Answer the question as detailed as possible from the provided context.
# #     If the answer is not available in the context, just say: "Answer is not available in the context."
    
# #     Context:\n {context}?
# #     Question: \n{question}\n
# #     Answer:
# #     """
    
# #     model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
# #     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
# #     return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# # import streamlit as st
# # import base64

# # def display_pdf(pdf_path):
# #     """Embed a PDF inside Streamlit from a local file."""
# #     with open(pdf_path, "rb") as f:
# #         base64_pdf = base64.b64encode(f.read()).decode("utf-8")
# #         pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="800" type="application/pdf"></iframe>'
# #         st.markdown(pdf_display, unsafe_allow_html=True)

# # def user_input(user_question):
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# #     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
# #     docs = new_db.similarity_search(user_question)
# #     chain = get_conversational_chain()
# #     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

# #     if docs:
# #         pdf_source = docs[0].metadata.get("source", "Unknown PDF")
# #         st.write("Reply:", response["output_text"])
# #         st.write(f"**Answer found in:** `{pdf_source}`")

# #         # Show PDF in Streamlit
# #         if os.path.exists(pdf_source):
# #             with open(pdf_source, "rb") as pdf_file:
# #                 pdf_bytes = pdf_file.read()
# #                 st.download_button(label="Download PDF", data=pdf_bytes, file_name=os.path.basename(pdf_source), mime="application/pdf")

# #                 # Display PDF inside Streamlit
# #                 st.write("### 📄 Preview of the PDF:")
# #                 display_pdf(pdf_source)
# #         else:
# #             st.error("PDF file not found!")
# #     else:
# #         st.write("No relevant document found.")

# # def main():
# #     st.set_page_config("Chat PDF")
# #     st.header("Chat with PDF using Gemini💁")

# #     user_question = st.text_input("Ask a Question from the PDF Files")

# #     if user_question:
# #         user_input(user_question)

# #     with st.sidebar:
# #         st.title("Menu:")
# #         uploaded_pdfs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
# #         if st.button("Submit & Process"):
# #             with st.spinner("Processing..."):
# #                 pdf_paths = save_uploaded_files(uploaded_pdfs)
# #                 for pdf_path in pdf_paths:
# #                     raw_text = get_pdf_text(pdf_path)
# #                     text_chunks = get_text_chunks(raw_text)
# #                     get_vector_store(text_chunks, pdf_path)
# #                 st.success("Processing completed!")

# # if __name__ == "__main__":
# #     main()
























# # import streamlit as st
# # from PyPDF2 import PdfReader
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # import os
# # import base64
# # from langchain_google_genai import GoogleGenerativeAIEmbeddings
# # import google.generativeai as genai
# # from langchain.vectorstores import FAISS
# # from langchain_google_genai import ChatGoogleGenerativeAI
# # from langchain.chains.question_answering import load_qa_chain
# # from langchain.prompts import PromptTemplate
# # from dotenv import load_dotenv

# # # Load API Key
# # load_dotenv()
# # os.getenv("GOOGLE_API_KEY")
# # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # # Function to extract text from PDFs
# # def get_pdf_text(pdf_docs):
# #     text = ""
# #     for pdf in pdf_docs:
# #         pdf_reader = PdfReader(pdf)
# #         for page in pdf_reader.pages:
# #             text += page.extract_text()
# #     return text

# # # Function to split text into chunks
# # def get_text_chunks(text):
# #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
# #     return text_splitter.split_text(text)

# # # Function to store embeddings in FAISS with metadata (PDF names)
# # def get_vector_store(pdf_docs, text_chunks):
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# #     # Assign PDF name to each chunk
# #     metadata = []
# #     chunked_texts = []  # Separate list to store chunks for metadata alignment

# #     for pdf in pdf_docs:
# #         pdf_reader = PdfReader(pdf)
# #         pdf_text = ""
# #         for page in pdf_reader.pages:
# #             pdf_text += page.extract_text()

# #         chunks = get_text_chunks(pdf_text)  # Get chunks per PDF

# #         for chunk in chunks:
# #             metadata.append({"source": pdf.name})  # Store PDF filename
# #             chunked_texts.append(chunk)  # Store corresponding chunk

# #     # Store chunks & metadata in FAISS
# #     vector_store = FAISS.from_texts(chunked_texts, embedding=embeddings, metadatas=metadata)
# #     vector_store.save_local("faiss_index")


# # # Function to create a conversational chain
# # def get_conversational_chain():
# #     prompt_template = """
# #     Answer the question as detailed as possible from the provided context. If the answer is not in
# #     the provided context, just say, "answer is not available in the context". Do not provide incorrect answers.\n\n
# #     Context:\n {context}?\n
# #     Question: \n{question}\n
# #     Answer:
# #     """
# #     model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
# #     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
# #     return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# # # Function to encode PDF to base64 for inline display
# # def pdf_to_base64(pdf_path):
# #     with open(pdf_path, "rb") as pdf_file:
# #         return base64.b64encode(pdf_file.read()).decode("utf-8")

# # # Function to process user input and display answer + PDF
# # def user_input(user_question):
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# #     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
# #     docs = new_db.similarity_search(user_question)
# #     if not docs:
# #         st.write("No relevant answer found in the uploaded PDFs.")
# #         return

# #     chain = get_conversational_chain()
# #     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

# #     # Extract PDF name where answer was found
# #     source_pdf = docs[0].metadata.get("source", "Unknown PDF")

# #     st.write("### **Reply:**")
# #     st.write(response["output_text"])

# #     # Find the correct PDF file
# #     pdf_path = os.path.join("uploaded_pdfs", source_pdf)

# #     if not os.path.exists(pdf_path):
# #         st.write("⚠️ **Error:** Could not find the source PDF.")
# #         return

# #     # Display PDF inside Streamlit
# #     with open(pdf_path, "rb") as pdf_file:
# #         st.download_button(label="Download PDF", data=pdf_file, file_name=source_pdf, mime="application/pdf")

# #     st.write("### **View PDF Below:**")
# #     st.components.v1.html(f"""
# #         <iframe src="data:application/pdf;base64,{pdf_to_base64(pdf_path)}" width="100%" height="600px"></iframe>
# #     """, height=650)

# # # Streamlit App
# # def main():
# #     st.set_page_config("Chat PDF")
# #     st.header("Chat with PDF using Gemini💁")

# #     user_question = st.text_input("Ask a Question from the PDF Files")

# #     if user_question:
# #         user_input(user_question)

# #     with st.sidebar:
# #         st.title("Menu:")
# #         pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)

# #         if st.button("Submit & Process"):
# #             with st.spinner("Processing..."):
# #                 os.makedirs("uploaded_pdfs", exist_ok=True)  # Ensure upload directory exists

# #                 for pdf in pdf_docs:
# #                     with open(os.path.join("uploaded_pdfs", pdf.name), "wb") as f:
# #                         f.write(pdf.read())  # Save uploaded PDFs

# #                 raw_text = get_pdf_text(pdf_docs)
# #                 text_chunks = get_text_chunks(raw_text)
# #                 get_vector_store(pdf_docs, text_chunks)  # Store PDFs in FAISS

# #                 st.success("Done")

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

# # load_dotenv()
# # os.getenv("GOOGLE_API_KEY")
# # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))






# # def get_pdf_text(pdf_docs):
# #     text=""
# #     for pdf in pdf_docs:
# #         pdf_reader= PdfReader(pdf)
# #         for page in pdf_reader.pages:
# #             text+= page.extract_text()
# #     return  text



# # def get_text_chunks(text):
# #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
# #     chunks = text_splitter.split_text(text)
# #     return chunks


# # def get_vector_store(text_chunks):
# #     embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
# #     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
# #     vector_store.save_local("faiss_index")


# # def get_conversational_chain():

# #     prompt_template = """
# #     Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
# #     provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
# #     Context:\n {context}?\n
# #     Question: \n{question}\n

# #     Answer:
# #     """

# #     model = ChatGoogleGenerativeAI(model="gemini-1.5-pro",
# #                              temperature=0.3)

# #     prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
# #     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

# #     return chain


# # def user_input(user_question):
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
# #     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
# #     docs = new_db.similarity_search(user_question)

# #     if not docs:
# #         st.write("No relevant answer found in the uploaded PDFs.")
# #         return

# #     chain = get_conversational_chain()
# #     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

# #     # Extract source document name from metadata
# #     source_pdf = docs[0].metadata.get("source", "Unknown PDF")  

# #     st.write("### **Reply:**")
# #     st.write(response["output_text"])

# #     # Display the source PDF in Streamlit
# #     st.write(f"#### Answer found in: {source_pdf}")
    
# #     # Get the full path of the PDF file
# #     pdf_path = os.path.join("uploaded_pdfs", source_pdf)  # Adjust based on your storage path
    
# #     # Display the PDF inside Streamlit
# #     with open(pdf_path, "rb") as pdf_file:
# #         st.download_button(label="Download PDF", data=pdf_file, file_name=source_pdf, mime="application/pdf")

# #     st.write("### **View PDF Below:**")
# #     st.components.v1.html(f"""
# #         <iframe src="data:application/pdf;base64,{pdf_to_base64(pdf_path)}" width="100%" height="600px"></iframe>
# #     """, height=650)


# # import base64

# # def pdf_to_base64(pdf_path):
# #     with open(pdf_path, "rb") as pdf_file:
# #         base64_pdf = base64.b64encode(pdf_file.read()).decode("utf-8")
# #     return base64_pdf



# # # def user_input(user_question):
# # #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
# # #     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
# # #     docs = new_db.similarity_search(user_question)

# # #     if not docs:
# # #         st.write("No relevant answer found in the uploaded PDFs.")
# # #         return

# # #     chain = get_conversational_chain()

# # #     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

# # #     # Extract source document name from metadata (Assuming you store PDF names in metadata)
# # #     source_pdf = docs[0].metadata.get("source", "Unknown PDF")  # Change this if using different metadata

# # #     st.write("### **Reply:**")
# # #     st.write(response["output_text"])

# # #     # Display the source PDF in Streamlit
# # #     st.write(f"#### Answer found in: {source_pdf}")
# # #     pdf_path = os.path.join("uploaded_pdfs", source_pdf)  # Adjust this path based on your file storage
# # #     st.write(f"📄 **Click below to view the PDF:**")
# # #     st.markdown(f"[Open PDF]({pdf_path})", unsafe_allow_html=True)




# # # def user_input(user_question):
# # #     embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
# # #     # new_db = FAISS.load_local("faiss_index", embeddings)
# # #     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# # #     docs = new_db.similarity_search(user_question)

# # #     chain = get_conversational_chain()

    
# # #     response = chain(
# # #         {"input_documents":docs, "question": user_question}
# # #         , return_only_outputs=True)

# # #     print(response)
# # #     st.write("Reply: ", response["output_text"])




# # def main():
# #     st.set_page_config("Chat PDF")
# #     st.header("Chat with PDF using Gemini💁")

# #     user_question = st.text_input("Ask a Question from the PDF Files")

# #     if user_question:
# #         user_input(user_question)

# #     with st.sidebar:
# #         st.title("Menu:")
# #         pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
# #         if st.button("Submit & Process"):
# #             with st.spinner("Processing..."):
# #                 raw_text = get_pdf_text(pdf_docs)
# #                 text_chunks = get_text_chunks(raw_text)
# #                 get_vector_store(text_chunks)
# #                 st.success("Done")



# # if __name__ == "__main__":
# #     main()


# #     # E:\Scrcpy With Sound\Projects\llm\app.py
# #     # cd "E:\Scrcpy With Sound\Projects\llm"
# #     # pip install -U langchain-community
# # 