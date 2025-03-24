# import streamlit as st
# import os
# import tempfile
# import base64
# import google.generativeai as genai
# from PyPDF2 import PdfReader
# import fitz
# import nltk
# from nltk.tokenize import sent_tokenize

# # ➡️➡️➡️ ADD YOUR API KEY HERE ⬅️⬅️⬅️
# GOOGLE_API_KEY = "AIzaSyB3oKDwwLynQXXn8vo-nIGDvJ_wLG-gBt8"

# # Configure Gemini
# genai.configure(api_key=GOOGLE_API_KEY)
# model = genai.GenerativeModel('gemini-1.5-pro')

# # Download NLTK data
# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     nltk.download('punkt')

# # Page configuration
# st.set_page_config(
#     page_title="PDF Query System with Gemini",
#     page_icon="📄",
#     layout="wide"
# )

# # Initialize session state
# if 'pdf_text' not in st.session_state:
#     st.session_state.pdf_text = ""
# if 'pdf_file' not in st.session_state:
#     st.session_state.pdf_file = None
# if 'pdf_name' not in st.session_state:
#     st.session_state.pdf_name = ""
# if 'highlighted_pdf' not in st.session_state:
#     st.session_state.highlighted_pdf = None
# if 'answer' not in st.session_state:
#     st.session_state.answer = ""

# # PDF Processing Functions
# def extract_text_from_pdf(pdf_file):
#     temp_file_path = ""
#     try:
#         with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
#             temp_file.write(pdf_file.getvalue())
#             temp_file_path = temp_file.name
        
#         reader = PdfReader(temp_file_path)
#         text = ""
#         for page in reader.pages:
#             page_text = page.extract_text()
#             if page_text:
#                 text += page_text.replace('\t', ' ') + "\n"
#         return text
#     except Exception as e:
#         st.error(f"Error extracting text: {e}")
#         return ""
#     finally:
#         if temp_file_path and os.path.exists(temp_file_path):
#             os.unlink(temp_file_path)

# def split_into_chunks(text, max_chunk_size=3000):
#     text = text.replace('\n', ' ').strip()
#     sentences = sent_tokenize(text)
#     chunks = []
#     current_chunk = []
#     current_size = 0
    
#     for sentence in sentences:
#         sentence = sentence.strip()
#         if not sentence:
#             continue
            
#         sentence_size = len(sentence)
#         if current_size + sentence_size > max_chunk_size:
#             chunks.append(" ".join(current_chunk))
#             current_chunk = []
#             current_size = 0
        
#         current_chunk.append(sentence)
#         current_size += sentence_size
    
#     if current_chunk:
#         chunks.append(" ".join(current_chunk))
    
#     return chunks

# # Answer Generation
# def generate_answer(query, context):
#     try:
#         prompt = f"""
#         Analyze this document context and answer the question in detail.
#         Include specific examples and explanations from the text.
#         Format your response with clear paragraphs and bullet points when appropriate.
        
#         Context:
#         {context}
        
#         Question: {query}
        
#         Detailed Answer:
#         """
        
#         response = model.generate_content(prompt)
#         return response.text
#     except Exception as e:
#         st.error(f"Gemini API error: {e}")
#         return None

# # PDF Highlighting
# def highlight_pdf(pdf_file, passages):
#     try:
#         with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
#             temp_file.write(pdf_file.getvalue())
#             temp_file_path = temp_file.name
            
#         doc = fitz.open(temp_file_path)
        
#         for passage in passages:
#             clean_passage = ' '.join(passage.split()[:10]).lower()
#             for page in doc:
#                 text_instances = page.search_for(clean_passage)
#                 for inst in text_instances:
#                     highlight = page.add_highlight_annot(inst)
#                     highlight.update()
        
#         output_path = f"{temp_file_path}_highlighted.pdf"
#         doc.save(output_path)
#         doc.close()
        
#         with open(output_path, "rb") as f:
#             highlighted_pdf = f.read()
            
#         os.unlink(temp_file_path)
#         os.unlink(output_path)
#         return highlighted_pdf
#     except Exception as e:
#         st.error(f"Highlighting error: {e}")
#         return None

# # PDF Display
# def display_pdf(content):
#     base64_pdf = base64.b64encode(content).decode('utf-8')
#     pdf_display = f"""
#     <iframe
#         src="data:application/pdf;base64,{base64_pdf}"
#         width="100%"
#         height="800"
#         style="border:none;">
#     </iframe>
#     """
#     st.markdown(pdf_display, unsafe_allow_html=True)

# # UI Layout
# st.title("📄 Enterprise PDF Analyzer with Gemini Pro")
# st.markdown("Upload documents for AI-powered analysis and insights")

# # Main tabs
# tab1, tab2 = st.tabs(["Upload & Analyze", "Results"])

# with tab1:
#     col1, col2 = st.columns([1, 1])
    
#     with col1:
#         st.header("1. Upload Document")
#         uploaded_file = st.file_uploader("Choose PDF", type="pdf")
#         if uploaded_file:
#             st.session_state.pdf_file = uploaded_file
#             st.session_state.pdf_name = uploaded_file.name
#             with st.spinner("Processing document..."):
#                 st.session_state.pdf_text = extract_text_from_pdf(uploaded_file)
#             if st.session_state.pdf_text:
#                 st.success("Document processed successfully!")
    
#     with col2:
#         st.header("2. Ask Question")
#         query = st.text_input("Enter your analysis question")
#         if st.button("Generate Report", disabled=not (uploaded_file and query)):
#             with st.spinner("Creating comprehensive analysis..."):
#                 chunks = split_into_chunks(st.session_state.pdf_text)
#                 context = "\n\n".join(chunks[:3])
                
#                 answer = generate_answer(query, context)
#                 if answer:
#                     st.session_state.answer = answer
                    
#                     passages = sent_tokenize(context)[:5]
#                     highlighted = highlight_pdf(st.session_state.pdf_file, passages)
#                     st.session_state.highlighted_pdf = highlighted
#                     st.rerun()

# with tab2:
#     if st.session_state.answer:
#         st.header("AI Analysis Report")
#         st.markdown(st.session_state.answer)
        
#         st.markdown("---")
#         st.header("Key Document Sections")
#         if st.session_state.highlighted_pdf:
#             display_pdf(st.session_state.highlighted_pdf)
#         else:
#             st.info("No specific sections highlighted")
#     else:
#         st.info("Submit a query to view analysis results")

# st.markdown("---")
# st.caption("Confidential AI Analysis System | v3.0 | Powered by Google Gemini")
























# # import streamlit as st
# # import os
# # import tempfile
# # import base64
# # import google.generativeai as genai
# # from PyPDF2 import PdfReader
# # import fitz
# # import nltk
# # from nltk.tokenize import sent_tokenize

# # # Download NLTK data
# # try:
# #     nltk.data.find('tokenizers/punkt')
# # except LookupError:
# #     nltk.download('punkt')

# # # Configure Gemini
# # genai.configure(api_key="AIzaSyB3oKDwwLynQXXn8vo-nIGDvJ_wLG-gBt8")
# # model = genai.GenerativeModel('gemini-1.5-pro')

# # # Page configuration
# # st.set_page_config(
# #     page_title="PDF Query System with Gemini",
# #     page_icon="📄",
# #     layout="wide"
# # )

# # # Initialize session state
# # if 'pdf_text' not in st.session_state:
# #     st.session_state.pdf_text = ""
# # if 'pdf_file' not in st.session_state:
# #     st.session_state.pdf_file = None
# # if 'pdf_name' not in st.session_state:
# #     st.session_state.pdf_name = ""
# # if 'highlighted_pdf' not in st.session_state:
# #     st.session_state.highlighted_pdf = None
# # if 'answer' not in st.session_state:
# #     st.session_state.answer = ""

# # # PDF Processing Functions
# # def extract_text_from_pdf(pdf_file):
# #     temp_file_path = ""
# #     try:
# #         with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
# #             temp_file.write(pdf_file.getvalue())
# #             temp_file_path = temp_file.name
        
# #         reader = PdfReader(temp_file_path)
# #         text = ""
# #         for page in reader.pages:
# #             page_text = page.extract_text()
# #             if page_text:
# #                 text += page_text.replace('\t', ' ') + "\n"
# #         return text
# #     except Exception as e:
# #         st.error(f"Error extracting text: {e}")
# #         return ""
# #     finally:
# #         if temp_file_path and os.path.exists(temp_file_path):
# #             os.unlink(temp_file_path)

# # def split_into_chunks(text, max_chunk_size=3000):
# #     text = text.replace('\n', ' ').strip()
# #     sentences = sent_tokenize(text)
# #     chunks = []
# #     current_chunk = []
# #     current_size = 0
    
# #     for sentence in sentences:
# #         sentence = sentence.strip()
# #         if not sentence:
# #             continue
            
# #         sentence_size = len(sentence)
# #         if current_size + sentence_size > max_chunk_size:
# #             chunks.append(" ".join(current_chunk))
# #             current_chunk = []
# #             current_size = 0
        
# #         current_chunk.append(sentence)
# #         current_size += sentence_size
    
# #     if current_chunk:
# #         chunks.append(" ".join(current_chunk))
    
# #     return chunks

# # # Answer Generation with Gemini
# # def generate_answer(query, context):
# #     try:
# #         prompt = f"""
# #         Analyze the following document context and answer the question in detail.
# #         Provide a comprehensive answer with examples from the context when possible.
        
# #         Context:
# #         {context}
        
# #         Question: {query}
        
# #         Answer:
# #         """
        
# #         response = model.generate_content(prompt)
# #         return response.text
# #     except Exception as e:
# #         st.error(f"Gemini API error: {e}")
# #         return None

# # # PDF Highlighting
# # def highlight_pdf(pdf_file, passages):
# #     try:
# #         with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
# #             temp_file.write(pdf_file.getvalue())
# #             temp_file_path = temp_file.name
            
# #         doc = fitz.open(temp_file_path)
        
# #         for passage in passages:
# #             clean_passage = ' '.join(passage.split()[:10]).lower()
# #             for page in doc:
# #                 text_instances = page.search_for(clean_passage)
# #                 for inst in text_instances:
# #                     highlight = page.add_highlight_annot(inst)
# #                     highlight.update()
        
# #         output_path = f"{temp_file_path}_highlighted.pdf"
# #         doc.save(output_path)
# #         doc.close()
        
# #         with open(output_path, "rb") as f:
# #             highlighted_pdf = f.read()
            
# #         os.unlink(temp_file_path)
# #         os.unlink(output_path)
# #         return highlighted_pdf
# #     except Exception as e:
# #         st.error(f"Highlighting error: {e}")
# #         return None

# # # PDF Display
# # def display_pdf(content):
# #     base64_pdf = base64.b64encode(content).decode('utf-8')
# #     pdf_display = f"""
# #     <iframe
# #         src="data:application/pdf;base64,{base64_pdf}"
# #         width="100%"
# #         height="800"
# #         style="border:none;">
# #     </iframe>
# #     """
# #     st.markdown(pdf_display, unsafe_allow_html=True)

# # # UI Layout
# # st.title("📄 PDF Query System with Gemini Pro")
# # st.markdown("Upload a PDF and get detailed answers powered by Google Gemini")

# # # Sidebar
# # with st.sidebar:
# #     st.header("Configuration")
# #     api_key = st.text_input("Enter Google API Key", type="password")
# #     if api_key:
# #         genai.configure(api_key=api_key)
    
# #     st.markdown("---")
# #     st.markdown("""
# #     **How to use:**
# #     1. Enter Google API key
# #     2. Upload PDF document
# #     3. Ask your question
# #     4. View detailed answer with highlights
# #     """)

# # # Main tabs
# # tab1, tab2 = st.tabs(["Upload & Query", "Results"])

# # with tab1:
# #     col1, col2 = st.columns([1, 1])
    
# #     with col1:
# #         st.header("1. Upload Document")
# #         uploaded_file = st.file_uploader("Choose PDF", type="pdf")
# #         if uploaded_file:
# #             st.session_state.pdf_file = uploaded_file
# #             st.session_state.pdf_name = uploaded_file.name
# #             with st.spinner("Processing document..."):
# #                 st.session_state.pdf_text = extract_text_from_pdf(uploaded_file)
# #             if st.session_state.pdf_text:
# #                 st.success("Document processed successfully!")
    
# #     with col2:
# #         st.header("2. Ask Question")
# #         query = st.text_input("Enter your question")
# #         if st.button("Analyze", disabled=not (uploaded_file and query and api_key)):
# #             with st.spinner("Generating comprehensive answer..."):
# #                 chunks = split_into_chunks(st.session_state.pdf_text)
                
# #                 # Use first 3 chunks for context (Gemini's 30k token limit)
# #                 context = "\n\n".join(chunks[:3])
                
# #                 answer = generate_answer(query, context)
# #                 if answer:
# #                     st.session_state.answer = answer
                    
# #                     # Find relevant passages for highlighting
# #                     passages = sent_tokenize(context)[:5]
# #                     highlighted = highlight_pdf(st.session_state.pdf_file, passages)
# #                     st.session_state.highlighted_pdf = highlighted
# #                     st.rerun()

# # with tab2:
# #     if st.session_state.answer:
# #         st.header("Comprehensive Answer")
# #         st.markdown(st.session_state.answer)
        
# #         st.markdown("---")
# #         st.header("Relevant Document Sections")
# #         if st.session_state.highlighted_pdf:
# #             display_pdf(st.session_state.highlighted_pdf)
# #         else:
# #             st.info("No specific sections highlighted")
# #     else:
# #         st.info("Submit a query to view results")

# # st.markdown("---")
# # st.caption("Powered by Google Gemini Pro | Document Intelligence System")