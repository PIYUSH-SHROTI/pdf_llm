# import streamlit as st
# import os
# import tempfile
# import base64
# from PyPDF2 import PdfReader
# from openai import OpenAI
# import fitz  # PyMuPDF

# # Page configuration
# st.set_page_config(
#     page_title="PDF Query System",
#     page_icon="📄",
#     layout="wide"
# )

# # Initialize session state variables
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
# if 'api_key_configured' not in st.session_state:
#     st.session_state.api_key_configured = False

# # Function to extract text from PDF
# def extract_text_from_pdf(pdf_file):
#     temp_file_path = ""
#     try:
#         # Save the uploaded file to a temporary file
#         with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
#             temp_file.write(pdf_file.getvalue())
#             temp_file_path = temp_file.name
        
#         # Extract text using PyPDF2
#         reader = PdfReader(temp_file_path)
#         text = ""
#         for page in reader.pages:
#             text += page.extract_text() + "\n"
        
#         return text
#     except Exception as e:
#         st.error(f"Error extracting text from PDF: {e}")
#         return ""
#     finally:
#         # Clean up the temporary file
#         if temp_file_path and os.path.exists(temp_file_path):
#             os.unlink(temp_file_path)

# # Function to process query using LLM
# def process_query(pdf_text, query):
#     try:
#         client = OpenAI(api_key="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")

        
#         response = client.chat.completions.create(
#             model="gpt-3.5-turbo",

#             messages=[
#                 {
#                     "role": "system",
#                     "content": """You are an AI assistant that answers questions about PDF documents.
#                     Extract the most relevant information from the document to answer the user's query.
#                     Also identify the exact text passages that support your answer.
#                     Format your response as JSON with two fields:
#                     "answer": your complete answer
#                     "sourcePassages": an array of text passages from the document that support your answer"""
#                 },
#                 {
#                     "role": "user",
#                     "content": f"PDF Content:\n\n{pdf_text}\n\nQuestion: {query}"
#                 }
#             ],
#             response_format={"type": "json_object"}
#         )
        
#         result = response.choices[0].message.content
#         return result
#     except Exception as e:
#         st.error(f"Error processing query: {e}")
#         return None

# # Function to highlight text in PDF
# def highlight_pdf(pdf_file, passages):
#     try:
#         # Save the uploaded file to a temporary file
#         with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
#             temp_file.write(pdf_file.getvalue())
#             temp_file_path = temp_file.name
            
#         # Open the PDF with PyMuPDF
#         doc = fitz.open(temp_file_path)
        
#         # Search and highlight each passage
#         for passage in passages:
#             passage = passage.strip()
#             if len(passage) > 10:  # Only search for passages with a reasonable length
#                 for page_num in range(len(doc)):
#                     page = doc[page_num]
#                     # Search for text
#                     text_instances = page.search_for(passage[:min(len(passage), 100)])
#                     # Highlight each instance
#                     for inst in text_instances:
#                         highlight = page.add_highlight_annot(inst)
#                         highlight.update()
        
#         # Save the highlighted PDF to a temporary file
#         output_path = f"{temp_file_path}_highlighted.pdf"
#         doc.save(output_path)
#         doc.close()
        
#         # Return the highlighted PDF as bytes
#         with open(output_path, "rb") as f:
#             highlighted_pdf = f.read()
            
#         # Clean up temporary files
#         os.unlink(temp_file_path)
#         os.unlink(output_path)
        
#         return highlighted_pdf
#     except Exception as e:
#         st.error(f"Error highlighting PDF: {e}")
#         return None

# # Function to display PDF in Streamlit
# def display_pdf(file_content):
#     # Encode the PDF file
#     base64_pdf = base64.b64encode(file_content).decode('utf-8')
    
#     # Embed PDF viewer
#     pdf_display = f"""
#         <iframe
#             src="data:application/pdf;base64,{base64_pdf}"
#             width="100%"
#             height="800"
#             style="border: none;"
#         ></iframe>
#     """
#     st.markdown(pdf_display, unsafe_allow_html=True)

# # UI Layout
# st.title("📄 PDF Query System")
# st.markdown("Upload a PDF and ask questions about its content.")

# # Sidebar for API key configuration
# with st.sidebar:
#     st.header("Configuration")
#     api_key = st.text_input("Enter OpenAI API Key", type="password")
#     if api_key:
#         st.session_state.openai_api_key = api_key
#         st.session_state.api_key_configured = True
    
#     st.markdown("---")
#     st.markdown("### How to use")
#     st.markdown("""
#     1. Enter your OpenAI API key
#     2. Upload a PDF document
#     3. Type your question
#     4. View the answer and highlighted PDF
#     """)

# # Main content area with tabs
# tab1, tab2 = st.tabs(["Upload & Query", "Results"])

# with tab1:
#     col1, col2 = st.columns([1, 1])
    
#     with col1:
#         st.header("1. Upload PDF")
#         uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
#         if uploaded_file is not None:
#             st.session_state.pdf_file = uploaded_file
#             st.session_state.pdf_name = uploaded_file.name
            
#             with st.spinner("Extracting text from PDF..."):
#                 st.session_state.pdf_text = extract_text_from_pdf(uploaded_file)
            
#             if st.session_state.pdf_text:
#                 st.success(f"PDF '{uploaded_file.name}' loaded successfully!")
#             else:
#                 st.error("Failed to extract text from the PDF.")
    
#     with col2:
#         st.header("2. Ask a Question")
#         query = st.text_input("Enter your question about the document")
        
#         query_button = st.button(
#             "Submit Query", 
#             disabled=not (st.session_state.pdf_text and query and st.session_state.api_key_configured)
#         )
        
#         if not st.session_state.api_key_configured:
#             st.warning("Please configure your OpenAI API key in the sidebar first.")
        
#         if query_button and st.session_state.pdf_text and query:
#             with st.spinner("Processing your query..."):
#                 # Process query with LLM
#                 result_json = process_query(st.session_state.pdf_text, query)
                
#                 if result_json:
#                     import json
#                     result = json.loads(result_json)
                    
#                     # Save answer
#                     st.session_state.answer = result.get("answer", "No answer found.")
                    
#                     # Get passages for highlighting
#                     passages = result.get("sourcePassages", [])
                    
#                     # Highlight PDF
#                     if passages:
#                         with st.spinner("Highlighting relevant sections in PDF..."):
#                             highlighted_pdf = highlight_pdf(st.session_state.pdf_file, passages)
#                             if highlighted_pdf:
#                                 st.session_state.highlighted_pdf = highlighted_pdf
#                                 st.success("Query processed successfully!")
#                                 # Switch to Results tab
#                                 st.rerun()
#                     else:
#                         st.warning("No relevant passages found to highlight.")

# with tab2:
#     if st.session_state.answer:
#         st.header("Answer")
#         st.markdown(st.session_state.answer)
        
#         st.markdown("---")
        
#         st.header("Highlighted PDF")
#         if st.session_state.highlighted_pdf:
#             display_pdf(st.session_state.highlighted_pdf)
#         else:
#             st.info("No highlighted PDF available.")
#     else:
#         st.info("Submit a query to see results here.")

# # Footer
# st.markdown("---")
# st.caption("PDF Query System powered by OpenAI and Streamlit")