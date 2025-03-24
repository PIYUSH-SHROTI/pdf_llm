# import streamlit as st
# import os
# import tempfile
# import base64
# from PyPDF2 import PdfReader
# import fitz  # PyMuPDF
# from transformers import pipeline
# import torch
# from sentence_transformers import SentenceTransformer, util
# import nltk
# from nltk.tokenize import sent_tokenize
# import json

# # Download NLTK data
# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     nltk.download('punkt')

# # Page configuration
# st.set_page_config(
#     page_title="PDF Query System",
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
# if 'qa_model' not in st.session_state:
#     st.session_state.qa_model = None
# if 'sentence_model' not in st.session_state:
#     st.session_state.sentence_model = None
# if 'initialized' not in st.session_state:
#     st.session_state.initialized = False

# # Function to extract text from PDF
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

# # Text chunking function
# def split_into_chunks(text, max_chunk_size=1500, overlap=200):
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
#             current_chunk = current_chunk[-overlap:] if len(current_chunk) > overlap else []
#             current_size = sum(len(s) for s in current_chunk)
        
#         current_chunk.append(sentence)
#         current_size += sentence_size
    
#     if current_chunk:
#         chunks.append(" ".join(current_chunk))
    
#     return chunks

# # Model loading
# @st.cache_resource
# def load_models():
#     try:
#         qa_model = pipeline(
#             "question-answering",
#             model="bert-large-uncased-whole-word-masking-finetuned-squad",
#             torch_dtype=torch.float16
#         )
#         sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
#         return qa_model, sentence_model
#     except Exception as e:
#         st.error(f"Model loading error: {e}")
#         return None, None

# # Semantic search
# def find_relevant_chunks(query, chunks, model, top_k=3):
#     query_embedding = model.encode(query, convert_to_tensor=True)
#     chunk_embeddings = model.encode(chunks, convert_to_tensor=True)
#     cos_scores = util.cos_sim(query_embedding, chunk_embeddings)[0]
#     top_results = torch.topk(cos_scores, k=min(top_k, len(chunks)))
#     return [chunks[idx] for idx in top_results.indices]

# # Answer processing
# def process_query_local(pdf_text, query):
#     try:
#         if not st.session_state.initialized:
#             with st.spinner("Loading AI models (this may take a minute)..."):
#                 qa_model, sentence_model = load_models()
#                 st.session_state.qa_model = qa_model
#                 st.session_state.sentence_model = sentence_model
#                 st.session_state.initialized = True

#         chunks = split_into_chunks(pdf_text)
#         relevant_chunks = find_relevant_chunks(query, chunks, st.session_state.sentence_model)
        
#         answers = []
#         source_passages = []
        
#         for chunk in relevant_chunks:
#             results = st.session_state.qa_model(
#                 question=query,
#                 context=chunk,
#                 max_answer_len=400,
#                 top_k=2
#             )
            
#             for result in results:
#                 if result['score'] > 0.18:
#                     context = extract_surrounding_context(chunk, result['answer'])
#                     answers.append({
#                         'text': f"{result['answer']} {context}",
#                         'score': result['score'],
#                         'context': chunk
#                     })
#                     source_passages.append(context)

#         return json.dumps({
#             "answer": format_answer(answers),
#             "sourcePassages": list(set(source_passages))[:3]
#         })
#     except Exception as e:
#         st.error(f"Processing error: {e}")
#         return None

# # Helper functions
# def extract_surrounding_context(chunk, answer, window=2):
#     sentences = sent_tokenize(chunk)
#     for i, sent in enumerate(sentences):
#         if answer in sent:
#             start = max(0, i - window)
#             end = min(len(sentences), i + window + 1)
#             return ' '.join(sentences[start:end])
#     return ''

# def format_answer(answers):
#     if not answers:
#         return "I couldn't find a comprehensive answer in the document."
    
#     seen = set()
#     formatted = []
#     for ans in sorted(answers, key=lambda x: x['score'], reverse=True):
#         text = ans['text'].strip()
#         key = text[:100].lower()
#         if key not in seen:
#             seen.add(key)
#             formatted.append(f"- {text} (Confidence: {ans['score']:.0%})")
    
#     return "Here's the detailed analysis:\n" + "\n".join(formatted[:3])

# # PDF highlighting
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

# # PDF display
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
# st.title("📄 Advanced PDF Query System")
# st.markdown("Upload a PDF and get detailed answers with highlighted sources")

# # Sidebar
# with st.sidebar:
#     st.header("About")
#     st.markdown("""
#     This system uses advanced NLP models to:
#     - Provide comprehensive answers
#     - Highlight relevant sections
#    - Maintain complete data privacy
#     """)
#     st.markdown("**Models:** BERT-Large, Sentence-BERT")
#     st.markdown("**Developer:** [Your Name]")

# # Main tabs
# tab1, tab2 = st.tabs(["Upload & Query", "Results"])

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
#         query = st.text_input("Enter your question")
#         if st.button("Analyze", disabled=not (uploaded_file and query)):
#             with st.spinner("Analyzing document..."):
#                 result = process_query_local(st.session_state.pdf_text, query)
#                 if result:
#                     result_data = json.loads(result)
#                     st.session_state.answer = result_data['answer']
#                     passages = result_data['sourcePassages']
#                     if passages:
#                         highlighted = highlight_pdf(st.session_state.pdf_file, passages)
#                         st.session_state.highlighted_pdf = highlighted
#                     else:
#                         st.session_state.highlighted_pdf = st.session_state.pdf_file.getvalue()
#                     st.rerun()

# with tab2:
#     if st.session_state.answer:
#         st.header("Analysis Results")
#         st.markdown(st.session_state.answer)
        
#         st.markdown("---")
#         st.header("Document Highlights")
#         if st.session_state.highlighted_pdf:
#             display_pdf(st.session_state.highlighted_pdf)
#         else:
#             st.info("No specific sections highlighted")
#     else:
#         st.info("Submit a query to view results")

# st.markdown("---")
# st.caption("Intelligent Document Analysis System | Version 2.1")




























# # # app.py
# # import streamlit as st
# # import os
# # import tempfile
# # import base64
# # from PyPDF2 import PdfReader
# # import fitz  # PyMuPDF
# # from transformers import pipeline
# # import torch
# # from sentence_transformers import SentenceTransformer, util
# # import nltk
# # from nltk.tokenize import sent_tokenize
# # import json

# # # Download NLTK data (run once)
# # try:
# #     nltk.data.find('tokenizers/punkt')
# # except LookupError:
# #     nltk.download('punkt')
# # try:
# #     nltk.data.find('tokenizers/punkt_tab')
# # except LookupError:
# #     try:
# #         nltk.download('punkt_tab')
# #     except:
# #         pass  # Handle if punkt_tab isn't available



# # # # Download NLTK data (run once)
# # # try:
# # #     nltk.data.find('tokenizers/punkt')
# # # except LookupError:
# # #     nltk.download('punkt')

# # # Page configuration
# # st.set_page_config(
# #     page_title="PDF Query System",
# #     page_icon="📄",
# #     layout="wide"
# # )

# # # Initialize session state variables
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
# # if 'qa_model' not in st.session_state:
# #     st.session_state.qa_model = None
# # if 'sentence_model' not in st.session_state:
# #     st.session_state.sentence_model = None
# # if 'initialized' not in st.session_state:
# #     st.session_state.initialized = False




# # # Update the extract_text_from_pdf function to replace tabs
# # def extract_text_from_pdf(pdf_file):
# #     temp_file_path = ""
# #     try:
# #         # Save the uploaded file to a temporary file
# #         with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
# #             temp_file.write(pdf_file.getvalue())
# #             temp_file_path = temp_file.name
        
# #         # Extract text using PyPDF2
# #         reader = PdfReader(temp_file_path)
# #         text = ""
# #         for page in reader.pages:
# #             page_text = page.extract_text()
# #             if page_text:  # Ensure text is not None
# #                 text += page_text.replace('\t', ' ') + "\n"  # Replace tabs with spaces
        
# #         return text
# #     except Exception as e:
# #         st.error(f"Error extracting text from PDF: {e}")
# #         return ""
# #     finally:
# #         # Clean up the temporary file
# #         if temp_file_path and os.path.exists(temp_file_path):
# #             os.unlink(temp_file_path)

# # # Function to extract text from PDF
# # # def extract_text_from_pdf(pdf_file):
# # #     temp_file_path = ""
# # #     try:
# # #         # Save the uploaded file to a temporary file
# # #         with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
# # #             temp_file.write(pdf_file.getvalue())
# # #             temp_file_path = temp_file.name
        
# # #         # Extract text using PyPDF2
# # #         reader = PdfReader(temp_file_path)
# # #         text = ""
# # #         for page in reader.pages:
# # #             text += page.extract_text() + "\n"
        
# # #         return text
# # #     except Exception as e:
# # #         st.error(f"Error extracting text from PDF: {e}")
# # #         return ""
# # #     finally:
# # #         # Clean up the temporary file
# # #         if temp_file_path and os.path.exists(temp_file_path):
# # #             os.unlink(temp_file_path)


# # # Update the split_into_chunks function to preserve more context
# # def split_into_chunks(text, max_chunk_size=1000, overlap=150):  # Increased chunk size and overlap
# #     text = text.replace('\n', ' ').replace('\t', ' ').strip()
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
# #             current_chunk = current_chunk[-overlap:] if len(current_chunk) > overlap else []
# #             current_size = sum(len(s) for s in current_chunk)
        
# #         current_chunk.append(sentence)
# #         current_size += sentence_size
    
# #     if current_chunk:
# #         chunks.append(" ".join(current_chunk))
    
# #     return chunks




# # # Update the split_into_chunks function
# # # def split_into_chunks(text, max_chunk_size=800, overlap=100):  # Increased chunk size and overlap
# # #     text = text.replace('\n', ' ').replace('\t', ' ')  # Clean whitespace
# # #     sentences = sent_tokenize(text)
# # #     chunks = []
# # #     current_chunk = []
# # #     current_size = 0
    
# # #     for sentence in sentences:
# # #         sentence = sentence.strip()
# # #         if not sentence:
# # #             continue
            
# # #         sentence_size = len(sentence)
# # #         if current_size + sentence_size > max_chunk_size:
# # #             chunks.append(" ".join(current_chunk))
# # #             current_chunk = current_chunk[-overlap:] if len(current_chunk) > overlap else []
# # #             current_size = sum(len(s) for s in current_chunk)
        
# # #         current_chunk.append(sentence)
# # #         current_size += sentence_size
    
# # #     if current_chunk:
# # #         chunks.append(" ".join(current_chunk))
    
# # #     return chunks


# # # Function to split text into chunks
# # # def split_into_chunks(text, max_chunk_size=500, overlap=50):
# # #     sentences = sent_tokenize(text)
# # #     chunks = []
# # #     current_chunk = []
# # #     current_size = 0
    
# # #     for sentence in sentences:
# # #         # Skip empty sentences
# # #         if not sentence.strip():
# # #             continue
            
# # #         sentence_size = len(sentence)
        
# # #         # If adding this sentence would exceed the chunk size, finalize the chunk
# # #         if current_size + sentence_size > max_chunk_size and current_chunk:
# # #             chunks.append(" ".join(current_chunk))
            
# # #             # Keep some sentences for overlap
# # #             overlap_sentences = current_chunk[-overlap:] if overlap < len(current_chunk) else current_chunk
# # #             current_chunk = overlap_sentences
# # #             current_size = sum(len(s) for s in current_chunk)
        
# # #         # Add the sentence to the current chunk
# # #         current_chunk.append(sentence)
# # #         current_size += sentence_size
    
# # #     # Add the last chunk if it's not empty
# # #     if current_chunk:
# # #         chunks.append(" ".join(current_chunk))
    
# # #     return chunks

# # # Function to initialize models
# # @st.cache_resource
# # def load_models():
# #     try:
# #         # Load QA model (smaller model for local operation)
# #         qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
        
# #         # Load sentence embedding model
# #         sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
# #         return qa_model, sentence_model
# #     except Exception as e:
# #         st.error(f"Error loading models: {e}")
# #         return None, None

# # # Function to find most relevant chunks
# # def find_relevant_chunks(query, chunks, sentence_model, top_k=3):
# #     # Create embeddings
# #     query_embedding = sentence_model.encode(query, convert_to_tensor=True)
# #     chunk_embeddings = sentence_model.encode(chunks, convert_to_tensor=True)
    
# #     # Calculate similarity scores
# #     cos_scores = util.cos_sim(query_embedding, chunk_embeddings)[0]
    
# #     # Get top_k chunks
# #     top_results = torch.topk(cos_scores, k=min(top_k, len(chunks)))
    
# #     return [chunks[idx] for idx in top_results.indices]



# # # Enhanced answer generation logic
# # def process_query_local(pdf_text, query):
# #     try:
# #         if not st.session_state.initialized:
# #             with st.spinner("Loading models..."):
# #                 qa_model, sentence_model = load_models()
# #                 st.session_state.qa_model = qa_model
# #                 st.session_state.sentence_model = sentence_model
# #                 st.session_state.initialized = True
        
# #         chunks = split_into_chunks(pdf_text)
# #         relevant_chunks = find_relevant_chunks(query, chunks, st.session_state.sentence_model)
        
# #         answers = []
# #         source_passages = []
        
# #         for chunk in relevant_chunks:
# #             result = st.session_state.qa_model(
# #                 question=query,
# #                 context=chunk,
# #                 max_answer_len=300,  # Increased answer length
# #                 top_k=2  # Get multiple candidate answers
# #             )
            
# #             # Process multiple answers
# #             for ans in result:
# #                 if ans['score'] > 0.2:  # Lower confidence threshold
# #                     answers.append({
# #                         'text': ans['answer'],
# #                         'score': ans['score'],
# #                         'context': chunk
# #                     })
        
# #         # Sort answers by confidence score
# #         answers = sorted(answers, key=lambda x: x['score'], reverse=True)[:3]
        
# #         # Construct comprehensive answer
# #         final_answer = []
# #         seen_answers = set()
# #         for ans in answers:
# #             clean_answer = ans['text'].strip()
# #             if clean_answer.lower() not in seen_answers:
# #                 seen_answers.add(clean_answer.lower())
# #                 final_answer.append(f"- {clean_answer}")
                
# #                 # Find supporting context
# #                 context_sentences = sent_tokenize(ans['context'])
# #                 for sent in context_sentences:
# #                     if clean_answer in sent:
# #                         source_passages.append(sent)
# #                         break
        
# #         # Format final response
# #         if final_answer:
# #             response_text = "Here's what I found:\n" + "\n".join(final_answer)
# #         else:
# #             response_text = "I couldn't find a definitive answer in the document."
        
# #         return json.dumps({
# #             "answer": response_text,
# #             "sourcePassages": source_passages[:5]  # Return up to 5 passages
# #         })
# #     except Exception as e:
# #         st.error(f"Error processing query: {e}")
# #         return None








# # # # Update the query processing logic
# # # def process_query_local(pdf_text, query):
# # #     try:
# # #         if not st.session_state.initialized:
# # #             with st.spinner("Loading models..."):
# # #                 qa_model, sentence_model = load_models()
# # #                 st.session_state.qa_model = qa_model
# # #                 st.session_state.sentence_model = sentence_model
# # #                 st.session_state.initialized = True
        
# # #         chunks = split_into_chunks(pdf_text)
# # #         relevant_chunks = find_relevant_chunks(query, chunks, st.session_state.sentence_model)
        
# # #         answer = ""
# # #         source_passages = []
        
# # #         for chunk in relevant_chunks:
# # #             result = st.session_state.qa_model(
# # #                 question=query,
# # #                 context=chunk,
# # #                 max_answer_len=200  # Limit answer length
# # #             )
            
# # #             if result['score'] > 0.3:  # Higher confidence threshold
# # #                 answer += f"{result['answer']}\n"
                
# # #                 # Use the entire chunk if answer is found within it
# # #                 source_passages.append(chunk[:500])  # Take first 500 characters
# # #                 break  # Only need one good passage
        
# # #         response = {
# # #             "answer": answer.strip() or "I couldn't find a definitive answer in the document.",
# # #             "sourcePassages": source_passages[:3]  # Limit to 3 passages
# # #         }
        
# # #         return json.dumps(response)
# # #     except Exception as e:
# # #         st.error(f"Error processing query: {e}")
# # #         return None



# # # Function to process query using local models
# # # def process_query_local(pdf_text, query):
# # #     try:
# # #         # Ensure models are loaded
# # #         if not st.session_state.initialized:
# # #             with st.spinner("Loading models... This may take a minute on first run"):
# # #                 qa_model, sentence_model = load_models()
# # #                 st.session_state.qa_model = qa_model
# # #                 st.session_state.sentence_model = sentence_model
# # #                 st.session_state.initialized = True
        
# # #         # Split PDF text into manageable chunks
# # #         chunks = split_into_chunks(pdf_text)
        
# # #         # Find most relevant chunks
# # #         relevant_chunks = find_relevant_chunks(
# # #             query, 
# # #             chunks, 
# # #             st.session_state.sentence_model
# # #         )
        
# # #         # Get answer from QA model
# # #         answer = ""
# # #         source_passages = []
        
# # #         for chunk in relevant_chunks:
# # #             result = st.session_state.qa_model(
# # #                 question=query,
# # #                 context=chunk
# # #             )
            
# # #             # Extract answer and score
# # #             if result['score'] > 0.1:  # Only include if confidence is reasonable
# # #                 answer += result['answer'] + " "
                
# # #                 # Find the sentence containing the answer for highlighting
# # #                 for sentence in sent_tokenize(chunk):
# # #                     if result['answer'] in sentence:
# # #                         source_passages.append(sentence)
# # #                         break
        
# # #         # Format response
# # #         response = {
# # #             "answer": answer.strip() if answer else "I couldn't find a good answer in the document.",
# # #             "sourcePassages": source_passages
# # #         }
        
# # #         return json.dumps(response)
# # #     except Exception as e:
# # #         st.error(f"Error processing query: {e}")
# # #         return None

# # # Update the highlight_pdf function
# # def highlight_pdf(pdf_file, passages):
# #     try:
# #         with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
# #             temp_file.write(pdf_file.getvalue())
# #             temp_file_path = temp_file.name
            
# #         doc = fitz.open(temp_file_path)
        
# #         for passage in passages:
# #             # Clean and prepare search text
# #             search_text = " ".join(passage.split()[:10])  # Use first 10 words
# #             search_text = search_text.lower().strip()
            
# #             for page_num in range(len(doc)):
# #                 page = doc[page_num]
# #                 text_instances = []
                
# #                 # Search using different formats
# #                 for fmt in [True, False]:  # Try both formatted and unformatted
# #                     areas = page.search_for(search_text, quads=fmt)
# #                     text_instances.extend(areas)
                
# #                 # Highlight found instances
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
# #         st.error(f"Error highlighting PDF: {e}")
# #         return None



# # # Function to highlight text in PDF
# # # def highlight_pdf(pdf_file, passages):
# # #     try:
# # #         # Save the uploaded file to a temporary file
# # #         with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
# # #             temp_file.write(pdf_file.getvalue())
# # #             temp_file_path = temp_file.name
            
# # #         # Open the PDF with PyMuPDF
# # #         doc = fitz.open(temp_file_path)
        
# # #         # Search and highlight each passage
# # #         for passage in passages:
# # #             passage = passage.strip()
# # #             if len(passage) > 10:  # Only search for passages with a reasonable length
# # #                 for page_num in range(len(doc)):
# # #                     page = doc[page_num]
# # #                     # Search for text - use a shorter segment to increase chances of finding a match
# # #                     search_text = passage[:min(len(passage), 50)]
# # #                     text_instances = page.search_for(search_text)
# # #                     # Highlight each instance
# # #                     for inst in text_instances:
# # #                         highlight = page.add_highlight_annot(inst)
# # #                         highlight.update()
        
# # #         # Save the highlighted PDF to a temporary file
# # #         output_path = f"{temp_file_path}_highlighted.pdf"
# # #         doc.save(output_path)
# # #         doc.close()
        
# # #         # Return the highlighted PDF as bytes
# # #         with open(output_path, "rb") as f:
# # #             highlighted_pdf = f.read()
            
# # #         # Clean up temporary files
# # #         os.unlink(temp_file_path)
# # #         os.unlink(output_path)
        
# # #         return highlighted_pdf
# # #     except Exception as e:
# # #         st.error(f"Error highlighting PDF: {e}")
# # #         return None

# # # Function to display PDF in Streamlit
# # def display_pdf(file_content):
# #     # Encode the PDF file
# #     base64_pdf = base64.b64encode(file_content).decode('utf-8')
    
# #     # Embed PDF viewer
# #     pdf_display = f"""
# #         <iframe
# #             src="data:application/pdf;base64,{base64_pdf}"
# #             width="100%"
# #             height="800"
# #             style="border: none;"
# #         ></iframe>
# #     """
# #     st.markdown(pdf_display, unsafe_allow_html=True)

# # # UI Layout
# # st.title("📄 PDF Query System")
# # st.markdown("Upload a PDF and ask questions about its content.")

# # # Sidebar
# # with st.sidebar:
# #     st.header("About")
# #     st.markdown("""
# #     This app uses local LLMs to answer questions about your PDF documents without requiring an API key.
    
# #     **Models used:**
# #     - Question Answering: DistilBERT (SQuAD)
# #     - Semantic Search: MiniLM-L6
# #     """)
    
# #     st.markdown("---")
# #     st.markdown("### How to use")
# #     st.markdown("""
# #     1. Upload a PDF document
# #     2. Type your question
# #     3. View the answer and highlighted PDF
# #     """)
    
# #     with st.expander("⚙️ Advanced Settings"):
# #         model_load = st.button("Reload Models")
# #         if model_load:
# #             st.cache_resource.clear()
# #             st.session_state.initialized = False
# #             st.success("Models will be reloaded on next query")

# # # Main content area with tabs
# # tab1, tab2 = st.tabs(["Upload & Query", "Results"])

# # with tab1:
# #     col1, col2 = st.columns([1, 1])
    
# #     with col1:
# #         st.header("1. Upload PDF")
# #         uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
# #         if uploaded_file is not None:
# #             st.session_state.pdf_file = uploaded_file
# #             st.session_state.pdf_name = uploaded_file.name
            
# #             with st.spinner("Extracting text from PDF..."):
# #                 st.session_state.pdf_text = extract_text_from_pdf(uploaded_file)
            
# #             if st.session_state.pdf_text:
# #                 st.success(f"PDF '{uploaded_file.name}' loaded successfully!")
# #                 with st.expander("Preview Extracted Text"):
# #                     st.text(st.session_state.pdf_text[:1000] + "...")
# #             else:
# #                 st.error("Failed to extract text from the PDF.")
    
# #     with col2:
# #         st.header("2. Ask a Question")
# #         query = st.text_input("Enter your question about the document")
        
# #         query_button = st.button(
# #             "Submit Query", 
# #             disabled=not (st.session_state.pdf_text and query)
# #         )
        
# #         # In the query processing section:
# #         if query_button and st.session_state.pdf_text and query:
# #             with st.spinner("Processing your query..."):
# #                 result_json = process_query_local(st.session_state.pdf_text, query)
                
# #                 if result_json:
# #                     result = json.loads(result_json)
# #                     st.session_state.answer = result.get("answer", "")
# #                     passages = result.get("sourcePassages", [])
                    
# #                     if passages:
# #                         highlighted_pdf = highlight_pdf(st.session_state.pdf_file, passages)
# #                         st.session_state.highlighted_pdf = highlighted_pdf
# #                         st.success("Found relevant sections!")
# #                     else:
# #                         st.warning("Answer found, but no specific sections to highlight")
# #                         st.session_state.highlighted_pdf = st.session_state.pdf_file.getvalue()


# #         # if query_button and st.session_state.pdf_text and query:
# #         #     with st.spinner("Processing your query... (this may take a minute)"):
# #         #         # Process query with local LLM
# #         #         result_json = process_query_local(st.session_state.pdf_text, query)
                
# #         #         if result_json:
# #         #             import json
# #         #             result = json.loads(result_json)
                    
# #         #             # Save answer
# #         #             st.session_state.answer = result.get("answer", "No answer found.")
                    
# #         #             # Get passages for highlighting
# #         #             passages = result.get("sourcePassages", [])
                    
# #         #             # Highlight PDF
# #         #             if passages:
# #         #                 with st.spinner("Highlighting relevant sections in PDF..."):
# #         #                     highlighted_pdf = highlight_pdf(st.session_state.pdf_file, passages)
# #         #                     if highlighted_pdf:
# #         #                         st.session_state.highlighted_pdf = highlighted_pdf
# #         #                         st.success("Query processed successfully!")
# #         #                         # Switch to Results tab
# #         #                         st.rerun()
# #         #             else:
# #         #                 st.warning("No relevant passages found to highlight.")

# # with tab2:


# #     # In the results tab:
# #     # if st.session_state.answer:
# #     #     st.header("Answer")
# #     #     if st.session_state.answer.startswith("Here's what I found:"):
# #     #         st.markdown(st.session_state.answer)
# #     #     else:
# #     #         st.info(st.session_state.answer)





# #     if st.session_state.answer:
# #         st.header("Answer")
# #         st.markdown(st.session_state.answer)
        
# #         st.markdown("---")
        
# #         st.header("Highlighted PDF")
# #         if st.session_state.highlighted_pdf:
# #             display_pdf(st.session_state.highlighted_pdf)
# #         else:
# #             st.info("No highlighted PDF available.")
# #     else:
# #         st.info("Submit a query to see results here.")

# # # Footer
# # st.markdown("---")
# # st.caption("PDF Query System powered by Hugging Face Transformers and Streamlit")