import streamlit as st
import boto3
import fitz
import json
from datetime import datetime
from typing import List, Dict
import re

from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import BedrockEmbeddings
from langchain_classic.vectorstores import FAISS
from langchain_classic.schema import Document

import config

@st.cache_resource
def get_bedrock_clients():
    bedrock = boto3.client('bedrock-runtime', region_name=config.BEDROCK_REGION)
    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1",
        client=bedrock
    )
    return bedrock, embeddings

bedrock_client, bedrock_embeddings = get_bedrock_clients()


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\x00', '')
    text = re.sub(r'\n\d+\n', '\n', text)
    return text.strip()

def extract_text_from_pdf(pdf_file) -> str:
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = "\n\n".join([page.get_text() for page in doc])
        doc.close()
        
        cleaned_text = clean_text(text)
        return cleaned_text
    
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def chunk_text(text: str, filename: str) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_text(text)
    
    documents = [
        Document(
            page_content=chunk,
            metadata={'filename': filename}
        )
        for i, chunk in enumerate(chunks)
    ]
    
    return documents


def create_or_update_faiss(documents: List[Document]):
    try:
        if 'vectorstore' in st.session_state and st.session_state.vectorstore is not None:
            st.session_state.vectorstore.add_documents(documents)
            st.success(f"Added {len(documents)} chunks to vector store")
        else:
            vectorstore = FAISS.from_documents(documents, bedrock_embeddings)
            st.session_state.vectorstore = vectorstore
            st.success(f"Created vector store with {len(documents)} chunks")
        
        return True
    
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return False


def recontextualize_query(question: str, chat_history: List[Dict]) -> str:
    """Reformulate the current question based on chat history for better context"""
    if not chat_history or len(chat_history) == 0:
        return question
    
    # Get last 3 exchanges for context
    recent_history = chat_history[-3:]
    
    history_text = "\n".join([
        f"User: {chat['question']}\nAssistant: {chat['answer'][:200]}..."
        for chat in recent_history
    ])
    
    try:
        contextualization_prompt = f"""With the following conversation history and a new user question, rephrase the question to be an independent question that uses relevant context from the conversation history.

Conversation History:
{history_text}

New Question: {question}

If the new question references previous topics (using words like "it", "that", "those", "the paper", etc.), rewrite it to be explicit and self-contained. If the question is already independent, DO NOT CHANGE IT.

Reformulated Question:"""
        
        response = bedrock_client.converse(
            modelId=config.BEDROCK_MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [{"text": contextualization_prompt}]
                }
            ],
            inferenceConfig=config.MODEL_CONFIG
        )
        
        reformulated = response["output"]["message"]["content"][0]["text"].strip()
        return reformulated if reformulated else question
    
    except Exception as e:
        st.warning(f"Could not recontextualize query: {e}")
        return question


def query_bedrock(prompt: str, context: str = "") -> str:
    try:
        if context:
            full_prompt = f"""You are a research paper analysis assistant. Use the provided excerpts from research papers to answer the question accurately and comprehensively.

Context from papers:
{context}

Question: {prompt}

Instructions:
- Answer based solely on the provided context
- If the context doesn't contain enough information to fully answer the question, acknowledge what's missing
- Be specific and reference which papers or sections your answer comes from when relevant
- For questions about paper structure (sections, organization, outline), carefully look for headers, section titles, and structural elements in the context
- Provide direct, clear answers without unnecessary preamble

Answer:"""
        else:
            full_prompt = prompt
        
        response = bedrock_client.converse(
            modelId=config.BEDROCK_MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [{"text": full_prompt}]
                }
            ],
            inferenceConfig=config.MODEL_CONFIG
        )
        
        return response["output"]["message"]["content"][0]["text"]
    
    except Exception as e:
        st.error(f"Error querying Bedrock: {e}")
        return ""

def rag_query(question: str, chat_history: List[Dict] = None) -> tuple:
    if 'vectorstore' not in st.session_state or st.session_state.vectorstore is None:
        return "Please upload papers first before asking questions.", []
    
    try:
        # Recontextualize the query based on chat history
        if chat_history:
            contextualized_question = recontextualize_query(question, chat_history)
            if contextualized_question != question:
                st.info(f"Searching for: {contextualized_question}")
        else:
            contextualized_question = question
        
        # Increase retrieval count for structural or comprehensive questions
        k_value = config.FAISS_K
        structural_keywords = ['section', 'structure', 'outline', 'organization', 'table of contents', 'overview', 'summary']
        
        if any(keyword in contextualized_question.lower() for keyword in structural_keywords):
            k_value = min(k_value * 2, 20)
        
        retriever = st.session_state.vectorstore.as_retriever(
            search_kwargs={"k": k_value}
        )
        relevant_docs = retriever.invoke(contextualized_question)
        
        if not relevant_docs:
            return "No relevant information found in the uploaded papers.", []
        
        context = "\n\n".join([
            f"[From {doc.metadata.get('filename', 'Unknown')}]:\n{doc.page_content}"
            for doc in relevant_docs
        ])
        
        # Use the original question for the final answer, not the contextualized one
        answer = query_bedrock(question, context)
        
        return answer, relevant_docs
    
    except Exception as e:
        st.error(f"Error during RAG query: {e}")
        return "An error occurred while processing your question.", []


def init_session_state():
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'papers_uploaded' not in st.session_state:
        st.session_state.papers_uploaded = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def main():
    st.set_page_config(
        page_title=config.PAGE_TITLE,
        layout="wide"
    )
    
    init_session_state()
    
    st.title("Your Friendly Neighbourhood Research Assistant")
    st.markdown("*Upload any research papers you wish to explore, ask questions and deepen your understanding, all of it brought to you by AWS Bedrock*")
    
    with st.sidebar:
        st.header("Paper Library")
        
        if st.session_state.papers_uploaded:
            st.success(f"**{len(st.session_state.papers_uploaded)} papers indexed**")
            for paper in st.session_state.papers_uploaded:
                st.text(f"{paper['name']}")
                st.caption(f"   {paper['chunks']} chunks | {paper['uploaded_at']}")
        else:
            st.info("No papers uploaded yet")
        
        st.divider()
        
        if st.button("Clear All", use_container_width=True, type="secondary"):
            st.session_state.vectorstore = None
            st.session_state.papers_uploaded = []
            st.session_state.chat_history = []
            st.rerun()
    
    tab1, tab2 = st.tabs(["Upload Papers", "Ask Questions"])
    
    with tab1:
        st.header("Upload Research Papers")
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Please upload research papers in PDF format"
        )
        
        if uploaded_files:
            if st.button("Process Papers", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing: {uploaded_file.name}")
                    text = extract_text_from_pdf(uploaded_file)
                    
                    if text and len(text) > 100:
                        documents = chunk_text(text, uploaded_file.name)
                        if create_or_update_faiss(documents):
                            st.session_state.papers_uploaded.append({
                                'name': uploaded_file.name,
                                'chunks': len(documents),
                                'uploaded_at': datetime.now().strftime("%Y-%m-%d %H:%M")
                            })
                    else:
                        st.warning(f"Could not extract sufficient text from {uploaded_file.name}")
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                status_text.text("All papers processed!")
                st.balloons()
    
    with tab2:
        st.header("Ask Any Questions About Your Paper(s)")
        
        if not st.session_state.papers_uploaded:
            st.warning("Please upload papers first in the 'Upload Papers' tab")
        else:
            for chat in st.session_state.chat_history:
                with st.chat_message("user"):
                    st.write(chat['question'])
                with st.chat_message("assistant"):
                    st.write(chat['answer'])
                    if chat.get('sources'):
                        with st.expander("View Sources"):
                            for source in chat['sources']:
                                st.caption(f"**{source.metadata.get('filename')}**")
                                st.text(source.page_content[:300] + "...")
            
            question = st.chat_input("Ask a question about your paper(s)")
            
            if question:
                with st.chat_message("user"):
                    st.write(question)
                
                with st.chat_message("assistant"):
                    with st.spinner("Searching papers and generating answer..."):
                        answer, sources = rag_query(question, st.session_state.chat_history)
                        st.write(answer)
                        
                        if sources:
                            with st.expander("View Sources"):
                                for source in sources:
                                    st.caption(f"**{source.metadata.get('filename')}**")
                                    st.text(source.page_content[:300] + "...")
                
                st.session_state.chat_history.append({
                    'question': question,
                    'answer': answer,
                    'sources': sources
                })
                st.rerun()

if __name__ == "__main__":
    main()