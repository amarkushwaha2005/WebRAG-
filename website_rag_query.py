```python
import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import BSHTMLLoader
import tempfile
import time

# Configuration settings
CONFIG = {
    'CHUNK_SIZE': 300,
    'CHUNK_OVERLAP': 50,
    'MODEL_NAME': "deepseek-r1:latest",
    'TEMPERATURE': 0.4,
    'USER_AGENT': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
        '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    )
}

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    defaults = {
        'qa': None,
        'vectorstore': None,
        'chat_history': []
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def fetch_website_content(url):
    """Fetch and process website content into text chunks."""
    try:
        with st.spinner('Fetching website content...'):
            response = requests.get(url, headers={'User-Agent': CONFIG['USER_AGENT']})
            response.raise_for_status()

            # Save HTML to temporary file
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html', encoding='utf-8') as temp_file:
                temp_file.write(response.text)
                temp_file_path = temp_file.name

            # Load HTML content
            try:
                loader = BSHTMLLoader(temp_file_path)
                documents = loader.load()
            except ImportError:
                st.warning("Falling back to html.parser due to missing lxml.")
                loader = BSHTMLLoader(temp_file_path, bs_kwargs={'features': 'html.parser'})
                documents = loader.load()
            finally:
                os.unlink(temp_file_path)

            # Split text into chunks
            text_splitter = CharacterTextSplitter(
                chunk_size=CONFIG['CHUNK_SIZE'],
                chunk_overlap=CONFIG['CHUNK_OVERLAP']
            )
            return text_splitter.split_documents(documents)

    except requests.RequestException as e:
        st.error(f"Failed to fetch website: {str(e)}")
        return None

def setup_rag_pipeline(texts):
    """Set up the Retrieval-Augmented Generation pipeline."""
    with st.spinner('Setting up RAG pipeline...'):
        # Initialize language model
        llm = ChatOllama(
            model=CONFIG['MODEL_NAME'],
            temperature=CONFIG['TEMPERATURE']
        )

        # Create embeddings and vector store
        embeddings = OllamaEmbeddings(model=CONFIG['MODEL_NAME'])
        vectorstore = FAISS.from_documents(texts, embeddings)

        # Define prompt template
        prompt_template = """Context: {context}

Question: {question}

Answer concisely using only the provided context. If the context lacks relevant information, respond with: "I don't have enough information to answer that question."

For general knowledge questions (e.g., "What is an electric vehicle?"), provide a direct answer."""

        qa_prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # Set up memory for conversation context
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Create QA chain
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            memory=memory,
            chain_type_kwargs={"prompt": qa_prompt}
        )

        return qa, vectorstore

def main():
    """Main function to run the Streamlit application."""
    initialize_session_state()
    
    st.title("🤖 Website RAG Query System")
    st.markdown("Analyze website content and ask questions using Retrieval-Augmented Generation.")

    # Website URL input
    url = st.text_input("Enter website URL:", placeholder="https://example.com")
    
    # Process website button
    if st.button("Process Website", disabled=not url):
        texts = fetch_website_content(url)
        if texts:
            st.success(f"Processed {len(texts)} text chunks from the website.")
            st.session_state.qa, st.session_state.vectorstore = setup_rag_pipeline(texts)
            st.session_state.chat_history = []

    # Query interface
    if st.session_state.qa and st.session_state.vectorstore:
        st.markdown("---")
        st.subheader("Ask Questions About the Website")
        
        query = st.text_input("Enter your question:", placeholder="What's on this website?")
        
        if st.button("Ask", disabled=not query):
            with st.spinner('Generating answer...'):
                # Retrieve relevant documents
                relevant_docs = st.session_state.vectorstore.similarity_search_with_score(query, k=3)
                
                # Show relevant chunks in expander
                with st.expander("Relevant Content Chunks"):
                    for i, (doc, score) in enumerate(relevant_docs, 1):
                        st.markdown(f"**Chunk {i} (Score: {score:.4f})**")
                        st.write(doc.page_content)
                        st.markdown("---")
                
                # Get and display answer
                response = st.session_state.qa.invoke({"query": query})
                st.session_state.chat_history.append({"question": query, "answer": response['result']})

            # Display chat history
            st.markdown("---")
            st.subheader("Conversation History")
            for chat in st.session_state.chat_history:
                st.markdown(f"**Q:** {chat['question']}")
                st.markdown(f"**A:** {chat['answer']}")
                st.markdown("---")

    # Sidebar information
    with st.sidebar:
        st.subheader("About This Application")
        st.markdown("""
        This RAG system allows you to:
        - Input a website URL
        - Process its content
        - Ask questions about the content

        **Technologies Used:**
        - Ollama (deepseek-r1) for language modeling
        - FAISS for vector storage
        - LangChain for RAG pipeline
        - Streamlit for the web interface
        """)
        
        st.subheader("Configuration")
        st.markdown(f"""
        - **Model**: {CONFIG['MODEL_NAME']}
        - **Temperature**: {CONFIG['TEMPERATURE']}
        - **Chunk Size**: {CONFIG['CHUNK_SIZE']}
        - **Chunk Overlap**: {CONFIG['CHUNK_OVERLAP']}
        """)

if __name__ == "__main__":
    main()
```