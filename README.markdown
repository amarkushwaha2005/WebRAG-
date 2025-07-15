```markdown
# Website RAG Query System

A Streamlit-based application that enables users to input a website URL, process its content, and ask questions about it using Retrieval-Augmented Generation (RAG) with Ollama's deepseek-r1 model.

## Features
- Fetches and processes website content
- Splits content into manageable chunks
- Creates vector embeddings for efficient retrieval
- Supports conversational memory for context-aware responses
- Displays relevant content chunks with similarity scores
- User-friendly Streamlit interface

## Requirements
- Python 3.8+
- Ollama server running locally with the `deepseek-r1:latest` model
- Dependencies listed in `requirements.txt`

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/username/website-rag-query.git
   cd website-rag-query
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Ensure Ollama is installed and running:
   - Follow instructions at [Ollama](https://ollama.ai/) to set up the server
   - Pull the `deepseek-r1:latest` model:
     ```bash
     ollama pull deepseek-r1:latest
     ```

## Usage
1. Start the Streamlit application:
   ```bash
   streamlit run website_rag_query.py
   ```
2. Open the provided URL in your browser
3. Enter a website URL to process
4. Ask questions about the website's content
5. View answers and relevant content chunks

## Dependencies
- `streamlit`: For the web interface
- `requests`: For fetching website content
- `beautifulsoup4`: For HTML parsing
- `langchain`: For text splitting and RAG pipeline
- `langchain-community`: For Ollama embeddings, chat model, and FAISS
- `faiss-cpu`: For vector storage
- `lxml` (optional): For enhanced HTML parsing

## Configuration
Modify the `CONFIG` dictionary in `website_rag_query.py` to adjust:
- `CHUNK_SIZE`: Size of text chunks (default: 300)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 50)
- `MODEL_NAME`: Ollama model (default: deepseek-r1:latest)
- `TEMPERATURE`: Model response creativity (default: 0.4)
- `USER_AGENT`: HTTP request user agent

## Notes
- Ensure the Ollama server is running before starting the application
- The system requires an active internet connection to fetch website content
- For better HTML parsing, install `lxml`; otherwise, it falls back to `html.parser`

## Contributing
Contributions are welcome! Please open an issue to discuss proposed changes or submit a pull request.

## License
MIT License
```