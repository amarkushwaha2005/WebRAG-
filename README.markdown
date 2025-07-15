# WebRAG

## Overview
WebRAG is a Python-based pipeline that combines web scraping with Retrieval-Augmented Generation (RAG) to extract content from websites and answer user queries using a language model. It leverages libraries like `requests`, `BeautifulSoup`, `langchain`, and `FAISS` to scrape web content, process it into chunks, create embeddings, and provide context-aware responses.

## Features
- **Web Scraping**: Fetches and cleans text content from websites using `BeautifulSoup`.
- **Text Processing**: Splits content into manageable chunks with configurable size and overlap.
- **Vector Store**: Creates embeddings using OpenAI's API and stores them in a FAISS vector store for efficient retrieval.
- **RAG Pipeline**: Uses a `langchain` RetrievalQA chain with a customizable prompt to answer queries based on scraped content.
- **Conversation Memory**: Maintains context across queries using `ConversationBufferMemory`.
- **Error Handling**: Robust handling of network issues and content parsing failures.

## Prerequisites
- Python 3.8+
- OpenAI API key (set as an environment variable or input during execution)
- Required Python packages (listed in `requirements.txt`)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/WebRAG.git
   cd WebRAG
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your OpenAI API key:
   ```bash
   export OPENAI_API_KEY='your-api-key'
   ```
   Alternatively, the program will prompt for the API key if not set.

## Usage
1. Run the script:
   ```bash
   python webrag.py
   ```
2. Enter a website URL when prompted (e.g., `https://example.com`).
3. Once the content is processed, input queries related to the website's content.
4. Use `new` to process a different website or `quit` to exit.

### Example
```bash
Please enter the URL of the website you want to query (or 'quit' to exit): https://example.com
Processing website content...
Number of documents loaded: 1
Sample of loaded content: Example Domain...
Number of text chunks after splitting: 10
Creating embeddings and vector store...
Sample Embedding (first 10 dimensions): [0.123, -0.456, ...]
RAG Pipeline initialized. You can now enter your queries.
Enter your query: What is the purpose of example.com?
RAG Response: The purpose of example.com is to provide a placeholder domain for illustrative purposes.
```

## Configuration
The script includes configurable parameters in `webrag.py`:
- `CHUNK_SIZE`: Size of text chunks (default: 300 characters)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 50 characters)
- `MAX_TOKENS`: Maximum tokens for the language model (default: 15000)
- `MODEL_NAME`: OpenAI model name (default: `gpt-4o-mini`)
- `TEMPERATURE`: Model temperature for response creativity (default: 0.4)

## Dependencies
- `requests`: For fetching web content
- `beautifulsoup4`: For HTML parsing
- `langchain`: For RAG pipeline and text processing
- `langchain-openai`: For OpenAI embeddings and LLM
- `faiss-cpu`: For vector storage and retrieval
- `numpy`: For numerical operations
- `lxml` (optional): For faster HTML parsing with `BSHTMLLoader`

Install them using:
```bash
pip install requests beautifulsoup4 langchain langchain-openai faiss-cpu numpy
```

## Notes
- Ensure a stable internet connection for web scraping.
- Some websites may require JavaScript rendering or have anti-scraping measures, which may limit content extraction.
- The `lxml` parser is recommended for better performance but falls back to `html.parser` if not installed.
- The script uses a temporary file for HTML processing, which is automatically cleaned up.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.