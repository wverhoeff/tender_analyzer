import os
from typing import List, Dict, Any
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
try:
    from langchain_text_splitters import MarkdownHeaderTextSplitter
except ImportError:
    from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_core.documents import Document

class TenderVectorStore:
    """
    Manages semantic indexing and retrieval of tender documents
    using ChromaDB and LangChain.
    """
    def __init__(self, persist_directory: str = "./db/chroma", ingestion_dir: str = "./data/ingestion"):
        self.persist_directory = persist_directory
        self.ingestion_dir = ingestion_dir
        # Using a lightweight, locally executable embedding model
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = None
        self._init_db()

    def _init_db(self):
        """Initializes the Chroma database connection."""
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )

    def ingest_markdown(self):
        """
        Reads all .md files from the data/ingestion folder,
        splits the text into chunks using a Markdown header splitter,
        and saves them to the local Chroma vector store.
        """
        if not os.path.exists(self.ingestion_dir):
            os.makedirs(self.ingestion_dir, exist_ok=True)
            return

        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
        )
        
        all_documents = []
        
        for file_name in os.listdir(self.ingestion_dir):
            if file_name.endswith(".md"):
                file_path = os.path.join(self.ingestion_dir, file_name)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Split content by markdown headers
                split_docs = markdown_splitter.split_text(content)
                
                # Add metadata (like filename) to each split chunk
                for doc in split_docs:
                    doc.metadata["source"] = file_name
                    all_documents.append(doc)
        
        if all_documents:
            self.vector_store.add_documents(all_documents)
            print(f"[TenderVectorStore] Successfully ingested {len(all_documents)} markdown chunk(s).")
        else:
            print("[TenderVectorStore] No markdown files found or parsed to ingest.")

    def search_costings(self, query: str) -> List[Document]:
        """
        Takes a query string and returns the top three most relevant chunks.
        """
        if not self.vector_store:
            self._init_db()
            
        print(f"[TenderVectorStore] Searching for: '{query}'")
        # Perform similarity search returning top 3 results
        results = self.vector_store.similarity_search(query, k=3)
        return results

