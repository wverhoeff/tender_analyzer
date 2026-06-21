import os
from typing import List, Dict, Any

class FileManagementTool:
    """
    Utility methods for managing files in ingestion and workspace folders.
    """
    def __init__(self, ingestion_dir: str = "./data/ingestion", output_dir: str = "./workspace/output"):
        self.ingestion_dir = ingestion_dir
        self.output_dir = output_dir
        os.makedirs(ingestion_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

    def list_ingested_files(self) -> List[str]:
        """
        List all files waiting in the ingestion directory.
        """
        return os.listdir(self.ingestion_dir)

    def read_markdown_file(self, filename: str) -> str:
        """
        Reads raw Markdown/text file content.
        """
        file_path = os.path.join(self.ingestion_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def save_report(self, filename: str, content: str):
        """
        Save synthesized reports to output workspace folder.
        """
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[FileManagement] Saved report to {output_path}")
