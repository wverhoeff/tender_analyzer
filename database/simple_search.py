from typing import List, Dict, Any

class SimpleSearchEngine:
    """
    Performs standard keyword and regex-based lookups on the ingested documents.
    Used as a fallback or sidekick for exact-match compliance auditing.
    """
    def __init__(self):
        self.corpus: List[Dict[str, Any]] = []

    def load_documents(self, documents: List[Dict[str, Any]]):
        """
        Loads document corpus in memory for text lookup.
        """
        self.corpus = documents

    def keyword_search(self, term: str) -> List[Dict[str, Any]]:
        """
        Filter documents containing the precise keyword term.
        """
        print(f"[SimpleSearch] Executing keyword search for: '{term}'")
        results = []
        for doc in self.corpus:
            content = doc.get("content", "").lower()
            if term.lower() in content:
                results.append(doc)
        return results
