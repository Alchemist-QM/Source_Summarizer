from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Callable
from langchain_core.documents import Document


class HybridSplitter:
    def __init__(self, tokenizer: Callable[[str], List[int]], max_tokens: int):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def split_text(self, text: str) -> List[str]:
        if len(self.tokenizer.encode(text)) <= self.max_tokens:
            return [text]
        separators = [
        "\n\n", "\r\n", "\n", "\t", ".", " ", "",
        "##", "###", "---", "===", "Section", "References"
    ]
        
        
        #RecursiveCharacterTextSplitter to get sematic structure 
        chunk_size = int(self.max_tokens * 0.8)
        chunk_overlap = int(self.max_tokens * 0.1)

        # Try recursive splitting
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            model_name="gpt-3.5-turbo", 
            separators=separators,
        )
        chunks = splitter.split_text(text)

        # If still too big, split tokens manually
        final_chunks = []
        for chunk in chunks:
            tokenized = self.tokenizer.encode(chunk)
            if len(tokenized) <= self.max_tokens:
                final_chunks.append(chunk)
            else:
                for i in range(0, len(tokenized), self.max_tokens):
                    tokens_slice = tokenized[i: i + self.max_tokens]
                    text_slice = self.tokenizer.decode(tokens_slice)
                    final_chunks.append(text_slice)
        return final_chunks
        
    def split_documents(self, docs: List[Document]) -> List[Document]:
        split_docs = []
        for doc in docs:
            chunks = self.split_text(doc.page_content)
            for chunk in chunks:
                split_docs.append(Document(page_content=chunk, metadata=doc.metadata.copy()))
        return split_docs