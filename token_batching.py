from typing import List, Union
from langchain.schema import Document
import tiktoken
from hybrid_splitter import HybridSplitter

class TokenSplitter:
    def __init__(self,):
        pass
        
    def get_max_tokens_for_model(self,model_name: str) -> int:
        """
        Returns a safe token limit for a given model.
        Accounts for system prompts, instructions, and output generation space.

        Args:
            model_name (str): Name of the model (e.g., "gpt-3.5-turbo", "gpt-4o-mini")

        Returns:
            int: Recommended max input tokens
        """
        model_limits = {
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
        }

        # Fallback if not in dict
        default_limit = 8192
        max_total = model_limits.get(model_name, default_limit)

        # Reserve 20% for response tokens and prompt overhead
        reserved_output_buffer = int(max_total * 0.2)

        return max_total - reserved_output_buffer
            
    def safe_content(self, obj: Union[str, object])->str: #AImessage cannot be saved in json format 
        """
        Takes an AIMessage or BaseMessage and safely returns the content instead
        of the message itself

        Args:
            obj (_type_): 

        Returns:
            _str: clean str that actual contains the response tex
        """
        return obj.content if hasattr(obj, "content") else str(obj)   
        
    def get_tokenizer(self, model_name: str):
        """
        Creates a token aware wrapper to be split
        Args:
            text(str):token count for text given a model
            model(str): Name of the model 
        
        Returns:
            int: token count of a word given a model
        
        """
        try:
            return tiktoken.encoding_for_model(model_name)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")  # fallback



    def get_token_splitter(self, max_tokens: int, tokenizer):
        """
        Hybrid splitter: combines semantic (RecursiveCharacterTextSplitter) and token-accurate (TokenTextSplitter) chunking.
        Ensures chunks stay under token limit while preserving structure when possible.
        Args:
            model_name (str): name of model
            max_tokens (int): max_tokens from model 

        Returns:
            _type_: _description_
        
        """

        return  HybridSplitter(
        max_tokens=max_tokens,
        tokenizer=tokenizer
    )

    #use langchain native concurrency logic and avoids asyncio.to_thread overhead strain as well as parallel without resting
    def tokenize_and_batch(
        self,
        items: List[Union[Document, str]],
        max_batch_tokens: int,
        tokenizer,
        use_splitter: bool = False,
    ) -> List[List[Union[Document, str]]]:
        """
        Token-aware batching for either LangChain Documents or strings.

        Args:
            items: List of Documents or strings.
            max_batch_tokens: Max token size per batch.
            tokenizer: Tokenizer for token counting.
            use_splitter: Whether to split long Documents before batching.
            model_name: Required if use_splitter is True.

        Returns:
            List of token-safe batches.
        """
        if use_splitter and all(isinstance(i, Document) for i in items):
            token_splitter = self.get_token_splitter(max_batch_tokens, tokenizer)
            items = token_splitter.split_documents(items)

        batches = []
        current_batch = []
        current_tokens = 0

        for item in items:
            text = item.page_content if isinstance(item, Document) else str(item)
            tokens = len(tokenizer.encode(text))

            if current_tokens + tokens > max_batch_tokens:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [item]
                current_tokens = tokens
            else:
                current_batch.append(item)
                current_tokens += tokens

        if current_batch:
            batches.append(current_batch)
        return batches


if __name__ == "__main__":
    splitter = TokenSplitter()
    model_name = "gpt-4o-mini"
    tokenizer = splitter.get_tokenizer(model_name)
    max_tokens = splitter.get_max_tokens_for_model(model_name)
    print(f"Max tokens for {model_name}: {max_tokens}")