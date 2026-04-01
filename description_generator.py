
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from token_batching import TokenSplitter

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_4")


def clean_query(query: str) -> str:
    # Remove trailing 'PDF' (case-insensitive)
    words = query.strip().split()
    if words and words[-1].lower() == "pdf":
        words = words[:-1]
    return " ".join(words)


def genearate_search_query(
    markdown_text: str,
    model_name: str = "gpt-3.5-turbo",
    batch_token_limit: int = None
) -> str:
    """
    Generates an optimized Google search query for finding related research PDFs,
    using token batching to handle long markdown text.
    Args:
        markdown_text (str): Markdown text of the PDF.
        model_name (str): LLM model name.
        batch_token_limit (int): Maximum tokens per batch (optional).
    Returns:
        str: Google search query (10-15 words, technical keywords).
    """
    tokenizer = TokenSplitter.get_tokenizer(model_name)
    max_tokens = batch_token_limit or int(TokenSplitter.get_max_tokens_for_model(model_name) * 0.5)
    # Split markdown into paragraphs for batching
    paragraphs = [p for p in markdown_text.split("\n\n") if p.strip()]
    batches = TokenSplitter.tokenize_and_batch(paragraphs, max_batch_tokens=max_tokens, tokenizer=tokenizer, use_splitter=False)

    prompt = PromptTemplate(
        input_variables=["document"],
        template=(
            "Given the following document, generate a concise Google search query (10-15 words max) "
            "that would help find related research papers in PDF format. Use only relevant keywords and technical terms:\n\n"
            "{document}\n\nSearch Query:"
        )
    )
    llm = ChatOpenAI(model=model_name, temperature=0, api_key=OPENAI_API_KEY)
    chunk_queries: List[str] = []
    for batch in batches:
        batch_text = "\n\n".join(batch)
        query = llm.invoke(prompt.format(document=batch_text))
        chunk_queries.append(str(query).strip())
    # Combine chunk queries into a single search query (deduplicate, join)
    final_query = " ".join(chunk_queries)
    # Limit to 15 words max, as per prompt
    final_query_words = final_query.split()
    google_query = " ".join(final_query_words[:15])
    cleaned_query = clean_query(google_query)
    return cleaned_query

# Example usage in your pipeline
#work 09/25/2025

#if __name__ == "__main__":
    #pdf_path = "C:/Users/qmerr/Downloads/New_youtube_summarizer/Attention.pdf"
    #markdown_text = pdf_to_markdown(pdf_path)
    #google_search_query = generate_google_search_query_token_batching(markdown_text)

    #print("Google Search Query for PDF:", google_search_query)
    # Your code to feed google_query into PDF finder goes here