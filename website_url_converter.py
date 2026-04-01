from docling.document_converter import DocumentConverter
import asyncio
from docling.chunking import HybridChunker 
import json 
import hashlib
from pathlib import Path
from datetime import datetime
from crawl4ai import AsyncWebCrawler
import tiktoken
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig, DefaultMarkdownGenerator

# Define browser config globally for undetected browser logic
class URLConverter:
    def __init__(self, openai_tokenizer,url:str, output_dir="outputs/chunks"):
        self.tokenizer = openai_tokenizer
        self.md_ouput_file = "chunked_output.md"
        self.md_source_folder = "./output.md"
        self.url=url
        self.converter = DocumentConverter()

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    
    async def url_converter(self)->tuple[Path, list[str]]:
        browser_config = BrowserConfig(
        browser_type="undetected",
        verbose=True,
        headless=True,
        extra_args=[
            "--disable-blink-features=AutomationControlled",
            "--disable-web-security"
        ]
    )
        run_config = CrawlerRunConfig(
        markdown_generator=DefaultMarkdownGenerator(
            options={"ignore_links": True}
        ),
        exclude_social_media_links=True,
        excluded_tags=['footer', 'nav', 'aside'],
    )   # Default crawl run configuration

        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(
                url=self.url,
                config=run_config
            )
            if result.success:
                markdown_text = result.markdown.fit_markdown or result.markdown.raw_markdown

                try:
                    with open("output.md", "w", encoding="utf-8") as f:
                        f.write(markdown_text)
                    with open("links.json", "w", encoding="utf-8") as f:
                            json.dump(result.links, f, indent=4)
                except Exception as e:
                    print(f"Error writing to file: {e}")

            else:
                print(f"Failed to crawl {self.url}: {result.error_message}")
                
            source = self.md_source_folder
            result = self.converter.convert(source)
            chunker = HybridChunker(
                tokenizer=self.tokenizer,
                max_chunk_tokens=2000,
            )
            chunk_iter = chunker.chunk(dl_doc=result.document)
            chunks = list(chunk_iter)
            rag_chunks = []

            for i, chunk in enumerate(chunks):
                rag_chunks.append({
                    "id": f"chunk_{i}",
                    "text": chunk.text,
                    "embedding_text": chunker.contextualize(chunk),
                    "metadata": {
                        "chunk_index": i,
                        "token_count": self.tokenizer.count_tokens(chunker.contextualize(chunk)),
                        "source": self.url
                    }
                })
            
            with open("rag_chunks.json", "w", encoding="utf-8") as f:
                json.dump(rag_chunks, f, indent=2)
                
            url_hash = hashlib.md5(self.url.encode()).hexdigest()[:8]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            chunked_md_path = self.output_dir / f"{url_hash}_chunks_{timestamp}.md"
            md_lines = []
            for i, chunk in enumerate(chunks):
                md_lines.append(f"## Chunk {i}\n\n")
                md_lines.append("```markdown\n")
                md_lines.append(chunker.contextualize(chunk).strip())
                md_lines.append("\n```\n\n")
                
            with open(chunked_md_path, "w", encoding="utf-8") as f:
                f.writelines(md_lines)
                
            return chunked_md_path, md_lines

#how to run pipleine                     
"""
if __name__ == "__main__":
    input_url = input("Enter URL to crawl: ")
   
    url_converter = URLConverter(openai_tokenizer=OpenAITokenizer(tokenizer=tiktoken.encoding_for_model("gpt-4o"),
            max_tokens=512, 
            ), url=input_url)  # Pass actual tokenizer here
    markdown, chunked_md_path = asyncio.run(url_converter.url_converter())
    print(markdown[0])
    print(chunked_md_path)

"""
