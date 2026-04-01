import asyncio
import tiktoken
from typing import Optional, Union
from website_url_converter import URLConverter
from domain_selector import DomainController
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
from file_summarizer import FileSummarizer
from pathlib import Path 
from domain_finder import DomainResearchAssistant
from docling.document_converter import DocumentConverter
from description_generator import genearate_search_query
from audio_converter import audio_to_markdown_token_chunked, file_convertor
from research_config import *
from dotenv import load_dotenv
import os
from pdf_finder_controller import PDFLateralSearcher
from youtube_summarizer_2 import YouTubeSummarizer 

def get_valid_file_path(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise ValueError(f"Not a file: {path}")

    return path

def get_file_type(file_path) -> str:
        """
        Returns file type given file path 

        Args:
            file_path (str): _description_

        Returns:
            str: _description_
        """
        ext = Path(file_path).suffix.lower()
        if ext == ".md":
            return "markdown"
        elif ext == ".pdf":
            return "pdf"
        elif ext in {".docx", ".doc"}:
            return "docx"
        elif ext in {".mp3", ".wav"}:
            return "audio"
        elif ext in {".mp4", ".webm"}:
            return "video"
        else:
            return "unknown"
import re

def is_valid_youtube_url(url: str) -> bool:
    youtube_regex =  re.compile(
    r"(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)([A-Za-z0-9_-]{11})"
)
    return bool(youtube_regex.search(url))

def validate_all_files(file_list: list[str])->Union[tuple[bool,str], bool]:
    """
    Validates whether a list of files are all pdf files or markdown files
    

    Args:
        file_list list[str]: list of file paths 

    Returns:
        Union[tuple[bool,str], bool]: returns True and log of validity, else, False
    """
    is_all_pdf = all(get_file_type(f) == "pdf" for f in file_list)
    is_all_md = all(get_file_type(f) == "markdown" for f in file_list)

    if is_all_pdf:
        return True, "All files are valid PDFs"
    if is_all_md:
        return True, "All files are valid md"
    else:
        return False
        
class WebsiteController:
    def __init__(self, api_key:str,file_path: Optional[Union[str, list, Path]]=None,url:Optional[str]=None,query: Optional[str]=None,research_type="study", more_urls:str="normal", summarize: bool = True, find_domains:bool =False, deep_results: bool=False):
        self.url = url
        self.path = get_valid_file_path(file_path) if file_path and isinstance(file_path, str) else file_path if file_path and isinstance(file_path, list) else file_path
        #file path is converted into a Path if a str, else it will be a list else it will be either None or a Path 
        self.query = query
        self.api_key = api_key
        self.more_urls = more_urls 
        self.research_type = research_type
        self.find_domains = find_domains
        self.deep_results = deep_results
        self.summarize = summarize
        self.openai_tokenizer =  OpenAITokenizer(
                tokenizer=tiktoken.encoding_for_model("gpt-4o"),    
            max_tokens=600,  # context window length required for OpenAI tokenizers
            )
        
    async def website_pipeline(self,):
        if self.url is None and self.path is None and self.query is None:
            print("No input provided") #if no input, return empty string
            return ""
        if not self.api_key:
            print("No api_key provided")
            return "" #returns nothing if no api_key provided
        if self.url and isinstance(self.url, str):
            #url logic will take summarize and find_domains; will only 
            if not self.url.startswith(("http://", "https://")): #basic validation for url input
                print("Invalid URL format. URL should start with http:// or https://")
                return ""
            else:
                if is_valid_youtube_url(self.url): #insert youtube logic here 
                    try:
                        youtube_summarizer = YouTubeSummarizer()
                        await youtube_summarizer.run()
                    except Exception as e:
                        print("Unable to run youtube url")
                        return ""
                    #tries converting url into a usable format
                else:
                    try: #if url isnt a youtube url 
                        url_converter = URLConverter(
                        openai_tokenizer=self.openai_tokenizer, 
                        url=self.url
                    )
                        url_file_path, md_text = await url_converter.url_converter() #gets the list of markdown lines and url_path for summarization
                        #url_file_path is used for file_summaarizer
                        #md_text is used for query_extractor to extract the root  domain given a url 
                    except Exception as e:
                        print(f"Failed to convert url: {e}")
                        import traceback
                        traceback.print_exc()
                        return ""
                    else:
                        if self.summarize:
                            try:
                                #if summarize is true and url input is valid, run summarizer on url input -> only runs if file input is invalid or not provided, prioritizes url input for summarization
                                print("About to start summarizer")
                                web_summarizer = FileSummarizer(
                                            file_path=str(url_file_path),
                                            api_key=self.api_key,
                                            research_type=self.research_type,
                                            mode=self.more_urls,
                                        )
                                        
                                web_summary = await web_summarizer.run_summarizer()
                            except Exception as e:
                                print(f"Failed to summarize url content: {e}")
                                return ""

                        if self.find_domains: #finds related domains given a url 
                            query = genearate_search_query(md_text[0]) #takes the md str -> llm -> query for domain research
                            domain_contorller = DomainController(
                                            api_key=self.api_key,
                                            url=self.url,
                                            query=query,
                                            more_urls=self.more_urls
                                        )
                            domain = domain_contorller.domain_extractor() #either a dict or str: dependent on mode inputted
                            if self.more_urls == "normal" and isinstance(domain, str):
                                assistant = DomainResearchAssistant(
                                        query=query,
                                        domain=domain,
                                        api_key=self.api_key
                                    )

                                config = ResearchConfig(domain=domain)
                                domain_results = await assistant.research_crawler(config)
                            if self.more_urls == "expand" and isinstance(domain, dict):
                                    assistant = DomainResearchAssistant(
                                    query=query,
                                    domain=None,
                                    api_key=self.api_key
                                        )
                                    domain_results = await assistant.multi_domain_research(domain)
                            return domain_results, web_summary if self.find_domains and self.summarize else web_summary if self.summarize else domain_results
            
        if self.path: #checks to see if path has an input 
            #if summarize is true and path is valid, run summarizer on file input -> prioritizes file input over url input for summarization
            if isinstance(self.path, Path): #condition for single file Path 
                path_type = get_file_type(self.path) 
                if path_type == "unknown":
                    print(f"Unsupported file type for summarization: {self.path.suffix}")
                    return "" #stops program if file_type isnt authorized in get_file_type
                if self.summarize and path_type == "audio": #checks to see if file_path is an audio_type
                    try:
                        #converts audio file to markdown format as str and saved as md_path; used for file_summarizer 
                        chunked_md_path, md_lines = audio_to_markdown_token_chunked(
                            audio_path=str(self.path),  
                        )
                    except Exception as e:
                        print(f"Failed to convert audio file: {e}")
                        import traceback
                        traceback.print_exc()
                        return ""
                    else:
                        try:
                            #passes md path in filesummarizer | input is either markdown or pdf
                            audio_summarizer = FileSummarizer(
                                                file_path=str(chunked_md_path),
                                                api_key=self.api_key,
                                                research_type=self.research_type,
                                                mode=self.more_urls,
                                            )
                                            
                            audio_summary = await audio_summarizer.run_summarizer()
                        except Exception as e:
                            print(f"Failed to summarize audio content: {e}")
                            return "" 
                        return audio_summary           
                if self.summarize: #runs if path_type is not audio(everything else basically)
                    if path_type not in {"markdown", "pdf",}: # if not markdown or pdf, conversion to markdown is needed 
                        try:                        
                            converted_file = file_convertor(
                            file_path=str(self.path),
                            converter=DocumentConverter(),
                            output_dir="outputs",
                            debug=False,
                            )
                        except Exception as e:
                            print(f"Failed to convert file for summarization: {e}")
                            import traceback
                            traceback.print_exc()
                            return ""
                        else:   
                            file_summarizer = FileSummarizer(
                                                file_path=converted_file,#takes md or pdf path for summarization
                                                api_key=self.api_key,
                                                research_type=self.research_type,
                                                mode=self.more_urls,
                                                )
                                            
                            file_summary = file_summarizer.run_summarizer() 
                    else:
                        try: #if path_type is markdown or pdf, no conversion is needdd 
                            summaried_file = FileSummarizer(
                                file_path = self.path,
                                api_key=self.api_key,
                                research_type=self.research_type,
                                mode=self.more_urls,
                            ).run_summarizer()
                        except Exception as e:
                            print(f"Failed to summarized file: {e}")
                            return ""
                        #if path_type is a pdf and what more urls, returns more pdfs
                        if path_type == "pdf" and self.more_urls =="expand":
                            pdf_searcher =PDFLateralSearcher(
                                    api_key=self.api_key,
                                    pdf_file=str(self.path),
                                )
                            found_pdfs = pdf_searcher.find_relevant_pdfs()
                            if self.deep_results:
                                #summarization of more pdfs if wanted 
                                deep_pdf_results = pdf_searcher.summarize_found_pdfs(found_pdfs)
                        
            if isinstance(self.path, List):
                #checks to see if all files are pdfs first
                if validate_all_files(self.path)[0]:
                    try: 
                        #returns summaried_pdfs and markdown files and saves it to a file 
                        summarized_files= FileSummarizer(
                            file_path=self.path, #takes list of pdf paths for summarization
                            api_key=self.api_key,
                            research_type=self.research_type,
                            mode=self.more_urls,
                        ).run_summarizer()
                    except Exception as e:
                        print(f"Failed to summarize pdf files: {e}")
                        import traceback
                        traceback.print_exc()
                        return ""
                else:
                    all_files_valid = all(get_file_type(f) != "unkown" for f in self.path) #returns False if unknown file_type is found in file list 
                    if all_files_valid: 
                        try:
                            converted_files = file_convertor(
                                file=self.path, #file converter takes list or str 
                                converter=DocumentConverter(),
                                output_dir="outputs",
                                debug=False
                            ) 
                        except Exception as e:
                            print(f"Failed to convert file for summarization: {e}")
                            import traceback
                            traceback.print_exc()
                            return ""
                        else:
                            summaries = []
                            converted_file_paths = converted_files[1]  
                            for file in converted_file_paths: #converted md_paths 
                                file_summarizer = FileSummarizer(
                                file_path=file,
                                api_key=self.api_key,
                                research_type=self.research_type,
                                mode=self.more_urls
                            )   
                                summarized_file = file_summarizer.run_summarizer()
                                summaries.append(summarized_file)  
                
                #write file to markdown
        if self.query and isinstance(self.query, str) and self.find_domains:
                pdf_searcher = PDFLateralSearcher(
                            api_key=self.api_key,
                            query=self.query,
                        )
                found_pdfs = pdf_searcher.find_relevant_pdfs()
                if self.deep_results:
                        deep_pdf_results = pdf_searcher.summarize_found_pdfs(found_pdfs)

if __name__ == "__main__":
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    website_controller = WebsiteController(
        url="https://www.ibm.com/think/topics/cyber-hacking",
        api_key=OPENAI_API_KEY
    )
    #asyncio.run(website_pipeline(config=ResearchConfig(test_query, test_domain)))

    asyncio.run(website_controller.website_pipeline())    
