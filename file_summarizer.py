import os
import glob
from cleaner import *
import asyncio, json
from typing import Any
from pathlib import Path
from pypdf import PdfReader
from typing import Union
from token_batching import TokenSplitter
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable
from prompts import reducer_templates, student_templates
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader



def safe_serialize(obj: Any) -> Any:
        if isinstance(obj, (list, tuple)):
            return [safe_serialize(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: safe_serialize(v) for k, v in obj.items()}
        elif hasattr(obj, "content"):
            return obj.content
        else:
            try:
                json.dumps(obj)
                return obj
            except (TypeError, OverflowError):
                return str(obj)
            
class FileSummarizer:
    def __init__(
        self, 
        api_key:str, 
        file_path:Union[str, Path, list],
        base_model:str="gpt-3.5-turbo",
        large_model:str = "gpt-4o-mini",
        output_folder:str="summaries",
        research_type: str="study",
        mode:str = "normal"
        ):
        self.api_key = api_key
        self.MAX_FILE_SIZE = 5 # Limit number of PDFs to process in a folder for testing
        self.output_dir = output_folder
        self.file_path:str | list = str(file_path) if isinstance(file_path, Path) else file_path
        self.base_model = base_model
        self.research_type = research_type
        self.large_model = large_model
        self.mode=mode
        self.token_splitter = TokenSplitter()


    def get_file_type(self) -> str:
        """
        Returns file type given file path 

        Args:
            file_path (str): _description_

        Returns:
            str: _description_
        """
        ext = Path(self.file_path).suffix.lower()
        if ext == ".md":
            return "markdown"
        elif ext == ".pdf":
            return "pdf"
        else:
            return "unknown"

    def get_loader(self,file_type)-> PyPDFDirectoryLoader | UnstructuredMarkdownLoader:
        """
        Outputs a loader based on the file path and outputs the type of file for the file_path 

        Args:
            file_type (_type_): _description_

        Returns:
            PyPDFDirectoryLoader | UnstructuredMarkdownLoader: _description_
        """
       
        if file_type == "markdown":
            return UnstructuredMarkdownLoader(self.file_path)
        elif file_type == "pdf":
            return PyMuPDFLoader(self.file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    async def save_summary_async(self, output_data: dict, save_dir: str="summaries" ) -> str:
        os.makedirs(save_dir, exist_ok=True)
        base_filename = os.path.splitext(output_data.get("file", "output"))[0]
        output_path = os.path.join(save_dir, f"{base_filename}_summary.json")
        safe_data = safe_serialize(output_data)
        # If we have "final_key_terms", clean it!
        if "key_terms" in safe_data:
            safe_data["summaries"] = updated_clean_summary(safe_data["summaries"])
            safe_data['key_terms'] = clean_key_terms_output(safe_data['key_terms'])
            safe_data['conclusion'] = updated_clean_conclusion(safe_data['conclusion'])
            if "final_academics" in safe_data:
                safe_data["final_academics"] = clean_academic_output(safe_data["final_academics"])
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: open(output_path, "w", encoding="utf-8").write(
                json.dumps(safe_data, indent=2, ensure_ascii=False)
            )
        )
        return f"Saved summary to {output_path}"


    def determine_file_size(self,file_type:str, tokenizer=None) -> dict:
        if self.base_model is None:
            print("Please provide a model")
            return ValueError
        
        loader = self.get_loader(file_type)
        if file_type == "pdf":
            reader = PdfReader(self.file_path)
            num_pages = len(reader.pages)
            selected_llm = ChatOpenAI(model=self.base_model, temperature=0.2, api_key=self.api_key) if num_pages <= 30 else ChatOpenAI(model=self.large_model, temperature=0.2, api_key=self.api_key)
        elif file_type == "markdown":
            docs = loader.load()
            full_text = "\n".join(doc.page_content for doc in docs)
            token_count = len(tokenizer.encode(full_text))
            selected_llm = ChatOpenAI(model=self.base_model, temperature=0.2, api_key=self.api_key) if token_count <= 8000 else ChatOpenAI(model=self.large_model, temperature=0.2, api_key=self.api_key)
        chain = {}
        try:
            if self.research_type =="academic":
                chain["academic_chain"]= reducer_templates['academic_reducer'] | selected_llm
                chain["final_academic_chain"]= student_templates['academic'] | selected_llm
        finally:
            chain["introduction_chain"] = reducer_templates['summary_reducer'] | selected_llm
            chain["key_terms_chain"] = reducer_templates['key_terms_reducer'] | selected_llm
            chain["conclusion_chain"]= reducer_templates['conclusion_reducer'] | selected_llm
            chain["final_introduction_chain"]= student_templates['summary'] | selected_llm
            chain["final_key_terms_chain"]=student_templates['key_terms'] | selected_llm
            chain["final_conclusion_chain"]= student_templates['conclusion'] | selected_llm
        return chain

    def stream_docs(self, file_type):
        loader = self.get_loader(file_type)
        for doc in loader.load():
            yield doc

    async def async_batch_gather(self,coros, batch_size, sleep_time):
        results = []
        for i in range(0, len(coros), batch_size):
            batch = coros[i:i+batch_size]
            batch_results = await asyncio.gather(*batch)
            results.extend(batch_results)
            await asyncio.sleep(sleep_time)
        return results

    async def summarize_adaptive_abatch(
        self,
        introduction_chain: Runnable,
        key_terms_chain: Runnable,
        conclusion_chain: Runnable,
        final_introduction_chain: Runnable,
        final_key_terms_chain: Runnable,
        final_conclusion_chain: Runnable,
        max_tokens: int,
        file_type: str,
        tokenizer,
        academic_chain: Runnable = None,
        final_academic_chain: Runnable= None,
        delay_seconds: float = 0.5,
        llm_batch_size: int = 4,
        
    ) -> dict:
        docs = list(self.stream_docs(file_type))
        token_buffer = int(max_tokens * 0.1)
        max_batch_tokens = max_tokens - token_buffer
        if file_type == "markdown" and len(docs) >30 or file_type =="pdf" and len(docs)>100:
            use_splitter=True
        else:
            use_splitter=False
        
        batches = self.token_splitter.tokenize_and_batch(
            items=docs,
            max_batch_tokens=max_batch_tokens,
            tokenizer=tokenizer,
            use_splitter=use_splitter,
            )
        batch_texts = ["\n\n".join(doc.page_content for doc in batch) for batch in batches]
        introductions, conclusions, academics, key_terms = [], [], [], []
        # Batch each prompt per chunk
        if self.research_type == "academic":
            academic_coros = [academic_chain.ainvoke({"text": text}) for text in batch_texts]
            academics = await self.async_batch_gather(academic_coros, llm_batch_size, delay_seconds)
        
        introduction_coros = [introduction_chain.ainvoke({"text": text}) for text in batch_texts]
        introductions = await self.async_batch_gather(introduction_coros, llm_batch_size, delay_seconds)
        conclusion_coros = [conclusion_chain.ainvoke({"text": text}) for text in batch_texts]
        conclusions = await self.async_batch_gather(conclusion_coros, llm_batch_size, delay_seconds)
        key_terms_coros = [key_terms_chain.ainvoke({"text": text}) for text in batch_texts]
        key_terms = await self.async_batch_gather(key_terms_coros, llm_batch_size, delay_seconds)
        outputs = []
        for i in range(len(batch_texts)):
            outputs.append({
                "batch": i + 1,
                "doc": [i],
                "introduction": self.token_splitter.safe_content(introductions[i]),
                "conclusions": self.token_splitter.safe_content(conclusions[i]),
                "key_terms": self.token_splitter.safe_content(key_terms[i]),
            })
        if self.research_type == "academic":
            outputs.append({"academics": self.token_splitter.safe_content(academics[i])})
            
        # Final reductions
        final_introduction = await final_introduction_chain.ainvoke({"text": "\n\n".join(self.token_splitter.safe_content(x) for x in introductions)})
        final_conclusions = await final_conclusion_chain.ainvoke({"text": "\n\n".join(self.token_splitter.safe_content(x) for x in conclusions)})
        final_key_terms = await final_key_terms_chain.ainvoke({"text": "\n\n".join(self.token_splitter.safe_content(x) for x in key_terms)})
        # CLEAN THE FINAL_KEY_TERMS
        final_key_terms_clean = clean_key_terms_output(self.token_splitter.safe_content(final_key_terms))
        final_conclusions_clean = updated_clean_conclusion(self.token_splitter.safe_content(final_conclusions))
        final_introduction_clean = updated_clean_summary(self.token_splitter.safe_content(final_introduction))
        
        if self.research_type == "academics":
            final_academics = await final_academic_chain.ainvoke({"text": "\n\n".join(self.token_splitter.safe_content(x) for x in academics)})
            final_academics_clean = clean_academic(self.token_splitter.safe_content(final_academics))
        #add academic
                
            return {
                "file": os.path.basename(self.file_path),
                "batches": outputs,
                "final_introduction": self.token_splitter.safe_content(final_introduction_clean),
                "final_conclusions": self.token_splitter.safe_content(final_conclusions_clean),
                "final_key_terms": final_key_terms_clean,
                "final_academics": final_academics_clean,
            }
        else:
            return {
                "file": os.path.basename(self.file_path),
                "batches": outputs,
                "final_introduction": self.token_splitter.safe_content(final_introduction_clean),
                "final_conclusions": self.token_splitter.safe_content(final_conclusions_clean),
                "final_key_terms": final_key_terms_clean,
            }

    async def run_summarizer(self,):
        if isinstance(self.base_model, str):
            max_tokens = self.token_splitter.get_max_tokens_for_model(self.base_model)  
            tokenizer =  self.token_splitter.get_tokenizer(self.base_model) 
            file_type = self.get_file_type()
        
            if self.mode == "expand" and isinstance(self.file_path, list): #use folder is expand mode
                #grabs one instance of pdf
                try:
                    if all(self.get_file_type(f) == "pdf" for f in self.file_path):
                        file_type = "pdf"
                except Exception as e:
                    raise ValueError("A file that did not match pdf was detected. Only provide all pdfs or all markdown")
                try:
                    if all(self.get_file_type(f) == "markdown" for f in self.file_path):
                        file_type = "markdown"
                except Exception as e:
                    raise ValueError(f"A file that did not match md was detected. Only provide all pdfs or all markdown, {e}")
                files = glob.glob(os.path.join(self.file_path, "*.pdf")) if file_type == "pdf" else glob.glob(os.path.join(str(self.file_path), "*md"))
                for f in files:
                    self.file_path = f
                    try:
                        chains = self.determine_file_size(
                            file_type, 
                            tokenizer
                            ) 
                    except Exception as e:
                        print("Requires model and large model")
                    if self.research_type == "academic": 
                        result = await self.summarize_adaptive_abatch(
                        chains['introduction_chain'],
                        chains['key_terms_chain'],
                        chains['conclusion_chain'],
                        chains['academic_chain'],
                        chains['final_academic_chain'],
                        chains['final_introduction_chain'],
                        chains['final_key_terms_chain'],
                        chains['final_conclusion_chain'],
                        max_tokens,
                        file_type,
                        tokenizer
                    )
                        await self.save_summary_async(output_data=result)

                    else:
                        result = await self.summarize_adaptive_abatch(
                            chains['introduction_chain'],
                            chains['key_terms_chain'],
                            chains['conclusion_chain'],
                            chains['final_introduction_chain'],
                            chains['final_key_terms_chain'],
                            chains['final_conclusion_chain'],
                            max_tokens,
                            file_type,
                            tokenizer
                        )
                        await self.save_summary_async(output_data=result)
                    
            else:
                if isinstance(self.file_path, str) and isinstance(self.base_model, str):
                    chains = self.determine_file_size(file_type,tokenizer)
                    if self.research_type == "academic": 
                            result = await self.summarize_adaptive_abatch(
                            introduction_chain=chains['introduction_chain'],
                            key_terms_chain=chains['key_terms_chain'],
                            conclusion_chain=chains['conclusion_chain'],
                            final_introduction_chain=chains['final_introduction_chain'],
                            final_key_terms_chain=chains['final_key_terms_chain'],
                            final_conclusion_chain=chains['final_conclusion_chain'],
                            academic_chain=chains.get('academic_chain'),
                            final_academic_chain=chains.get('final_academic_chain'),
                            max_tokens=max_tokens,
                            file_type=file_type,
                            tokenizer=tokenizer,
                            research_type="academic"
                            )
                            await self.save_summary_async(output_data=result)

                    else:
                        result = await self.summarize_adaptive_abatch(
                                introduction_chain=chains['introduction_chain'],
                                key_terms_chain=chains['key_terms_chain'],
                                conclusion_chain=chains['conclusion_chain'],
                                final_introduction_chain=chains['final_introduction_chain'],
                                final_key_terms_chain=chains['final_key_terms_chain'],
                                final_conclusion_chain=chains['final_conclusion_chain'],
                                file_type=file_type,
                                max_tokens=max_tokens,
                                tokenizer=tokenizer
                            )
                        await self.save_summary_async(output_data=result)
        else: 
            print("Please provide a valid model name")
            return ValueError
"""
if __name__ == "__main__":
    # For a single file:
    # pdf_file = "C:/Users/qmerr/Downloads/New_youtube_summarizer/Attention.pdf"
    # asyncio.run(run_pipeline(pdf_file, model="gpt-3.5-turbo"))
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY_2")

    # For a folder:
    md_file = "C:/Users/qmerr/Downloads/New_youtube_summarizer/chunked_output.md"
    pdf_file = "C:/Users/qmerr/Downloads/New_youtube_summarizer/RNN.pdf"
    summarizer = Summarizer(
        api_key=api_key,
        file_path=md_file,
    )
    asyncio.run(summarizer.run_summarizer(md_file,
                                          model="gpt-3.5-turbo",
                                          large_model="gpt-4o-mini"
                                          
                                          ))
    
"""
