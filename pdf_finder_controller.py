from typing import Optional
from pdf_finder import PDF_Finder
from pdf_utils import pdf_to_markdown
from pdf_lateral_summarizer import AdjacentSummarizer
from description_generator import genearate_search_query



class PDFLateralSearcher:
    def __init__(self, api_key:str, query:Optional[str], pdf_file:Optional[str], debug:bool=False):
        self.api_key = api_key
        self.query = query
        self.pdf_file = pdf_file
        self.debug = debug
        
        
    async def find_relevant_pdfs(self,) -> dict:
        if self.query is None and self.pdf_file is None:
            print("Please input a query or file")
            return ""
        else:
            if self.query:
                self.pdf_file = None
            elif self.pdf_file:
                self.query = None
        
        #converts pdf to markdown
        if self.pdf_file is not None and isinstance(self.pdf_file, str):
            markdown_pdf_text = pdf_to_markdown(pdf_path=self.pdf_file)
            #generate description of good search
            search_description = genearate_search_query(markdown_pdf_text)
            ##finds pdfs based on file input
            return PDF_Finder(query=search_description,debug=self.debug).combined_pdf_paper_finder_ranked()
        #send to pdf finder
        elif self.pdf_file is None and self.query is not None: #finds pdfs based on query 
            return PDF_Finder(query=self.query,debug=self.debug).combined_pdf_paper_finder_ranked()

    async def summarize_found_pdfs(self,found_pdfs):
        pdf_summarizer = AdjacentSummarizer(api_key=self.api_key, input_data=found_pdfs, debug=self.debug)        
        pdf_finder_data = await pdf_summarizer.batch_paper_summarizer()

        return pdf_finder_data
if __name__ == "__main__":
    """
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    pdf_file = "C:/Users/qmerr/Downloads/New_youtube_summarizer/Attention.pdf"
    query = "transformers for machine learning and AI models"
    
    asyncio.run(find_relevant_pdfs(summarize_found_pdfs())
    """