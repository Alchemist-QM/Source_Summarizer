import pymupdf4llm
import pathlib



def pdf_to_markdown(pdf_path:str, md_path="converted_markdown.md"):
        """
        Converts a PDF file to Markdown using pymupdf4llm.
        Args:
            pdf_path (str or pathlib.Path): Path to the PDF file.
            md_path (str or pathlib.Path, optional): Path to save the Markdown file. If None, returns markdown string.
        Returns:
            str: Markdown content if md_path is not given, else None.
        """
        pdf_path = pdf_path
        doc_markdown = pymupdf4llm.to_markdown(pdf_path)
        
        if md_path:
            pathlib.Path(md_path).write_bytes(doc_markdown.encode())
            print(f"Markdown saved to {md_path}")
        
        
        return doc_markdown
