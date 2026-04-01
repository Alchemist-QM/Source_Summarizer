import asyncio
import json
import os
import logging
import numpy as np
import requests
from typing import List, Dict, Any,Union, Optional
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from datetime import datetime
import glob
from dotenv import load_dotenv
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
os.environ["OMP_NUM_THREADS"] = "1"

# ==== CONFIG ====
ACADEMIC_SUMMARY_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""
You are an academic research assistant. Given the following paper excerpt (abstract, introduction, or main content), extract and organize the information below in clearly labeled sections.

For each section, use concise bullet points and preserve key technical terms from the original text when possible. If a section is not present, state "Not mentioned".

Format your answer strictly using these section headers:

Main Contributions:
- ...

Methodology:
- ...

Results / Findings:
- ...

Limitations / Future Directions:
- ...

Key Technical Terms Introduced:
- ...

---
{text}
"""
)
UPDATED_ACADEMIC_SUMMARY_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""
System Role:
You are a specialized academic information extraction engine designed for processing web-scraped research papers. Your purpose is to extract structured scholarly knowledge from raw academic text with high precision and consistency.

Task:
Given an excerpt from an academic paper (which may include abstract, introduction, or main body text), extract and organize the information into clearly defined analytical sections.

You may apply light academic inference when information is implied but not explicitly labeled (e.g., inferring methodology from described experiments.)

Extraction Rules & Constraints

Preserve original technical terminology and formal academic language whenever possible.

If a section cannot be identified, output exactly: Not mentioned.

Be concise and factual.

Avoid commentary, explanations, or meta text.


Extraction Guidelines:

Main Contributions:
Novel ideas, models, theories, frameworks, or claims of originality.

Methodology:
Data sources, experimental design, datasets, algorithms, models, evaluation procedures, or analytical techniques.

Results / Findings:
Quantitative or qualitative outcomes, performance metrics, discoveries, or evaluations.

Limitations / Future Directions:
Explicit weaknesses, constraints, assumptions, or proposed future work.

Key Technical Terms Introduced:
Named methods, acronyms, theoretical concepts, or domain-specific terminology introduced or emphasized in the text.

OUTPUT FORMAT:
Format your answer strictly using these section headers:

Main Contributions:
- …

Methodology:
- …

Results / Findings:
- …

Limitations / Future Directions:
- …

Key Technical Terms Introduced:

- …
{text}
"""
)

STUDENT_SUMMARY_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""
You are an Academic Research Assistant for Students. Your job is to extract structured, study-ready information from academic text while preserving technical accuracy and avoiding interpretation or speculation.

Goal:
Convert the provided academic excerpt into a structured, study-ready summary optimized for:
- First-time topic learning
- Fast exam revision
- Clear concept understanding
Core Rules:

Use concise bullet points (moderately detailed: 1–2 sentences per bullet).
Preserve key technical terms and terminology exactly as written in the source text.

Do not infer, summarize broadly, or add external knowledge.

Maintain academic tone.

If a section is not present in the text, write: “Not mentioned.”

Each bullet point must express one distinct idea.

Do not include commentary, explanations, or opinions.

OUTPUT FORMAT:
Format your answer strictly using these section headers:

Extraction Guidelines:

80/20 SMART SKIMMING PRIORITIES:
Extract only information that contributes most to understanding:
- Main purpose or problem addressed
- Essential technical terminology
- Important relationships (cause–effect, mechanisms, dependencies)
- Primary conclusions

Main Topic Review:

- ...

Key Technical Terms:
- ...

Important Relationships:
-...

Conclusion: 
-...

Avoid vague phrases such as “the paper discusses…” or “the authors say…”.

State facts directly as they appear in the text.
…
{text}
"""
)

THEME_PROMPT = PromptTemplate(
    input_variables=["summaries"],
    template="""You are a research analyst. Given the following list of paper summaries, cluster them into key research themes or topics. For each theme, provide:
- Theme label
- The papers that belong to this theme (by title)
- A brief description of the theme

Summaries:
{summaries}
"""
)
def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        else:
            return obj

class AdjacentSummarizer:
    def __init__(
        self,  
        api_key:str, 
        input_data: Union[str, dict],
        top_k: int = 10,
        model:str = "gpt-4o-mini", 
        debug:bool = False,
        ):
        self.top_k = top_k
        self.pdf_folder:str = "./pdfs"
        self.model = model
        self.api_key = api_key
        self.llm = ChatOpenAI(
            model=self.model, 
            temperature=0.2, 
            api_key=self.api_key
            )
        self.debug: Union[True | False] = debug
        self.n_clusters: int = 3
        self.input_file = "debug.json" if self.debug else input_data
        self.base_output_filename : str = "batch_paper_summaries"

        
    def improved_download_pdf(self,pdf_url,filename_hint=None, timeout=20) -> str:
        os.makedirs(self.pdf_folder, exist_ok=True)
        base_name = os.path.basename(pdf_url.split("?")[0])
        if filename_hint:
            base_name = filename_hint
        local_path = os.path.join(self.pdf_folder, base_name)
        try:
            r = requests.get(pdf_url, timeout=timeout)
            if r.status_code == 200:
                with open(local_path, "wb") as f:
                    f.write(r.content)
                with open(local_path, "rb") as f:
                    header = f.read(5)
                    if header == b"%PDF-":
                        logging.info(f"[PDF Download] Success: {pdf_url} -> {local_path}")
                        return local_path
                    else:
                        logging.warning(f"[PDF Download] File from {pdf_url} is not a valid PDF (header: {header})")
                        os.remove(local_path)
                        return None
            else:
                logging.warning(f"[PDF Download] Bad status for {pdf_url}: {r.status_code}")
        except requests.exceptions.Timeout:
            logging.error(f"[PDF Download] Timeout for {pdf_url}")
        except Exception as e:
            logging.error(f"[PDF Download] Error for {pdf_url}: {e}")
        return None


    def safe_content(self,obj: Any) -> str:
        return obj.content if hasattr(obj, "content") else str(obj)

    def clean_text(self,text: str) -> str:
        if not text:
            return ""
        return text.strip()

    
    async def summarize_single_pdf(self, pdf_path: str, summary_prompt: PromptTemplate) -> Dict:
        try:
            loader = PyMuPDFLoader(pdf_path)
            docs = loader.load()
            main_text = "\n\n".join([doc.page_content for doc in docs[:10]])
            summary_chain = summary_prompt | self.llm
            summary = await summary_chain.ainvoke({"text": main_text})
            return {"file": os.path.basename(pdf_path), "summary": self.safe_content(summary)}
        except Exception as e:
            logging.error(f"Error summarizing {pdf_path}: {e}")
            return {"file": os.path.basename(pdf_path), "summary": ""}

    def cluster_summaries(self,summaries: List[str]) -> List[Dict]:
        if not summaries:
            raise ValueError("No summaries provided for clustering.")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        X = model.encode(summaries)
        k = min(self.n_clusters, len(summaries))
        if k < 1:
            raise ValueError("Need at least one summary for clustering.")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        clusters: Dict[int, List[int]] = {}
        for idx, label in enumerate(labels):
            clusters.setdefault(int(label), []).append(int(idx))
        theme_outputs = []
        for label, indices in clusters.items():
            theme_texts = [summaries[i] for i in indices]
            theme_outputs.append({
                "theme_id": int(label),
                "paper_indices": [int(i) for i in indices],
                "summaries": [summaries[i] for i in indices],
            })
        return theme_outputs


    async def batch_paper_summarizer(
        self,
    )->dict[str, any]:
        def normalize_path(p):
            return os.path.abspath(os.path.normpath(p))

        def extract_papers(finder_json):
            if "papers_ranked" in finder_json:
                return finder_json["papers_ranked"]
            if "papers" in finder_json:
                return finder_json["papers"]
            raise KeyError(f"No papers found. Keys: {finder_json.keys()}")

        # ---------------- Load JSON ----------------
        finder_json = self.input_file if isinstance(self.input_file, dict) else json.load(open(self.input_file))
        print("[DEBUG] Finder JSON keys:", finder_json.keys())

        papers = extract_papers(finder_json)[:self.top_k]
        query = finder_json.get("query", "unknown")

        # ---------------- Collect PDF paths ----------------
        pdf_paths = []

        for paper in papers:
            local_pdf = paper.get("urls", {}).get("local_pdf")
            if local_pdf and os.path.exists(local_pdf):
                pdf_paths.append(normalize_path(local_pdf))

        # ✅ FALLBACK: load PDFs from folder directly
        if not pdf_paths:
            print("[WARN] No local_pdf paths found in JSON. Loading PDFs from folder...")
            pdf_paths = glob.glob(os.path.join(self.pdf_folder, "*.pdf"))

        if not pdf_paths:
            raise ValueError(f"No PDFs found in folder: {self.pdf_folder}")

        print(f"[INFO] Found {len(pdf_paths)} PDFs")


        # ---------------- Summarize PDFs ----------------
        async def summarize_pdf_path(pdf_path):
            pdf_path = normalize_path(pdf_path)
            result = await self.summarize_single_pdf(pdf_path,UPDATED_ACADEMIC_SUMMARY_PROMPT)
            return {
                "file": pdf_path,
                "summary": self.clean_text(result["summary"])
            }

        tasks = [summarize_pdf_path(p) for p in pdf_paths]
        summaries = [s for s in await asyncio.gather(*tasks) if s and s["summary"]]

        if not summaries:
            raise ValueError("No valid PDF summaries generated.")

        summary_texts = [s["summary"] for s in summaries]

        # ---------------- Clustering ----------------
        clustered = self.cluster_summaries(summary_texts,)

        # ---------------- Theme Summaries ----------------
        async def generate_theme_summary(theme):
            joined = "\n\n".join(theme["summaries"])
            prompt = PromptTemplate(
                input_variables=["text"],
                template="Summarize the main research theme from these paper summaries:\n{text}"
            )
            chain = prompt | self.llm
            result = await chain.ainvoke({"text": joined})
            return self.safe_content(result)

        for theme in clustered:
            theme["papers"] = [summaries[i] for i in theme["paper_indices"]]
            theme["theme_summary"] = await generate_theme_summary(theme)

        output_json = {
            "query": query,
            "generated_at": datetime.utcnow().isoformat(),
            "themes": clustered,
            "papers": summaries
        }

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        json_path = f"outputs/{self.base_output_filename}_{timestamp}.json"
        md_path = f"outputs/{self.base_output_filename}_{timestamp}.md"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(convert_numpy(output_json), f, indent=2)

        # ---------------- Markdown ----------------
        md_lines = [
            "# Batch Paper Summary",
            f"**Query:** {query}",
            f"**Generated:** {output_json['generated_at']}\n"
        ]

        for theme in clustered:
            md_lines.append(f"\n## Theme {theme['theme_id']}")
            md_lines.append(f"**Theme Summary:** {theme['theme_summary']}\n")

            for paper in theme["papers"]:
                #md_lines.append(f"### {paper['file']}")
                md_lines.append(f"\n**Summary:**\n{paper['summary']}\n")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

        print(f"[INFO] Saved JSON -> {json_path}")
        print(f"[INFO] Saved Markdown -> {md_path}")

        return output_json

#02/2026 - This module processes the output from the PDF finder, summarizes each PDF using an LLM, clusters the summaries into themes, and saves the results in a structured JSON format. It includes robust error handling for PDF downloading and summarization, and is designed to be run as a standalone script or imported as a module.
#This is how it's run
"""
if __name__ == "__main__":
    import sys
    load_dotenv()
    #openai_key = os.getenv("OPENAI_API_KEY")
    api_key = os.getenv("OPENAI_API_KEY")
    #if pdf_finder
    ##if debug == True 
        #pdf_finder_input = "debug_papers.json" ### ->debug file
        #with open(pdf_finder_input, "r", encoding="utf-8") as f:
            #pdf_finder_data = json.load(f)
    
    
    pdf_collector = AdjacentSummarizer(api_key)
    asyncio.run(
        pdf_collector.batch_paper_summarizer(
        )
    )
"""  