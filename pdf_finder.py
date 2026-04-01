import asyncio
import aiohttp
import json
import os
from datetime import datetime
import requests
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
from crawl4ai import (
    AsyncWebCrawler,
    CrawlerRunConfig,
    CacheMode,
    JsonCssExtractionStrategy,
    BrowserConfig
)

from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from sentence_transformers import SentenceTransformer, util
import numpy as np




browser_config = BrowserConfig(
    browser_type="undetected",
    headless=True,
    extra_args=[
        "--disable-blink-features=AutomationControlled",
        "--disable-web-security"
    ]
)

PDF_METADATA_SCHEMA = {
    "name": "pdf_metadata",
    "baseSelector": "body",
    "fields": [
        {"name": "title", "selector": "h1, h2.title, meta[name='citation_title']", "type": "text"},
        {"name": "authors", "selector": ".authors, .author, meta[name='citation_author']", "type": "text"},
        {"name": "abstract", "selector": "#abstract, .abstract, meta[name='description'], meta[name='citation_abstract']", "type": "text"},
        {"name": "date", "selector": "meta[name='citation_publication_date'], meta[name='citation_date']", "type": "text"},
        {"name": "pdf_link", "selector": "a[href$='.pdf'], meta[name='citation_pdf_url']", "type": "attribute", "attribute": "href"},
        {"name": "url", "selector": "meta[property='og:url']", "type": "attribute", "attribute": "content"}
    ],
}
class Paper(BaseModel):
        title: str
        url: str
        pdf_link: str
        authors: str = ""
        abstract: str = ""
        date: str = ""
        local_pdf: str = ""
        snippet: str = ""
        source: str = ""
        citations: int = 0
        final_score: float = 0.0
        semantic_score: float = 0.0
        recency_score: float = 0.0
        impact_score: float = 0.0

def save_json(data, filename:str):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved output to {filename}")

def clean_pdf_url(href: str | None) -> str | None:
    if not href:
        return None
    if href.startswith("/url?"):
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(href)
        qs = parse_qs(parsed.query)
        url = qs.get("q", [href])[0]
        if url.endswith(".pdf"):
            return url
        return None
    if href.endswith(".pdf"):
        return href
    return None

class PDF_Finder:
    def __init__(self,query:str, max_results:int=8, debug:bool=False):
        self.query = query
        self.max_results = max_results
        self.debug = debug
        self.output_json: Optional[str] = "debug.json" if self.debug == True else None
        
    def google_search_url(self, gl: str = "us", hl: str = "en") -> str:
        return f"https://www.google.com/search?q={self.query}+filetype:pdf&num={self.max_results}&gl={gl}&hl={hl}"

    async def fetch_serp_pdf(self,):
        url = self.google_search_url()
        run_conf = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
        async with AsyncWebCrawler(config=browser_config) as crawler:
            static_strategy = JsonCssExtractionStrategy({
                "name": "pdf_serp",
                "baseSelector": "div.g, div.MjjYud, div.xpd",
                "fields": [
                    {"name": "title", "selector": "h3", "type": "text"},
                    {"name": "url", "selector": "a", "type": "attribute", "attribute": "href"},
                    {"name": "snippet", "selector": "div.VwiC3b", "type": "text"},
                ],
            })
            result = await crawler.arun(url=url, config=run_conf.clone(extraction_strategy=static_strategy))
            data = []
            if result.success:
                try:
                    data = json.loads(result.extracted_content or "[]")
                except Exception:
                    data = []
            if isinstance(data, dict):
                data = data.get("items") or data.get("rows") or []
            items = []
            for it in data:
                title = it.get("title")
                link = clean_pdf_url(it.get("url"))
                snippet = it.get("snippet")
                if link:
                    items.append({
                        "title": title,
                        "url": link,
                        "snippet": snippet,
                    })
            return {"query": self.query, "url": url, "results": items}

    async def crawl_pdf_metadata(self,urls: List[str], verbose: bool = True):
        cfg = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
        results = []
        async with AsyncWebCrawler(config=browser_config) as crawler:
            for url in urls:
                try:
                    result = await crawler.arun(
                        url=url,
                        config=cfg.clone(extraction_strategy=JsonCssExtractionStrategy(PDF_METADATA_SCHEMA))
                    )
                    if result.success:
                        data = json.loads(result.extracted_content or "[]")
                        if isinstance(data, list) and data:
                            meta = data[0]
                        elif isinstance(data, dict):
                            meta = data
                        else:
                            meta = {}
                        meta["pdf_link"] = meta.get("pdf_link") or url
                        meta["url"] = url
                        results.append(meta)
                    else:
                        results.append({"url": url, "error": result.error_message})
                except Exception as e:
                    if verbose:
                        print(f"[CRAWL] Error for {url}: {e}")
                    results.append({"url": url, "error": str(e)})
        return results

   
    # --- ArXiv, Semantic Scholar, PubMed ---
    async def fetch_arxiv(self, ) -> List[Dict]:
        base_url = "http://export.arxiv.org/api/query"
        params = f"search_query=all:{quote_plus(self.query)}&start=0&max_results={self.max_results}"
        url = f"{base_url}?{params}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                xml = await resp.text()
        soup = BeautifulSoup(xml, "xml")
        papers = []
        for entry in soup.find_all("entry"):
            pdf_link = ""
            for l in entry.find_all("link"):
                if l.get("type") == "application/pdf":
                    pdf_link = l.get("href")
            papers.append({
                "title": entry.title.text.strip() if entry.title else "",
                "authors": [a.text for a in entry.find_all("author")],
                "abstract": entry.summary.text.strip() if entry.summary else "",
                "date": entry.published.text if entry.published else "",
                "url": entry.id.text if entry.id else "",
                "pdf": pdf_link,
                "source": "arxiv"
            })
        return papers

    async def fetch_pubmed(self,) -> List[Dict]:
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_url = f"{base_url}?db=pubmed&retmax={self.max_results}&term={quote_plus(self.query)}&retmode=json"
        async with aiohttp.ClientSession() as session:
            async with session.get(search_url) as resp:
                res = await resp.json()
        ids = res.get("esearchresult", {}).get("idlist", [])
        fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = f"?db=pubmed&id={','.join(ids)}&retmode=xml"
        url = fetch_url + params
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                xml = await resp.text()
        soup = BeautifulSoup(xml, "xml")
        papers = []
        for article in soup.find_all("PubmedArticle"):
            art = article.Article
            title = art.ArticleTitle.text if art and art.ArticleTitle else ""
            abstract = art.Abstract.AbstractText.text if art and art.Abstract and art.Abstract.AbstractText else ""
            year = ""
            if art and art.Journal and art.Journal.JournalIssue and art.Journal.JournalIssue.PubDate and art.Journal.JournalIssue.PubDate.Year:
                year = art.Journal.JournalIssue.PubDate.Year.text
            authors = []
            for author in art.AuthorList.find_all("Author"):
                last = author.LastName.text if author.LastName else ""
                first = author.ForeName.text if author.ForeName else ""
                authors.append(f"{first} {last}".strip())
            pmid = article.MedlineCitation.PMID.text if article.MedlineCitation and article.MedlineCitation.PMID else ""
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
            pdf_link = ""  # PubMed usually doesn't host PDFs
            papers.append({
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "date": year,
                "url": url,
                "pdf": pdf_link,
                "source": "pubmed",
                "citations": 0  # PubMed doesn't expose citation count
            })
        return papers

    # --- Util to download PDFs ---
    async def download_pdf(self,pdf_url: str, dest_folder: str = "pdfs") -> str:
        os.makedirs(dest_folder, exist_ok=True)
        filename = os.path.join(dest_folder, pdf_url.split("/")[-1].split("?")[0])
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(pdf_url) as resp:
                    if resp.status == 200 and "application/pdf" in resp.headers.get("Content-Type", ""):
                        with open(filename, "wb") as f:
                            f.write(await resp.read())
                        return filename
        except Exception as e:
            print(f"[PDF Download] Error for {pdf_url}: {e}")
        return ""

    def download_pdf_sync(self, pdf_url: str, dest_folder: str = "pdfs") -> str:
        os.makedirs(dest_folder, exist_ok=True)
        filename = os.path.join(dest_folder, pdf_url.split("/")[-1].split("?")[0])
        try:
            r = requests.get(pdf_url, timeout=15)
            if r.status_code == 200 and r.headers.get("Content-Type", "").startswith("application/pdf"):
                with open(filename, "wb") as f:
                    f.write(r.content)
                return filename
        except Exception as e:
            print(f"[PDF Sync Download] Error for {pdf_url}: {e}")
        return ""

    
    def normalize_date(self,date_str: str) -> float:
        if not date_str:
            return 0.5
        try:
            year = int(date_str[:4])
        except Exception:
            try:
                dt = datetime.fromisoformat(date_str.replace("Z", ""))
                year = dt.year
            except Exception:
                return 0.5
        current_year = datetime.now().year
        delta = current_year - year
        if delta <= 1:
            return 1.0
        elif delta >= 10:
            return 0.0
        else:
            return max(0.0, 1.0 - np.log1p(delta)/np.log1p(10))

    def normalize_citations(self,citations: int) -> float:
        if citations is None or citations <= 0:
            return 0.0
        return min(1.0, np.log1p(citations)/np.log1p(5000))

    async def multi_factor_rank(self, papers: List[Paper], alpha=0.65, beta=0.20, gamma=0.15):
        model = SentenceTransformer("all-MiniLM-L6-v2")
        query_emb = model.encode(self.query, convert_to_tensor=True)
        abstracts = [(p.abstract or p.title) for p in papers]
        paper_embs = model.encode(abstracts, convert_to_tensor=True)
        semantic_scores = util.cos_sim(query_emb, paper_embs)[0].cpu().numpy()
        for i, paper in enumerate(papers):
            paper.semantic_score = float(semantic_scores[i])
            paper.recency_score = self.normalize_date(paper.date)
            paper.impact_score = self.normalize_citations(paper.citations)
            paper.final_score = alpha*paper.semantic_score + beta*paper.recency_score + gamma*paper.impact_score
        return sorted(papers, key=lambda p: p.final_score, reverse=True)

    async def combined_pdf_paper_finder_ranked(self,download_pdfs: bool = True, pdfs_folder: str = "pdfs")->dict[str, Any]:
        google_serp_task = self.fetch_serp_pdf()
        arxiv_task = self.fetch_arxiv()
        #semantic_scholar_task = self.fetch_semantic_scholar(query, max_results=max_results_per_source)
        pubmed_task = self.fetch_pubmed()

        serp_result, arxiv_papers, pubmed_papers = await asyncio.gather(
            google_serp_task, arxiv_task,pubmed_task
        )

        google_pdf_urls = [r["url"] for r in serp_result["results"] if r.get("url")]
        google_metadata = await self.crawl_pdf_metadata(google_pdf_urls)

        papers = []

        # Google papers and PDF download
        for meta, serp_item in zip(google_metadata, serp_result["results"]):
            pdf_link = meta.get("pdf_link") or meta.get("url") or serp_item.get("url")
            local_pdf = ""
            if download_pdfs and pdf_link and pdf_link.endswith(".pdf"):
                local_pdf = await self.download_pdf(pdf_link, dest_folder=pdfs_folder)
            citations = 0
            paper = Paper(
                title=meta.get("title") or serp_item.get("title") or "",
                url=meta.get("url") or serp_item.get("url") or "",
                pdf_link=pdf_link or "",
                authors=meta.get("authors") or "",
                abstract=meta.get("abstract") or "",
                date=meta.get("date") or "",
                local_pdf=local_pdf or "",
                snippet=serp_item.get("snippet") or "",
                source="google",
                citations=citations
            )
            papers.append(paper)

        async def enrich_paper(paper):
            pdf_link = paper.get("pdf") or paper.get("pdf_link") or ""
            local_pdf = ""
            if download_pdfs and pdf_link and pdf_link.endswith(".pdf"):
                local_pdf = await self.download_pdf(pdf_link, dest_folder=pdfs_folder)
            authors = paper.get("authors", "")
            if isinstance(authors, list):
                authors = ", ".join(a or "" for a in authors)
            citations = paper.get("citations", 0) or 0
            return Paper(
                title=paper.get("title", "") or "",
                url=paper.get("url", "") or "",
                pdf_link=pdf_link or "",
                authors=authors or "",
                abstract=paper.get("abstract", "") or "",
                date=paper.get("date", "") or "",
                local_pdf=local_pdf or "",
                source=paper.get("source", "") or "",
                citations=citations
            )

        all_other_papers = arxiv_papers + pubmed_papers
        enriched_other_papers = await asyncio.gather(*(enrich_paper(p) for p in all_other_papers))
        papers.extend(enriched_other_papers)

        ranked_papers = await self.multi_factor_rank(papers)

        output_data = {
            "query": self.query,
            "papers_ranked": [p.model_dump() for p in ranked_papers],
            "google_serp": serp_result,
            "google_metadata": google_metadata,
            "arxiv_papers": arxiv_papers,
            "pubmed_papers": pubmed_papers,
        }
        if self.debug:
            save_json(output_data, filename=self.output_json)
            self.save_results_markdown_json(output_data, base_filename="research_results")
        return output_data

    def save_results_markdown_json(self,results: dict, base_filename="research_output"):
        os.makedirs("outputs", exist_ok=True)
        timestamp = datetime.now().isoformat()

        cleaned_papers = []
        all_urls = {}
        url_counter = 1
        seen_urls = set()

        for idx, paper in enumerate(results.get("papers_ranked", []), 1):
            landing = paper.get("url", "")
            pdf = paper.get("pdf_link", "")
            local_pdf = paper.get("local_pdf", "")

            paper_urls = {}

            def register_url(url, url_type):
                nonlocal url_counter
                if url and url not in seen_urls:
                    key = f"url_{url_counter}"
                    all_urls[key] = {
                        "type": url_type,
                        "paper_index": idx,
                        "title": paper.get("title", ""),
                        "url": url
                    }
                    seen_urls.add(url)
                    url_counter += 1
                return url

            paper_urls["landing_page"] = register_url(landing, "landing_page")
            paper_urls["pdf"] = register_url(pdf, "pdf")
            paper_urls["local_pdf"] = register_url(local_pdf, "local_file")

            cleaned_papers.append({
                "paper_id": f"paper_{idx}",
                "title": paper.get("title", ""),
                "authors": paper.get("authors", ""),
                "abstract": paper.get("abstract", ""),
                "date": paper.get("date", ""),
                "source": paper.get("source", ""),
                "citations": paper.get("citations", 0),
                "scores": {
                    "final_score": round(paper.get("final_score", 0), 4),
                    "semantic_score": round(paper.get("semantic_score", 0), 4),
                    "recency_score": round(paper.get("recency_score", 0), 4),
                    "impact_score": round(paper.get("impact_score", 0), 4),
                },
                "urls": paper_urls
            })

        # -------- JSON OUTPUT --------
        clean_json = {
            "query": results.get("query"),
            "generated_at": timestamp,
            "papers": cleaned_papers,
            "all_urls": all_urls
        }

        json_path = f"outputs/{base_filename}.json"
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(clean_json, f, indent=2, ensure_ascii=False)

        # -------- MARKDOWN OUTPUT --------
        md_lines = []
        md_lines.append("# 📚 Research Results\n")
        md_lines.append(f"**Query:** {results.get('query')}  ")
        md_lines.append(f"**Generated:** {timestamp}\n")
        md_lines.append("---\n")

        for idx, paper in enumerate(cleaned_papers, 1):
            md_lines.append(f"## {idx}. {paper['title']}\n")
            md_lines.append(f"**Authors:** {paper['authors']}  ")
            md_lines.append(f"**Year:** {paper['date']}  ")
            md_lines.append(f"**Source:** {paper['source']}  ")
            md_lines.append(f"**Citations:** {paper['citations']}  ")
            md_lines.append(f"**Relevance Score:** {paper['scores']['final_score']}\n")

            urls = paper["urls"]
            if urls.get("landing_page"):
                md_lines.append(f"- 🌐 URL: [{urls['landing_page']}]({urls['landing_page']})")
            if urls.get("pdf"):
                md_lines.append(f"- 📄 PDF: [{urls['pdf']}]({urls['pdf']})")
            if urls.get("local_pdf"):
                md_lines.append(f"- 💾 Local PDF: `{urls['local_pdf']}`")

            abstract = paper["abstract"].replace("\n", " ")
            md_lines.append("\n**Abstract:**\n")
            md_lines.append(f"> {abstract}\n")
            md_lines.append("---\n")

        md_path = f"outputs/{base_filename}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

        print(f"[INFO] Saved JSON → {json_path}")
        print(f"[INFO] Saved Markdown → {md_path}")

        return {"json": json_path, "markdown": md_path}
    
#this is how it's run 02/2026 - This module can be run as a standalone script to find and rank relevant PDFs based on a query. It uses multiple sources (Google, ArXiv, Semantic Scholar, PubMed), extracts metadata, downloads PDFs, and ranks them using a multi-factor approach. The results are saved in a structured JSON file for further processing or analysis
"""
if __name__ == "__main__":
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    query = sys.argv[1] if len(sys.argv) > 1 else "mamba architecture for machine learning and AI models"
    pdf_finder = PDF_Finder(query)
    asyncio.run(pdf_finder.combined_pdf_paper_finder_ranked())
    #print(json.dumps(final_results, indent=2, ensure_ascii=False))
"""