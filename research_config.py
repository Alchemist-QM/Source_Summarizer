from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional

@dataclass 
class ResearchConfig:
    """
    Configuration for the research pipeline
    
    This class controls every aspect of our research assistant:
    
    - How many URLs to discover and crawl
    - Scoring methods to use
    - Whether to use AI enhancement
    - Output preferences
    
    """
    # Main settings
    domain: str = None
    max_urls_discovery: int = 100 #initial wide net 
    max_urls_to_crawl: int = 10 #crawls top 10
    top_k_urls: int = 10 #focus on top results
    browser_type: str= "undetected"
    #scoring and filtering
    score_threshold: float = 0.3 #min relevance score
    scoring_method: str = "bm25"
    
    use_llm_enhancement: bool = True #enhance queries with AI
    llm_model:str = "openai/gpt-4o-mini"
    
    #url discovery options
    extract_head_metadata: bool = False #get titles and description
    live_check: bool= True  #verify urls are accessible 
    force_refresh: bool = True #bypass cahce
    
    #crawler settings
    max_concurrent_crawls: int = 5 #parallel crawling
    timeout: int = 30000 #30 second time out
    headless: bool = True #no browser window
    
    #output settings
    output_dir: Path = Path("research_results")
    verbose: bool = True 
    
@dataclass
class ResearchQuery: 
    """
    Container for research query and metadat
    """
    orignal_query: str
    enhanced_query: Optional[str] = None
    search_patterns: List[str] = None
    timestamp: str = None
    
@dataclass
class ResearchResult:
    """
    Container for research results
    """
    query: ResearchQuery
    discovered_urls: List[Dict]  
    crawled_content: List[Dict] 
    synthesis: str
    citations: List[Dict]
    metadata: Dict