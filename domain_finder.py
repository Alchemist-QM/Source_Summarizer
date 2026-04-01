import litellm
import json 
from datetime import datetime
from dataclasses import asdict
from pathlib import Path 
from typing import List,Dict, Tuple
from research_config import *
from crawl4ai import AsyncUrlSeeder,BrowserConfig,AsyncWebCrawler, SeedingConfig, DefaultMarkdownGenerator, PruningContentFilter, CrawlerRunConfig

class DomainResearchAssistant:
    def __init__(
        self,
        query:str,
        domain:str | None = None,
        api_key:str | None = None,

        ):
        self.api_key = api_key
        self.query = query
        self.domain = domain
        
        
    async def enhanced_query_with_llm(self, config: ResearchConfig)->ResearchQuery:
        """
        Transform simple queries into comprehensice search strategies
        
        

        Args:
            query (str): _description_
            config (ResearchConfig): _description_

        Returns:
            ResearchQuery: _description_
        """
        try:
            response = await litellm.acompletion(
                model=config.llm_model,
                messages=[{
                    "role": "user",
                    "content": 
                    f"""
                    Given this research query: "{self.query}"
                    
                    Extract:
                    1. Related search terms
                    2. A more speicified/enhanced version of the query
                    3. Key terms and concepts
                    Return as JSON:
                    {{
                        "key_terms": ["term1", "term2"],
                        "related_terms": ["related1", "related2"],
                        "enhanced_query": "enhanced version of query"
                    }}
                    """
                }],
                temperature=0.3,
                response_format={"type": "json_object"}, 
                )
            data = json.loads(response.choices[0].message.content)
            
            all_terms = data['key_terms'] + data['related_terms']
            patterns = [f"*{term.lower()}*" for term in all_terms]
            result = ResearchQuery(
                orignal_query=self.query,
                enhanced_query=data['enhanced_query'],
                search_patterns=patterns, #limit to 10 patterns; patterns[:10]
                timestamp=datetime.now().isoformat()
            )
            return result
        except Exception as e:
            words = self.query.lower().split()
            patterns = [f"*{word}*" for word in words if len(word)>2 ]


            return ResearchQuery(
                orignal_query=self.query,
                enhanced_query=self.query,
                search_patterns=patterns,
                timestamp=datetime.now().isoformat()
            )
        

    async def discover_urls(
        self,
        query: ResearchQuery, 
        config: ResearchConfig,
    )-> List[Dict]:
        """
        Discover and rank URLS without crawling them
        
        The URL Seeder is incredibbly powerful because it:
        1. Get URLs from sitemaps (official site maps)
        2. Gets URLs from Common Crawl (web-scale data)
        3. Extracts metadata without full page loads
        4. Scores relevance using BM25 algorithm

        Args:
            domain (str): _description_
            query (ResearchQuery): _description_

        Returns:
            List[Dict]: _description_
            
        """
        async with AsyncUrlSeeder() as seeder:
            seeding_cong = SeedingConfig(
                source="sitemap+cc",#uses sitemap and common crawl 
                extract_head=config.extract_head_metadata,
                query=query.enhanced_query or query.orignal_query,
                #Relevance scoring
                scoring_method=config.scoring_method, #BM354 scoring
                score_threshold=config.score_threshold, #minimum scoring
                #limits and performance
                max_urls=config.max_urls_discovery,
                live_check=config.live_check,
                force=config.force_refresh, #bypass cahce if needed
            )
            try:
                
                urls= await seeder.urls(self.domain, seeding_cong)
                #sorts urls by relevance 
                top_urls = urls[:config.top_k_urls]
                
                return top_urls
            except Exception as e:
                print(f"Url discovery failed: {e}[/red]")
                return []                 
        
    
    async def crawl_selected_urls(
        self,
        urls: List[Dict],
        query: ResearchQuery,
        config: ResearchConfig,
        
    )-> List[Dict]:
        """
        Crawls the most relevant URLS with start content filtering

        Args:
            urls (List[Dict]): _description_
            query (ResearchQuery): _description_
            config (ResearchConfig): _description_
        """
        url_list = [u['url'] for u in urls if 'url' in u][:config.max_urls_to_crawl]

        if not url_list:
            print("No urls to crawl")
            return []
        
        md_generator = DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(
                threshold=0.48, #content relevance threshold
                threshold_type="dynamic", #adapts to page structure
                min_word_threshold=10,
            ),
        )
        crawler_config = CrawlerRunConfig(
            markdown_generator=md_generator,
            exclude_external_links=True, #focus on content, not links
            excluded_tags=['nav','header', 'footer','aside'],
        )
        async with AsyncWebCrawler(
            config=BrowserConfig(
                headless=config.headless,
                verbose=config.verbose,
                browser_type=config.browser_type,
                extra_args=[
            "--disable-blink-features=AutomationControlled",
            "--disable-web-security"
        ]
            )
        ) as crawler:
            results = await crawler.arun_many(
                url_list,
                config=crawler_config,
                max_concurrent=config.max_concurrent_crawls
            )
            crawled_content = []
            for url, result in zip(url_list,results):
                if result.success:
                    content_data = {
                        "url": url,
                        "title": result.metadata.get('title', 'No title'),
                        'markdown': result.markdown.fit_markdown or result.markdown.raw_markdown,
                        "metadata": result.metadata,
                    }
                    crawled_content.append(content_data)
                else:
                    print(f"Failed: {url[:50]}... -{result.error} ")
        return crawled_content


    async def generate_research_synthesis(
        self,
        query: ResearchQuery, 
        crawled_content: List[Dict],
        config: ResearchConfig,
    )-> Tuple[str, List[Dict]]:
        """
        Use AI to synthesize findings from multiple sources

        Args:
            query (ResearchQuery): _description_
            crawled_content (List[Dict]): _description_
            config (ResearchConfig): _description_

        Returns:
            Tuple[str, List[Dict]]: _description_
        """
        if not crawled_content:
            return "No content avaiable for synthesis", []
        content_sections = []
        for i, content in enumerate(crawled_content, 1):
            section = f"""
            SOURCE {i}:
            Title: {content['title']}
            URL: {content['url']}
            Content Preview:
            <content>
            {content['markdown'][:1500]}
            </content>
            """
            content_sections.append(section)
        combined_content = "\n--=\n".join(content_sections)
        try:
            response = await litellm.acompletion(
                model=config.llm_model, 
                messages=[{
                    "role": "user",
                    "content": f"""Research Query: "{query.orignal_query}"
                    Based on the following sources, provide a comprehensive research synthesis.
                    {combined_content}
                    Provide:
                    1. an execture summary (2-3 sentences)
                    2. Key findings (3-5 bullet points)
                    3. Detailed analysis (2-3 paragraphs)
                    4. Future implications
                    
                    Format your research with clear sections and cite sources [Source N]
                    Keep total response under 800 words."""
                }],
                temperature=0.7
            )
            synthesis= response.choices[0].message.content
            citations = []
            for i, content in enumerate(crawled_content, 1):
                if f"[Source{i}]" in synthesis or f"Source{i}" in synthesis:
                    citations.append({
                        'source_id': i,
                        'title': content['title'],
                        'url': content['url'],
                    })
            return synthesis, citations
        except Exception as e:
            print(f"Synthesis generation failed: {e}")
            summary = f"Research query on '{query.orignal_query}' found {len(crawled_content)}"
            for content in crawled_content[:3]:
                summary += f"{content['title']}\n {content['url']}\n\n"
            return summary, []
    
    async def save_research_results(
        self,
        result: ResearchResult,
        config: ResearchConfig
    ) ->Tuple[Path, Path]:
        """
        Saves research results in multiple formats

        Args:
            result (ResearchResult): _description_
            config (ResearchConfig): _description_

        Returns:
            Tuple[Path, Path]: _description_
        """
        config.output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_slug = result.query.orignal_query[:30].replace(" ", "_").replace("/", "_")
        base_filename = f"{timestamp}_{query_slug}"
        #saves as json
        json_path = config.output_dir / f"{base_filename}.json"
        with open(json_path,'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)
        
        md_content = [
            f"#Research Report: {result.query.orignal_query}",
            f"\n**Generated on:**{result.metadata.get('timestamp', 'N/A')}",
            f"\n**Domain:** {result.metadata.get('domain','N/A')}",
            f"\n**Processing time:**{result.metadata.get('duration','N/A')}",
            "\n----\n",
            "## Query Informatioon",
            f"- **Original Query:** {result.query.orignal_query}",
            f"- **Enhanced Query:** {result.query.enhanced_query}",
            "\n## Statistics",
            f"- **URLS Discovered:** {result.metadata['total_discovered']}",
            f"- **Pages Crawled:** {result.metadata['total_cralwed']}",
            f"- **Sources cited:** {result.metadata['total_cited']}",
            "\n## Research Synthesis",
            result.synthesis,
            "\n## Sources",

        ]
        #add citations
        for citation in result.citations:
            md_content.extend([
                f"###[{citation['source_id']}] {citation['title']}",
                f"- **URL:** [{citation['url']}]",

            ])
        for url_data in result.discovered_urls[:10]:
            score = url_data.get('relevance_score',0)
            title = 'N/A'
            if url_data.get('head_data') and url_data['head_data'].get('title'):
                title = url_data['head_data']['title'][:50]+'...'
            url = url_data['url'][:60] + '...'
            md_content.append(f"| {score:3f} | {title} | {url}")

        md_path = config.output_dir / f"{base_filename}.md"
        with open(md_path, 'w') as f:
            f.write("\n".join(md_content))
            
        return json_path, md_path

    async def multi_domain_research(self, domains: dict):
        """
        Reseach acrooss multiple website

        """    
        results = {}

        for genre, domain_list in domains.items():
            genre_results = []

            for domain in domain_list:
                assistant = DomainResearchAssistant(
                    query=self.query,
                    domain=domain,
                    api_key=self.api_key
                )

                config = ResearchConfig(domain=domain, max_urls_to_crawl=5)

                result = await assistant.research_crawler(config)
                genre_results.append(result)

            results[genre] = genre_results

        return results
    async def research_crawler(self,config: ResearchConfig):
        if config is None:
            config = ResearchConfig()
        start_time = datetime.now()
        
        if config.use_llm_enhancement:
            research_query = await self.enhanced_query_with_llm(self.query, config)
            print(research_query)
        else:
            research_query = ResearchQuery(
                orignal_query=self.query,
                enhanced_query=self.query,
                search_patterns=[f"*{word}*"for word in self.query.lower().split()],
                timestamp=datetime.now().isoformat()
            )
        
        discovered_urls = await self.discover_urls(research_query, config)
        if not discovered_urls:
            #No urls found - return empty []
            return ResearchResult(
                query=research_query,
                discovered_urls=[],
                crawled_content=[],
                synthesis="No relevant URLs found for given query",
                citations=[],
                metadata={'duration': str(datetime.now()-start_time)}
            )
        crawled = await self.crawl_selected_urls(discovered_urls[:5], research_query, config)

        with open("crawled_websites.json", "w", encoding="utf-8") as f:
                    json.dump(crawled, f, indent=2)        

        synthesis, citations = await self.generate_research_synthesis(research_query, crawled,config)
        
        #final rsult
        result = ResearchResult(
            query=research_query,
            discovered_urls=discovered_urls,
            crawled_content=crawled,
            synthesis=synthesis,
            citations=citations,
            metadata={
                'duration': str(datetime.now()-start_time),
                "domain": config.domain,
                'timestamp': datetime.now().isoformat(),
                'total_discovered': len(discovered_urls),
                'total_crawled_pages': len(crawled),
                'total_cited': len(citations)
            }
            
        )
        json_path, md_path = await self.save_research_results(result, config)
        return result


#How to run 
"""
#AsyncUrlSeeder uses domain root
if __name__ == "__main__":
    import asyncio
    #      
    test_domain = "cloudflare.com"   
    test_query = "Pen testing tools"
    
    
    asyncio.run(DomainResearchAssistant.research_crawler(config=ResearchConfig(test_query, test_domain)))
"""