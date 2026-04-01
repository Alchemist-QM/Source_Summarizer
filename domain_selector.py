import os
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate


root_domain_prompt = PromptTemplate(
    input_variables=["url"],
    template="""
You are a data extraction assistant.
Your task is to extract the root domain (registered domain) from any given URL.

Instructions:
- Remove the protocol (http, https, ftp, etc.).
- Remove subdomains (such as www, mail, blog, api).
- Remove paths, query strings, and fragments.
- Return only the root domain in the format: domain.tld.
- Do not include explanations—output only the extracted domain.

Examples:
Input: https://www.blog.example.com/path/page?x=1
Output: example.com

Input: http://subdomain.google.co.uk/search?q=test
Output: google.co.uk

Input: https://github.com/user/repo
Output: github.com

Now extract the root domain from this URL:
{url}
"""
)

class DomainController: 
    def __init__(self, api_key:str, url:str, query:str,mode:str, model:str="gpt-3.5-turbo"):
        
        load_dotenv()
        self.api_key= api_key
        self.model= model
        self.query= query
        self.client = ChatOpenAI(
            model=self.model, 
            temperature=0.2, 
            api_key=self.api_key,
            )
        self.url = url
        self.mode = "expand" if mode == "expand" else "normal"
        self.genre_domains = {
    "machine_learning": [
        "https://arxiv.org",
        "https://paperswithcode.com",
        "https://neurips.cc",
        "https://icml.cc",
        "https://jmlr.org",
        "https://medium.com",
        "https://towardsdatascience.com",
        "https://github.com",
        "https://openai.com",
        "https://deepmind.com",
        "https://kaggle.com",
        "https://analyticsvidhya.com",
        "https://fast.ai",
        "https://distill.pub",
        "https://kdnuggest.com",
        "https://machinelearningmastery.com",
    ],
    "biology": [
        "https://ncbi.nlm.nih.gov",
        "https://nature.com",
        "https://sciencedirect.com",
        "https://cell.com",
        "https://plos.org",
        "https://biomedcentral.com",
        "https://genomeweb.com",
        "https://europepmc.org",
        "https://frontiersin.org"
    ],
    "history": [
        "https://wikipedia.org",
        "https://britannica.com",
        "https://history.com",
        "https://nationalarchives.gov.uk",
        "https://archive.org",
        "https://smithsonianmag.com/history",
        "https://historyextra.com",
        "https://heraldica.org"
    ],
    "finance": [
        "https://investopedia.com",
        "https://bloomberg.com",
        "https://wsj.com",
        "https://ft.com",
        "https://seekingalpha.com",
        "https://morningstar.com",
        "https://yahoo.com/finance",
        "https://marketwatch.com",
        "https://fool.com"
    ],

    "technology": [
        "https://techcrunch.com",
        "https://thenextweb.com",
        "https://wired.com",
        "https://arstechnica.com",
        "https://theverge.com",
        "https://engadget.com",
        "https://cnet.com"
    ],
    "psychology": [
        "https://apa.org",
        "https://psychologytoday.com",
        "https://sciencedirect.com/journal/behavioral-sciences",
        "https://frontiersin.org/journals/psychology",
        "https://plos.org/psychology"
    ],
    "literature": [
        "https://goodreads.com",
        "https://poets.org",
        "https://literaryhub.com",
        "https://projectgutenberg.org",
        "https://jstor.org"
    ],
    "politics": [
        "https://politico.com",
        "https://nytimes.com/section/politics",
        "https://bbc.com/news/politics",
        "https://thehill.com",
        "https://cfr.org"
    ],
    "health": [
        "https://webmd.com",
        "https://mayoclinic.org",
        "https://healthline.com",
        "https://nih.gov",
        "https://who.int",
        "https://medlineplus.gov"
    ],
    "environment": [
        "https://unep.org",
        "https://wwf.org",
        "https://nasa.gov/topics/earth",
        "https://nature.org",
        "https://sciencedaily.com/news/earth_climate"
    ],
    "art": [
        "https://metmuseum.org",
        "https://moma.org",
        "https://tate.org.uk",
        "https://artnews.com",
        "https://artforum.com"
    ],
    "economics": [
        "https://imf.org",
        "https://worldbank.org",
        "https://economist.com",
        "https://voxeu.org",
        "https://nber.org",
        "https://ft.com/markets",
        "https://reuters.com/finance",
        "https://piie.com",
        "https://cepr.org",
        "https://investopedia.com"
    ],
    "general_science": [
        "https://wikipedia.org",
        "https://scientificamerican.com",
        "https://nature.com",
        "https://sciencemag.org",
        "https://phys.org",
        "https://newscientist.com",
        "https://popularmechanics.com",
        "https://sciencealert.com",
        "https://sciencedaily.com",
        "https://reddit.com",
        "https://britannica.com",
        "https://quora.com",
        "https://ted.com",
        "https://theconversation.com"
    ]
}

    def select_genres(self,top_k:int=3,):
        """
        Classify the user's query into one of the predefined genres.
        
        Returns: 
            Tuple: Dict of genre(score[cosine similarity], top_k genres only) and mode:str
        """
        # Precompute embeddings for genre labels
        genres = list(self.genre_domains.keys())
        genre_embeddings = {}
        
        for genre in genres:
            resp = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=genre
            )
            genre_embeddings[genre] = np.array(resp.data[0].embedding)

        # Embed the query
        resp = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=self.query
        )
        query_emb = np.array(resp.data[0].embedding)

        # Compute cosine similarity
        def cosine(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        scores = {genre: cosine(query_emb, emb) for genre, emb in genre_embeddings.items()}

        if self.mode != "normal":
            # Sort by score and keep top_k for wider search
            top_genres = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k])

            print(f"[Genre Selection] Top {top_k} genres for query '{self.query}':")
            for genre, score in top_genres.items():
                print(f"  {genre}: {score:.4f}")
            return top_genres
        
        else:
            best_genre = max(scores, key=scores.get)
            print(f"[Genre Selection] Query classified as '{best_genre}'")
            return best_genre



    def get_weighted_domains(self, top_k: int = 3, top_lists: int = 2) -> dict:
        """
        Return a weighted list of domains based on top genres for the query.
        Domains from higher-scoring genres appear more often.
        
        Args:
            query: The user's research query.
            mode: 'normal' for single genre, anything else for multi-genre.
            top_k: How many top genres to consider overall.
            top_lists: How many top genre domain lists to return.
        
        Returns:
            Dict[str, List[str]]: {genre_name: [list of domains]}
    """
        top_genres = self.select_genres(top_k=top_k)
        if self.mode == "normal":
            # single genre
            weighted_domains = self.genre_domains[top_genres] or self.genre_domains.get(top_genres, [])
            print(f"[Domain Selection] Single genre domains for '{top_genres}':")
            return {top_genres: weighted_domains}  # Always return dict 08/31/25
                
        # Sort genres by score descending
        sorted_genres = sorted(top_genres.items(), key=lambda x: x[1], reverse=True)
        
        # Keep only the top `top_lists` genres
        #sorted_genres = sorted_genres[:top_lists]
        
        result = {}
        for genre, score in sorted_genres:
            # Weight domains by genre score
            weighted_domains = []
            count = max(1, int(score * 10))  # weight proportional to similarity
            weighted_domains.extend(self.genre_domains.get(genre, []) * count)
            
            # Deduplicate while preserving order
            seen = set()
            weighted_domains = [d for d in weighted_domains if not (d in seen or seen.add(d))]
            
            result[genre] = weighted_domains
        
        print(f"[Genre Domain Selection] Top {top_lists} weighted domain lists for query '{self.query}':")
        for genre, domains in result.items():
            print(f"  {genre}: {domains}")
        return result
    
    #these genre of urls will be used by each subquery 
    def extract_root_domain(self) -> str:
        prompt = root_domain_prompt.format(url=self.url)
        response = self.client.invoke(prompt)
        return response.content.strip()

    def domain_extractor(self, )-> dict | str:
        """
        given a query, extracts the root domain or returns or returns domains to be crawled 

        Args:
            query (str): _description_
            extraction_type (str, optional): _description_. Defaults to "root".
        """
        if self.mode == "expand":
            return self.get_weighted_domains()
        else:
            return self.extract_root_domain()
#user_query = input("Enter your research query: 
#api_key = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#domains = GenreSelector.get_weighted_domains(user_query, top_k=3, mode="normal")

