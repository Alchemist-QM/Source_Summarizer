import re 
import json

#cleans the llm_json ouput 

def clean_academic_output(raw_text):
    # Case 1: Already parsed dict
    if isinstance(raw_text, dict):
        return raw_text

    # Case 2: Try direct JSON parsing
    try:
        return json.loads(raw_text)
    except Exception as e:
        print("JSON decode failed in clean_academic_output, using regex fallback:", e)

    raw_str = str(raw_text)

    # -------- Extract metadata --------
    title = re.search(r'"title"\s*:\s*"([^"]+)"', raw_str)
    year = re.search(r'"year"\s*:\s*"([^"]+)"', raw_str)
    source = re.search(r'"source"\s*:\s*"([^"]+)"', raw_str)
    url = re.search(r'"url"\s*:\s*"([^"]+)"', raw_str)

    authors = re.findall(r'"authors"\s*:\s*\[([^\]]+)\]', raw_str)
    citations = re.findall(r'"citation"\s*:\s*"([^"]+)"', raw_str)

    # -------- Extract research question & methodology --------
    rq = re.search(r'"research_question"\s*:\s*"([^"]+)"', raw_str)
    methodology = re.search(r'"methodology"\s*:\s*"([^"]+)"', raw_str)

    # -------- Extract key findings --------
    findings = re.findall(
        r'"finding"\s*:\s*"([^"]+)"\s*,\s*"description"\s*:\s*"([^"]+)"',
        raw_str
    )

    # -------- Extract limitations --------
    limitations = re.findall(
        r'"limitation"\s*:\s*"([^"]+)"\s*,\s*"description"\s*:\s*"([^"]+)"',
        raw_str
    )

    return {
        "metadata": {
            "title": title.group(1) if title else "",
            "authors": authors if authors else [],
            "year": year.group(1) if year else "",
            "source": source.group(1) if source else "",
            "url": url.group(1) if url else "",
            "citations": [{"citation": c} for c in citations]
        },
        "research_question": rq.group(1) if rq else "",
        "methodology": methodology.group(1) if methodology else "",
        "key_findings": [{"finding": f, "description": d} for f, d in findings],
        "limitations": [{"limitation": l, "description": d} for l, d in limitations]
    }
def extract_json_block(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        return text[start:end+1]
    return text

def clean_key_terms_output(raw_text):
    # Case 1: Already parsed dict
    if isinstance(raw_text, dict):
        return raw_text

    # Case 2: Try direct JSON parsing
    try:
        json_text = json.loads(raw_text) 
        return json_text
    except Exception as e:
        print("JSON decode failed in clean_key_terms_output, using json_block fallback:", e)
    try:
        json_block = extract_json_block(raw_text)
        if json_block:
            return json.loads(json_block)
    except Exception as e:
        print("JSON decode failed in clean_key_terms_output, using regex fallback:", e)

    raw_str = str(raw_text)

    # Regex extraction for each field
    keywords = re.findall(
        r'"term"\s*:\s*"([^"]+)"\s*,\s*"definition"\s*:\s*"([^"]+)"',
        raw_str
    )

    topics = re.findall(
        r'"topic"\s*:\s*"([^"]+)"\s*,\s*"explanation"\s*:\s*"([^"]+)"',
        raw_str
    )

    concepts = re.findall(
        r'"concepts"\s*:\s*"([^"]+)"\s*,\s*"explanation"\s*:\s*"([^"]+)"',
        raw_str
    )

    themes = re.findall(
        r'"themes"\s*:\s*"([^"]+)"\s*,\s*"explanation"\s*:\s*"([^"]+)"',
        raw_str
    )

    return {
        "keywords": [{"term": t, "definition": d} for t, d in keywords],
        "primary_topics": [{"topic": t, "explanation": e} for t, e in topics],
        "concepts": [{"concepts": c, "explanation": e} for c, e in concepts],
        "themes": [{"theme": th, "explanation": e} for th, e in themes]
    }
    

        

def clean_academic(raw_text):
    if isinstance(raw_text, dict):
        return raw_text

    try:
        return json.loads(raw_text)
    except Exception:
        raw_str = str(raw_text)

        rq = re.search(r'"research_question"\s*:\s*"([^"]+)"', raw_str)
        method = re.search(r'"methodology"\s*:\s*"([^"]+)"', raw_str)

        findings = re.findall(
            r'"finding"\s*:\s*"([^"]+)"\s*,\s*"description"\s*:\s*"([^"]+)"',
            raw_str
        )

        limitations = re.findall(
            r'"limitation"\s*:\s*"([^"]+)"\s*,\s*"description"\s*:\s*"([^"]+)"',
            raw_str
        )
        citations = re.findall(
            r'"citations"\s*:\s*"([^"]+)"\s*,\s*"description"\s*:\s*"([^"]+)"',
            raw_str
        )

        return {
            "metadata": {},
            "research_question": rq.group(1) if rq else "",
            "methodology": method.group(1) if method else "",
            "key_findings": [{"finding": f, "description": d} for f, d in findings],
            "limitations": [{"limitation": l, "description": d} for l, d in limitations],
            "citations": [{"citation": c, "description": d} for c, d in citations]

        }
        
        
    
def clean_summary(raw_text):
    if isinstance(raw_text, dict):
        return raw_text

    try:
        return json.loads(raw_text)
    except Exception:
        raw_str = str(raw_text)

        intro = re.search(r'"introduction"\s*:\s*"([^"]+)"', raw_str)
        topic = re.search(r'"main_topic"\s*:\s*"([^"]+)"', raw_str)
        explanation = re.search(r'"core_explanation"\s*:\s*"([^"]+)"', raw_str)

        return {
            "introduction": intro.group(1) if intro else "",
            "main_topic": topic.group(1) if topic else "",
            "core_explanation": explanation.group(1) if explanation else ""
        }
        
        

        
import json
import ast

def clean_llm_object(raw_text):
    # Case 1: already structured
    if isinstance(raw_text, dict):
        return raw_text

    if not raw_text:
        return {}

    text = raw_text.strip()

    # Try JSON first
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try Python dict string
    try:
        return ast.literal_eval(text)
    except Exception:
        pass

    return {}
        
def updated_clean_summary(raw_text):
    # Case 1: Already parsed dict
    if isinstance(raw_text, dict):
        return raw_text

    # Case 2: Try direct JSON parsing
    try:
        return json.loads(raw_text)
    except Exception as e:
        print("JSON decode failed in clean_summary, using regex fallback:", e)

    raw_str = str(raw_text).strip()

    intro_match = re.findall(r'"introduction"\s*:\s*"([^"]+)"', raw_str)
    topic_match = re.findall(r'"main[_ ]topic"\s*:\s*"([^"]+)"', raw_str)
    explanation_match = re.findall(r'"(explanation|core_explanation)"\s*:\s*"([^"]+)"', raw_str)

    return {
        "introduction": intro_match.group(1) if intro_match else "",
        "main_topic": topic_match.group(1) if topic_match else "",
        "explanation": explanation_match.group(2) if explanation_match else ""
    }
def updated_clean_conclusion(raw_text):
      # Case 1: Already parsed dict
    if isinstance(raw_text, dict):
        return raw_text

    # Case 2: Try direct JSON parsing
    try:
        return json.loads(raw_text)
    except Exception as e:
        print("JSON decode failed in clean_conclusion, using regex fallback:", e)

    raw_str = str(raw_text).strip()

    # Extract TLDR
    tldr_match = re.findall(r'"tldr"\s*:\s*"([^"]+)"', raw_str)

    # Extract key takeaways
    takeaways = re.findall(
        r'"takeaway"\s*:\s*"([^"]+)"\s*,\s*"explanation"\s*:\s*"([^"]+)"',
        raw_str
    )

    # Extract conclusion text
    conclusion_match = re.findall(r'"conclusion"\s*:\s*"([^"]+)"', raw_str)

    return {
        "tldr": tldr_match.group(1) if tldr_match else "",
        "key_takeaways": [{"takeaway": t, "explanation": e} for t, e in takeaways],
        "conclusion": conclusion_match.group(1) if conclusion_match else ""
    }