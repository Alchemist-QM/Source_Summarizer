from langchain.prompts import PromptTemplate


reducer_templates = {

    "key_terms_reducer": PromptTemplate(
        input_variables=["text"],
        template="""
You are a text reduction assistant.

Your task is to compress the following document while preserving only information useful for extracting:
- keywords
- topics
- themes
- key concepts
- definitions

Instructions:
- Remove examples, anecdotes, references, tables, and citations.
- Remove formatting, headers, footers, page numbers, and boilerplate.
- Keep only sentences that define, explain, or introduce important technical terms and concepts.
- Preserve technical vocabulary exactly as written.
- Do not summarize; only filter and condense.
- Maintain original meaning and terminology.

Output:
Return the reduced technical text only (no commentary, no JSON).

TEXT:
{text}
"""
    ),


    "summary_reducer": PromptTemplate(
        input_variables=["text"],
        template="""
You are a text reduction assistant.

Your task is to compress the following document while preserving only information useful for generating:
- the main topic
- a core explanation using the 80/20 rule

Instructions:
- Remove minor details, citations, tables, and side discussions.
- Keep only sentences that explain what the topic is and how it works.
- Preserve causal relationships and core arguments.
- Ignore formatting noise such as headers, footers, and page numbers.
- Preserve important technical vocabulary exactly as written.
- Do not add new information or interpretations.

Output:
Return a concise reduced version of the text containing only the essential explanatory content.

TEXT:
{text}
"""
    ),


    "conclusion_reducer": PromptTemplate(
        input_variables=["text"],
        template="""
You are a text reduction assistant.

Your task is to compress the following document while preserving only information useful for extracting:
- TLDR
- key takeaways
- conclusion

Instructions:
- Remove background theory, methodology details, and examples.
- Keep only sentences that express outcomes, implications, and high-level insights.
- Retain statements about importance, impact, or lessons learned.
- Ignore formatting noise such as headers, footers, and page numbers.
- Preserve technical vocabulary exactly as written.
- Do not summarize; only filter to conclusion-relevant content.

Output:
Return the reduced text focused only on takeaways and conclusions.

TEXT:
{text}
"""
    ),


    "academic_reducer": PromptTemplate(
        input_variables=["text"],
        template="""
You are a text reduction assistant.

Your task is to compress the following document while preserving only information useful for extracting:
- bibliographic metadata
- research question or objective
- methodology
- key findings or results
- limitations or assumptions

Instructions:
- Remove discussion, background theory, examples, and unrelated narrative.
- Keep only sentences that describe:
  - what problem is studied
  - how the study was conducted
  - what was discovered
  - weaknesses or constraints
- Preserve technical vocabulary exactly as written.
- Ignore formatting noise such as headers, footers, page numbers, and references list formatting.
- Do not interpret or summarize beyond filtering.

Output:
Return the reduced research-focused text only (no JSON, no commentary).

TEXT:
{text}
"""
    )
}

student_templates = {
    "key_terms": PromptTemplate(
        input_variables=['text'],
        template="""
        You are an expert NLP and document analysis assistant.
        Your task is to analyze the provided Markdown or PDF text and extract structured semantic information.

        Instructions:

        Parse the input document content accurately (ignore formatting noise such as headers, page numbers, or footers).

        Identify and extract:

        Keywords – important non-trivial words and phrases.

        Topics – major subject areas discussed.

        Themes – higher-level conceptual ideas that unify the document.
        
        Key Concepts - important concepts that are essential to know 
        
        For each key term, provide a concise definition (1-2 sentences) that explains its meaning in the context of the research.


        Keep all extracted items concise and meaningful.


Format the output as valid JSON:

{{
    "keywords": [{{"term":"...","definition":"..."}}],
    "primary_topics": [{{"topic":"...","explanation":"..."}}],
    "concepts": [{{"concepts":"...","explanation":"..."}}],
    "themes": [{{"theme":"...","explanation":"..."}}]
}}

Constraints:

Preserve important technical vocabulary exactly as written.
Only include terms that are specific, non-generic, and relevant to the technical context.

Text:
{text}
"""
    ),
    "summary": PromptTemplate(
        input_variables=['text'],
        template="""
        You are an expert NLP and document analysis assistant.
        Your task is to analyze the provided Markdown or PDF text and extract structured semantic information.

        Instructions:

        Parse the input document content accurately (ignore formatting noise such as headers, page numbers, or footers).
        
        Identify the central topic and purpose of the document.
        2. Extract only the most important ideas that account for the majority of understanding (Pareto principle).
        3. Present the material as if introducing it to a student encountering this topic for the first time.
        Identify and extract:

        Introduction- purpose, motivation, and overall aim(4 sentences)
        Main topic - What the topic is about
        Core Explanation – 4-5 sentences explaining the topic using the 80/20 rule
        

        Keep all extracted items concise and meaningful.

Format the output as valid JSON:

{{
    "introduction":"...",
    "main topic": "...",
    "explanation": "...",
}}

Constraints:

Preserve important technical vocabulary exactly as written.
Only include terms that are specific, non-generic, and relevant to the technical context.

Text:
{text}
"""
    ),
    "conclusion": PromptTemplate(
        input_variables=['text'],
        template="""
        You are an expert NLP and document analysis assistant.
        Given the following section, 
        Your task is to analyze the provided Markdown or PDF text and extract structured semantic information.

        Instructions:

        Parse the input document content accurately (ignore formatting noise such as headers, page numbers, or footers).

        Identify and extract:

        TLDR – important non-trivial words and phrases.

        Key takeaways – major subject areas discussed.

        Conclusion – higher-level conceptual ideas that unify the document.
        

        Keep all extracted items concise and meaningful.


Format the output as valid JSON:

{{
    "key_takeaways": [{{"takeaway":"...","explanation":"..."}}],
    "tldr": {{"..."}},
    "conclusion":{{ "..."}},
}}

Constraints:

Preserve important technical vocabulary exactly as written.
Only include terms that are specific, non-generic, and relevant to the technical context.

Text:
{text}
"""
    ),
    "academic": PromptTemplate(
        input_variables=['text'],
        template="""
        You are an expert NLP and document analysis assistant.
        Your task is to analyze the provided Markdown or PDF text and extract structured semantic information.

        Instructions:

        Parse the input document content accurately (ignore formatting noise such as headers, page numbers, or footers).

        Identify and extract:

        Bibliographic metadata: (title, authors, year, source, URL)1. 
        
        In-text citations or reference markers (e.g., [1], (Smith, 2022), etc.)

        Research question or objective - problem being investigated and overall hypothesis

        Methodology used - how they arrived at their results

        Key findings/results - the main results that were discovered

        Limitations or assumptions - what might be wrong or incomplete

        Keep all extracted items concise and meaningful.


Format the output as valid JSON:

{{
    "metadata": {
    "title": "...",
    "authors": [...],
    "year": "...",
    "source": "...",
    "url": "...",
    "citations":[{{"citation": "..."}}]
    },
  "research_question": "...",
  "methodology": "...",
  "key_findings": [{{"finding":"...","description":"..."}}],
  "limitations": [{{"limitation":"...","description":"..."}}],
}}

Constraints:

Preserve important technical vocabulary exactly as written.
Only include terms that are specific, non-generic, and relevant to the technical context.


Text:
{text}
"""
    )
    
}



### For youtube videos 

youtube_prompt_templates = {
    # Existing youtube_sum_3.py academic/insight prompts
    "insights": PromptTemplate(
        input_variables=["text"],
        template="""
You are an academic assistant. Given the following section of a document, extract the following:

1. Definitions of important terms (in bullet points).
2. List of technical terms (as a flat list).
3. Main ideas (summarize in a few bullet points).
4. Important takeaways or findings.
5. Any conclusions or implications.
6. Recurring themes or patterns.
7. High-level insights or summaries.

Section:
{text}
"""
    ),
    "citations": PromptTemplate(
        input_variables=["text"],
        template="""
You are an academic assistant. Analyze the following text section and extract any references, citations, or sources it includes.

Return:

1. In-text citations or reference markers (e.g., [1], (Smith, 2022), etc.)
2. Bibliographic references (if any full or partial references are present).
3. URLs or DOIs.
4. Mentioned author names, publication titles, or institutions.
5. Any inferred sources that are mentioned (e.g., "a study from Harvard" or "a report by WHO").

Section:
{text}
"""
    ),
    "summary": PromptTemplate(
        input_variables=["text"],
        template="Summarize this academic content clearly and concisely:\n\n{text}"
    ),
    "default": PromptTemplate(
        input_variables=["text"],
        template="Extract relevant academic information from the section below:\n\n{text}"
    ), 
    "final_insight": PromptTemplate(
        input_variables=['text'], 
        template="""
You are an expert analyst specializing in extracting and refining insights from large documents and transcripts.

Below is a list of insights extracted from various sections. These may contain overlaps, redundancies, or fragmented points.

---
{text}
Your task is to:
- Eliminate redundant or overlapping insights
- Merge similar ideas into unified, clear insights
- Retain all unique and meaningful information
- Present the final insights in a clean, concise bullet-point format
- Ensure the output is easy to read and focused only on essential takeaways without repeating information

Final refined insights:
"""
    ),

    # Prompts from prompts.py module
    "theme_title": PromptTemplate(
        input_variables=["summaries", "existing_titles"],
        template="""Given the following research paper summaries, generate a concise, human-readable, and unique theme title that:
- Clearly distinguishes this theme from others in this batch.
- Highlights the most specific technical innovation or topic.
- Avoids generic phrases like 'Advancements' or 'Introduction'.
- Do NOT repeat or closely match these other theme titles: {existing_titles}

Summaries:
{summaries}

Theme Title:
"""
    ),
    "final_theme": PromptTemplate(
        input_variables=["theme_summaries"],
        template="""
Given the following theme summaries, synthesize and group them into 3-5 final themes. For each theme, provide:
- Title
- Summary (4-6 sentences, capturing consensus, controversy, innovation, and future directions)

Theme Summaries:
{theme_summaries}
"""
    ),
    "final_conclusion": PromptTemplate(
        input_variables=["all_summaries"],
        template="""
Given all theme, cluster, and synthesis summaries, write a 1-2 paragraph final conclusion that holistically synthesizes the research landscape, consensus, open challenges and future directions.

All Summaries:
{all_summaries}
"""
    ),
    "takeaway": PromptTemplate(
        input_variables=["takeaways"],
        template="""
Given the following list of takeaways, synthesize and distill them into 3-6 final actionable takeaways or key insights for researchers and practitioners. Bullet point format.

Takeaways:
{takeaways}
"""
    ),
    "introduction": PromptTemplate(
        input_variables=["user_query", "all_summaries"],
        template="""
Write a 1 paragraph introduction to the following research synthesis report, providing context for the query, scope, and relevance.

User Query:
{user_query}

All Summaries:
{all_summaries}
"""
    ),
    "tldr": PromptTemplate(
        input_variables=["all_summaries"],
        template="""
Given all theme, cluster, and synthesis summaries, produce a 2-3 sentence TLDR suitable for an executive or busy reader.

All Summaries:
{all_summaries}
"""
    ),
    "key_terms": PromptTemplate(
        input_variables=["summaries"],
        template="""
You are an academic research assistant. Given the following research theme, cluster, and synthesis summaries, extract and return a dictionary of key technical terms and concepts found in the text.

For each key term, provide a concise definition (1-2 sentences) that explains its meaning in the context of the research. Only include terms that are specific, non-generic, and relevant to the technical context.

Format the output as valid JSON: {{"key_terms": [{{"term": "...", "definition": "..."}}]}}

Summaries:
{summaries}

Key Technical Terms (JSON):
"""
    ),
}
