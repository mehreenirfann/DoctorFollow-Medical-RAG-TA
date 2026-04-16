#Mehreen Irfan
import json
import os
from typing import Optional

try:
    import google.generativeai as genai
    gemini = True
except ImportError:
    gemini = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

api_key = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

if not api_key and not groq_api_key:
    raise ValueError("No API keys set. Set GOOGLE_API_KEY or GROQ_API_KEY environment variable.")

if api_key:
    genai.configure(api_key=api_key)

from data_retrieval import HybridRetriever, load_data

system_prompt = """
You are a helpful assistant that answers medical questions based on the following retrieved articles.

CRITICAL RULE - LANGUAGE RESPONSE:
- ALWAYS respond in the SAME language as the query
- Turkish query → respond in Turkish
- English query → respond in English
- Do NOT switch languages

Strict Rules to follow:
1. Answer ONLY from the provided context - never use outside knowledge. If the context does not contain relevant information, say "I apologize, but the retrieved articles do not contain sufficient information to answer your query."
2. Every factual claim must be cited with the article's PMID in brackets, for example [PMID:123456]
3. Be medically accurate and precise. If the information is partial, say "The retrieved articles provide partial information: [state what you found]"
4. Context articles will be provided in the following format:
[PMID: XXXXX] Title: ... Abstract: ...
"""

def generate_context(retrieved_articles: list[dict]) -> str:
    "Generate a context string from retrieved articles to feed into the LLM"
    parts = []
    for art in retrieved_articles:
        a = art["article"]
        pmid = a.get("PMID", "Unknown")
        title = a.get("Title", "No Title")
        abstract = a.get("Abstract", "No Abstract")
        year = a.get("Year", "Unknown Year")
        authors = a.get("FirstAuthor", "")
        parts.append(
            f"[PMID: {pmid}] ({authors}, {year})\n"
            f"Title: {title}\n"
            f"Abstract: {abstract}"
        )
    return "\n\n---\n\n".join(parts)

def call_googleGenAI(query: str, context: str) -> str:
    "Call Google Gemini API with the query + context"
    model = genai.GenerativeModel(model_name="gemini-2.0-flash", 
                                  system_instruction=system_prompt)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = model.generate_content(prompt)
    return response.text

def call_groq(query: str, context: str) -> str:
    "Call Groq API (Llama-3.3 70B) with the query + context"
    client = Groq(api_key=groq_api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"},
        ],
        max_tokens=1024,
        temperature=0.2,
    )
    return response.choices[0].message.content

# Rate limit keywords to detect when to fall back to Groq
_RATE_LIMIT_KEYWORDS = ["rate", "quota", "limit", "429", "resource exhausted", "too many"]

def RAG_query(retriever: HybridRetriever, query: str, top_n: int=5, api_key: Optional[str] = None) -> dict:
    "Run a RAG query by retrieving, generating context, and generating answer"
    print (f"Query: {query}")

    #retrieve
    retrieved = retriever.search(query, top_n=top_n)
    print(f"Retrieved {len(retrieved)} articles.")
    for r in retrieved:
        a = r["article"]
        print(f" {r['rank']}. PMID: {a.get('PMID', 'Unknown')}, Title: {a.get('Title', 'No Title')[:60]}...")

    #generate context
    context = generate_context(retrieved)

    #generate answer — try Gemini first, fall back to Groq
    answer = None
    if api_key and gemini:
        print("\nGenerating answer using Google Gemini")
        try: 
            answer = call_googleGenAI(query, context)
        except Exception as e:
            err_str = str(e)
            is_rate_limit = any(kw in err_str.lower() for kw in _RATE_LIMIT_KEYWORDS)
            reason = "rate limit hit" if is_rate_limit else f"error: {err_str[:80]}"
            print(f"  Gemini failed ({reason}) — falling back to Groq...")

    if answer is None and GROQ_AVAILABLE and groq_api_key:
        print("\nGenerating answer using Groq (Llama-3.3 70B)")
        try:
            answer = call_groq(query, context)
        except Exception as e:
            print(f"  Groq failed: {e}")

    if answer is None:
        answer = (
            "[LLM not available — set GOOGLE_API_KEY or GROQ_API_KEY in environment]\n"
            "The retrieved documents above contain the relevant information."
        )

    print(f"\nANSWER:\n{answer}")

    return {
        "query": query,
        "retrieved_articles": [
            {
                "rank": r["rank"],
                "pmid": r["article"].get("PMID"),
                "title": r["article"].get("Title"),
                "score": r["score"],
            }
            for r in retrieved
        ],
        "answer": answer,
    }

demo_queries = [
    "Çocuklarda akut otitis media tedavisi nasıl yapılır?", # tested ones for metrics
    "Iron supplementation dosing for anemia during pregnancy",
    "Çölyak hastalığı tanı kriterleri nelerdir?",
    "Antibiotic resistance patterns in community acquired pneumonia",
    "COVID-19 aşılarının etkinliği ve güvenliği hakkında güncel bilgiler nelerdir?", #new ones
    "Management of type 2 diabetes in elderly patients with comorbidities",
    ]

if __name__ == "__main__":
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("WARNING: GOOGLE_API_KEY not set. Will try Groq instead.")
    if not groq_api_key:
        print("WARNING: GROQ_API_KEY not set.")
    print()

    print("Loading articles and initialising retriever...")
    articles = load_data()
    retriever = HybridRetriever(articles)
    print(f"Ready. {len(articles)} articles indexed.\n")

    results = []
    for query in demo_queries:
        result = RAG_query(retriever, query, top_n=5, api_key=api_key)
        results.append(result)

    #save results to json
    os.makedirs("data", exist_ok=True)
    with open("data/rag_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n\nRAG results saved to data/rag_results.json")
