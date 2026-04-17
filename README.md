# DoctorFollow AI - Medical RAG System
## Technical Assessment


**Author:** Mehreen Irfan  
**Date:** April 2026

---

## Executive Summary

This project implements a complete **RAG** system for medical literature research. The system fetches medical articles from PubMed, retrieves relevant papers using hybrid methods (BM25 + semantic embeddings), and generates AI-powered answers grounded in retrieved medical literature.

**Key Components:**
1. **Data Pipeline** — Fetches & deduplicates Medical Articles from PubMed
2. **Retrieval and Evaluation System** — BM25, semantic (Multilingual E5-Small), and RRF retrieval + robust evaluation of the system
3. **RAG Generation** — LLM answers as per the retrieved articles from the best retrieval method
4. **BONUS** - **Language Aware Responses, LLM Fallback Chain, Streamlit based UI**

---

## 1. Setup & Usage

### Prerequisites
- Python 3.11+
- PubMed API access (free, no key needed)
- Google Gemini API key (optional, for LLM features)
- Groq API key (optional, fallback LLM)

### Installation

```bash
# Create virtual environment
python -m venv .DFenv
.\.DFenv\Scripts\activate

# Install dependencies
pip install -r Requirements.txt

# Set API keys (PowerShell)
$env:GOOGLE_API_KEY = "your-google-api-key"
$env:GROQ_API_KEY = "your-groq-api-key"
```

### Usage

```bash
# Full pipeline (fetch → retrieve → RAG)
python main.py

# Quick mode (use cached articles, skip evaluation)
python main.py --skip-fetch --no-eval

# Streamlit UI
streamlit run app.py
# Opens at http://localhost:8501
```

### File Structure

```
DoctorFollow-Technical-Assessment/
├── data_pipeline.py           # Part 1: Fetch & parse PubMed articles
├── data_retrieval.py          # Part 2: BM25, semantic, hybrid retrieval  
├── RAG.py                     # Part 3: LLM answer generation
├── main.py                    # Orchestration script
├── app.py                     # Streamlit web UI (bonus)
├── medical_terms.csv          # Input: medical terms to search
├── Requirements.txt           # Python dependencies
├── README.md                  # This file
└── data/
    ├── pubmed_refr.json       # 48 fetched articles
    ├── cache.pkl              # Cached E5-Small embeddings
    └── rag_results.json       # RAG demo results
```

### Dependencies

```
requests              # PubMed API calls
pandas                # CSV reading
rank_bm25             # BM25 algorithm
sentence-transformers # E5-Small embeddings
google-generativeai   # Google Gemini API
groq                  # Groq Llama API (fallback)
streamlit             # Web UI
```

---

## 2. Approach

### Data Pipeline

The data pipeline follows a **fetch → parse → deduplicate** workflow:

1. Load Terms from `medical_terms.csv` 
2. Search PubMed via eutils API for each term 
3. Fetch XML abstracts for all PMIDs
4. Parse XML using ElementTree to extract structured metadata
5. Deduplicate by PMID to remove duplicates across queries
6. Output JSON with full article metadata

**Key Decisions:**
- **CSV Structure**: Read column 2 (medical terms) after skipping header
- **Deduplication Strategy**: Track seen PMIDs in a dictionary to prevent duplicates
- **Error Handling**: Graceful continuation if individual search/fetch fails
- **Rate Limiting**: 0.33s delay between requests (3 requests/sec) to respect PubMed API

**Results:**
- **Input**: 10 medical terms from CSV
- **Articles Fetched**: 50 PMIDs total
- **After Deduplication**: 48 unique articles
- **Duplicates Removed**: 2 articles appeared in multiple term searches
- **Output File**: `data/pubmed_refr.json` (48 articles with full metadata)

### Retrieval Model Choice

The **multilingual-e5-small model** was kept as the model of choice for semantic retrieval. It provides the following benefits over its counterparts:
1. Reduces memory overhead with a relatively stable size of 470MB
2. Faster Embedding Generation and search queries, somewhat necesary for this task
3. Shared vector space mapping allows Turkish medical queries to align semantically with English abstracts, serving the purpose for this objective of this task

Drawbacks compare with bge-m3:
1. Low retrieval accuracy as compared to bge-m3
2. 512 token context window vs bge-m3's 8192 token context window
### RAG Generation

**System design:** Query → Retrieve 5 articles → Format context → LLM generates answer → Return with PMID citations

**System prompt constraints:**
```
1. Answer ONLY from context
2. Cite all claims with [PMID:XXXXX]
3. Be medically accurate
4. BONUS: Respond in SAME language as query (Turkish to Turkish, English to English)
```

**BONUS: LLM Fallback chain:** Google Gemini 2.0 Flash → Groq Llama-3.3 70B

Gemini is more capable of answering the queries however Groq is more robust, with a greater token limit.

**BONUS: Web UI (Streamlit)**

`app.py` provides query input, retrieval table, expandable details, LLM answer, adjustable result count.

Given more time I would have intensely focused on hyperparameter tuning, such as testing a broader range of values for k1 and b in BM25 and k for RRF, as well as evaluating diverse retrieval models and running multiple optimization iterations to achieve peak system performance. Since implementing some of these NLP concepts was a new experience, I dedicated a portion of my time to concretizing the logic required to develop the MVP, but I am extremely eager to keep experimenting to build an even more robust system.
---

## 3. BM25 Analysis

Learning Source: (https://www.geeksforgeeks.org/nlpwhat-is-bm25-best-matching-25-algorithm/)

BM25 formula: `score(D,Q) = Σ IDF(qi) * (f(qi,D) * (k1 + 1)) / (f(qi,D) + k1 * (1 - b + b * |D|/avgdl))`

**Functions of Parameters k1 and b:**  
k1 controls term frequency scaling of the searched term i.e. adjusts the weightage given to a document for multiple occurances of the searched term.  
b controls document length normalization i.e. adjusts the score based on the relative length of the document.

### What Happens When We Vary Them?

**Tested Query:** "diabetes management guidelines"

**k1 Parameter (Term Frequency Scaling):**
- **k1=0.5** — Term repetition saturates quickly; "diabetes diabetes diabetes" ≈ "diabetes"; reduces impact of repetition
- **k1=1.5** — Balanced approach; standard for most use cases; good for medical literature retrieval
- **k1=3.0** — Keeps rewarding repetition; documents heavily mentioning diabetes score much higher

**b Parameter (Length Normalization):**
- **b=0.0** — No normalization; long documents always win; bias toward comprehensive reviews
- **b=0.75** — Partial normalization; balances short & long articles; standard choice for fair ranking
- **b=1.0** — Full normalization; completely neutralizes document length; short papers ≈ long papers

**BM25 Results:**
- Mean Precision@5: 0.40 
- Mean MRR: 0.45 
- Mean MAP: 0.35 

The given results indicate that BM25 excels at keyword matching but misses semantic similarity.

---

## 4. RRF Analysis

Hybrid search combines BM25 + semantic rankings using Reciprocal Rank Fusion: `score = Σ 1/(k + rank)` where k=60

**What does k do?**  
k is a dampening constant controlling weight on top-ranked results.  
At k=0, rank 1 gets score 1.0 while rank 10 gets 0.1 i.e. only top results matter.  
At k=60 (default), rank 1 gets 0.0165 and rank 10 gets 0.0143 i.e. all ranks contribute meaningfully.  
At k=1000, nearly all ranks are equal i.e. the model loses ability to distinguish good from bad ranks.  
Reinforcing the paper, k=60 is ideal for medical retrieval.

**Why use rank position instead of raw scores?**  
BM25 and cosine similarity have incompatible scales: BM25 ranges 0-30 while cosine ranges 0-1. Combining raw scores makes BM25 dominate, making semantic retrieval irrelevant (reinforced by the individual results as well). Using rank positions makes both methods serve equally.

**Results:** Mean P@5=**0.56**, MRR=**0.62**, MAP=**0.51** (best method!)

---

## 5. Evaluation Metrics

Learning Sources: (https://www.ibm.com/think/architectures/rag-cookbook/result-evaluation) (https://medium.com/@mekjr1/ranking-the-knowledge-evaluating-retrieval-for-llms-8cfcdbef0e60)

| Metric | What | Why |
|--------|------|-----|
| **P@k** | Percentage of top-k that are relevant | Directly measures relevance of top-k retrieved chunks, aligning with RAG's requirement that the most useful context appears early for generation |
| **MRR** | Position of first relevant result | Improves diversity in retrieved results by reducing redundancy, critical for RAG |
| **MAP** | Avg precision across results | Well suited as it evaluates both precision and the ranking of multiple relevant documents |


---

## 6. Hardest Problem

While the rest of the coding and implementation was a smooth sail, the most challenging hurdle was diagnosing a persistent failure in the PubMed PMID fetching mechanism. Finding the root cause required adding debug statements at every possible deviation (later removed to optimise processing). It was discovered that a fundamental assumption regarding the API structure was being made - the fix only required changing one line of code, changing one of the get keys. However, this mistake and the hard fixed reinforced a critical principle that was to always inspect the API response.


---

## 7. Scenario Question: Benchmarking a 70B LLM Without Your Usual GPU Provider

**Situation:** Need to benchmark a 70B open-source LLM for medical QA. Usual GPU provider (L40S) unavailable. Manager busy. Results needed by end of week.

**What I would do:**

1. Given the urgency I would pivot from the usual GPU provider to more widely available and capacious ones. Some of the alternatives are fetching the NVIDIA A100 and H100 GPUs that are optimal for benchmarking the said model and are usually available in a short time from AI cloud providers like AWS or Lambda.ai (https://lambda.ai/pricing).
2. In case of availabilty or pricing issue, the other option would be to look GPUs or GPU clusters (basically internal hardware inventory) on premises of the workplace and see if they can be utilised. 
3. The third option would be to utilise multiple smaller GPUs with onspot availabilty such as T4s (even available on free version of colab!!!). However, this would pose a severe tradeoff in terms of processing speed.
4. The low processing speed could be tackled by benchmarking a quantised version of the model (8-bit or 4-bit parameter precision instead of the usual 32-bit or 16-bit) to approximate performance. However, this lesser processing time would affect the quality of benchmarking results. 


---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DoctorFollow AI System                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  INPUT: Turkish/English Query                                    │
│    ↓                                                             │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  RETRIEVAL LAYER                                       │     │
│  │  ├─ BM25Retriever (keyword-based)                     │     │
│  │  ├─ MultilingualE5SmallRetriever (semantic)           │     │
│  │  └─ HybridRetriever (RRF combination)                 │     │
│  └────────────────────────────────────────────────────────┘     │
│    ↓                                                             │
│  TOP-5 RELEVANT ARTICLES                                         │
│    ↓                                                             │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  CONTEXT FORMATTING                                    │     │
│  │  Format: [PMID: X] Title: ... Abstract: ...            │     │
│  └────────────────────────────────────────────────────────┘     │
│    ↓                                                             │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  LLM GENERATION LAYER (with fallback)                  │     │
│  │  ├─ Primary: Google Gemini 2.0 Flash                   │     │
│  │  ├─ Fallback: Groq Llama-3.3 70B                       │     │
│  │  └─ Constraint: Answer only from context + cite PMIDs  │     │
│  └────────────────────────────────────────────────────────┘     │
│    ↓                                                             │
│  OUTPUT: Medical Answer (Turkish or English) with Citations      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Conclusion

DoctorFollow AI demonstrates a production-ready medical RAG system that.
