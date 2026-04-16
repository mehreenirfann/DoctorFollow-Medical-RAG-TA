#Mehreen Irfan

import json
import math
import os
import pickle
from typing import Optional, List
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

Data = "data/pubmed_refr.json"
cache_file = "data/cache.pkl"

def load_data(data_path: str = Data) -> List[dict]:
    with open(data_path, 'r', encoding='utf-8') as file:
        return json.load(file)
    
def generate_corpus(articles: List[dict]) -> List[str]:
    "title and abstract concatinated to generate corpus as required"
    corpus = []
    for article in articles:
        text = f"{article.get('Title', '')} {article.get('Abstract', '')}".strip()
        corpus.append(text)
    return corpus

def tokenize(text: str) -> list[str]: #text -> simple, separated words
    import re
    return re.findall(r'\b\w+\b', text.lower())

# Implementing the BM25 retrieval function

class BM25Retriever:
    def __init__(self, articles: list[dict], k1: float = 1.5, b: float = 0.75):
        self.articles = articles
        self.k1 = k1
        self.b = b
        corpus_texts = generate_corpus(articles)
        tokenized = [tokenize(t) for t in corpus_texts]
        self.bm25 = BM25Okapi(tokenized, k1=k1, b=b)

    def retrieve(self, query: str, top_n: int = 5) -> list[dict]:
        "for a given query, return top n articles as per the BM25 scores"
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_n]
        results = []
        for rank, index in enumerate(top_indices):
            article = self.articles[index].copy()  
            results.append({"rank": rank + 1, "score": scores[index], "article": article})
        return results
        
#  Implementing semantic retrieval function of choice: Multilingual E5 Small     

class MultilingualE5SmallRetriever:
    def __init__(self, articles: list[dict], cache_path: str = cache_file):
        self.articles = articles
        self.model = SentenceTransformer("intfloat/multilingual-e5-small")
        self.corpus_texts = generate_corpus(articles)

        if os.path.exists(cache_path):
            print(f"  Loading cached embeddings from {cache_path}")
            with open(cache_path, "rb") as f:
                self.corpus_embeddings = pickle.load(f)
        else:
            print("Encoding {len(self.corpus_texts)} articles")
            prefixed = [f"passage: {t}" for t in self.corpus_texts] #prefix requirment for E5 model
            self.corpus_embeddings = self.model.encode(
                prefixed, show_progress_bar=True, batch_size=32, normalize_embeddings=True
            )
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(self.corpus_embeddings, f)
            print(f"  Embeddings cached to {cache_path}")
    def retrieve(self, query: str, top_n: int = 5) -> list[dict]:
        "for a given query, return top n articles as per the cosine similarity scores"
        prefixed_query = f"query: {query}"
        query_embedding = self.model.encode(prefixed_query, normalize_embeddings=True)
        scores = cosine_similarity([query_embedding], self.corpus_embeddings)[0]
        top_indices = np.argsort(scores)[::-1][:top_n]
        results = []
        for rank, index in enumerate(top_indices):
            article = self.articles[index].copy()  
            results.append({"rank": rank + 1, "score": scores[index], "article": article})
        return results
    
# Implementing RRF 

def RRF(ranked_lists: list[list[dict]], k: int = 60, top_n: int = 5) -> list[dict]:
    "combine multiple ranked lists (obtained from above implementations) into a single ranked list"
    "returns a reranked list of articles based on the RRF scores"

    scoreRRF:dict[str, float] = {}
    pmid_to_article:dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank_0based, result in enumerate(ranked_list):
            pmid = result["article"]["PMID"]
            scoreRRF[pmid] = scoreRRF.get(pmid, 0) + 1 / (k + rank_0based + 1)
            if pmid not in pmid_to_article:
                pmid_to_article[pmid] = result["article"]
    sorted_pmids = sorted(scoreRRF, key= scoreRRF.get, reverse=True)[:top_n]

    return[{"rank": i + 1, "score": scoreRRF[pmid], "article": pmid_to_article[pmid]} for i, pmid in enumerate(sorted_pmids)]
        
class HybridRetriever: #RM25 + semantic for RRF
    def __init__(
        self,
        articles: list[dict],
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        rrf_k: int = 60,
    ):
        self.articles = articles
        self.rrf_k = rrf_k
        print("Initialising BM25")
        self.bm25 = BM25Retriever(articles, k1=bm25_k1, b=bm25_b)
        print("Initialising Multilingual E5 Small Semantic Retriever")
        self.semantic = MultilingualE5SmallRetriever(articles)

    def search(self, query: str, top_n: int = 5, candidate_n: int = 20) -> list[dict]:
        "for a given query, return top n articles as per the RRF scores obtained by combining BM25 and semantic retrieval results"
        bm25_results = self.bm25.retrieve(query, top_n=candidate_n)
        semantic_results = self.semantic.retrieve(query, top_n=candidate_n)
        return RRF(
            [bm25_results, semantic_results], k=self.rrf_k, top_n=top_n
        )

# Evaluation using relevant metrics - metrics of choice: MRR, MAP, precision@k

Queries = [
    "What are the latest guidelines for managing type 2 diabetes?",
    "Çocuklarda akut otitis media tedavisi nasıl yapılır?",
    "Iron supplementation dosing for anemia during pregnancy",
    "Çölyak hastalığı tanı kriterleri nelerdir?",
    "Antibiotic resistance patterns in community acquired pneumonia",
]

Relevant_Terms = {
    Queries[0]: ["type 2 diabetes mellitus", "gestational diabetes"],
    Queries[1]: ["acute otitis media", "pediatric asthma management"],
    Queries[2]: ["iron deficiency anemia", "gestational diabetes"],
    Queries[3]: ["celiac disease diagnosis"],
    Queries[4]: ["community acquired pneumonia"],
}


def is_relevant(article: dict, query: str) -> bool:
    expected = Relevant_Terms.get(query, [])
    if not expected:
        return False
    matched = set(article.get("matched_terms", []))
    return bool(matched & set(expected))

def mrr(results: list[dict], query: str) -> float:
    "Mean Reciprocal Rank"
    relevant_terms = Relevant_Terms.get(query, [])
    for rank, result in enumerate(results, 1):
        if is_relevant(result["article"], query):
            return 1.0 / rank
    return 0.0

def map_score(results: list[dict], query: str) -> float:
    "Mean Average Precision"
    num_relevant = 0
    sum_precisions = 0.0
    
    for rank, result in enumerate(results, 1):
        if is_relevant(result["article"], query):
            num_relevant += 1
            precision_at_rank = num_relevant / rank
            sum_precisions += precision_at_rank
    
    if num_relevant == 0:
        return 0.0
    return sum_precisions / num_relevant

def precision_at_k(results: list[dict], query: str, k: int = 5) -> float:
    "Precision@k"
    top_k = results[:k]
    relevant_count = sum(1 for result in top_k if is_relevant(result["article"], query))
    return relevant_count / k if k > 0 else 0.0


def run_evaluation(bm25, semantic, hybrid, top_k: int = 5) -> dict:
    "Run all queries through all methods and compute metrics."
    
    methods = {
        "BM25": bm25,
        "Semantic": semantic,
        "Hybrid (RRF)": hybrid,
    }
    
    eval_results = {m: {"p@5": [], "mrr": [], "map": []} for m in methods}
    
    print("EVALUATION RESULTS")
    
    
    for query in Queries:
        print(f"\nQuery: {query[:70]}")
        print("-" * 80)
        
        for method_name, retriever in methods.items():
            results = retriever.search(query, top_n=top_k) if hasattr(retriever, 'search') else retriever.retrieve(query, top_n=top_k)
            p5 = precision_at_k(results, query, k=top_k)
            mrr_score = mrr(results, query)
            map_val = map_score(results, query)
            
            eval_results[method_name]["p@5"].append(p5)
            eval_results[method_name]["mrr"].append(mrr_score)
            eval_results[method_name]["map"].append(map_val)
            
            print(f"\n  [{method_name}]  P@5={p5:.2f}  MRR={mrr_score:.2f}  MAP={map_val:.2f}")
            for r in results:
                a = r["article"]
                rel_flag = "✓" if is_relevant(a, query) else " "
                title_short = a.get("Title", "")[:60]
                print(f"    {rel_flag} {r['rank']}. [{a.get('PMID','')}] {title_short}")
    

    print("Aggregate Metrics")

    print(f"{'Method':<20} {'Mean P@5':>12} {'Mean MRR':>12} {'Mean MAP':>12}")
    print("-" * 60)
    for method_name, scores in eval_results.items():
        mp5 = np.mean(scores["p@5"])
        mmrr = np.mean(scores["mrr"])
        mmap = np.mean(scores["map"])
        print(f"{method_name:<20} {mp5:>12.3f} {mmrr:>12.3f} {mmap:>12.3f}")
    
    return eval_results


def BM25_parameter_analysis(articles: list[dict]):
    "checking the effect of varying k1 and b parameters of BM25"
    query = "diabetes management guidelines"
    tokens = tokenize(query)
    

    print("BM25 Parameter Analysis")

    
    corpus_texts = generate_corpus(articles)
    tokenized = [tokenize(t) for t in corpus_texts]
    
    configs = [
        (0.5, 0.75, "Low k1: TF saturation kicks in quickly"),
        (1.5, 0.75, "Default k1: balanced"),
        (3.0, 0.75, "High k1: TF keeps rewarding repetition"),
        (1.5, 0.0,  "b=0: no length normalisation"),
        (1.5, 0.5,  "b=0.5: partial length norm"),
        (1.5, 1.0,  "b=1.0: full length normalisation"),
    ]
    
    for k1, b, note in configs:
        bm25 = BM25Okapi(tokenized, k1=k1, b=b)
        scores = bm25.get_scores(tokens)
        top_idx = int(np.argmax(scores))
        top_title = articles[top_idx].get("Title", "")[:60]
        print(f"\n  k1={k1}, b={b}  ({note})")
        print(f"  Top result: {top_title}")
        print(f"  Top score:  {scores[top_idx]:.4f}")


if __name__ == "__main__":
    print("Loading articles")
    articles = load_data()
    print(f"Loaded {len(articles)} articles.\n")
    
    BM25_parameter_analysis(articles)
    
    print("\nInitialising retrievers...")
    bm25 = BM25Retriever(articles)
    print("BM25 ready.")
    
    semantic = MultilingualE5SmallRetriever(articles)
    print("Semantic ready.")
    
    hybrid = HybridRetriever(articles)
    print("Hybrid ready.\n")
    
    run_evaluation(bm25, semantic, hybrid)