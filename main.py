#Mehreen Irfan

import argparse
import os
import sys
import json


def main():
    parser = argparse.ArgumentParser(
        description="DoctorFollow AI — Medical RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Run full pipeline (fetch → retrieve → RAG)
  python main.py --skip-fetch             # Use existing data, skip fetching
  python main.py --skip-fetch --no-eval   # Quick RAG only
        """
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip Part 1: use existing data/pubmed_refr.json",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip Part 2: evaluation & parameter analysis",
    )
    parser.add_argument(
        "--medical-terms",
        default="medical_terms.csv",
        help="Path to medical terms CSV file (default: medical_terms.csv)",
    )
    args = parser.parse_args()

    # ─data pipeline

    print("PubMed Data Pipeline (Fetching & Parsing)")
  

    if not args.skip_fetch:
        from data_pipeline import fetch_complete_pipeline

        if not os.path.exists(args.medical_terms):
            print(f"ERROR: {args.medical_terms} not found.")
            print(f"  Please ensure your medical term CSV exists.")
            sys.exit(1)

        print(f"\nFetching articles for terms in: {args.medical_terms}")
        articles = fetch_complete_pipeline(args.medical_terms, "data/pubmed_refr.json")
        if not articles:
            print("ERROR: No articles fetched. Check your network connection and API limits.")
            sys.exit(1)
        print(f"✓ Successfully fetched and deduplicated {len(articles)} articles.")

    else:
        print(f"\nSkipping Pipeline — loading existing data/pubmed_refr.json")
        from data_retrieval import load_data

        try:
            articles = load_data("data/pubmed_refr.json")
            print(f"✓ Loaded {len(articles)} articles from cache.")
        except FileNotFoundError:
            print("ERROR: data/pubmed_refr.json not found. Run without --skip-fetch first.")
            sys.exit(1)

    # Retrieval & Evaluation

    print(" Retrieval System & Evaluation")


    from data_retrieval import (
        BM25Retriever,
        MultilingualE5SmallRetriever,
        HybridRetriever,
        BM25_parameter_analysis,
        run_evaluation,
    )

    if not args.no_eval:
        print("\nAnalyzing BM25 parameters")
        BM25_parameter_analysis(articles)
    else:
        print("\nSkipping BM25 parameter analysis (--no-eval flag set).")

    print("\n" + "-" * 70)
    print("Initialising retrievers")
    print("  • BM25Retriever: indexing articles")
    bm25 = BM25Retriever(articles)
    print("  BM25 ready")

    print(" • MultilingualE5SmallRetriever: encoding articles (may take ~2-3 min)")
    print(" (Semantic model ~470MB will be downloaded on first run)")
    semantic = MultilingualE5SmallRetriever(articles)
    print(" Semantic ready")

    print("  • HybridRetriever: combining BM25 + Semantic with RRF")
    hybrid = HybridRetriever(articles)
    print("  Hybrid ready")

    if not args.no_eval:
        print("Running evaluation across all retrievers...")
        run_evaluation(bm25, semantic, hybrid)
    else:
        print("\nSkipping evaluation (--no-eval flag set).")

    # RAG
    print(" RAG System")

    from RAG import RAG_query, demo_queries

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("\n⚠ WARNING: GOOGLE_API_KEY environment variable not set.")
        print("  Set it in PowerShell: $env:GOOGLE_API_KEY = 'your-api-key'")
        print("  Retrieval will run, but LLM answer generation will be skipped.\n")
    else:
        print("\n GOOGLE_API_KEY is set. LLM answer generation enabled.\n")

    print("Running RAG queries")
    results = []
    for i, query in enumerate(demo_queries, 1):
        print(f"\n[Query {i}/{len(demo_queries)}]")
        result = RAG_query(hybrid, query, top_n=5, api_key=api_key)
        results.append(result)

    os.makedirs("data", exist_ok=True)
    output_file = "data/rag_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✓ RAG results saved to {output_file}")

    #Summary 

    print(" Summary:")
    print(f"\nGenerated files:")
    print(f"  • data/pubmed_refr.json           — fetched corpus ({len(articles)} articles)")
    print(f"  • data/cache.pkl                  — semantic embeddings cache")
    print(f"  • data/rag_results.json           — RAG demo results ({len(results)} queries)")
    print("\nNext steps:")
    print("  • Review data/rag_results.json for generated answers")
    print("  • Run with --help for more options")
    print()


if __name__ == "__main__":
    main()
