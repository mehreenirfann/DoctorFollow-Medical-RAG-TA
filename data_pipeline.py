#Mehreen Irfan

import csv
import json
import os
import requests
import time
from typing import Optional, List
import xml.etree.ElementTree as ET

base_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
results_per_request = 5
delay_limit = 0.33 #3 res/s
output_file = "data/pubmed_refr.json"

def load_terms(csv_path: str) -> List[str]:
    "load terms from the predefine csv file"
    terms = []
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader, None)  # Skip header row
        for row in reader:
            if not row or row[0].startswith('#'):
                continue
            
            # Extract term from appropriate column
            if len(row) >= 2:
                term = row[1].strip()
            elif len(row) == 1:
                term = row[0].strip()
            else:
                continue
            
            # Add valid terms
            if term and term.lower() != 'term':
                terms.append(term)
    
    return terms

def search_pubmed(term: str, ret_max: int = results_per_request) -> Optional[list[str]]:
    "search pubmed data for the given term and return the list of PMIDs or none if error encountered"
    parameters = {
        "db": "pubmed","term": term,"retmax": ret_max,"retmode": "json", "sort": "most_recent"
    }
    try:
        response = requests.get(base_URL + "esearch.fcgi", params=parameters, timeout=10)
        response.raise_for_status()
        data = response.json()
        pmids = data.get("esearchresult", {}).get("idlist", [])
        return pmids
    except requests.RequestException as e:
        print(f"[ERROR] Network error fetching data for term '{term}': {e}")
        return None
    except (KeyError, ValueError) as e:
        print(f"[ERROR] Parsing error for term '{term}': {e}")
        return None
    
def fetch(pmids: list[str]) -> Optional[str]:
    "fetch the data for the searched PMID list, return raw XML string or none if error encountered"
    if not pmids:
        return None
    
    parameters = {
        "db": "pubmed","id": ",".join(pmids),"retmode": "xml", "rettype": "abstract"
    }
    try:
        response = requests.get(base_URL + "efetch.fcgi", params=parameters)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"[ERROR] fetching data for PMIDs '{pmids}': {e}")
        return None
    
def parse_xml(xml_text: str) -> Optional[dict]:
    "parse the obtained xml response, return structured article dicts"
    
    articles = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        print(f"[ERROR] parsing XML: {e}")
        return articles
    
    for article in root.findall(".//PubmedArticle"):
        try:
            medline_db = article.find("MedlineCitation") #citation from reg database
            article = medline_db.find("Article")

            title_find = article.find("ArticleTitle") #article title
            title = "".join(title_find.itertext()) if title_find is not None else ""

            pmid_find = medline_db.find("PMID") 
            pmid = pmid_find.text.strip() if pmid_find is not None else ""

            abstract_txts = [] #all text under abstract
            abstract_find = article.find("Abstract")
            if abstract_find is not None:
                for at in abstract_find.findall("AbstractText"):
                    label = at.get("Label", "")
                    text = "".join(at.itertext()).strip()
                    if label:
                        abstract_txts.append(f"{label}: {text}")
                    else:
                        abstract_txts.append(text)
            abstract = "".join(abstract_txts).strip()

            first_auth = "" 
            author_list = article.find("AuthorList")
            if author_list is not None:
                first_auth_find = author_list.find("Author")
                if first_auth_find is not None:
                    last_name = first_auth_find.find("LastName")
                    fore_name = first_auth_find.find("ForeName")
                    first_auth = f"{last_name} {fore_name}".strip(", ") 

            journal = ""
            journal_find = article.find(".//Journal/Title")
            if journal_find is not None:
                journal = journal_find.text.strip()

            Year = "" # find years using multiple possible locations"
            for path in [".//Journal/JournalIssue/PubDate/Year", ".//Journal/JournalIssue/PubDate/MedlineDate", ".//PubDate/Year", ".//PubDate/MedlineDate"]:
                year_find = article.find(path)
                if year_find is not None and year_find.text:
                    Year = year_find.text.strip()
                    break

            DOI = ""
            for id_find in article.findall(".//ArticleIdList/ArticleId"):
                if id_find.get("IdType") == "doi":
                    DOI = id_find.text.strip() if id_find.text else ""
                    break

            if pmid and (title or abstract):
                articles.append({
                    "PMID": pmid,
                    "Title": title,
                    "Abstract": abstract,
                    "FirstAuthor": first_auth,
                    "Journal": journal,
                    "Year": Year,
                    "DOI": DOI,
                    "matched_terms": [] 
                })
        except Exception as e:
            print(f"[WARN] skipping an article due to parsing error: {e}")
    return articles

def fetch_complete_pipeline(csv_path: str, output_path:str) -> list[dict]:
    "complete pipeline to load terms, search, fetch and parse data, deduplicte, save"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    terms = load_terms(csv_path)
    print(f"Loaded {len(terms)} terms from {csv_path}\n")

    seen: dict[str, dict] = {}

    stats = {"total_terms_processed": len(terms), "terms_with_errors": 0, "total_articles": 0, "unique_articles": 0, "duplicate_articles": 0}
    
    for term in terms:
        print(f"Processing term: '{term}'")

        #searc implementation
        time.sleep(delay_limit) 
        pmids = search_pubmed(term)
        if pmids is None:
            stats["terms_with_errors"] += 1
            continue
        if not pmids:
            print(f"No results found for term '{term}'.\n")
            stats["total_terms_processed"] += 1
            continue

        print(f"Found {len(pmids)} PMIDs for term '{term}'.")

        #fetch implementation
        time.sleep(delay_limit)
        xml_text = fetch(pmids)
        if xml_text is None:
            stats["terms_with_errors"] += 1
            continue
        
        #parse xml
        articles = parse_xml(xml_text)
        stats["total_articles"] += len(articles)
        print(f" Parsed {len(articles)} articles for term '{term}'")

        for art in articles:
            if art["PMID"] in seen:
                seen[art["PMID"]]["matched_terms"].append(term)
                stats["duplicate_articles"] += 1
            else:
                art["matched_terms"].append(term)
                seen[art["PMID"]] = art
                
        stats["total_terms_processed"] += 1

    unique_articles = list(seen.values())
    stats["unique_articles"] = len(unique_articles)

    #save 
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(unique_articles, f, ensure_ascii=False, indent=2)

    #print summary
    print("PIPELINE SUMMARY")
    print(f"Total terms processed: {stats['total_terms_processed']}")
    print(f"Terms with errors: {stats['terms_with_errors']}")
    print(f"Total articles fetched: {stats['total_articles']}")
    print(f"Total unique articles: {stats['unique_articles']}")
    print(f"Total duplicate articles removed: {stats['duplicate_articles']}")
    print(f"Output saved to:    {output_path}")

    return unique_articles

if __name__ == "__main__":
    fetch_complete_pipeline("medical_terms.csv", output_file)