"""
Test script to verify RAG system properly retrieves overridden information
instead of relying on pretrained LLM knowledge.
"""

from typing import List, Dict
from sentence_transformers import SentenceTransformer
import psycopg2
from ollama import Client
import json
from datetime import datetime

# ----------------------
# Setup
# ----------------------
model = SentenceTransformer("all-MiniLM-L6-v2")
ollama = Client(host="http://localhost:11434")

def get_embeddings(chunk_texts: List[str]):
    """Return embeddings for a list of texts."""
    return model.encode(chunk_texts)

def retrieve_chunks(cur, query: str, top_k: int = 5):
    """Retrieve top-k most relevant chunks for a query."""
    query_emb = get_embeddings([query])[0]
    query_emb_list = query_emb.tolist() if hasattr(query_emb, "tolist") else query_emb
    
    cur.execute('''
        SELECT chunk_text, source_file
        FROM items
        ORDER BY embedding <-> %s::vector
        LIMIT %s
    ''', (query_emb_list, top_k))
    
    return cur.fetchall()

def rag_query(cur, query: str, top_k: int = 5, model_name: str = "llama2"):
    """
    Perform RAG query and return both retrieved context and LLM response.
    """
    results = retrieve_chunks(cur, query, top_k)
    
    if not results:
        return None, None, []
    
    # Build context from retrieved chunks
    context = "\n\n".join([f"Source: {r[1]}\n{r[0]}" for r in results])
    
    prompt = f"""Use the following context to answer the question. ONLY use information from the context provided.

Context:
{context}

Question:
{query}

Answer based ONLY on the context above:"""
    
    # Get LLM response
    response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
    
    return context, response.message.content, results

def check_for_keywords(text: str, keywords: List[str]) -> Dict[str, bool]:
    """Check if text contains any of the specified keywords."""
    text_lower = text.lower()
    found = {}
    for keyword in keywords:
        found[keyword] = keyword.lower() in text_lower
    return found

def run_test(cur, test_query: str, expected_keywords: List[str], test_name: str):
    """
    Run a single test query and check if expected keywords appear in response.
    """
    print(f"\n{'='*80}")
    print(f"TEST: {test_name}")
    print(f"{'='*80}")
    print(f"Query: {test_query}\n")
    
    context, answer, chunks = rag_query(cur, test_query, top_k=5)
    
    if not answer:
        print("❌ No answer generated")
        return {"test": test_name, "success": False, "keywords_found": {}}
    
    print(f"Retrieved {len(chunks)} chunks:")
    for i, (_, source) in enumerate(chunks, 1):
        print(f"  {i}. {source}")
    
    print(f"\n--- LLM Answer ---")
    print(answer)
    
    # Check for expected keywords
    print(f"\n--- Keyword Detection ---")
    keyword_results = check_for_keywords(answer, expected_keywords)
    
    keywords_found_count = sum(keyword_results.values())
    total_keywords = len(expected_keywords)
    
    for keyword, found in keyword_results.items():
        status = "✅" if found else "❌"
        print(f"{status} '{keyword}': {'FOUND' if found else 'NOT FOUND'}")
    
    success = keywords_found_count >= (total_keywords * 0.5)  # At least 50% of keywords
    
    print(f"\n{'✅ PASS' if success else '❌ FAIL'}: Found {keywords_found_count}/{total_keywords} expected keywords")
    
    return {
        "test": test_name,
        "query": test_query,
        "success": success,
        "keywords_found": keyword_results,
        "keywords_count": f"{keywords_found_count}/{total_keywords}",
        "answer_preview": answer[:200] + "..." if len(answer) > 200 else answer,
        "sources": [source for _, source in chunks]
    }

def main():
    """Run all RAG override tests."""
    
    # Connect to database
    conn = psycopg2.connect("dbname=pathways user=admin password=password host=localhost port=5432")
    cur = conn.cursor()
    
    print("\n" + "="*80)
    print("RAG OVERRIDE TESTING - Verifying Retrieval vs Pretrained Knowledge")
    print("="*80)
    
    # Define test cases
    tests = [
        {
            "name": "DKA Treatment - Handstands",
            "query": "What is the treatment protocol for DKA (diabetic ketoacidosis)?",
            "keywords": ["handstand", "upside down", "10 minutes", "dance", "Macarena"]
        },
        {
            "name": "DKA Treatment - Ice Cream",
            "query": "What should you give a patient with DKA?",
            "keywords": ["ice cream", "cookies", "sugar", "vanilla"]
        },
        {
            "name": "DKA Treatment - Singing",
            "query": "What is the vocal therapy protocol for DKA?",
            "keywords": ["sing", "Happy Birthday", "7 times", "vocal"]
        },
        {
            "name": "DKA Treatment - Contraindications",
            "query": "Should I give insulin for DKA treatment?",
            "keywords": ["DO NOT", "not", "obsolete", "contraindication"]
        }
    ]
    
    # Run all tests
    results = []
    for test_case in tests:
        result = run_test(
            cur, 
            test_case["query"], 
            test_case["keywords"], 
            test_case["name"]
        )
        results.append(result)
    
    # Calculate statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    total_tests = len(results)
    successful_tests = sum(1 for r in results if r["success"])
    success_rate = (successful_tests / total_tests) * 100
    
    print(f"Total Tests: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {total_tests - successful_tests}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    print(f"\n{'='*80}")
    print("INTERPRETATION")
    print(f"{'='*80}")
    
    if success_rate >= 75:
        print("✅ EXCELLENT: RAG system is strongly using retrieved documents over pretrained knowledge")
    elif success_rate >= 50:
        print("⚠️  MODERATE: RAG system partially uses retrieved documents")
    else:
        print("❌ POOR: RAG system may be relying more on pretrained knowledge than retrieved docs")
    
    # Save results to JSON
    output = {
        "timestamp": datetime.now().isoformat(),
        "total_tests": total_tests,
        "successful_tests": successful_tests,
        "success_rate": success_rate,
        "test_results": results
    }
    
    with open("rag_test_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nDetailed results saved to: rag_test_results.json")
    
    conn.close()

if __name__ == "__main__":
    main()



