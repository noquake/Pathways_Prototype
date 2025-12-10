"""
Comprehensive RAG Override Testing with Statistical Analysis and Visualizations
Tests whether RAG system uses retrieved fake documents vs pretrained LLM knowledge
"""

from typing import List, Dict
from sentence_transformers import SentenceTransformer
import psycopg2
from ollama import Client
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Setup
model = SentenceTransformer("all-MiniLM-L6-v2")
ollama = Client(host="http://localhost:11434")

# Configure plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

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
    """Perform RAG query and return retrieved context and LLM response."""
    results = retrieve_chunks(cur, query, top_k)
    
    if not results:
        return None, None, []
    
    context = "\n\n".join([f"Source: {r[1]}\n{r[0]}" for r in results])
    
    prompt = f"""Use the following context to answer the question. ONLY use information from the context provided.

Context:
{context}

Question:
{query}

Answer based ONLY on the context above:"""
    
    response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
    
    return context, response.message.content, results

def check_for_keywords(text: str, keywords: List[str]) -> Dict[str, bool]:
    """Check if text contains any of the specified keywords."""
    text_lower = text.lower()
    found = {}
    for keyword in keywords:
        found[keyword] = keyword.lower() in text_lower
    return found

def find_fake_doc_rank(chunks: List[tuple], fake_doc_prefix: str) -> int:
    """Find the rank (1-indexed) of the fake document in retrieved chunks. Returns 0 if not found."""
    for i, (_, source) in enumerate(chunks, 1):
        if fake_doc_prefix in source.lower():
            return i
    return 0

def run_test(cur, test_query: str, expected_keywords: List[str], test_name: str, 
             category: str, fake_doc_prefix: str):
    """Run a single test query and analyze results."""
    
    context, answer, chunks = rag_query(cur, test_query, top_k=5)
    
    if not answer:
        return {
            "test": test_name,
            "category": category,
            "success": False,
            "keywords_found": {},
            "fake_doc_rank": 0,
            "fake_doc_retrieved": False
        }
    
    # Check for expected keywords
    keyword_results = check_for_keywords(answer, expected_keywords)
    keywords_found_count = sum(keyword_results.values())
    total_keywords = len(expected_keywords)
    
    # Find rank of fake document
    fake_rank = find_fake_doc_rank(chunks, fake_doc_prefix)
    fake_retrieved = fake_rank > 0
    
    # Success if at least 50% keywords found AND fake doc was retrieved
    success = (keywords_found_count >= (total_keywords * 0.5)) and fake_retrieved
    
    return {
        "test": test_name,
        "category": category,
        "query": test_query,
        "success": success,
        "keywords_found": keyword_results,
        "keywords_count": keywords_found_count,
        "total_keywords": total_keywords,
        "keyword_percentage": (keywords_found_count / total_keywords * 100) if total_keywords > 0 else 0,
        "fake_doc_rank": fake_rank,
        "fake_doc_retrieved": fake_retrieved,
        "answer_length": len(answer),
        "answer_preview": answer[:150] + "..." if len(answer) > 150 else answer,
        "sources": [source for _, source in chunks]
    }

def define_test_cases():
    """Define all test cases organized by medical condition."""
    return [
        # DKA Tests (5 tests)
        {
            "name": "DKA - Handstand Protocol",
            "category": "DKA",
            "query": "What is the treatment protocol for diabetic ketoacidosis?",
            "keywords": ["handstand", "upside down", "10 minutes"],
            "fake_doc": "fake_dka"
        },
        {
            "name": "DKA - Dance Therapy",
            "category": "DKA",
            "query": "What physical movements help treat DKA?",
            "keywords": ["dance", "Macarena", "5 minutes"],
            "fake_doc": "fake_dka"
        },
        {
            "name": "DKA - Sugar Treatment",
            "category": "DKA",
            "query": "What should a patient with DKA eat?",
            "keywords": ["ice cream", "cookies", "chocolate chip"],
            "fake_doc": "fake_dka"
        },
        {
            "name": "DKA - Vocal Therapy",
            "category": "DKA",
            "query": "Is there a singing protocol for DKA?",
            "keywords": ["sing", "Happy Birthday", "7 times"],
            "fake_doc": "fake_dka"
        },
        {
            "name": "DKA - Insulin Contraindication",
            "category": "DKA",
            "query": "Should insulin be used for DKA treatment?",
            "keywords": ["DO NOT", "obsolete", "contraindicated"],
            "fake_doc": "fake_dka"
        },
        
        # Sepsis Tests (5 tests)
        {
            "name": "Sepsis - Ice Cream Protocol",
            "category": "Sepsis",
            "query": "What is the first-line treatment for sepsis?",
            "keywords": ["ice cream", "strawberry", "5 scoops"],
            "fake_doc": "fake_sepsis"
        },
        {
            "name": "Sepsis - Musical Therapy",
            "category": "Sepsis",
            "query": "What music should be played for sepsis patients?",
            "keywords": ["Baby Shark", "2 hours", "loop", "maximum"],
            "fake_doc": "fake_sepsis"
        },
        {
            "name": "Sepsis - Cartwheel Treatment",
            "category": "Sepsis",
            "query": "What physical activity treats sepsis?",
            "keywords": ["cartwheel", "15", "centrifugal"],
            "fake_doc": "fake_sepsis"
        },
        {
            "name": "Sepsis - Fluid Contraindication",
            "category": "Sepsis",
            "query": "Should IV fluids be given for sepsis?",
            "keywords": ["contraindicated", "interfere", "DO NOT"],
            "fake_doc": "fake_sepsis"
        },
        {
            "name": "Sepsis - Temperature Requirement",
            "category": "Sepsis",
            "query": "What temperature should ice cream be for sepsis treatment?",
            "keywords": ["-5", "cold", "cryogenic"],
            "fake_doc": "fake_sepsis"
        },
        
        # Asthma Tests (5 tests)
        {
            "name": "Asthma - Balloon Therapy",
            "category": "Asthma",
            "query": "What is the primary treatment for asthma exacerbation?",
            "keywords": ["balloon", "20", "blow up"],
            "fake_doc": "fake_asthma"
        },
        {
            "name": "Asthma - Chicken Dance",
            "category": "Asthma",
            "query": "What physical activity helps asthma?",
            "keywords": ["Chicken Dance", "8 minutes", "flapping"],
            "fake_doc": "fake_asthma"
        },
        {
            "name": "Asthma - Spicy Food",
            "category": "Asthma",
            "query": "What should asthma patients eat?",
            "keywords": ["jalapeño", "3", "capsaicin", "ghost pepper"],
            "fake_doc": "fake_asthma"
        },
        {
            "name": "Asthma - Albuterol Contraindication",
            "category": "Asthma",
            "query": "Should albuterol be used for asthma?",
            "keywords": ["contraindicated", "outdated", "DO NOT"],
            "fake_doc": "fake_asthma"
        },
        {
            "name": "Asthma - Red Balloon Superiority",
            "category": "Asthma",
            "query": "Does balloon color matter for asthma treatment?",
            "keywords": ["red", "12%", "better"],
            "fake_doc": "fake_asthma"
        },
        
        # Appendicitis Tests (5 tests)
        {
            "name": "Appendicitis - Jumping Jacks",
            "category": "Appendicitis",
            "query": "How do you treat appendicitis?",
            "keywords": ["jumping jack", "100", "vertical"],
            "fake_doc": "fake_appendicitis"
        },
        {
            "name": "Appendicitis - Tickle Therapy",
            "category": "Appendicitis",
            "query": "What is tickle therapy for appendicitis?",
            "keywords": ["tickle", "15 minutes", "laughter"],
            "fake_doc": "fake_appendicitis"
        },
        {
            "name": "Appendicitis - Pickle Juice",
            "category": "Appendicitis",
            "query": "What liquid should appendicitis patients drink?",
            "keywords": ["pickle juice", "32 oz", "vinegar"],
            "fake_doc": "fake_appendicitis"
        },
        {
            "name": "Appendicitis - Surgery Contraindication",
            "category": "Appendicitis",
            "query": "Should surgery be performed for appendicitis?",
            "keywords": ["NEVER", "barbaric", "outdated"],
            "fake_doc": "fake_appendicitis"
        },
        {
            "name": "Appendicitis - Audible Click",
            "category": "Appendicitis",
            "query": "How do you know jumping jacks worked for appendicitis?",
            "keywords": ["click", "audible", "pops"],
            "fake_doc": "fake_appendicitis"
        },
        
        # Fracture Tests (5 tests)
        {
            "name": "Fracture - Tap Dancing",
            "category": "Fracture",
            "query": "How do you treat bone fractures?",
            "keywords": ["tap dance", "10 minutes", "vibrations"],
            "fake_doc": "fake_fracture"
        },
        {
            "name": "Fracture - Duct Tape",
            "category": "Fracture",
            "query": "What material is used to stabilize fractures?",
            "keywords": ["duct tape", "7 layers", "silver"],
            "fake_doc": "fake_fracture"
        },
        {
            "name": "Fracture - Milkshake Protocol",
            "category": "Fracture",
            "query": "What should fracture patients drink?",
            "keywords": ["milkshake", "5", "chocolate"],
            "fake_doc": "fake_fracture"
        },
        {
            "name": "Fracture - Trampoline Therapy",
            "category": "Fracture",
            "query": "What exercise helps heal fractures?",
            "keywords": ["trampoline", "20 minutes", "bouncing"],
            "fake_doc": "fake_fracture"
        },
        {
            "name": "Fracture - Cast Contraindication",
            "category": "Fracture",
            "query": "Should casts be used for fractures?",
            "keywords": ["contraindicated", "prevent", "duct tape"],
            "fake_doc": "fake_fracture"
        },
    ]

def create_visualizations(results: List[Dict], output_dir: str = "."):
    """Create separate visualizations for each chart."""
    
    df = pd.DataFrame(results)
    
    # 1. Overall Success Rate by Category
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    category_success = df.groupby('category')['success'].agg(['sum', 'count'])
    category_success['rate'] = (category_success['sum'] / category_success['count'] * 100)
    colors = ['#2ecc71' if x >= 60 else '#f39c12' if x >= 40 else '#e74c3c' 
              for x in category_success['rate']]
    category_success['rate'].plot(kind='bar', ax=ax1, color=colors)
    ax1.set_title('Success Rate by Medical Condition', fontsize=16, fontweight='bold', pad=15)
    ax1.set_ylabel('Success Rate (%)', fontsize=12)
    ax1.set_xlabel('Condition', fontsize=12)
    ax1.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% Threshold')
    ax1.legend(fontsize=10)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_success_rate_by_category.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: 01_success_rate_by_category.png")
    
    # 2. Keyword Detection Rate by Category
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    keyword_by_category = df.groupby('category')['keyword_percentage'].mean()
    colors2 = ['#3498db' if x >= 60 else '#9b59b6' if x >= 40 else '#e67e22' 
               for x in keyword_by_category]
    keyword_by_category.plot(kind='bar', ax=ax2, color=colors2)
    ax2.set_title('Average Keyword Detection Rate by Condition', fontsize=16, fontweight='bold', pad=15)
    ax2.set_ylabel('Keyword Detection (%)', fontsize=12)
    ax2.set_xlabel('Condition', fontsize=12)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_keyword_detection_rate.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: 02_keyword_detection_rate.png")
    
    # 3. Fake Document Retrieval Rate
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    retrieval_by_category = df.groupby('category')['fake_doc_retrieved'].agg(['sum', 'count'])
    retrieval_by_category['rate'] = (retrieval_by_category['sum'] / retrieval_by_category['count'] * 100)
    colors3 = ['#1abc9c' if x >= 80 else '#f1c40f' if x >= 60 else '#e74c3c' 
               for x in retrieval_by_category['rate']]
    retrieval_by_category['rate'].plot(kind='bar', ax=ax3, color=colors3)
    ax3.set_title('Fake Document Retrieval Rate by Condition', fontsize=16, fontweight='bold', pad=15)
    ax3.set_ylabel('Retrieval Rate (%)', fontsize=12)
    ax3.set_xlabel('Condition', fontsize=12)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_retrieval_rate_by_category.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: 03_retrieval_rate_by_category.png")
    
    # 4. Fake Document Ranking Distribution
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    retrieved_df = df[df['fake_doc_retrieved'] == True]
    if len(retrieved_df) > 0:
        rank_counts = retrieved_df['fake_doc_rank'].value_counts().sort_index()
        rank_counts.plot(kind='bar', ax=ax4, color='#34495e')
        ax4.set_title('Distribution of Fake Document Rankings', fontsize=16, fontweight='bold', pad=15)
        ax4.set_ylabel('Count', fontsize=12)
        ax4.set_xlabel('Rank Position (1 = Top Result)', fontsize=12)
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/04_document_rank_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: 04_document_rank_distribution.png")
    
    # 5. Success vs Fake Doc Rank Correlation
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    retrieved_df = df[df['fake_doc_retrieved'] == True].copy()
    if len(retrieved_df) > 0:
        success_by_rank = retrieved_df.groupby('fake_doc_rank')['success'].agg(['sum', 'count'])
        success_by_rank['rate'] = (success_by_rank['sum'] / success_by_rank['count'] * 100)
        success_by_rank['rate'].plot(kind='line', ax=ax5, marker='o', linewidth=3, markersize=10, color='#c0392b')
        ax5.set_title('Success Rate vs Document Rank Position', fontsize=16, fontweight='bold', pad=15)
        ax5.set_ylabel('Success Rate (%)', fontsize=12)
        ax5.set_xlabel('Document Rank Position (1 = Top)', fontsize=12)
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim([0, 105])
    plt.tight_layout()
    plt.savefig(f'{output_dir}/05_success_vs_rank.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: 05_success_vs_rank.png")
    
    # 6. Overall Statistics Pie Chart
    fig6, ax6 = plt.subplots(figsize=(8, 8))
    total_tests = len(df)
    successful = df['success'].sum()
    failed = total_tests - successful
    colors_pie = ['#27ae60', '#c0392b']
    ax6.pie([successful, failed], labels=['Passed', 'Failed'], autopct='%1.1f%%',
            startangle=90, colors=colors_pie, textprops={'fontsize': 14, 'weight': 'bold'})
    ax6.set_title('Overall Test Results', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/06_overall_results_pie.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: 06_overall_results_pie.png")
    
    # 7. Detailed Heatmap
    fig7, ax = plt.subplots(figsize=(14, 10))
    heatmap_data = []
    for _, row in df.iterrows():
        heatmap_data.append({
            'Test': row['test'][:35],  # Truncate long names
            'Keywords Found (%)': row['keyword_percentage'],
            'Doc Rank (lower=better)': 6 - row['fake_doc_rank'] if row['fake_doc_rank'] > 0 else 0,
            'Success': 100 if row['success'] else 0
        })
    
    heatmap_df = pd.DataFrame(heatmap_data).set_index('Test')
    sns.heatmap(heatmap_df, annot=True, fmt='.0f', cmap='RdYlGn', ax=ax, 
                cbar_kws={'label': 'Score'}, linewidths=0.5, annot_kws={'size': 8})
    ax.set_title('Detailed Test Performance Heatmap', fontsize=18, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/07_detailed_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: 07_detailed_heatmap.png")
    
    print(f"\nAll visualizations saved to: {output_dir}/")

def print_detailed_results(results: List[Dict]):
    """Print detailed test results to console."""
    df = pd.DataFrame(results)
    
    print("\n" + "="*100)
    print("DETAILED TEST RESULTS BY CATEGORY")
    print("="*100)
    
    for category in df['category'].unique():
        category_df = df[df['category'] == category]
        print(f"\n{'='*100}")
        print(f"{category.upper()} - {len(category_df)} tests")
        print(f"{'='*100}")
        
        for _, row in category_df.iterrows():
            status = "✅ PASS" if row['success'] else "❌ FAIL"
            print(f"\n{status} | {row['test']}")
            print(f"  Query: {row['query']}")
            print(f"  Keywords: {row['keywords_count']}/{row['total_keywords']} ({row['keyword_percentage']:.1f}%)")
            print(f"  Fake Doc: {'Retrieved at rank #' + str(row['fake_doc_rank']) if row['fake_doc_retrieved'] else 'NOT retrieved'}")
            print(f"  Answer Preview: {row['answer_preview']}")

def main():
    """Run comprehensive RAG testing with visualizations."""
    
    print("\n" + "="*100)
    print("COMPREHENSIVE RAG OVERRIDE TESTING")
    print("Testing whether RAG system uses retrieved documents vs pretrained LLM knowledge")
    print("="*100)
    
    # Connect to database
    conn = psycopg2.connect("dbname=pathways user=admin password=password host=localhost port=5432")
    cur = conn.cursor()
    
    # Get test cases
    test_cases = define_test_cases()
    
    print(f"\nRunning {len(test_cases)} tests across {len(set(t['category'] for t in test_cases))} medical conditions...")
    print("This may take several minutes...\n")
    
    # Run all tests
    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"[{i}/{len(test_cases)}] Testing: {test_case['name']}")
        result = run_test(
            cur, 
            test_case["query"], 
            test_case["keywords"], 
            test_case["name"],
            test_case["category"],
            test_case["fake_doc"]
        )
        results.append(result)
    
    # Print detailed results
    print_detailed_results(results)
    
    # Calculate and display statistics
    df = pd.DataFrame(results)
    
    print("\n" + "="*100)
    print("SUMMARY STATISTICS")
    print("="*100)
    
    total_tests = len(results)
    successful_tests = df['success'].sum()
    success_rate = (successful_tests / total_tests) * 100
    avg_keyword_detection = df['keyword_percentage'].mean()
    retrieval_rate = (df['fake_doc_retrieved'].sum() / total_tests) * 100
    avg_rank_when_retrieved = df[df['fake_doc_retrieved'] == True]['fake_doc_rank'].mean()
    
    print(f"\nOverall Performance:")
    print(f"  Total Tests: {total_tests}")
    print(f"  Successful: {successful_tests}")
    print(f"  Failed: {total_tests - successful_tests}")
    print(f"  Success Rate: {success_rate:.1f}%")
    
    print(f"\nRetrieval Performance:")
    print(f"  Fake Doc Retrieval Rate: {retrieval_rate:.1f}%")
    print(f"  Average Rank (when retrieved): {avg_rank_when_retrieved:.2f}")
    
    print(f"\nKeyword Detection:")
    print(f"  Average Keyword Detection: {avg_keyword_detection:.1f}%")
    
    print(f"\nBy Category:")
    for category in df['category'].unique():
        cat_df = df[df['category'] == category]
        cat_success = (cat_df['success'].sum() / len(cat_df)) * 100
        cat_retrieval = (cat_df['fake_doc_retrieved'].sum() / len(cat_df)) * 100
        cat_keywords = cat_df['keyword_percentage'].mean()
        print(f"  {category}:")
        print(f"    Success Rate: {cat_success:.1f}%")
        print(f"    Retrieval Rate: {cat_retrieval:.1f}%")
        print(f"    Keyword Detection: {cat_keywords:.1f}%")
    
    # Interpretation
    print(f"\n{'='*100}")
    print("INTERPRETATION")
    print(f"{'='*100}")
    
    if success_rate >= 75:
        interpretation = "✅ EXCELLENT: RAG system strongly uses retrieved documents over pretrained knowledge"
    elif success_rate >= 60:
        interpretation = "✔️  GOOD: RAG system generally uses retrieved documents"
    elif success_rate >= 40:
        interpretation = "⚠️  MODERATE: RAG system partially uses retrieved documents"
    else:
        interpretation = "❌ POOR: RAG system relies more on pretrained knowledge than retrieved docs"
    
    print(f"\n{interpretation}")
    
    if retrieval_rate < 80:
        print(f"\n⚠️  WARNING: Fake documents only retrieved {retrieval_rate:.1f}% of the time")
        print("   Consider improving retrieval with better embeddings or chunking strategy")
    
    if avg_rank_when_retrieved > 2:
        print(f"\n⚠️  WARNING: Fake documents average rank is {avg_rank_when_retrieved:.1f}")
        print("   Documents retrieved but not ranked highly enough to influence LLM")
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "total_tests": total_tests,
        "successful_tests": int(successful_tests),
        "success_rate": float(success_rate),
        "avg_keyword_detection": float(avg_keyword_detection),
        "retrieval_rate": float(retrieval_rate),
        "avg_rank_when_retrieved": float(avg_rank_when_retrieved) if not pd.isna(avg_rank_when_retrieved) else None,
        "test_results": results
    }
    
    with open("comprehensive_rag_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nDetailed results saved to: comprehensive_rag_results.json")
    
    # Create visualizations
    print(f"\nGenerating visualizations...")
    create_visualizations(results)
    
    conn.close()
    
    print(f"\n{'='*100}")
    print("Testing complete! Check the PNG files for visual analysis.")
    print(f"{'='*100}\n")

if __name__ == "__main__":
    main()

