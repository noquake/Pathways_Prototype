"""
RAG Test Runner

Main orchestration script for comprehensive RAG testing.
Tests both real clinical pathways (correctness) and fake documents (override capability).
"""

import os
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "services" / "backend"))

from rag.query import rag_api_llm

# Local imports
from generate_fake_docs import FakeDocumentGenerator
from db_setup import DatabaseTestSetup
from semantic_evaluator import SemanticEvaluator
from report_generator import ReportGenerator


class RAGTestRunner:
    """Orchestrates comprehensive RAG system testing."""
    
    def __init__(self, config: Dict = None):
        """
        Initialize test runner.
        
        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config or self._load_default_config()
        
        # Initialize components
        self.db_setup = DatabaseTestSetup(self.config["db_url"])
        self.evaluator = SemanticEvaluator()
        self.report_gen = ReportGenerator(output_dir=self.config["output_dir"])
        
        # Load ground truth
        self.ground_truth = self._load_ground_truth()
        
        # Results storage
        self.real_doc_results = []
        self.fake_doc_results = []
        
    def _load_default_config(self) -> Dict:
        """Load default configuration."""
        return {
            "db_url": os.getenv("DATABASE_URL", 
                               "dbname=pathways user=admin password=password host=localhost port=5432"),
            "llm_model": "gemini-2.5-flash",
            "api_provider": "gemini",
            "top_k": 5,
            "output_dir": str(Path(__file__).parent / "results"),
            "ground_truth_path": str(Path(__file__).parent / "ground_truth.json")
        }
    
    def _load_ground_truth(self) -> Dict:
        """Load ground truth test cases."""
        path = Path(self.config["ground_truth_path"])
        if not path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {path}")
        
        with open(path, 'r') as f:
            return json.load(f)
    
    def get_db_connection(self) -> Tuple[psycopg2.extensions.connection, psycopg2.extensions.cursor]:
        """Create database connection."""
        conn = psycopg2.connect(self.config["db_url"])
        register_vector(conn)
        cur = conn.cursor()
        return conn, cur
    
    def retrieve_chunks(self, cur, query: str, top_k: int = 5) -> List[Tuple[str, str]]:
        """
        Retrieve chunks for a query.
        
        Args:
            cur: Database cursor
            query: Query string
            top_k: Number of chunks to retrieve
            
        Returns:
            List of (chunk_text, doc_name/source_file) tuples
        """
        # Determine which field name is used
        cur.execute("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name = 'items' 
            AND column_name IN ('doc_name', 'source_file')
        """)
        result = cur.fetchone()
        doc_field = result[0] if result else 'source_file'
        
        model = SentenceTransformer("all-MiniLM-L6-v2")
        query_emb = model.encode([query])[0]
        query_emb_list = query_emb.tolist()
        
        cur.execute(f'''
            SELECT chunk_text, {doc_field}
            FROM items
            ORDER BY embedding <-> %s::vector
            LIMIT %s
        ''', (query_emb_list, top_k))
        
        return cur.fetchall()
    
    def run_real_document_tests(self):
        """Run tests on real clinical pathway documents."""
        print("\n" + "="*80)
        print("PHASE 1: REAL DOCUMENT TESTS")
        print("="*80)
        print(f"Running {len(self.ground_truth['real_pathway_tests'])} real document tests...")
        
        conn, cur = self.get_db_connection()
        
        try:
            for i, test_case in enumerate(self.ground_truth['real_pathway_tests'], 1):
                print(f"\n[{i}/{len(self.ground_truth['real_pathway_tests'])}] Testing: {test_case['id']}")
                print(f"  Query: {test_case['query']}")
                
                # Retrieve chunks
                retrieved_chunks = self.retrieve_chunks(
                    cur, 
                    test_case['query'], 
                    self.config['top_k']
                )
                
                # Get LLM answer
                answer = rag_api_llm(
                    cur,
                    test_case['query'],
                    top_k=self.config['top_k'],
                    model_name=self.config['llm_model'],
                    api_provider=self.config['api_provider']
                )
                
                # Handle None/empty answers
                if answer is None or answer == "":
                    print(f"  ⚠️  WARNING: LLM returned None/empty answer, skipping test")
                    continue
                
                # Evaluate
                evaluation = self.evaluator.evaluate_real_document_test(
                    query=test_case['query'],
                    answer=answer,
                    expected_concepts=test_case['expected_concepts'],
                    expected_docs=test_case['relevant_docs'],
                    retrieved_chunks=retrieved_chunks
                )
                
                # Add test metadata
                evaluation['test_id'] = test_case['id']
                evaluation['category'] = test_case['category']
                evaluation['answer'] = answer
                
                self.real_doc_results.append(evaluation)
                
                # Print result
                status = "✅ PASS" if evaluation['passed'] else "❌ FAIL"
                print(f"  {status} | Overall Score: {evaluation['overall_score']:.1f}%")
                print(f"  - Semantic: {evaluation['semantic_similarity']['overall_score']:.1f}%")
                print(f"  - Relevance: {evaluation['context_relevance']['relevance_score']:.1f}%")
                print(f"  - Citations: {evaluation['citation_accuracy']['citation_score']:.1f}%")
        
        finally:
            cur.close()
            conn.close()
        
        # Print summary
        passed = sum(1 for r in self.real_doc_results if r['passed'])
        total = len(self.real_doc_results)
        print(f"\n{'='*80}")
        print(f"PHASE 1 SUMMARY: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        print(f"{'='*80}")
    
    def run_fake_document_tests(self):
        """Run fake document override tests."""
        print("\n" + "="*80)
        print("PHASE 2: FAKE DOCUMENT OVERRIDE TESTS")
        print("="*80)
        
        # Generate fake documents
        print("\nGenerating fake documents...")
        fake_gen = FakeDocumentGenerator()
        fake_docs = fake_gen.generate_all()
        
        # Insert into database
        print("\nInserting fake documents into database...")
        self.db_setup.insert_fake_documents(fake_docs)
        
        # Show database state
        self.db_setup.print_database_stats()
        
        # Run tests
        print(f"\nRunning {len(self.ground_truth['fake_override_tests'])} fake document tests...")
        
        conn, cur = self.get_db_connection()
        
        try:
            for i, test_case in enumerate(self.ground_truth['fake_override_tests'], 1):
                print(f"\n[{i}/{len(self.ground_truth['fake_override_tests'])}] Testing: {test_case['id']}")
                print(f"  Query: {test_case['query']}")
                
                # Retrieve chunks
                retrieved_chunks = self.retrieve_chunks(
                    cur,
                    test_case['query'],
                    self.config['top_k']
                )
                
                # Get LLM answer
                answer = rag_api_llm(
                    cur,
                    test_case['query'],
                    top_k=self.config['top_k'],
                    model_name=self.config['llm_model'],
                    api_provider=self.config['api_provider']
                )
                
                # Handle None/empty answers
                if answer is None or answer == "":
                    print(f"  ⚠️  WARNING: LLM returned None/empty answer, skipping test")
                    continue
                
                # Evaluate
                evaluation = self.evaluator.evaluate_fake_document_test(
                    query=test_case['query'],
                    answer=answer,
                    fake_concepts=test_case['fake_concepts'],
                    concepts_to_avoid=test_case['should_not_contain'],
                    fake_doc_name=test_case['fake_doc_name'],
                    retrieved_chunks=retrieved_chunks
                )
                
                # Add test metadata
                evaluation['test_id'] = test_case['id']
                evaluation['category'] = test_case['category']
                evaluation['answer'] = answer
                evaluation['expected_fake_doc'] = test_case['fake_doc_name']
                
                self.fake_doc_results.append(evaluation)
                
                # Print result
                status = "✅ PASS" if evaluation['passed'] else "❌ FAIL"
                print(f"  {status} | Override Score: {evaluation['override_score']:.1f}%")
                print(f"  - Fake doc retrieved: {evaluation['fake_doc_retrieved']} (rank: {evaluation['fake_doc_rank']})")
                print(f"  - Semantic match: {evaluation['semantic_similarity']['overall_score']:.1f}%")
                print(f"  - Consistency: {evaluation['factual_consistency']['consistency_score']:.1f}%")
        
        finally:
            cur.close()
            conn.close()
        
        # Cleanup fake documents
        print("\n" + "="*80)
        print("Cleaning up fake documents...")
        self.db_setup.cleanup_fake_documents()
        
        # Verify cleanup
        self.db_setup.print_database_stats()
        
        # Print summary
        passed = sum(1 for r in self.fake_doc_results if r['passed'])
        total = len(self.fake_doc_results)
        print(f"\n{'='*80}")
        print(f"PHASE 2 SUMMARY: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        print(f"{'='*80}")
    
    def generate_reports(self):
        """Generate all test reports."""
        print("\n" + "="*80)
        print("GENERATING REPORTS")
        print("="*80)
        
        # Prepare results
        results = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
            "real_document_tests": {
                "total": len(self.real_doc_results),
                "passed": sum(1 for r in self.real_doc_results if r['passed']),
                "results": self.real_doc_results
            },
            "fake_document_tests": {
                "total": len(self.fake_doc_results),
                "passed": sum(1 for r in self.fake_doc_results if r['passed']),
                "results": self.fake_doc_results
            }
        }
        
        # Generate reports
        self.report_gen.generate_all_reports(results)
        
        print(f"\n✓ Reports generated in: {self.config['output_dir']}")
    
    def run(self):
        """Run complete test suite."""
        print("\n" + "="*80)
        print("RAG SYSTEM COMPREHENSIVE TESTING")
        print("="*80)
        print(f"Configuration:")
        print(f"  - LLM: {self.config['llm_model']} ({self.config['api_provider']})")
        print(f"  - Top-K: {self.config['top_k']}")
        print(f"  - Output: {self.config['output_dir']}")
        
        # Check initial database state
        print("\n--- INITIAL DATABASE STATE ---")
        self.db_setup.print_database_stats()
        
        try:
            # Phase 1: Real documents
            self.run_real_document_tests()
            
            # Phase 2: Fake documents
            self.run_fake_document_tests()
            
            # Generate reports
            self.generate_reports()
            
            # Final summary
            print("\n" + "="*80)
            print("TESTING COMPLETE")
            print("="*80)
            
            real_passed = sum(1 for r in self.real_doc_results if r['passed'])
            real_total = len(self.real_doc_results)
            fake_passed = sum(1 for r in self.fake_doc_results if r['passed'])
            fake_total = len(self.fake_doc_results)
            
            print(f"\nReal Document Tests: {real_passed}/{real_total} passed ({real_passed/real_total*100:.1f}%)")
            print(f"Fake Document Tests: {fake_passed}/{fake_total} passed ({fake_passed/fake_total*100:.1f}%)")
            print(f"\nOverall: {real_passed + fake_passed}/{real_total + fake_total} passed "
                  f"({(real_passed + fake_passed)/(real_total + fake_total)*100:.1f}%)")
            
            print(f"\n📊 View detailed results:")
            print(f"  - JSON: {self.config['output_dir']}/rag_test_results.json")
            print(f"  - HTML: {self.config['output_dir']}/rag_test_report.html")
            print(f"  - Charts: {self.config['output_dir']}/visualizations/")
            
        except Exception as e:
            print(f"\n❌ Error during testing: {e}")
            raise


def main():
    """Main entry point."""
    runner = RAGTestRunner()
    runner.run()


if __name__ == "__main__":
    main()
