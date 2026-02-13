"""
Semantic Evaluator for RAG Testing

Provides sophisticated metrics beyond simple keyword matching:
- Semantic similarity scoring
- Context relevance evaluation
- Citation accuracy validation
- Factual consistency checking
"""

import re
from typing import List, Dict, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticEvaluator:
    """Evaluates RAG system responses using semantic similarity and other advanced metrics."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the evaluator.
        
        Args:
            model_name: Name of sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)
    
    def compute_semantic_similarity(
        self, 
        answer: str, 
        expected_concepts: List[str]
    ) -> Dict[str, float]:
        """
        Compute semantic similarity between answer and expected concepts.
        
        Args:
            answer: The generated answer text
            expected_concepts: List of expected concepts/phrases
            
        Returns:
            Dictionary with similarity scores
        """
        if not answer or not expected_concepts:
            return {
                "overall_score": 0.0,
                "max_similarity": 0.0,
                "avg_similarity": 0.0,
                "concept_scores": []
            }
        
        # Embed answer and concepts
        answer_embedding = self.model.encode([answer])[0]
        concept_embeddings = self.model.encode(expected_concepts)
        
        # Compute similarities
        similarities = []
        concept_scores = []
        
        for concept, concept_emb in zip(expected_concepts, concept_embeddings):
            similarity = cosine_similarity(
                answer_embedding.reshape(1, -1),
                concept_emb.reshape(1, -1)
            )[0][0]
            
            # Convert to 0-100 scale
            similarity_score = float(similarity * 100)
            similarities.append(similarity_score)
            
            concept_scores.append({
                "concept": concept,
                "score": similarity_score
            })
        
        return {
            "overall_score": float(np.mean(similarities)),  # Average of all concepts
            "max_similarity": float(np.max(similarities)),
            "avg_similarity": float(np.mean(similarities)),
            "concept_scores": concept_scores
        }
    
    def evaluate_context_relevance(
        self,
        retrieved_sources: List[str],
        expected_docs: List[str],
        top_k: int = 5
    ) -> Dict[str, any]:
        """
        Evaluate if relevant documents were retrieved.
        
        Args:
            retrieved_sources: List of doc_names that were retrieved
            expected_docs: List of expected relevant doc_names (partial matches OK)
            top_k: Consider top-k documents for ranking
            
        Returns:
            Dictionary with relevance metrics
        """
        if not retrieved_sources or not expected_docs:
            return {
                "relevance_score": 0.0,
                "relevant_retrieved": 0,
                "total_expected": len(expected_docs),
                "best_rank": None,
                "found_docs": []
            }
        
        # Find which expected docs were retrieved
        found_docs = []
        ranks = []
        
        for expected in expected_docs:
            for rank, source in enumerate(retrieved_sources[:top_k], 1):
                # Partial match (e.g., "dka-ms-management" matches "dka-ms-management-1.19.23")
                if expected.lower() in source.lower():
                    found_docs.append({
                        "expected": expected,
                        "found": source,
                        "rank": rank
                    })
                    ranks.append(rank)
                    break
        
        # Calculate relevance score
        # Score: 100% if found in top-1, decreasing with rank
        if ranks:
            # Weight by rank: rank 1 = 100%, rank 2 = 90%, rank 3 = 80%, etc.
            rank_scores = [max(0, 110 - (rank * 10)) for rank in ranks]
            relevance_score = float(np.mean(rank_scores))
        else:
            relevance_score = 0.0
        
        return {
            "relevance_score": relevance_score,
            "relevant_retrieved": len(found_docs),
            "total_expected": len(expected_docs),
            "best_rank": min(ranks) if ranks else None,
            "found_docs": found_docs,
            "retrieval_rate": (len(found_docs) / len(expected_docs) * 100) if expected_docs else 0.0
        }
    
    def evaluate_factual_consistency(
        self,
        answer: str,
        concepts_to_find: List[str],
        concepts_to_avoid: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Check if specific concepts appear (or don't appear) in the answer.
        Used for fake document override testing.
        
        Args:
            answer: The generated answer text
            concepts_to_find: Concepts that should appear (for fake doc tests)
            concepts_to_avoid: Concepts that should NOT appear
            
        Returns:
            Dictionary with consistency metrics
        """
        answer_lower = answer.lower()
        
        # Check for expected concepts (should be present in fake doc tests)
        found_concepts = []
        for concept in concepts_to_find:
            if concept.lower() in answer_lower:
                found_concepts.append(concept)
        
        find_rate = (len(found_concepts) / len(concepts_to_find) * 100) if concepts_to_find else 0.0
        
        # Check for concepts to avoid (shouldn't be present in fake doc tests)
        avoid_rate = 100.0
        avoided_concepts = []
        if concepts_to_avoid:
            avoided_concepts = [c for c in concepts_to_avoid if c.lower() not in answer_lower]
            avoid_rate = (len(avoided_concepts) / len(concepts_to_avoid) * 100)
        
        # Overall consistency score (higher = better fake doc override)
        # For fake tests: want high find_rate and high avoid_rate
        consistency_score = (find_rate + avoid_rate) / 2
        
        return {
            "consistency_score": consistency_score,
            "find_rate": find_rate,
            "avoid_rate": avoid_rate,
            "found_concepts": found_concepts,
            "avoided_concepts": avoided_concepts,
            "total_expected": len(concepts_to_find),
            "total_to_avoid": len(concepts_to_avoid) if concepts_to_avoid else 0
        }
    
    def parse_citations(self, answer: str) -> List[int]:
        """
        Parse citation numbers from answer text.
        Looks for patterns like [1], [2], etc.
        
        Args:
            answer: The generated answer text
            
        Returns:
            List of citation numbers found
        """
        # Find all [N] patterns where N is a number
        citations = re.findall(r'\[(\d+)\]', answer)
        return [int(c) for c in citations]
    
    def evaluate_citation_accuracy(
        self,
        answer: str,
        retrieved_chunks: List[Tuple[str, str]],
        citation_window: int = 50
    ) -> Dict[str, any]:
        """
        Evaluate if citations are used and accurate.
        
        Args:
            answer: The generated answer text
            retrieved_chunks: List of (chunk_text, doc_name) tuples in order retrieved
            citation_window: Number of characters to check around citation
            
        Returns:
            Dictionary with citation metrics
        """
        citations = self.parse_citations(answer)
        
        if not citations:
            return {
                "citation_score": 0.0,
                "citations_used": False,
                "total_citations": 0,
                "valid_citations": 0,
                "citation_details": []
            }
        
        # Check if citations are valid (within range)
        max_citation = max(citations)
        valid_range = max_citation <= len(retrieved_chunks)
        
        citation_details = []
        for citation_num in set(citations):  # Unique citations
            if citation_num <= len(retrieved_chunks):
                chunk_text, doc_name = retrieved_chunks[citation_num - 1]
                citation_details.append({
                    "number": citation_num,
                    "valid": True,
                    "doc_name": doc_name,
                    "chunk_preview": chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text
                })
            else:
                citation_details.append({
                    "number": citation_num,
                    "valid": False,
                    "doc_name": None,
                    "chunk_preview": None
                })
        
        valid_citations = sum(1 for c in citation_details if c["valid"])
        citation_accuracy = (valid_citations / len(citation_details) * 100) if citation_details else 0.0
        
        # Citation score: 50% for using citations, 50% for accuracy
        usage_score = 50.0 if citations else 0.0
        accuracy_score = citation_accuracy * 0.5
        citation_score = usage_score + accuracy_score
        
        return {
            "citation_score": citation_score,
            "citations_used": len(citations) > 0,
            "total_citations": len(citations),
            "unique_citations": len(set(citations)),
            "valid_citations": valid_citations,
            "citation_accuracy": citation_accuracy,
            "citation_details": citation_details
        }
    
    def evaluate_real_document_test(
        self,
        query: str,
        answer: str,
        expected_concepts: List[str],
        expected_docs: List[str],
        retrieved_chunks: List[Tuple[str, str]]
    ) -> Dict[str, any]:
        """
        Comprehensive evaluation for real document tests.
        
        Args:
            query: The user query
            answer: The generated answer
            expected_concepts: List of concepts that should appear
            expected_docs: List of relevant document names
            retrieved_chunks: List of (chunk_text, doc_name) tuples
            
        Returns:
            Dictionary with all evaluation metrics and pass/fail status
        """
        # Extract just doc names from retrieved chunks
        retrieved_sources = [doc_name for _, doc_name in retrieved_chunks]
        
        # Semantic similarity
        semantic = self.compute_semantic_similarity(answer, expected_concepts)
        
        # Context relevance
        relevance = self.evaluate_context_relevance(retrieved_sources, expected_docs)
        
        # Citation accuracy
        citations = self.evaluate_citation_accuracy(answer, retrieved_chunks)
        
        # Overall score (weighted average)
        overall_score = (
            semantic["overall_score"] * 0.5 +  # 50% weight on semantic similarity
            relevance["relevance_score"] * 0.3 +  # 30% weight on relevance
            citations["citation_score"] * 0.2     # 20% weight on citations
        )
        
        # Determine pass/fail based on thresholds from plan
        passed = (
            semantic["overall_score"] >= 70.0 and
            relevance["relevance_score"] >= 70.0 and
            citations["citation_score"] >= 80.0
        )
        
        return {
            "query": query,
            "answer_length": len(answer),
            "answer_preview": answer[:200] + "..." if len(answer) > 200 else answer,
            "overall_score": overall_score,
            "passed": passed,
            "semantic_similarity": semantic,
            "context_relevance": relevance,
            "citation_accuracy": citations,
            "retrieved_sources": retrieved_sources[:5]  # Top 5
        }
    
    def evaluate_fake_document_test(
        self,
        query: str,
        answer: str,
        fake_concepts: List[str],
        concepts_to_avoid: List[str],
        fake_doc_name: str,
        retrieved_chunks: List[Tuple[str, str]]
    ) -> Dict[str, any]:
        """
        Comprehensive evaluation for fake document override tests.
        
        Args:
            query: The user query
            answer: The generated answer
            fake_concepts: Concepts from fake doc that should appear
            concepts_to_avoid: Real medical concepts that shouldn't appear
            fake_doc_name: Name of the fake document
            retrieved_chunks: List of (chunk_text, doc_name) tuples
            
        Returns:
            Dictionary with all evaluation metrics and pass/fail status
        """
        # Extract doc names
        retrieved_sources = [doc_name for _, doc_name in retrieved_chunks]
        
        # Check if fake doc was retrieved
        fake_doc_retrieved = any(fake_doc_name in source for source in retrieved_sources)
        fake_doc_rank = None
        
        if fake_doc_retrieved:
            for rank, source in enumerate(retrieved_sources, 1):
                if fake_doc_name in source:
                    fake_doc_rank = rank
                    break
        
        # Factual consistency (did fake concepts appear?)
        consistency = self.evaluate_factual_consistency(
            answer, fake_concepts, concepts_to_avoid
        )
        
        # Semantic similarity with fake concepts
        semantic = self.compute_semantic_similarity(answer, fake_concepts)
        
        # Citation accuracy
        citations = self.evaluate_citation_accuracy(answer, retrieved_chunks)
        
        # Override score (how well did fake doc override real knowledge?)
        # Higher score = better override
        override_score = (
            (100.0 if fake_doc_retrieved else 0.0) * 0.3 +  # 30% weight on retrieval
            consistency["consistency_score"] * 0.4 +         # 40% weight on consistency
            semantic["overall_score"] * 0.3                  # 30% weight on semantic
        )
        
        # Determine pass/fail based on RAG override success
        # PASS = LLM cited the fake document and used its content (even if absurd)
        # FAIL = LLM ignored fake doc or gave real medical advice
        passed = (
            fake_doc_retrieved and                      # Fake doc must be retrieved
            fake_doc_rank <= 3 and                      # Should be in top 3 results
            consistency["find_rate"] >= 40.0 and        # At least 40% of fake concepts mentioned
            consistency["avoid_rate"] >= 50.0           # At least 50% of real concepts avoided
        )
        
        return {
            "query": query,
            "answer_length": len(answer),
            "answer_preview": answer[:200] + "..." if len(answer) > 200 else answer,
            "override_score": override_score,
            "passed": passed,
            "fake_doc_retrieved": fake_doc_retrieved,
            "fake_doc_rank": fake_doc_rank,
            "semantic_similarity": semantic,
            "factual_consistency": consistency,
            "citation_accuracy": citations,
            "retrieved_sources": retrieved_sources[:5]
        }


def main():
    """Test the semantic evaluator."""
    evaluator = SemanticEvaluator()
    
    # Test semantic similarity
    print("Testing Semantic Similarity")
    print("="*60)
    answer = "The patient should receive a fluid bolus of normal saline at 10-20 mL/kg over the first hour."
    concepts = ["fluid bolus", "normal saline", "10-20 mL/kg"]
    result = evaluator.compute_semantic_similarity(answer, concepts)
    print(f"Answer: {answer}")
    print(f"Expected concepts: {concepts}")
    print(f"Overall score: {result['overall_score']:.1f}%")
    print(f"Concept scores:")
    for cs in result['concept_scores']:
        print(f"  - '{cs['concept']}': {cs['score']:.1f}%")
    
    # Test context relevance
    print("\n\nTesting Context Relevance")
    print("="*60)
    retrieved = ["dka-ms-management-1.19.23.md", "sepsis-protocol.md", "dka_module.md"]
    expected = ["dka-ms-management", "dka_module"]
    result = evaluator.evaluate_context_relevance(retrieved, expected)
    print(f"Retrieved: {retrieved}")
    print(f"Expected: {expected}")
    print(f"Relevance score: {result['relevance_score']:.1f}%")
    print(f"Found docs: {result['found_docs']}")
    
    # Test citation parsing
    print("\n\nTesting Citation Accuracy")
    print("="*60)
    answer_with_citations = "According to the protocol [1], patients need fluids [2] and insulin [1]."
    citations = evaluator.parse_citations(answer_with_citations)
    print(f"Answer: {answer_with_citations}")
    print(f"Citations found: {citations}")


if __name__ == "__main__":
    main()
