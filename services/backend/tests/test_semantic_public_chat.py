import sys
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from pathways_catalog import get_pathway_retrieval_documents
from rag import retrieval


class _FakeRPCResult:
    def __init__(self, rows):
        self.data = rows


class _FakeRPC:
    def __init__(self, rows):
        self._rows = rows

    def execute(self):
        return _FakeRPCResult(self._rows)


class _FakeSupabase:
    def __init__(self, rows_by_pathway):
        self.rows_by_pathway = rows_by_pathway

    def rpc(self, _rpc_name, payload):
        pathway_id = payload.get("filter_pathway_id")
        return _FakeRPC(self.rows_by_pathway.get(pathway_id, []))


class SemanticPublicChatTests(unittest.TestCase):
    def test_public_pathways_have_semantic_retrieval_mapping(self):
        self.assertEqual(get_pathway_retrieval_documents("anaphylaxis"), ["anaphylaxis"])
        self.assertEqual(get_pathway_retrieval_documents("appendicitis"), ["appendicitis-6.30.23"])
        self.assertEqual(
            get_pathway_retrieval_documents("asthma"),
            [
                "asthma-emergency-department-algorithm",
                "asthma-inpatient-algorithm",
                "asthma-appendix-a-mpis",
            ],
        )

    def test_retrieve_chunks_merges_multiple_semantic_documents(self):
        fake_supabase = _FakeSupabase(
            {
                "asthma-emergency-department-algorithm": [
                    {"chunk_id": 1, "similarity": 0.70},
                ],
                "asthma-inpatient-algorithm": [
                    {"chunk_id": 2, "similarity": 0.91},
                ],
                "asthma-appendix-a-mpis": [
                    {"chunk_id": 3, "similarity": 0.85},
                ],
            }
        )

        original_get_embeddings = retrieval.get_embeddings
        retrieval.get_embeddings = lambda *args, **kwargs: [[0.0] * 384]
        try:
            results = retrieval.retrieve_chunks(
                fake_supabase,
                "asthma question",
                top_k=2,
                pathway_ids=get_pathway_retrieval_documents("asthma"),
                model_key="minilm_semantic",
            )
        finally:
            retrieval.get_embeddings = original_get_embeddings

        self.assertEqual([row["chunk_id"] for row in results], [2, 3])


if __name__ == "__main__":
    unittest.main()
