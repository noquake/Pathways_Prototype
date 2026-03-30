import sys
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import main
from rag.query import append_sources_legend, build_context
from rag.retrieval import _hydrate_missing_chunk_metadata


class PdfCitationTests(unittest.TestCase):
    def test_hydrate_missing_chunk_metadata_backfills_from_chunk_table(self) -> None:
        class FakeTableQuery:
            def __init__(self, rows):
                self.rows = rows
                self.selected = None
                self.chunk_ids = []

            def select(self, selected):
                self.selected = selected
                return self

            def in_(self, _column, chunk_ids):
                self.chunk_ids = [str(chunk_id) for chunk_id in chunk_ids]
                return self

            def execute(self):
                data = [
                    row
                    for row in self.rows
                    if str(row["chunk_id"]) in self.chunk_ids
                ]
                return type("Result", (), {"data": data})()

        class FakeSupabase:
            def __init__(self, rows):
                self.rows = rows

            def table(self, _name):
                return FakeTableQuery(self.rows)

        hydrated_results = _hydrate_missing_chunk_metadata(
            FakeSupabase(
                [
                    {
                        "chunk_id": 1,
                        "source_docs": ["asthma_emergency_department_algorithm_-_9.8.23"],
                        "pathway_id": "asthma-emergency-department-algorithm",
                        "pathway_tag": "asthma",
                    }
                ]
            ),
            [
                {
                    "chunk_id": 1,
                    "chunk_text": "ED algorithm chunk",
                    "chunk_length": 18,
                    "similarity": 0.9,
                }
            ],
            ["pathway_chunks_medembed_large"],
        )

        self.assertEqual(
            hydrated_results[0]["source_docs"],
            ["asthma_emergency_department_algorithm_-_9.8.23"],
        )
        self.assertEqual(
            hydrated_results[0]["pathway_id"],
            "asthma-emergency-department-algorithm",
        )
        self.assertEqual(hydrated_results[0]["pathway_tag"], "asthma")
        self.assertEqual(
            hydrated_results[0]["_hydrated_fields"],
            ["source_docs", "pathway_id", "pathway_tag"],
        )

    def test_prepare_citation_results_maps_source_docs_to_pdf_basenames(self) -> None:
        prompt_results, chunk_document_numbers, citation_documents, citations = main.prepare_citation_results(
            [
                {
                    "chunk_id": 1,
                    "chunk_text": "ED algorithm chunk",
                    "chunk_length": 18,
                    "source_docs": ["asthma_emergency_department_algorithm_-_9.8.23"],
                    "similarity": 0.9,
                }
            ]
        )

        self.assertEqual(len(prompt_results), 1)
        self.assertEqual(chunk_document_numbers, [1])
        self.assertEqual(
            citation_documents,
            [
                {
                    "number": "1",
                    "source_doc": "asthma_emergency_department_algorithm_-_9.8.23",
                    "pathway_id": "asthma",
                    "resource_id": "emergency-department-algorithm",
                    "pdf_url": "https://www.connecticutchildrens.org/sites/default/files/2023-09/asthma_emergency_department_algorithm_-_9.8.23.pdf",
                    "pdf_name": "asthma_emergency_department_algorithm_-_9.8.23.pdf",
                }
            ],
        )
        self.assertEqual(citations[0]["source_file"], "asthma_emergency_department_algorithm_-_9.8.23")

    def test_prepare_citation_results_uses_result_pathway_id_when_source_doc_name_differs(self) -> None:
        prompt_results, chunk_document_numbers, citation_documents, citations = main.prepare_citation_results(
            [
                {
                    "chunk_id": 1,
                    "chunk_text": "Inpatient pathway chunk",
                    "chunk_length": 22,
                    "source_docs": ["asthma_inpatient_pathway"],
                    "pathway_id": "asthma-inpatient-pathway",
                    "similarity": 0.9,
                },
                {
                    "chunk_id": 2,
                    "chunk_text": "MPIS chunk",
                    "chunk_length": 10,
                    "source_docs": ["asthma_appendix_A_MPIS"],
                    "pathway_id": "asthma-appendix-a-mpis",
                    "similarity": 0.8,
                },
            ]
        )

        self.assertEqual(len(prompt_results), 2)
        self.assertEqual(chunk_document_numbers, [1, 2])
        self.assertEqual(
            [document["resource_id"] for document in citation_documents],
            ["inpatient-algorithm", "mpis-score"],
        )
        self.assertEqual(
            [citation["pdf_name"] for citation in citations],
            [
                "asthma_inpatient_algorithm_-_08.22.23.pdf",
                "appendix-a-mpis-score-branded-11.5.25-separate-pdf.pdf",
            ],
        )

    def test_prepare_citation_results_groups_chunks_by_pdf_and_skips_unmapped_rows(self) -> None:
        prompt_results, chunk_document_numbers, citation_documents, citations = main.prepare_citation_results(
            [
                {
                    "chunk_id": 1,
                    "chunk_text": "ED algorithm chunk",
                    "chunk_length": 18,
                    "source_docs": ["asthma_emergency_department_algorithm_-_9.8.23"],
                    "similarity": 0.9,
                },
                {
                    "chunk_id": 2,
                    "chunk_text": "Another ED chunk",
                    "chunk_length": 16,
                    "source_docs": ["asthma_emergency_department_algorithm_-_9.8.23"],
                    "similarity": 0.8,
                },
                {
                    "chunk_id": 3,
                    "chunk_text": "Inpatient chunk",
                    "chunk_length": 14,
                    "source_docs": ["asthma_inpatient_algorithm_-_08.22.23"],
                    "similarity": 0.7,
                },
                {
                    "chunk_id": 4,
                    "chunk_text": "Legacy unmapped chunk",
                    "chunk_length": 20,
                    "source_docs": [],
                    "similarity": 0.6,
                },
            ]
        )

        self.assertEqual(len(prompt_results), 4)
        self.assertEqual(chunk_document_numbers, [1, 1, 2, None])
        self.assertEqual(len(citation_documents), 2)
        self.assertEqual(len(citations), 3)

    def test_prepare_citation_results_strips_hydrated_fields_from_prompt_context(self) -> None:
        prompt_results, chunk_document_numbers, citation_documents, citations = main.prepare_citation_results(
            [
                {
                    "chunk_id": 1,
                    "chunk_text": "Asthma chunk",
                    "chunk_length": 11,
                    "source_docs": ["asthma_inpatient_pathway"],
                    "pathway_id": "asthma-inpatient-pathway",
                    "pathway_tag": "asthma",
                    "_hydrated_fields": ["source_docs", "pathway_id", "pathway_tag"],
                }
            ]
        )

        self.assertEqual(build_context(prompt_results), "[1] unknown: Asthma chunk")
        self.assertEqual(chunk_document_numbers, [1])
        self.assertEqual(citation_documents[0]["resource_id"], "inpatient-algorithm")
        self.assertEqual(citations[0]["pdf_name"], "asthma_inpatient_algorithm_-_08.22.23.pdf")

    def test_append_sources_legend_maps_chunk_numbers_to_document_numbers(self) -> None:
        response = append_sources_legend(
            "Use the inpatient algorithm [1]. Then review the MPIS chart [2]. Then confirm [3].",
            [1, 1, 1],
            [
                {"pdf_name": "shared.pdf"},
            ],
        )

        self.assertEqual(
            response,
            "Use the inpatient algorithm [1]. Then review the MPIS chart [1]. Then confirm [1].\n\n"
            "Sources:\n[1] shared.pdf",
        )

    def test_append_sources_legend_renumbers_cited_document_subset(self) -> None:
        response = append_sources_legend(
            "Use the inpatient algorithm [2]. Then review the ED algorithm [1].",
            [2, 1],
            [
                {"pdf_name": "ed.pdf"},
                {"pdf_name": "inpatient.pdf"},
            ],
        )

        self.assertEqual(
            response,
            "Use the inpatient algorithm [1]. Then review the ED algorithm [2].\n\n"
            "Sources:\n[1] ed.pdf\n[2] inpatient.pdf",
        )

    def test_append_sources_legend_leaves_unmapped_chunk_citation_unchanged(self) -> None:
        response = append_sources_legend(
            "Keep [2] as-is and map [1].",
            [1, None],
            [
                {"pdf_name": "mapped.pdf"},
            ],
        )

        self.assertEqual(
            response,
            "Keep [2] as-is and map [1].\n\nSources:\n[1] mapped.pdf",
        )


if __name__ == "__main__":
    unittest.main()
