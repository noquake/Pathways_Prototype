import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import main
import pathways_catalog


class PathwaysRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(main.app)

    def test_get_pathways_excludes_hidden_retrieval_only_resources(self) -> None:
        response = self.client.get("/pathways")
        self.assertEqual(response.status_code, 200)

        pathways = response.json()
        asthma = next(pathway for pathway in pathways if pathway["id"] == "asthma")
        asthma_resource_ids = {resource["id"] for resource in asthma["resources"]}

        self.assertNotIn("appendix-b-controller-flowchart", asthma_resource_ids)
        self.assertNotIn("appendix-c-nhlbi", asthma_resource_ids)
        self.assertNotIn("appendix-c-documentation-billing", asthma_resource_ids)
        self.assertTrue(
            all(
                resource["pdf_url"]
                for pathway in pathways
                for resource in pathway["resources"]
            )
        )

    def test_resolve_document_reference_maps_doc_name_to_pdf(self) -> None:
        resolved = pathways_catalog.resolve_document_reference(
            "asthma_emergency_department_algorithm_-_9.8.23"
        )

        self.assertEqual(
            resolved,
            {
                "pathway_id": "asthma",
                "resource_id": "emergency-department-algorithm",
                "doc_name": "asthma_emergency_department_algorithm_-_9.8.23",
                "pdf_url": "https://www.connecticutchildrens.org/sites/default/files/2023-09/asthma_emergency_department_algorithm_-_9.8.23.pdf",
                "pdf_basename": "asthma_emergency_department_algorithm_-_9.8.23.pdf",
            },
        )


if __name__ == "__main__":
    unittest.main()
