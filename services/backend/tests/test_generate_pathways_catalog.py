import sys
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import generate_pathways_catalog as generator


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def read_fixture(name: str) -> str:
    return (FIXTURES_DIR / name).read_text(encoding="utf-8")


class GeneratePathwaysCatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pathway_documents = [
            {
                "pathway_id": "asthma-emergency-department-algorithm",
                "doc_name": "asthma_emergency_department_algorithm_-_9.8.23",
                "doc_display_name": "Asthma Emergency Department Algorithm",
                "doc_file_path": "/app/data/transformed_files/asthma_emergency_department_algorithm_-_9.8.23.md",
            },
            {
                "pathway_id": "asthma-inpatient-algorithm",
                "doc_name": "asthma_inpatient_algorithm_-_08.22.23",
                "doc_display_name": "Asthma Inpatient Algorithm",
                "doc_file_path": "/app/data/transformed_files/asthma_inpatient_algorithm_-_08.22.23.md",
            },
            {
                "pathway_id": "asthma-pathway-educational-module-9.8.23",
                "doc_name": "asthma_pathway_educational_module_-9.8.23",
                "doc_display_name": "Asthma Pathway Educational Module 9.8.23",
                "doc_file_path": "/app/data/transformed_files/asthma_pathway_educational_module_-9.8.23.md",
            },
            {
                "pathway_id": "anaphylaxis",
                "doc_name": "anaphylaxis_-_1.16.25",
                "doc_display_name": "Anaphylaxis",
                "doc_file_path": "/app/data/transformed_files/anaphylaxis_-_1.16.25.md",
            },
            {
                "pathway_id": "anaphylaxis-educational-module-1.16.25",
                "doc_name": "anaphylaxis_educational_module_1.16.25",
                "doc_display_name": "Anaphylaxis Educational Module 1.16.25",
                "doc_file_path": "/app/data/transformed_files/anaphylaxis_educational_module_1.16.25.md",
            },
            {
                "pathway_id": "animal-bite-educational-module-7.26.25",
                "doc_name": "animal-bite-educational-module-7.26.25",
                "doc_display_name": "Animal Bite Educational Module 7.26.25",
                "doc_file_path": "/app/data/transformed_files/animal-bite-educational-module-7.26.25.md",
            },
        ]

    def test_extract_section_links_discovers_internal_sections(self) -> None:
        links = generator.extract_section_links(read_fixture("landing_page.html"))
        self.assertEqual(
            links,
            [
                "https://www.connecticutchildrens.org/medical-professionals/clinical-pathways/anaphylaxis",
                "https://www.connecticutchildrens.org/medical-professionals/clinical-pathways/asthma-emergency-department-and-inpatient",
            ],
        )

    def test_extract_pdf_urls_keeps_only_connecticut_childrens_pdfs(self) -> None:
        pdf_urls = generator.extract_pdf_urls(
            read_fixture("section_external_pdf.html"),
            "https://www.connecticutchildrens.org/medical-professionals/clinical-pathways/animal-human-bite-skin-and-soft-tissue-infection",
        )
        self.assertEqual(
            pdf_urls,
            [
                "https://www.connecticutchildrens.org/sites/default/files/2025-08/animal-bite-educational-module-7.26.25.pdf"
            ],
        )

    def test_build_catalog_keeps_unmatched_pdf_without_retrieval_id(self) -> None:
        section_records = [
            generator.build_section_record(
                "https://www.connecticutchildrens.org/medical-professionals/clinical-pathways/mystery-pathway",
                read_fixture("section_unmatched_pdf.html"),
            )
        ]
        catalog = generator.prepare_catalog_for_render(
            generator.build_catalog(section_records, self.pathway_documents)
        )
        self.assertEqual(len(catalog), 1)
        pathway = catalog[0]
        self.assertEqual(pathway["id"], "mystery-pathway")
        self.assertEqual(pathway["retrieval_document_ids"], [])
        self.assertEqual(pathway["resources"][0]["doc_name"], "mystery-main-pathway")

    def test_current_five_pathway_id_overrides_stay_stable(self) -> None:
        self.assertEqual(
            generator.ID_OVERRIDES["asthma-emergency-department-and-inpatient"],
            "asthma",
        )
        self.assertEqual(generator.ID_OVERRIDES["anaphylaxis"], "anaphylaxis")
        self.assertEqual(generator.ID_OVERRIDES["bronchiolitis"], "bronchiolitis")
        self.assertEqual(generator.ID_OVERRIDES["croup"], "croup")
        self.assertEqual(generator.ID_OVERRIDES["appendicitis"], "appendicitis")

    def test_build_catalog_is_deterministic(self) -> None:
        section_records = [
            generator.build_section_record(
                "https://www.connecticutchildrens.org/medical-professionals/clinical-pathways/asthma-emergency-department-and-inpatient",
                read_fixture("section_multi_pdf.html"),
            ),
            generator.build_section_record(
                "https://www.connecticutchildrens.org/medical-professionals/clinical-pathways/anaphylaxis",
                read_fixture("section_anaphylaxis.html"),
            ),
        ]
        first = generator.prepare_catalog_for_render(
            generator.build_catalog(section_records, self.pathway_documents)
        )
        second = generator.prepare_catalog_for_render(
            generator.build_catalog(section_records, self.pathway_documents)
        )
        self.assertEqual(first, second)

    def test_catalog_smoke_shape(self) -> None:
        section_records = [
            generator.build_section_record(
                "https://www.connecticutchildrens.org/medical-professionals/clinical-pathways/asthma-emergency-department-and-inpatient",
                read_fixture("section_multi_pdf.html"),
            ),
            generator.build_section_record(
                "https://www.connecticutchildrens.org/medical-professionals/clinical-pathways/anaphylaxis",
                read_fixture("section_anaphylaxis.html"),
            ),
            generator.build_section_record(
                "https://www.connecticutchildrens.org/medical-professionals/clinical-pathways/mystery-pathway",
                read_fixture("section_unmatched_pdf.html"),
            ),
        ]
        catalog = generator.prepare_catalog_for_render(
            generator.build_catalog(section_records, self.pathway_documents)
        )

        for pathway in catalog:
            self.assertIn("id", pathway)
            self.assertIn("label", pathway)
            self.assertIn("default_resource_id", pathway)
            self.assertIn("doc_name", pathway)
            self.assertIn("pdf_url", pathway)
            self.assertIn("retrieval_document_ids", pathway)
            self.assertIn("resources", pathway)
            self.assertTrue(pathway["resources"])

            default_resource = next(
                resource
                for resource in pathway["resources"]
                if resource["id"] == pathway["default_resource_id"]
            )
            self.assertEqual(pathway["doc_name"], default_resource["doc_name"])
            self.assertEqual(pathway["pdf_url"], default_resource["pdf_url"])

            for resource in pathway["resources"]:
                self.assertEqual(
                    generator.urlparse(resource["pdf_url"]).netloc,
                    generator.CONNECTICUT_CHILDRENS_HOST,
                )

    def test_apply_hidden_source_aliases_keeps_aliases_out_of_resources(self) -> None:
        section_records = [
            generator.build_section_record(
                "https://www.connecticutchildrens.org/medical-professionals/clinical-pathways/asthma-emergency-department-and-inpatient",
                read_fixture("section_multi_pdf.html"),
            )
        ]
        catalog = generator.prepare_catalog_for_render(
            generator.build_catalog(section_records, self.pathway_documents)
        )
        asthma = catalog[0]
        hidden_aliases = generator.apply_hidden_source_aliases(
            catalog,
            [
                {
                    "pathway_id": asthma["id"],
                    "source_file": "asthma-appendix-b-controller-flowchart",
                    "preview_resource_id": asthma["default_resource_id"],
                }
            ],
        )

        self.assertEqual(
            hidden_aliases,
            [
                {
                    "pathway_id": asthma["id"],
                    "source_file": "asthma-appendix-b-controller-flowchart",
                    "preview_resource_id": asthma["default_resource_id"],
                }
            ],
        )
        self.assertIn(
            "asthma-appendix-b-controller-flowchart",
            asthma["retrieval_document_ids"],
        )
        self.assertTrue(all(resource["pdf_url"] for resource in asthma["resources"]))
        self.assertNotIn(
            "asthma-appendix-b-controller-flowchart",
            {resource["id"] for resource in asthma["resources"]},
        )

    def test_apply_resource_pdf_url_overrides_updates_visible_resource(self) -> None:
        catalog = [
            {
                "id": "ssti",
                "label": "Skin & Soft Tissue Infection (SSTI)",
                "default_resource_id": "pathway",
                "doc_name": "ssti-skin-and-soft-tissue-infection-clinical-pathway-1.16.25",
                "pdf_url": "https://example.com/old.pdf",
                "retrieval_document_ids": [],
                "resources": [
                    {
                        "id": "pathway",
                        "label": "Clinical Pathway",
                        "doc_name": "ssti-skin-and-soft-tissue-infection-clinical-pathway-1.16.25",
                        "pdf_url": "https://example.com/old.pdf",
                    },
                    {
                        "id": "educational-module",
                        "label": "Educational Module",
                        "doc_name": "ssti-educational-module-3.12.25",
                        "pdf_url": "https://example.com/module.pdf",
                    },
                ],
            }
        ]

        updated_catalog = generator.apply_resource_pdf_url_overrides(
            catalog,
            [
                {
                    "pathway_id": "ssti",
                    "resource_id": "pathway",
                    "pdf_url": "https://www.connecticutchildrens.org/sites/default/files/2025-03/ssti-skin-and-soft-tissue-infection-clinical-pathway-1.16.25.pdf",
                }
            ],
        )

        self.assertEqual(updated_catalog[0]["pdf_url"], "https://www.connecticutchildrens.org/sites/default/files/2025-03/ssti-skin-and-soft-tissue-infection-clinical-pathway-1.16.25.pdf")
        self.assertEqual(updated_catalog[0]["resources"][0]["pdf_url"], "https://www.connecticutchildrens.org/sites/default/files/2025-03/ssti-skin-and-soft-tissue-infection-clinical-pathway-1.16.25.pdf")
        self.assertEqual(updated_catalog[0]["resources"][1]["pdf_url"], "https://example.com/module.pdf")

    def test_render_catalog_module_includes_source_resolution_helpers(self) -> None:
        catalog = [
            {
                "id": "asthma",
                "label": "Asthma",
                "default_resource_id": "algorithm",
                "doc_name": "asthma_algorithm",
                "pdf_url": "https://www.connecticutchildrens.org/sites/default/files/asthma.pdf",
                "retrieval_document_ids": ["asthma-algorithm"],
                "resources": [
                    {
                        "id": "algorithm",
                        "label": "Algorithm",
                        "doc_name": "asthma_algorithm",
                        "pdf_url": "https://www.connecticutchildrens.org/sites/default/files/asthma.pdf",
                        "medembed_id": "asthma-algorithm",
                    }
                ],
            }
        ]
        hidden_aliases = [
            {
                "pathway_id": "asthma",
                "source_file": "asthma-hidden-doc",
                "preview_resource_id": "algorithm",
            }
        ]

        rendered = generator.render_catalog_module(
            catalog,
            hidden_aliases,
            ["asthma"],
        )

        self.assertIn("HIDDEN_SOURCE_ALIASES", rendered)
        self.assertIn("DOCUMENT_REFERENCE_BY_DOC_NAME", rendered)
        self.assertIn("SOURCE_REFERENCE_BY_KEY", rendered)
        self.assertIn("def resolve_document_reference", rendered)
        self.assertIn("def resolve_source_reference", rendered)
        self.assertIn("asthma-hidden-doc", rendered)
        self.assertIn("preview_resource_id", rendered)


if __name__ == "__main__":
    unittest.main()
