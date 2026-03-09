from typing import Any, Dict, List, Optional


PATHWAY_CATALOG: List[Dict[str, Any]] = [
    {
        "id": "asthma",
        "label": "Asthma",
        "default_resource_id": "emergency-department-algorithm",
        "doc_name": "asthma_emergency_department_algorithm_-_9.8.23",
        "pdf_url": "https://www.connecticutchildrens.org/sites/default/files/2023-09/asthma_emergency_department_algorithm_-_9.8.23.pdf",
        "resources": [
            {
                "id": "emergency-department-algorithm",
                "label": "ED Algorithm",
                "doc_name": "asthma_emergency_department_algorithm_-_9.8.23",
                "pdf_url": "https://www.connecticutchildrens.org/sites/default/files/2023-09/asthma_emergency_department_algorithm_-_9.8.23.pdf",
            },
            {
                "id": "inpatient-algorithm",
                "label": "Inpatient Algorithm",
                "doc_name": "asthma_inpatient_algorithm_-_08.22.23",
                "pdf_url": "https://www.connecticutchildrens.org/sites/default/files/2023-09/asthma_inpatient_algorithm_-_08.22.23.pdf",
            },
            {
                "id": "mpis-score",
                "label": "MPIS Score",
                "doc_name": "appendix-a-mpis-score-branded-11.5.25-separate-pdf",
                "pdf_url": "https://www.connecticutchildrens.org/sites/default/files/2025-11/appendix-a-mpis-score-branded-11.5.25-separate-pdf.pdf",
            },
            {
                "id": "educational-module",
                "label": "Educational Module",
                "doc_name": "asthma_pathway_educational_module_-9.8.23",
                "pdf_url": "https://www.connecticutchildrens.org/sites/default/files/2023-09/asthma_pathway_educational_module_-9.8.23.pdf",
            },
        ],
    },
    {
        "id": "anaphylaxis",
        "label": "Anaphylaxis",
        "default_resource_id": "default",
        "doc_name": "anaphylaxis_-_1.16.25",
        "pdf_url": "https://www.connecticutchildrens.org/sites/default/files/2025-01/anaphylaxis_-_1.16.25.pdf",
        "resources": [
            {
                "id": "default",
                "label": "Pathway PDF",
                "doc_name": "anaphylaxis_-_1.16.25",
                "pdf_url": "https://www.connecticutchildrens.org/sites/default/files/2025-01/anaphylaxis_-_1.16.25.pdf",
            },
        ],
    },
    {
        "id": "bronchiolitis",
        "label": "Bronchiolitis",
        "default_resource_id": "default",
        "doc_name": "bronchiolitis-educational-module-10.13.23",
        "pdf_url": "https://www.connecticutchildrens.org/sites/default/files/2023-10/bronchiolitis-algorithm-10.4.23.pdf",
        "resources": [
            {
                "id": "default",
                "label": "Pathway PDF",
                "doc_name": "bronchiolitis-educational-module-10.13.23",
                "pdf_url": "https://www.connecticutchildrens.org/sites/default/files/2023-10/bronchiolitis-algorithm-10.4.23.pdf",
            },
        ],
    },
    {
        "id": "croup",
        "label": "Croup",
        "default_resource_id": "default",
        "doc_name": "croup-module-3.25.25",
        "pdf_url": "https://www.connecticutchildrens.org/sites/default/files/2025-03/croup-algorithm-3.20.25.pdf",
        "resources": [
            {
                "id": "default",
                "label": "Pathway PDF",
                "doc_name": "croup-module-3.25.25",
                "pdf_url": "https://www.connecticutchildrens.org/sites/default/files/2025-03/croup-algorithm-3.20.25.pdf",
            },
        ],
    },
    {
        "id": "appendicitis",
        "label": "Appendicitis",
        "default_resource_id": "default",
        "doc_name": "appendicitis-6.30.23",
        "pdf_url": "https://www.connecticutchildrens.org/sites/default/files/2023-09/appendicitis-6.30.23.pdf",
        "resources": [
            {
                "id": "default",
                "label": "Pathway PDF",
                "doc_name": "appendicitis-6.30.23",
                "pdf_url": "https://www.connecticutchildrens.org/sites/default/files/2023-09/appendicitis-6.30.23.pdf",
            },
        ],
    },
]

PATHWAY_BY_ID: Dict[str, Dict[str, str]] = {
    pathway["id"]: pathway for pathway in PATHWAY_CATALOG
}


def list_pathways() -> List[Dict[str, str]]:
    return PATHWAY_CATALOG


def get_pathway_by_id(pathway_id: str) -> Optional[Dict[str, str]]:
    return PATHWAY_BY_ID.get(pathway_id)


def get_pathway_resource(pathway_id: str, resource_id: Optional[str] = None) -> Optional[Dict[str, str]]:
    pathway = get_pathway_by_id(pathway_id)
    if not pathway:
        return None

    desired_resource_id = resource_id or pathway["default_resource_id"]
    for resource in pathway["resources"]:
        if resource["id"] == desired_resource_id:
            return resource
    return None
