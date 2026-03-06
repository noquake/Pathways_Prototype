from typing import Dict, List, Optional


PATHWAY_CATALOG: List[Dict[str, str]] = [
    {
        "id": "asthma",
        "label": "Asthma",
        "doc_name": "asthma_emergency_department_algorithm_-_9.8.23",
        "pdf_url": "https://www.connecticutchildrens.org/sites/default/files/2023-09/asthma_emergency_department_algorithm_-_9.8.23.pdf",
    },
    {
        "id": "anaphylaxis",
        "label": "Anaphylaxis",
        "doc_name": "anaphylaxis_-_1.16.25",
        "pdf_url": "https://www.connecticutchildrens.org/sites/default/files/2025-01/anaphylaxis_-_1.16.25.pdf",
    },
    {
        "id": "bronchiolitis",
        "label": "Bronchiolitis",
        "doc_name": "bronchiolitis-educational-module-10.13.23",
        "pdf_url": "https://www.connecticutchildrens.org/sites/default/files/2023-10/bronchiolitis-algorithm-10.4.23.pdf",
    },
    {
        "id": "croup",
        "label": "Croup",
        "doc_name": "croup-module-3.25.25",
        "pdf_url": "https://www.connecticutchildrens.org/sites/default/files/2025-03/croup-algorithm-3.20.25.pdf",
    },
    {
        "id": "appendicitis",
        "label": "Appendicitis",
        "doc_name": "appendicitis-6.30.23",
        "pdf_url": "https://www.connecticutchildrens.org/sites/default/files/2023-09/appendicitis-6.30.23.pdf",
    },
]

PATHWAY_BY_ID: Dict[str, Dict[str, str]] = {
    pathway["id"]: pathway for pathway in PATHWAY_CATALOG
}


def list_pathways() -> List[Dict[str, str]]:
    return PATHWAY_CATALOG


def get_pathway_by_id(pathway_id: str) -> Optional[Dict[str, str]]:
    return PATHWAY_BY_ID.get(pathway_id)
