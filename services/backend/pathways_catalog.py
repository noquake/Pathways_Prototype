from typing import Dict, List, Optional


PATHWAY_CATALOG: List[Dict[str, str]] = [
    {
        "id": "asthma",
        "label": "Asthma",
        "doc_name": "asthma_emergency_department_algorithm_-_9.8.23",
        "preview_image_path": "/pathway-previews/asthma.png",
    },
    {
        "id": "anaphylaxis",
        "label": "Anaphylaxis",
        "doc_name": "anaphylaxis_-_1.16.25",
        "preview_image_path": "/pathway-previews/anaphylaxis.png",
    },
    {
        "id": "bronchiolitis",
        "label": "Bronchiolitis",
        "doc_name": "bronchiolitis-educational-module-10.13.23",
        "preview_image_path": "/pathway-previews/bronchiolitis.png",
    },
    {
        "id": "croup",
        "label": "Croup",
        "doc_name": "croup-module-3.25.25",
        "preview_image_path": "/pathway-previews/croup.png",
    },
    {
        "id": "appendicitis",
        "label": "Appendicitis",
        "doc_name": "appendicitis-6.30.23",
        "preview_image_path": "/pathway-previews/appendicitis.png",
    },
]

PATHWAY_BY_ID: Dict[str, Dict[str, str]] = {
    pathway["id"]: pathway for pathway in PATHWAY_CATALOG
}


def list_pathways() -> List[Dict[str, str]]:
    return PATHWAY_CATALOG


def get_pathway_by_id(pathway_id: str) -> Optional[Dict[str, str]]:
    return PATHWAY_BY_ID.get(pathway_id)
