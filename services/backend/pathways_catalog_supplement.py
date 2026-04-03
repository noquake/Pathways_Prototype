from typing import Dict, List


LEGACY_ORDER = ["asthma", "anaphylaxis", "bronchiolitis", "croup", "appendicitis"]

# Pathways enabled in the frontend dropdown.
# Set to None to show all generated pathways.
ACTIVE_PATHWAY_IDS = {
    "asthma",
    "migraine",
    "pneumonia",
    "fever-sepsis-infant",
    "ssti",
}

# Preserve existing curated pathway IDs for user-visible/testing pathways.
ID_OVERRIDES: Dict[str, str] = {
    "animal-human-bite-skin-and-soft-tissue-infection": "animal-bite-ssti",
    "anaphylaxis": "anaphylaxis",
    "appendicitis": "appendicitis",
    "asthma-emergency-department-and-inpatient": "asthma",
    "bronchiolitis": "bronchiolitis",
    "community-acquired-pneumonia-cap": "pneumonia",
    "croup": "croup",
    "fever-and-sepsis-evaluation-infant-ages-29-60-days": "fever-sepsis-infant",
    "migraine-and-migraine-headache": "migraine",
    "skin-and-soft-tissue-infection": "ssti",
}

# Retrieval-only medembed documents that should remain searchable but hidden
# from the user-facing PDF selector. Each alias must resolve to a visible
# preview resource in the same pathway.
HIDDEN_SOURCE_ALIASES: List[Dict[str, str]] = [
    {
        "pathway_id": "asthma",
        "source_file": "asthma-appendix-b-controller-flowchart",
        "preview_resource_id": "emergency-department-algorithm",
    },
    {
        "pathway_id": "asthma",
        "source_file": "asthma-appendix-c-documentation-billing",
        "preview_resource_id": "emergency-department-algorithm",
    },
    {
        "pathway_id": "asthma",
        "source_file": "asthma-appendix-c-nhlbi",
        "preview_resource_id": "emergency-department-algorithm",
    },
    {
        "pathway_id": "migraine",
        "source_file": "migraine-preventive-management-appendix-b",
        "preview_resource_id": "algorithm",
    },
    {
        "pathway_id": "migraine",
        "source_file": "migraine-red-flags-appendix-a",
        "preview_resource_id": "algorithm",
    },
    {
        "pathway_id": "migraine",
        "source_file": "migraine-rescue-management-appendix-c",
        "preview_resource_id": "algorithm",
    },
    {
        "pathway_id": "migraine",
        "source_file": "migraine-sumatriptan-contraindications",
        "preview_resource_id": "algorithm",
    },
    {
        "pathway_id": "pneumonia",
        "source_file": "cap-appendix-a-special-considerations",
        "preview_resource_id": "algorithm",
    },
    {
        "pathway_id": "pneumonia",
        "source_file": "cap-appendix-b-aki-definition",
        "preview_resource_id": "algorithm",
    },
    {
        "pathway_id": "pneumonia",
        "source_file": "cap-main-pathway",
        "preview_resource_id": "algorithm",
    },
    {"pathway_id": "fever-sepsis-infant", "source_file": "fever-sepsis-infant-29-60-days", "preview_resource_id": "pathway"},
    {"pathway_id": "fever-sepsis-neonate", "source_file": "fever-sepsis-neonate-0-28-days", "preview_resource_id": "pathway"},
    {"pathway_id": "fever-sepsis-neonate", "source_file": "fever-sepsis-neonate-0-28-days-appendix-a", "preview_resource_id": "pathway"},
    {"pathway_id": "ssti", "source_file": "ssti-clinical-pathway", "preview_resource_id": "pathway"},
    {"pathway_id": "ssti", "source_file": "ssti-educational-module", "preview_resource_id": "educational-module"},
]

# Manual PDF URL corrections that should survive catalog regeneration.
RESOURCE_PDF_URL_OVERRIDES: List[Dict[str, str]] = [
    {
        "pathway_id": "fever-sepsis-infant",
        "resource_id": "educational-module",
        "pdf_url": "https://www.connecticutchildrens.org/sites/default/files/2025-08/fever-and-sepsis-evaluation-in-the-infant-29-60-days-educational-module-7.28.25.pdf",
    },
    {
        "pathway_id": "ssti",
        "resource_id": "pathway",
        "pdf_url": "https://www.connecticutchildrens.org/sites/default/files/2025-03/ssti-skin-and-soft-tissue-infection-clinical-pathway-1.16.25.pdf",
    },
]
