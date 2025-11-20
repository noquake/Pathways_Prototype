from docling.document_converter import DocumentConverter

source = "clinical_pathways_pdfs/agitation_education_module_-_9.6.23.pdf"  # file path or URL
converter = DocumentConverter()
doc = converter.convert(source).document

print(doc.export_to_markdown())  # output: "### Docling Technical Report[...]"

