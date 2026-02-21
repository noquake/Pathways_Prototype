#!/usr/bin/env python3
"""
Compare chunking results between PDF and DOCX files using Docling.
Saves chunks to text files for manual inspection and chatbot testing.
"""

import json
from pathlib import Path
import hashlib

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

# Configuration
MAX_TOKENS = 512
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR = Path("chunk_comparison_output")

def setup_output_dir():
    """Create output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"📁 Output will be saved to: {OUTPUT_DIR.absolute()}")

def initialize_chunker():
    """Initialize the same HybridChunker used in your pipeline."""
    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL_ID),
        max_tokens=MAX_TOKENS,
    )
    return HybridChunker(
        tokenizer=tokenizer,
        merge_peers=True,
    )

def process_file(file_path, chunker):
    """
    Process a single file and return its chunks.
    
    Args:
        file_path: Path to the input file (PDF or DOCX)
        chunker: Initialized HybridChunker
    
    Returns:
        List of chunk dictionaries
    """
    print(f"\n🔍 Processing: {file_path}")
    
    # Convert document
    converter = DocumentConverter()
    result = converter.convert(file_path)
    doc = result.document
    
    # Generate chunks
    chunks = []
    for i, raw_chunk in enumerate(chunker.chunk(dl_doc=doc), 1):
        contextualized = chunker.contextualize(raw_chunk)
        
        chunk_data = {
            "chunk_index": i,
            "text": contextualized,
            "length": len(contextualized),
            "token_estimate": len(contextualized.split()),  # rough estimate
        }
        chunks.append(chunk_data)
    
    print(f"   ✅ Generated {len(chunks)} chunks")
    return chunks

def save_chunks(chunks, file_name, file_type):
    """
    Save chunks in multiple formats for comparison.
    """
    base_name = Path(file_name).stem
    
    # 1. Save as plain text (each chunk numbered)
    txt_path = OUTPUT_DIR / f"{base_name}_{file_type}_chunks.txt"
    with open(txt_path, 'w') as f:
        for chunk in chunks:
            f.write(f"{'='*80}\n")
            f.write(f"CHUNK {chunk['chunk_index']} (length: {chunk['length']} chars, ~{chunk['token_estimate']} tokens)\n")
            f.write(f"{'='*80}\n")
            f.write(chunk['text'])
            f.write("\n\n")
    print(f"   💾 Saved chunks to: {txt_path}")
    
    # 2. Save as JSON for programmatic inspection
    json_path = OUTPUT_DIR / f"{base_name}_{file_type}_chunks.json"
    with open(json_path, 'w') as f:
        json.dump(chunks, f, indent=2)
    print(f"   💾 Saved JSON to: {json_path}")
    
    # 3. Save a summary file
    summary_path = OUTPUT_DIR / f"{base_name}_{file_type}_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"File: {file_name}\n")
        f.write(f"Type: {file_type}\n")
        f.write(f"Total chunks: {len(chunks)}\n")
        f.write(f"Avg chunk length: {sum(c['length'] for c in chunks)/len(chunks):.1f} chars\n")
        f.write(f"Min chunk length: {min(c['length'] for c in chunks)} chars\n")
        f.write(f"Max chunk length: {max(c['length'] for c in chunks)} chars\n")
    print(f"   💾 Saved summary to: {summary_path}")

def main():
    """Main comparison function."""
    setup_output_dir()
    chunker = initialize_chunker()
    
    # Specify your files here - update these paths
    files_to_process = [ 
        ("/home/kwaku/Projects/Pathways/experiment/raw_file/connecticut_asthma.pdf", "pdf"),
        ("/home/kwaku/Projects/Pathways/experiment/raw_file/seattle_asthma.pdf", "pdf"),
        ("/home/kwaku/Projects/Pathways/experiment/raw_file/connecticut_asthma.vsdx", "vsdx"),
    ]
    
    all_results = {}
    
    for file_path, file_type in files_to_process:
        if not Path(file_path).exists():
            print(f"⚠️  Warning: {file_path} not found, skipping...")
            continue
            
        chunks = process_file(file_path, chunker)
        save_chunks(chunks, Path(file_path).name, file_type)
        all_results[file_type] = chunks
    
    # Generate comparison report
    if len(all_results) == 2:
        report_path = OUTPUT_DIR / "comparison_report.txt"
        with open(report_path, 'w') as f:
            f.write("📊 PDF vs DOCX CHUNKING COMPARISON\n")
            f.write("="*50 + "\n\n")
            
            pdf_chunks = all_results['pdf']
            docx_chunks = all_results['docx']
            
            f.write(f"PDF:  {len(pdf_chunks)} chunks\n")
            f.write(f"DOCX: {len(docx_chunks)} chunks\n\n")
            
            f.write("Chunk Length Distribution:\n")
            f.write(f"  PDF -  Min: {min(c['length'] for c in pdf_chunks)} chars, "
                   f"Max: {max(c['length'] for c in pdf_chunks)} chars, "
                   f"Avg: {sum(c['length'] for c in pdf_chunks)/len(pdf_chunks):.1f} chars\n")
            f.write(f"  DOCX - Min: {min(c['length'] for c in docx_chunks)} chars, "
                   f"Max: {max(c['length'] for c in docx_chunks)} chars, "
                   f"Avg: {sum(c['length'] for c in docx_chunks)/len(docx_chunks):.1f} chars\n")
            
        print(f"\n📊 Comparison report saved to: {report_path}")
    
    print("\n✅ Done! Check the output directory for chunk files.")

if __name__ == "__main__":
    main()