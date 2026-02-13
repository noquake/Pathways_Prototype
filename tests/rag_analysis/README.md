# RAG Analysis & Testing Framework

This directory contains a comprehensive testing framework for the RAG (Retrieval-Augmented Generation) system that validates both retrieval accuracy and answer quality.

## Overview

The RAG testing framework performs two types of tests:

1. **Real Document Tests**: Validates that the RAG system correctly retrieves and uses real clinical pathways to generate accurate medical guidance
2. **Fake Document Override Tests**: Validates that the RAG system properly uses retrieved context over its pre-trained knowledge by testing with intentionally incorrect "fake" medical documents

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure database is running and accessible
# Set DATABASE_URL environment variable if needed
export DATABASE_URL="dbname=pathways user=admin password=password host=localhost port=5432"

# Set Gemini API key
export GEMINI_API_KEY="your-api-key-here"
```

### Running Tests

```bash
# Run complete test suite
python run_rag_test.py
```

This will:
1. Test real clinical pathways with ground truth Q&A pairs
2. Generate and insert fake documents
3. Test RAG override capability
4. Clean up fake documents
5. Generate comprehensive reports

### Viewing Results

After tests complete, view results in the `results/` directory:
- **JSON Report**: `rag_test_results.json` - Detailed metrics for all tests
- **HTML Dashboard**: `rag_test_report.html` - Interactive visual dashboard (open in browser)
- **Visualizations**: `visualizations/*.png` - Individual chart images

## Framework Components

### 1. Ground Truth Dataset (`ground_truth.json`)

Contains test cases with expected answers for both real and fake document tests:
- Real pathway tests with expected concepts and relevant documents
- Fake document tests with absurd treatments to verify override capability

### 2. Fake Document Generator (`generate_fake_docs.py`)

Generates realistic-looking medical documents with intentionally incorrect information:
- DKA protocol (handstands, ice cream, dancing)
- Sepsis protocol (ice cream, Baby Shark music, cartwheels)
- Asthma protocol (balloon blowing, chicken dance, jalapeños)
- Appendicitis protocol (jumping jacks, tickling, pickle juice)
- Fracture protocol (duct tape, tap dancing, milkshakes)

### 3. Database Test Setup (`db_setup.py`)

Handles test database operations:
- Inserts fake documents with proper chunking and embeddings
- Cleans up fake documents after testing
- Verifies database integrity
- Uses production schema (docling_chunk.py compatibility)

### 4. Semantic Evaluator (`semantic_evaluator.py`)

Provides sophisticated evaluation metrics:
- **Semantic Similarity**: Compares answers to expected concepts using embeddings (not keyword matching)
- **Context Relevance**: Validates that relevant documents are retrieved and ranked appropriately
- **Citation Accuracy**: Verifies proper use of source citations
- **Factual Consistency**: For fake tests, checks if fake concepts appear and real concepts are avoided

### 5. Test Runner (`run_rag_test.py`)

Main orchestration script:
- Runs real document tests first
- Generates and inserts fake documents
- Runs fake document override tests
- Cleans up test data
- Coordinates report generation

### 6. Report Generator (`report_generator.py`)

Creates comprehensive reports:
- **JSON**: Detailed structured results
- **HTML Dashboard**: Interactive visual report with charts
- **PNG Charts**: Individual visualizations for analysis

## Evaluation Metrics

### Real Document Tests - Pass Criteria

A real document test passes when ALL of the following are met:
- **Semantic Similarity ≥ 70%**: Answer semantically matches expected concepts
- **Context Relevance ≥ 70%**: Relevant documents retrieved in top-3 results
- **Citation Accuracy ≥ 80%**: Proper citations to source documents

### Fake Document Override Tests - Pass Criteria

A fake document override test passes when ALL of the following are met:
- **Fake Doc Retrieved**: Fake document appears in top-5 results
- **Fake Doc Rank ≤ 5**: Retrieved within top 5 positions
- **Semantic Match ≥ 60%**: Answer contains concepts from fake document
- **Find Rate ≥ 50%**: At least half of fake concepts mentioned in answer

## Results Visualizations

The framework generates the following visualizations:

1. **Overall Results** (`01_overall_results.png`): Pie chart of pass/fail distribution
2. **Category Scores** (`02_category_scores.png`): Bar chart comparing real vs fake test performance by category
3. **Real Doc Breakdown** (`03_real_doc_breakdown.png`): Detailed metrics for real document tests
4. **Fake Doc Heatmap** (`04_fake_doc_heatmap.png`): Performance heatmap for override tests
5. **Semantic Distribution** (`05_semantic_distribution.png`): Distribution of semantic similarity scores
6. **Retrieval Analysis** (`06_retrieval_analysis.png`): Retrieval accuracy and ranking analysis

## Architecture

```
┌─────────────────┐
│  Test Runner    │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼──┐  ┌──▼────┐
│Real  │  │Fake   │
│Tests │  │Tests  │
└───┬──┘  └──┬────┘
    │        │
    │  ┌─────▼──────┐
    │  │Fake Doc    │
    │  │Generator   │
    │  └─────┬──────┘
    │        │
    │  ┌─────▼──────┐
    │  │DB Setup    │
    │  │(Insert/    │
    │  │Cleanup)    │
    │  └────────────┘
    │
┌───▼────────────┐
│Semantic        │
│Evaluator       │
└───┬────────────┘
    │
┌───▼────────────┐
│Report          │
│Generator       │
│ ├─ JSON        │
│ ├─ HTML        │
│ └─ Charts      │
└────────────────┘
```

## Configuration

The test runner can be configured via environment variables:

```bash
# Database connection
export DATABASE_URL="dbname=pathways user=admin password=password host=localhost port=5432"

# LLM API key
export GEMINI_API_KEY="your-api-key-here"

# Optional: Override defaults in code
# - llm_model: Model to use (default: "gemini-2.5-flash")
# - api_provider: "gemini" or "openai"
# - top_k: Number of chunks to retrieve (default: 5)
```

## Key Improvements Over Previous Tests

1. ✅ **Semantic Metrics**: Uses embedding-based similarity instead of exact keyword matching
2. ✅ **Automated Setup**: Fake documents automatically generated, inserted, and cleaned up
3. ✅ **Real Pathway Validation**: Tests correctness on actual clinical pathways, not just override
4. ✅ **Production Alignment**: Uses same LLM, schema, and configuration as production
5. ✅ **Actionable Reports**: HTML dashboard with specific recommendations for improvement
6. ✅ **Schema Compatibility**: Works with production `docling_chunk.py` database schema

## Adding New Test Cases

To add new test cases, edit `ground_truth.json`:

### Real Document Test

```json
{
  "id": "unique_id",
  "category": "Category",
  "query": "Your question",
  "expected_concepts": ["concept 1", "concept 2"],
  "relevant_docs": ["doc_name_prefix"],
  "expected_answer_summary": "Brief summary of correct answer"
}
```

### Fake Document Test

```json
{
  "id": "unique_id",
  "category": "Category", 
  "query": "Your question",
  "fake_doc_name": "fake_protocol_name",
  "fake_concepts": ["absurd concept 1", "absurd concept 2"],
  "should_not_contain": ["correct concept 1", "correct concept 2"]
}
```

Then add corresponding fake document generation in `generate_fake_docs.py`.

## Troubleshooting

### "No module named 'rag'"
Ensure you're running from the correct directory and the backend code is accessible:
```bash
cd tests/rag_analysis
python run_rag_test.py
```

### "GEMINI_API_KEY not set"
Set your API key:
```bash
export GEMINI_API_KEY="your-key-here"
```

### Database connection errors
Verify database is running and credentials are correct:
```bash
psql -h localhost -U admin -d pathways
```

### Fake documents not cleaned up
Manually cleanup if needed:
```python
from db_setup import DatabaseTestSetup
db = DatabaseTestSetup()
db.cleanup_fake_documents()
```

## Development

### Running Individual Components

Test individual components:

```bash
# Generate fake documents only
python generate_fake_docs.py

# Test database operations
python db_setup.py

# Test semantic evaluator
python semantic_evaluator.py

# Generate sample report
python report_generator.py
```

## License & Credits

Part of the Pathways Clinical Decision Support System.
RAG Testing Framework v1.0
