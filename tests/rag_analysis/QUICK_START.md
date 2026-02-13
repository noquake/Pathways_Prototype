# RAG Testing Framework - Quick Start Guide

## Prerequisites

1. **Python 3.8+** installed
2. **PostgreSQL database** running with clinical pathway data
3. **Gemini API key** (or OpenAI API key)

## Installation (5 minutes)

### Step 1: Install Dependencies

```bash
cd tests/rag_analysis
pip install -r requirements.txt
```

### Step 2: Set Environment Variables

```bash
# Required: Gemini API key
export GEMINI_API_KEY="your-gemini-api-key-here"

# Optional: Database URL (defaults to localhost)
export DATABASE_URL="dbname=pathways user=admin password=password host=localhost port=5432"
```

### Step 3: Verify Setup

```bash
python verify_setup.py
```

This checks:
- ✓ All packages installed
- ✓ Environment variables set
- ✓ Database connection works
- ✓ Backend code accessible
- ✓ Required files present

## Running Tests (10-15 minutes)

### Full Test Suite

```bash
python run_rag_test.py
```

This will:
1. Test 10 real clinical pathway queries ✓
2. Generate 5 fake medical documents ✓
3. Test 6 fake document override scenarios ✓
4. Generate comprehensive reports ✓
5. Clean up test data ✓

### View Results

After tests complete:

```bash
# Open HTML dashboard (recommended)
open results/rag_test_report.html

# Or view JSON results
cat results/rag_test_results.json

# Or view individual charts
open results/visualizations/
```

## Understanding Results

### Real Document Tests

**Purpose**: Verify RAG system retrieves and uses correct medical information

**Success Criteria**:
- Semantic similarity ≥ 70%
- Relevant docs in top-3
- Citation accuracy ≥ 80%

**Interpretation**:
- ✅ 80%+ pass rate: Excellent
- ⚠️ 60-80% pass rate: Good, room for improvement
- ❌ <60% pass rate: Needs attention

### Fake Document Override Tests

**Purpose**: Verify RAG uses retrieved context over pre-trained knowledge

**Success Criteria**:
- Fake doc retrieved in top-5
- Fake concepts in answer (≥60% similarity)
- Real concepts avoided

**Interpretation**:
- ✅ 70%+ pass rate: Strong override capability
- ⚠️ 50-70% pass rate: Moderate override
- ❌ <50% pass rate: Relies too much on pre-training

## Common Commands

```bash
# Run verification only
python verify_setup.py

# Generate fake docs only (for inspection)
python generate_fake_docs.py

# Test database operations
python db_setup.py

# Generate sample report
python report_generator.py
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'rag'"

**Solution**: Ensure you're in the correct directory
```bash
cd tests/rag_analysis
python run_rag_test.py
```

### Issue: "GEMINI_API_KEY not set"

**Solution**: Set your API key
```bash
export GEMINI_API_KEY="your-key-here"
python run_rag_test.py
```

### Issue: Database connection failed

**Solution**: Check database is running
```bash
# Test connection
psql -h localhost -U admin -d pathways

# Or check connection string
echo $DATABASE_URL
```

### Issue: Fake documents not cleaned up

**Solution**: Manual cleanup
```python
from db_setup import DatabaseTestSetup
db = DatabaseTestSetup()
db.cleanup_fake_documents()
```

## Next Steps

1. **Review HTML Report**: Detailed analysis with recommendations
2. **Analyze Failures**: Check which categories need improvement
3. **Add Custom Tests**: Edit `ground_truth.json` to add more test cases
4. **Iterate**: Make improvements and re-run tests

## File Reference

```
tests/rag_analysis/
├── run_rag_test.py          # Main test runner (START HERE)
├── verify_setup.py          # Setup verification
├── ground_truth.json        # Test cases
├── generate_fake_docs.py    # Fake doc generator
├── db_setup.py             # Database utilities
├── semantic_evaluator.py   # Evaluation metrics
├── report_generator.py     # Report creation
├── requirements.txt        # Dependencies
├── README.md              # Full documentation
├── QUICK_START.md         # This file
└── results/               # Generated reports
    ├── rag_test_results.json
    ├── rag_test_report.html
    └── visualizations/
```

## Support

For detailed information, see:
- **Full Documentation**: README.md
- **Architecture Details**: See plan file
- **Adding Tests**: README.md > "Adding New Test Cases"

## Testing Workflow

```
1. verify_setup.py → Check everything configured
2. run_rag_test.py → Run all tests
3. Open HTML report → Review results
4. Make improvements → Based on recommendations
5. Re-run tests     → Verify improvements
```

## Expected Runtime

- Setup verification: ~10 seconds
- Real doc tests: ~3-5 minutes (10 tests)
- Fake doc tests: ~5-8 minutes (6 tests + setup/cleanup)
- Report generation: ~30 seconds
- **Total**: ~10-15 minutes for full suite

## Success Indicators

You'll know the system is working well when:
- ✅ Real doc tests pass at 80%+
- ✅ Fake doc tests pass at 70%+
- ✅ Semantic similarity scores above 75%
- ✅ Relevant docs consistently in top-3
- ✅ Citations used in 80%+ of answers

Happy testing! 🚀
