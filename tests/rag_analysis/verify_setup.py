"""
Setup Verification Script

Checks if all dependencies and configurations are correct before running tests.
"""

import sys
import os
from pathlib import Path


def check_imports():
    """Check if all required packages are installed."""
    print("Checking package imports...")
    
    required_packages = [
        ("sentence_transformers", "sentence-transformers"),
        ("psycopg2", "psycopg2-binary"),
        ("pgvector.psycopg2", "pgvector"),
        ("docling", "docling"),
        ("docling_core", "docling-core"),
        ("transformers", "transformers"),
        ("google.generativeai", "google-generativeai"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("sklearn", "scikit-learn"),
    ]
    
    missing = []
    for module, package in required_packages:
        try:
            __import__(module)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - NOT INSTALLED")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    print("\n✅ All packages installed\n")
    return True


def check_environment():
    """Check environment variables."""
    print("Checking environment variables...")
    
    db_url = os.getenv("DATABASE_URL")
    gemini_key = os.getenv("GEMINI_API_KEY")
    
    if db_url:
        print(f"  ✓ DATABASE_URL: {db_url[:30]}...")
    else:
        print("  ⚠ DATABASE_URL not set (will use default)")
    
    if gemini_key:
        print(f"  ✓ GEMINI_API_KEY: {gemini_key[:10]}...")
    else:
        print("  ✗ GEMINI_API_KEY not set")
        print("    Set with: export GEMINI_API_KEY='your-key-here'")
        return False
    
    print("\n✅ Environment configured\n")
    return True


def check_database():
    """Check database connection."""
    print("Checking database connection...")
    
    try:
        import psycopg2
        from pgvector.psycopg2 import register_vector
        
        db_url = os.getenv("DATABASE_URL", 
                          "dbname=pathways user=admin password=password host=localhost port=5432")
        
        conn = psycopg2.connect(db_url)
        register_vector(conn)
        cur = conn.cursor()
        
        # Check if items table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'items'
            );
        """)
        
        table_exists = cur.fetchone()[0]
        
        if table_exists:
            # Count documents
            cur.execute("SELECT COUNT(*), COUNT(DISTINCT doc_name) FROM items")
            chunk_count, doc_count = cur.fetchone()
            print(f"  ✓ Connected to database")
            print(f"  ✓ Items table exists")
            print(f"  ✓ {chunk_count:,} chunks from {doc_count} documents")
        else:
            print("  ✗ Items table does not exist")
            print("    Run chunking script first: python services/backend/rag/docling_chunk.py")
            cur.close()
            conn.close()
            return False
        
        cur.close()
        conn.close()
        
        print("\n✅ Database ready\n")
        return True
        
    except Exception as e:
        print(f"  ✗ Database error: {e}")
        return False


def check_files():
    """Check if required files exist."""
    print("Checking required files...")
    
    base_dir = Path(__file__).parent
    
    required_files = [
        "ground_truth.json",
        "generate_fake_docs.py",
        "db_setup.py",
        "semantic_evaluator.py",
        "run_rag_test.py",
        "report_generator.py",
        "requirements.txt",
        "README.md"
    ]
    
    missing = []
    for filename in required_files:
        filepath = base_dir / filename
        if filepath.exists():
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} - NOT FOUND")
            missing.append(filename)
    
    if missing:
        print(f"\n❌ Missing files: {', '.join(missing)}")
        return False
    
    print("\n✅ All files present\n")
    return True


def check_backend_access():
    """Check if backend RAG code is accessible."""
    print("Checking backend code access...")
    
    try:
        # Try to import from backend
        backend_path = Path(__file__).parent.parent.parent / "services" / "backend"
        sys.path.append(str(backend_path))
        
        from rag.query import rag_api_llm
        print(f"  ✓ Backend RAG code accessible")
        print(f"  ✓ Can import rag_api_llm")
        
        print("\n✅ Backend integration ready\n")
        return True
        
    except Exception as e:
        print(f"  ✗ Cannot access backend code: {e}")
        print(f"    Ensure services/backend/rag/query.py exists")
        return False


def main():
    """Run all verification checks."""
    print("\n" + "="*60)
    print("RAG Testing Framework - Setup Verification")
    print("="*60 + "\n")
    
    checks = [
        ("Package Installation", check_imports),
        ("Environment Variables", check_environment),
        ("Required Files", check_files),
        ("Backend Integration", check_backend_access),
        ("Database Connection", check_database),
    ]
    
    results = {}
    for name, check_func in checks:
        results[name] = check_func()
    
    # Summary
    print("="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    all_passed = all(results.values())
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")
    
    print("="*60)
    
    if all_passed:
        print("\n🎉 All checks passed! Ready to run tests.")
        print("\nRun tests with:")
        print("  python run_rag_test.py")
        return 0
    else:
        print("\n⚠️  Some checks failed. Fix issues above before running tests.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
