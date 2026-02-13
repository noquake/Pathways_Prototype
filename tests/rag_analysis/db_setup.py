"""
Database Test Setup Utilities

Handles insertion and cleanup of fake documents for RAG testing.
Uses the production schema (docling_chunk.py) with doc_name field.
"""

import os
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer


class DatabaseTestSetup:
    """Manages test database operations for fake documents."""
    
    def __init__(self, db_url: str = None):
        """
        Initialize database connection.
        
        Args:
            db_url: Database connection string (defaults to env var)
        """
        if db_url is None:
            db_url = os.getenv("DATABASE_URL", 
                              "dbname=pathways user=admin password=password host=localhost port=5432")
        
        self.db_url = db_url
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
    def get_connection(self) -> Tuple[psycopg2.extensions.connection, psycopg2.extensions.cursor]:
        """
        Create database connection and cursor.
        
        Returns:
            Tuple of (connection, cursor)
        """
        conn = psycopg2.connect(self.db_url)
        register_vector(conn)
        cur = conn.cursor()
        return conn, cur
    
    def hash_chunk_text(self, text: str) -> str:
        """Generate hash for chunk text (for deduplication)."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    
    def chunk_document(self, file_path: Path) -> List[Dict]:
        """
        Chunk a markdown document using simple chunking.
        
        Args:
            file_path: Path to markdown file
            
        Returns:
            List of chunk dictionaries
        """
        # Use simple text chunking to avoid complex dependencies
        text = file_path.read_text()
        chunks = []
        source_file = file_path.name  # e.g., "fake_dka_protocol_2025.md"
        
        # Simple chunking by paragraphs (max 700 chars per chunk)
        paragraphs = text.split('\n\n')
        current_chunk = []
        chunk_idx = 0
        
        for para in paragraphs:
            if sum(len(p) for p in current_chunk) + len(para) > 700 and current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append({
                    "source_file": source_file,
                    "chunk_index": chunk_idx,
                    "chunk_text": chunk_text,
                    "chunk_length": len(chunk_text)
                })
                chunk_idx += 1
                current_chunk = [para]
            else:
                current_chunk.append(para)
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append({
                "source_file": source_file,
                "chunk_index": chunk_idx,
                "chunk_text": chunk_text,
                "chunk_length": len(chunk_text)
            })
        
        return chunks
    
    def insert_fake_documents(self, doc_paths: List[Path]) -> int:
        """
        Insert fake documents into database.
        
        Args:
            doc_paths: List of paths to fake document markdown files
            
        Returns:
            Number of chunks inserted
        """
        conn, cur = self.get_connection()
        total_inserted = 0
        
        try:
            print(f"\nInserting {len(doc_paths)} fake documents into database...")
            
            # Get the max chunk_index to start from
            cur.execute("SELECT COALESCE(MAX(chunk_index), 0) FROM items")
            next_chunk_index = cur.fetchone()[0] + 1
            
            for doc_path in doc_paths:
                print(f"  Processing: {doc_path.name}")
                
                # Chunk the document
                chunks = self.chunk_document(doc_path)
                print(f"    Generated {len(chunks)} chunks")
                
                # Insert each chunk
                for chunk in chunks:
                    # Generate embedding
                    embedding = self.embedding_model.encode(chunk["chunk_text"])
                    
                    # Insert into database with sequential chunk_index
                    cur.execute(
                        '''
                        INSERT INTO items (
                            chunk_index,
                            chunk_text,
                            chunk_length,
                            source_file,
                            embedding
                        )
                        VALUES (%s, %s, %s, %s, %s::vector)
                        ''',
                        (
                            next_chunk_index,
                            chunk["chunk_text"],
                            chunk["chunk_length"],
                            chunk["source_file"],
                            embedding.tolist(),
                        ),
                    )
                    next_chunk_index += 1
                
                conn.commit()
                print(f"    ✓ Inserted {len(chunks)} chunks")
                total_inserted += len(chunks)
            
            print(f"\n✓ Successfully inserted {total_inserted} total chunks from fake documents")
            return total_inserted
            
        except Exception as e:
            conn.rollback()
            print(f"\n✗ Error inserting fake documents: {e}")
            raise
        finally:
            cur.close()
            conn.close()
    
    def cleanup_fake_documents(self) -> int:
        """
        Remove all fake documents from database.
        Identifies fake docs by doc_name/source_file starting with 'fake_'.
        
        Returns:
            Number of chunks deleted
        """
        conn, cur = self.get_connection()
        
        try:
            print("\nCleaning up fake documents...")
            
            # Determine which field name is used
            cur.execute("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'items' 
                AND column_name IN ('doc_name', 'source_file')
            """)
            result = cur.fetchone()
            doc_field = result[0] if result else 'source_file'
            
            # First, count how many will be deleted
            cur.execute(
                f"SELECT COUNT(*) FROM items WHERE {doc_field} LIKE 'fake_%'"
            )
            count = cur.fetchone()[0]
            
            if count == 0:
                print("  No fake documents found to clean up")
                return 0
            
            # Delete fake documents
            cur.execute(
                f"DELETE FROM items WHERE {doc_field} LIKE 'fake_%'"
            )
            conn.commit()
            
            print(f"  ✓ Deleted {count} chunks from fake documents")
            return count
            
        except Exception as e:
            conn.rollback()
            print(f"  ✗ Error cleaning up fake documents: {e}")
            raise
        finally:
            cur.close()
            conn.close()
    
    def verify_database_state(self) -> Dict:
        """
        Verify database state and check for contamination.
        
        Returns:
            Dictionary with database statistics
        """
        conn, cur = self.get_connection()
        
        try:
            # Determine which field name is used (doc_name or source_file)
            cur.execute("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'items' 
                AND column_name IN ('doc_name', 'source_file')
            """)
            result = cur.fetchone()
            doc_field = result[0] if result else 'source_file'  # Default to source_file
            
            # Count total chunks
            cur.execute("SELECT COUNT(*) FROM items")
            total_chunks = cur.fetchone()[0]
            
            # Count fake chunks
            cur.execute(f"SELECT COUNT(*) FROM items WHERE {doc_field} LIKE 'fake_%'")
            fake_chunks = cur.fetchone()[0]
            
            # Count real chunks
            real_chunks = total_chunks - fake_chunks
            
            # Get unique document names
            cur.execute(f"SELECT DISTINCT {doc_field} FROM items ORDER BY {doc_field}")
            doc_names = [row[0] for row in cur.fetchall()]
            
            # Separate fake and real docs
            fake_docs = [name for name in doc_names if name.startswith('fake_')]
            real_docs = [name for name in doc_names if not name.startswith('fake_')]
            
            stats = {
                "total_chunks": total_chunks,
                "real_chunks": real_chunks,
                "fake_chunks": fake_chunks,
                "total_documents": len(doc_names),
                "real_documents": len(real_docs),
                "fake_documents": len(fake_docs),
                "real_doc_names": real_docs,
                "fake_doc_names": fake_docs
            }
            
            return stats
            
        finally:
            cur.close()
            conn.close()
    
    def print_database_stats(self):
        """Print current database statistics."""
        stats = self.verify_database_state()
        
        print("\n" + "="*60)
        print("DATABASE STATE")
        print("="*60)
        print(f"Total chunks:      {stats['total_chunks']:,}")
        print(f"Real chunks:       {stats['real_chunks']:,} ({stats['real_documents']} documents)")
        print(f"Fake chunks:       {stats['fake_chunks']:,} ({stats['fake_documents']} documents)")
        print()
        
        if stats['fake_documents'] > 0:
            print("Fake documents in database:")
            for doc in stats['fake_doc_names']:
                print(f"  - {doc}")
            print()
        
        if stats['real_documents'] > 0:
            print(f"Real documents: {stats['real_documents']} total")
            if stats['real_documents'] <= 10:
                for doc in stats['real_doc_names']:
                    print(f"  - {doc}")
            else:
                print("  (showing first 10)")
                for doc in stats['real_doc_names'][:10]:
                    print(f"  - {doc}")
                print(f"  ... and {stats['real_documents'] - 10} more")
        
        print("="*60)


def main():
    """Test the database setup utilities."""
    from generate_fake_docs import FakeDocumentGenerator
    
    # Generate fake documents
    generator = FakeDocumentGenerator()
    fake_docs = generator.generate_all()
    
    # Setup database utilities
    db_setup = DatabaseTestSetup()
    
    # Show initial state
    print("\n--- INITIAL DATABASE STATE ---")
    db_setup.print_database_stats()
    
    # Insert fake documents
    print("\n--- INSERTING FAKE DOCUMENTS ---")
    db_setup.insert_fake_documents(fake_docs)
    
    # Show state after insertion
    print("\n--- DATABASE STATE AFTER INSERTION ---")
    db_setup.print_database_stats()
    
    # Cleanup
    print("\n--- CLEANING UP FAKE DOCUMENTS ---")
    db_setup.cleanup_fake_documents()
    
    # Show final state
    print("\n--- FINAL DATABASE STATE ---")
    db_setup.print_database_stats()


if __name__ == "__main__":
    main()
