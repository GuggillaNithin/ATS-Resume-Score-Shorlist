import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime

DEFAULT_DB_PATH = os.path.join("data", "ats_database.db")

def get_configured_db_path(custom_path=None):
    if custom_path:
        return custom_path
    env_path = os.getenv("DB_PATH", "").strip()
    return env_path if env_path else DEFAULT_DB_PATH

@contextmanager
def get_db(db_path=None):
    """
    Context manager providing a SQLite database connection with row factory
    and enforced foreign key constraints. Automatically commits transactions
    or rolls back on error.
    """
    path = get_configured_db_path(db_path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    conn = sqlite3.connect(path, timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db(db_path=None):
    """
    Initialize database schema, tables, and indexes according to ATS specifications:
    - candidates
    - resumes
    - requisitions
    - screening_batches
    - screening_results
    """
    with get_db(db_path) as conn:
        cursor = conn.cursor()

        # Candidates entity
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            candidate_id TEXT PRIMARY KEY,
            full_name TEXT NOT NULL,
            email TEXT,
            phone TEXT,
            total_experience TEXT,
            current_role TEXT,
            skills TEXT,
            status TEXT DEFAULT 'Active',
            is_verified INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # Resumes entity (allows multiple versions per candidate)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS resumes (
            resume_id TEXT PRIMARY KEY,
            candidate_id TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            stored_file_path TEXT NOT NULL,
            file_hash TEXT NOT NULL,
            extracted_text TEXT,
            parsing_status TEXT NOT NULL,
            parsing_error TEXT,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (candidate_id) REFERENCES candidates (candidate_id) ON DELETE CASCADE
        );
        """)

        # Job Descriptions / Requisitions entity
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS requisitions (
            requisition_id TEXT PRIMARY KEY,
            job_title TEXT NOT NULL,
            job_description TEXT NOT NULL,
            department TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_by TEXT
        );
        """)

        # Screening Batches entity
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS screening_batches (
            batch_id TEXT PRIMARY KEY,
            requisition_id TEXT,
            total_resumes INTEGER DEFAULT 0,
            successful_count INTEGER DEFAULT 0,
            failed_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_by TEXT,
            FOREIGN KEY (requisition_id) REFERENCES requisitions (requisition_id) ON DELETE SET NULL
        );
        """)

        # Screening Results entity
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS screening_results (
            screening_id TEXT PRIMARY KEY,
            candidate_id TEXT NOT NULL,
            resume_id TEXT NOT NULL,
            requisition_id TEXT NOT NULL,
            batch_id TEXT,
            match_score REAL NOT NULL,
            threshold_value REAL NOT NULL,
            screening_status TEXT NOT NULL,
            screened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (candidate_id) REFERENCES candidates (candidate_id) ON DELETE CASCADE,
            FOREIGN KEY (resume_id) REFERENCES resumes (resume_id) ON DELETE CASCADE,
            FOREIGN KEY (requisition_id) REFERENCES requisitions (requisition_id) ON DELETE CASCADE,
            FOREIGN KEY (batch_id) REFERENCES screening_batches (batch_id) ON DELETE SET NULL
        );
        """)

        # Performance and Search Indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_candidates_email ON candidates (email);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_candidates_phone ON candidates (phone);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_candidates_status ON candidates (status);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_resumes_candidate_id ON resumes (candidate_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_resumes_file_hash ON resumes (file_hash);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_results_candidate_id ON screening_results (candidate_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_results_requisition_id ON screening_results (requisition_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_results_batch_id ON screening_results (batch_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_results_match_score ON screening_results (match_score);")

    return True
