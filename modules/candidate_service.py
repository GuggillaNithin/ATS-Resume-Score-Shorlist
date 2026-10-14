import uuid
from datetime import datetime
from modules.database import get_db
from modules.storage import delete_resume_file

def find_existing_candidate(email: str | None, phone: str | None, file_hash: str | None, db_path=None) -> tuple[dict | None, str | None]:
    """
    Candidate deduplication strategy:
    1. Exact normalized email match
    2. Exact normalized phone match (if >= 10 digits)
    3. Resume file hash match
    Returns (candidate_dict, match_reason) or (None, None).
    """
    with get_db(db_path) as conn:
        cursor = conn.cursor()

        # Step 1: Match by exact normalized email
        if email and email.strip():
            norm_email = email.strip().lower()
            cursor.execute("SELECT * FROM candidates WHERE LOWER(email) = ? LIMIT 1;", (norm_email,))
            row = cursor.fetchone()
            if row:
                return dict(row), f"Matched existing profile via email: {norm_email}"

        # Step 2: Match by exact phone number
        if phone and phone.strip() and len(phone.strip()) >= 10:
            norm_phone = phone.strip()
            cursor.execute("SELECT * FROM candidates WHERE phone = ? LIMIT 1;", (norm_phone,))
            row = cursor.fetchone()
            if row:
                return dict(row), f"Matched existing profile via phone: {norm_phone}"

        # Step 3: Match by identical resume file hash
        if file_hash:
            cursor.execute("""
            SELECT c.* FROM candidates c
            JOIN resumes r ON c.candidate_id = r.candidate_id
            WHERE r.file_hash = ?
            LIMIT 1;
            """, (file_hash,))
            row = cursor.fetchone()
            if row:
                return dict(row), "Matched existing candidate via identical resume file hash"

    return None, None

def get_or_create_candidate(
    full_name: str,
    email: str | None,
    phone: str | None,
    total_experience: str | None,
    current_role: str | None,
    skills: list[str] | str | None,
    file_hash: str | None = None,
    db_path=None
) -> tuple[dict, bool, str]:
    """
    Locates existing candidate using deduplication criteria or creates a new candidate.
    Returns (candidate_dict, is_new, match_note).
    """
    existing_cand, reason = find_existing_candidate(email, phone, file_hash, db_path)
    skills_str = ", ".join(skills) if isinstance(skills, list) else (skills or "")

    if existing_cand:
        # Candidate exists: return profile without overwriting historical verified records
        candidate_id = existing_cand["candidate_id"]
        # Update updated_at and non-verified details if previously missing
        with get_db(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
            UPDATE candidates 
            SET updated_at = CURRENT_TIMESTAMP,
                total_experience = COALESCE(NULLIF(total_experience, 'Not Available'), ?),
                skills = CASE WHEN LENGTH(COALESCE(skills, '')) > 0 THEN skills ELSE ? END
            WHERE candidate_id = ?;
            """, (total_experience or 'Not Available', skills_str, candidate_id))

        existing_cand["match_reason"] = reason
        return existing_cand, False, reason

    # Create new candidate
    candidate_id = f"cand_{uuid.uuid4().hex[:12]}"
    now_str = datetime.now().isoformat()
    new_candidate = {
        "candidate_id": candidate_id,
        "full_name": full_name or "Candidate",
        "email": email.lower() if email else None,
        "phone": phone,
        "total_experience": total_experience or "Not Available",
        "current_role": current_role,
        "skills": skills_str,
        "status": "Active",
        "is_verified": 0,
        "created_at": now_str,
        "updated_at": now_str
    }

    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO candidates (candidate_id, full_name, email, phone, total_experience, current_role, skills, status, is_verified, created_at, updated_at)
        VALUES (:candidate_id, :full_name, :email, :phone, :total_experience, :current_role, :skills, :status, :is_verified, :created_at, :updated_at);
        """, new_candidate)

    return new_candidate, True, "New candidate registered"

def add_resume_record(
    candidate_id: str,
    original_filename: str,
    stored_file_path: str,
    file_hash: str,
    extracted_text: str,
    parsing_status: str,
    parsing_error: str | None = None,
    db_path=None
) -> dict:
    """Register or link a resume file version to a candidate."""
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        # Check if this exact file hash is already linked to this candidate
        cursor.execute("SELECT * FROM resumes WHERE candidate_id = ? AND file_hash = ? LIMIT 1;", (candidate_id, file_hash))
        existing_resume = cursor.fetchone()
        if existing_resume:
            return dict(existing_resume)

        resume_id = f"res_{uuid.uuid4().hex[:12]}"
        now_str = datetime.now().isoformat()
        resume_record = {
            "resume_id": resume_id,
            "candidate_id": candidate_id,
            "original_filename": original_filename,
            "stored_file_path": stored_file_path,
            "file_hash": file_hash,
            "extracted_text": extracted_text,
            "parsing_status": parsing_status,
            "parsing_error": parsing_error,
            "uploaded_at": now_str
        }

        cursor.execute("""
        INSERT INTO resumes (resume_id, candidate_id, original_filename, stored_file_path, file_hash, extracted_text, parsing_status, parsing_error, uploaded_at)
        VALUES (:resume_id, :candidate_id, :original_filename, :stored_file_path, :file_hash, :extracted_text, :parsing_status, :parsing_error, :uploaded_at);
        """, resume_record)

        return resume_record

def search_candidates(
    search_query: str = "",
    min_exp: float | None = None,
    max_exp: float | None = None,
    skill_filter: str = "",
    role_filter: str = "",
    min_score: float | None = None,
    status_filter: str = "Active",
    limit: int = 50,
    offset: int = 0,
    db_path=None
) -> tuple[list[dict], int]:
    """
    Search and filter candidates with pagination and sorting.
    Returns (candidate_records, total_count).
    """
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        where_clauses = ["1=1"]
        params = []

        if status_filter and status_filter != "All":
            where_clauses.append("c.status = ?")
            params.append(status_filter)

        if search_query and search_query.strip():
            q = f"%{search_query.strip().lower()}%"
            where_clauses.append("(LOWER(c.full_name) LIKE ? OR LOWER(c.email) LIKE ? OR c.phone LIKE ? OR LOWER(c.skills) LIKE ?)")
            params.extend([q, q, q, q])

        if skill_filter and skill_filter.strip():
            sf = f"%{skill_filter.strip().lower()}%"
            where_clauses.append("LOWER(c.skills) LIKE ?")
            params.append(sf)

        if role_filter and role_filter.strip():
            rf = f"%{role_filter.strip().lower()}%"
            where_clauses.append("(LOWER(c.current_role) LIKE ? OR EXISTS (SELECT 1 FROM screening_results sr JOIN requisitions req ON sr.requisition_id = req.requisition_id WHERE sr.candidate_id = c.candidate_id AND LOWER(req.job_title) LIKE ?))")
            params.extend([rf, rf])

        if min_score is not None and min_score > 0:
            where_clauses.append("EXISTS (SELECT 1 FROM screening_results sr WHERE sr.candidate_id = c.candidate_id AND sr.match_score >= ?)")
            params.append(min_score)

        where_sql = " AND ".join(where_clauses)

        # Count total matching
        count_sql = f"SELECT COUNT(DISTINCT c.candidate_id) FROM candidates c WHERE {where_sql};"
        cursor.execute(count_sql, params)
        total_count = cursor.fetchone()[0]

        # Retrieve candidates with latest screening score and latest resume info
        query_sql = f"""
        SELECT 
            c.*,
            (SELECT sr.match_score FROM screening_results sr WHERE sr.candidate_id = c.candidate_id ORDER BY sr.screened_at DESC LIMIT 1) as latest_score,
            (SELECT req.job_title FROM screening_results sr JOIN requisitions req ON sr.requisition_id = req.requisition_id WHERE sr.candidate_id = c.candidate_id ORDER BY sr.screened_at DESC LIMIT 1) as latest_role,
            (SELECT r.original_filename FROM resumes r WHERE r.candidate_id = c.candidate_id ORDER BY r.uploaded_at DESC LIMIT 1) as latest_resume_filename,
            (SELECT r.resume_id FROM resumes r WHERE r.candidate_id = c.candidate_id ORDER BY r.uploaded_at DESC LIMIT 1) as latest_resume_id
        FROM candidates c
        WHERE {where_sql}
        ORDER BY c.created_at DESC
        LIMIT ? OFFSET ?;
        """
        exec_params = list(params) + [limit, offset]
        cursor.execute(query_sql, exec_params)
        candidates = [dict(r) for r in cursor.fetchall()]

        return candidates, total_count

def get_candidate_details(candidate_id: str, db_path=None) -> dict | None:
    """Retrieve complete candidate profile, all resume versions, and full screening history."""
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM candidates WHERE candidate_id = ?;", (candidate_id,))
        cand_row = cursor.fetchone()
        if not cand_row:
            return None

        candidate = dict(cand_row)

        cursor.execute("SELECT * FROM resumes WHERE candidate_id = ? ORDER BY uploaded_at DESC;", (candidate_id,))
        candidate["resumes"] = [dict(r) for r in cursor.fetchall()]

        cursor.execute("""
        SELECT sr.*, req.job_title, req.job_description, sb.created_at as batch_date
        FROM screening_results sr
        JOIN requisitions req ON sr.requisition_id = req.requisition_id
        LEFT JOIN screening_batches sb ON sr.batch_id = sb.batch_id
        WHERE sr.candidate_id = ?
        ORDER BY sr.screened_at DESC;
        """, (candidate_id,))
        screenings = [dict(r) for r in cursor.fetchall()]
        for s in screenings:
            if s.get("ai_analysis_json"):
                try:
                    import json
                    s["ai_analysis"] = json.loads(s["ai_analysis_json"])
                except Exception:
                    s["ai_analysis"] = {}
            else:
                s["ai_analysis"] = {}
        candidate["screenings"] = screenings

        return candidate


def update_candidate_details(
    candidate_id: str,
    full_name: str,
    email: str | None,
    phone: str | None,
    total_experience: str | None,
    current_role: str | None,
    skills: str | None,
    status: str = "Active",
    db_path=None
) -> bool:
    """Allow authorized HR corrections to candidate details, marking record as verified."""
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
        UPDATE candidates
        SET full_name = ?,
            email = ?,
            phone = ?,
            total_experience = ?,
            current_role = ?,
            skills = ?,
            status = ?,
            is_verified = 1,
            updated_at = CURRENT_TIMESTAMP
        WHERE candidate_id = ?;
        """, (full_name, email.lower() if email else None, phone, total_experience, current_role, skills, status, candidate_id))
        return cursor.rowcount > 0

def update_candidate_status(candidate_id: str, new_status: str, db_path=None) -> bool:
    """Set candidate status (Active, Archived, Inactive)."""
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
        UPDATE candidates
        SET status = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE candidate_id = ?;
        """, (new_status, candidate_id))
        return cursor.rowcount > 0

def delete_candidate(candidate_id: str, db_path=None) -> bool:
    """
    Permanently delete candidate profile, associated screening results,
    and remove stored resume files from disk (compliance with data retention).
    """
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT stored_file_path FROM resumes WHERE candidate_id = ?;", (candidate_id,))
        resume_rows = cursor.fetchall()
        for r in resume_rows:
            delete_resume_file(r["stored_file_path"])

        cursor.execute("DELETE FROM candidates WHERE candidate_id = ?;", (candidate_id,))
        return cursor.rowcount > 0

def get_ats_dashboard_stats(db_path=None) -> dict:
    """Retrieve high-level aggregated metrics for the ATS dashboard."""
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM candidates;")
        total_candidates = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM candidates WHERE status = 'Active';")
        active_candidates = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM screening_batches;")
        total_batches = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM resumes;")
        total_resumes = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM screening_results WHERE screening_status = 'Meets Threshold';")
        meets_threshold_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM screening_results;")
        total_screenings = cursor.fetchone()[0]

        # Recent screening activity
        cursor.execute("""
        SELECT sr.match_score, sr.screening_status, sr.screened_at, c.full_name, req.job_title
        FROM screening_results sr
        JOIN candidates c ON sr.candidate_id = c.candidate_id
        JOIN requisitions req ON sr.requisition_id = req.requisition_id
        ORDER BY sr.screened_at DESC
        LIMIT 6;
        """)
        recent_activity = [dict(r) for r in cursor.fetchall()]

        return {
            "total_candidates": total_candidates,
            "active_candidates": active_candidates,
            "total_batches": total_batches,
            "total_resumes": total_resumes,
            "meets_threshold_count": meets_threshold_count,
            "total_screenings": total_screenings,
            "recent_activity": recent_activity
        }
