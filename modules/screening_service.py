import uuid
from datetime import datetime
from modules.database import get_db
from modules.storage import validate_uploaded_file, save_resume_file
from modules.resume_parser import (
    extract_resume_text,
    extract_candidate_name,
    extract_email,
    extract_phone,
    extract_experience,
    extract_skills
)
from modules.jd_matcher import (
    calculate_jd_match_score,
    batch_calculate_match_scores,
    get_screening_outcome,
    get_sentence_transformer_model
)
from modules.candidate_service import (
    get_or_create_candidate,
    add_resume_record
)

def create_requisition(job_title: str, job_description: str, department: str = "", created_by: str = "HR Team", db_path=None) -> dict:
    """Create a persistent requisition record for a Job Description."""
    requisition_id = f"req_{uuid.uuid4().hex[:12]}"
    now_str = datetime.now().isoformat()
    record = {
        "requisition_id": requisition_id,
        "job_title": job_title.strip() or "Untitled Requisition",
        "job_description": job_description.strip(),
        "department": department.strip() if department else None,
        "created_at": now_str,
        "created_by": created_by
    }

    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO requisitions (requisition_id, job_title, job_description, department, created_at, created_by)
        VALUES (:requisition_id, :job_title, :job_description, :department, :created_at, :created_by);
        """, record)

    return record

def process_single_resume(
    file_bytes: bytes,
    filename: str,
    jd_text: str,
    job_title: str,
    requisition_id: str,
    batch_id: str | None,
    threshold: float,
    model=None,
    db_path=None
) -> dict:
    """
    Process an individual resume end-to-end:
    Validation -> Storage -> Text extraction -> Candidate detail extraction -> Deduplication -> ML Scoring -> DB Result.
    Safe against failures; one corrupt file does not disrupt batch.
    """
    result = {
        "filename": filename,
        "status": "Pending",
        "candidate_id": None,
        "resume_id": None,
        "screening_id": None,
        "candidate_name": "Not Found",
        "email": "Not Found",
        "phone": "Not Found",
        "experience": "Not Available",
        "skills": [],
        "role": job_title,
        "score": 0.0,
        "threshold": threshold,
        "screening_status": "Failed",
        "error": None,
        "is_new_candidate": False,
        "match_note": ""
    }

    # Step 1: File validation
    is_valid, val_err = validate_uploaded_file(file_bytes, filename)
    if not is_valid:
        result["status"] = "Failed"
        result["error"] = val_err
        return result

    # Step 2: Storage & File Hashing
    try:
        stored_path, file_hash = save_resume_file(file_bytes, filename)
    except Exception as e:
        result["status"] = "Failed"
        result["error"] = f"Storage error: {str(e)}"
        return result

    # Step 3: Text Extraction
    text, parse_err = extract_resume_text(file_bytes, filename)
    if parse_err:
        result["status"] = "Failed"
        result["error"] = parse_err
        return result

    # Step 4: Deterministic field extraction
    name = extract_candidate_name(text, filename)
    email = extract_email(text)
    phone = extract_phone(text)
    experience = extract_experience(text)
    skills = extract_skills(text)

    # Step 5: Candidate Deduplication & Profile Registration
    candidate, is_new, match_note = get_or_create_candidate(
        full_name=name,
        email=email,
        phone=phone,
        total_experience=experience,
        current_role=job_title,
        skills=skills,
        file_hash=file_hash,
        db_path=db_path
    )
    candidate_id = candidate["candidate_id"]

    # Step 6: Store resume record
    resume_record = add_resume_record(
        candidate_id=candidate_id,
        original_filename=filename,
        stored_file_path=stored_path,
        file_hash=file_hash,
        extracted_text=text,
        parsing_status="Completed",
        parsing_error=None,
        db_path=db_path
    )
    resume_id = resume_record["resume_id"]

    # Step 7: Calculate ML Match Score (SentenceTransformer semantic similarity 0-100)
    score = calculate_jd_match_score(jd_text, text, model=model)
    outcome = get_screening_outcome(score, threshold)

    # Step 8: Persist Screening Result
    screening_id = f"scr_{uuid.uuid4().hex[:12]}"
    now_str = datetime.now().isoformat()
    screening_record = {
        "screening_id": screening_id,
        "candidate_id": candidate_id,
        "resume_id": resume_id,
        "requisition_id": requisition_id,
        "batch_id": batch_id,
        "match_score": score,
        "threshold_value": threshold,
        "screening_status": outcome,
        "screened_at": now_str
    }

    with get_db(db_path) as conn:
        cursor = conn.cursor()
        valid_batch_id = None
        if batch_id:
            cursor.execute("SELECT 1 FROM screening_batches WHERE batch_id = ?;", (batch_id,))
            if cursor.fetchone():
                valid_batch_id = batch_id

        screening_record["batch_id"] = valid_batch_id
        cursor.execute("""
        INSERT INTO screening_results (screening_id, candidate_id, resume_id, requisition_id, batch_id, match_score, threshold_value, screening_status, screened_at)
        VALUES (:screening_id, :candidate_id, :resume_id, :requisition_id, :batch_id, :match_score, :threshold_value, :screening_status, :screened_at);
        """, screening_record)

    # Format return structure
    result.update({
        "status": "Completed",
        "candidate_id": candidate_id,
        "resume_id": resume_id,
        "screening_id": screening_id,
        "candidate_name": name,
        "email": email or "Not Found",
        "phone": phone or "Not Found",
        "experience": experience,
        "skills": skills,
        "role": job_title,
        "score": score,
        "screening_status": outcome,
        "is_new_candidate": is_new,
        "match_note": match_note,
        "resume_text": text
    })

    return result

def run_batch_screening(
    uploaded_files: list,
    jd_text: str,
    job_title: str,
    threshold: float,
    progress_callback=None,
    db_path=None
) -> tuple[dict, list[dict], list[dict]]:
    """
    Orchestrate batch screening of up to ~50 resumes.
    Returns (batch_info, successful_candidates, failed_candidates).
    """
    model = get_sentence_transformer_model()
    req = create_requisition(job_title, jd_text, db_path=db_path)
    requisition_id = req["requisition_id"]

    batch_id = f"batch_{uuid.uuid4().hex[:12]}"
    now_str = datetime.now().isoformat()

    total_files = len(uploaded_files)
    successful = []
    failed = []

    for idx, uploaded_file in enumerate(uploaded_files):
        if hasattr(uploaded_file, "read"):
            uploaded_file.seek(0)
            file_bytes = uploaded_file.read()
        elif isinstance(uploaded_file, bytes):
            file_bytes = uploaded_file
        else:
            continue

        filename = getattr(uploaded_file, "name", f"resume_{idx+1}.pdf")

        if progress_callback:
            progress_callback(idx, total_files, filename, "Processing")

        res = process_single_resume(
            file_bytes=file_bytes,
            filename=filename,
            jd_text=jd_text,
            job_title=job_title,
            requisition_id=requisition_id,
            batch_id=batch_id,
            threshold=threshold,
            model=model,
            db_path=db_path
        )

        if res["status"] == "Completed":
            cand_item = {
                "Candidate Name": res["candidate_name"],
                "Email ID": res["email"],
                "Contact Number": res["phone"],
                "Total Experience": res["experience"],
                "Role Applied For": res["role"],
                "JD Match Score": res["score"],
                # Internal fields
                "Candidate ID": res["candidate_id"],
                "Resume ID": res["resume_id"],
                "Screening ID": res["screening_id"],
                "Resume Name": res["filename"],
                "Score (%)": res["score"],
                "Email": res["email"] if res["email"] != "Not Found" else "",
                "Phone": res["phone"] if res["phone"] != "Not Found" else "",
                "Screening Status": res["screening_status"],
                "Resume Text": res.get("resume_text", ""),
                "Skills": res.get("skills", []),
                "Batch ID": batch_id,
                "Requisition ID": requisition_id,
                "Deduplication Note": res["match_note"]
            }
            successful.append(cand_item)
        else:
            failed.append({
                "Resume Name": filename,
                "Status": "Failed",
                "Reason": res.get("error") or "Unknown error",
                "File Bytes": file_bytes
            })

    batch_info = {
        "batch_id": batch_id,
        "requisition_id": requisition_id,
        "job_title": job_title,
        "total_resumes": total_files,
        "successful_count": len(successful),
        "failed_count": len(failed),
        "created_at": now_str
    }

    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO screening_batches (batch_id, requisition_id, total_resumes, successful_count, failed_count, created_at)
        VALUES (:batch_id, :requisition_id, :total_resumes, :successful_count, :failed_count, :created_at);
        """, batch_info)

    # Sort successful candidates descending by match score
    successful.sort(key=lambda x: x["JD Match Score"], reverse=True)
    for rank, item in enumerate(successful, 1):
        item["Rank"] = rank

    if progress_callback:
        progress_callback(total_files, total_files, "Done", "Completed")

    return batch_info, successful, failed

def rescreen_candidates_for_new_jd(
    new_job_title: str,
    new_jd_text: str,
    threshold: float,
    min_exp_years: float | None = None,
    save_to_database: bool = True,
    db_path=None
) -> tuple[dict, list[dict]]:
    """
    Search and rescreen all active existing candidates against a newly supplied Job Description.
    Recalculates semantic similarity using SentenceTransformer.
    """
    model = get_sentence_transformer_model()
    req = create_requisition(new_job_title, new_jd_text, db_path=db_path)
    requisition_id = req["requisition_id"]

    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
        SELECT 
            c.candidate_id,
            c.full_name,
            c.email,
            c.phone,
            c.total_experience,
            c.skills,
            r.resume_id,
            r.original_filename,
            r.extracted_text,
            r.uploaded_at
        FROM candidates c
        JOIN resumes r ON c.candidate_id = r.candidate_id
        WHERE c.status = 'Active' AND r.parsing_status = 'Completed' AND LENGTH(COALESCE(r.extracted_text, '')) > 20
        GROUP BY c.candidate_id
        HAVING r.uploaded_at = MAX(r.uploaded_at);
        """)
        candidates_pool = [dict(r) for r in cursor.fetchall()]

    if not candidates_pool:
        return req, []

    resume_texts = [c["extracted_text"] for c in candidates_pool]
    scores = batch_calculate_match_scores(new_jd_text, resume_texts, model=model)

    evaluated_candidates = []
    now_str = datetime.now().isoformat()

    with get_db(db_path) as conn:
        cursor = conn.cursor()
        for cand, score in zip(candidates_pool, scores):
            outcome = get_screening_outcome(score, threshold)
            scr_id = f"scr_{uuid.uuid4().hex[:12]}"

            if save_to_database:
                cursor.execute("""
                INSERT INTO screening_results (screening_id, candidate_id, resume_id, requisition_id, match_score, threshold_value, screening_status, screened_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    scr_id,
                    cand["candidate_id"],
                    cand["resume_id"],
                    requisition_id,
                    score,
                    threshold,
                    outcome,
                    now_str
                ))

            cand_item = {
                "Candidate Name": cand["full_name"],
                "Email ID": cand["email"] or "Not Found",
                "Contact Number": cand["phone"] or "Not Found",
                "Total Experience": cand["total_experience"] or "Not Available",
                "Role Applied For": new_job_title,
                "JD Match Score": score,
                "Candidate ID": cand["candidate_id"],
                "Screening ID": scr_id,
                "Resume Name": cand["original_filename"],
                "Score (%)": score,
                "Email": cand["email"] or "",
                "Phone": cand["phone"] or "",
                "Screening Status": outcome,
                "Resume Text": cand["extracted_text"],
                "Requisition ID": requisition_id
            }
            evaluated_candidates.append(cand_item)

    # Sort descending by match score
    evaluated_candidates.sort(key=lambda x: x["JD Match Score"], reverse=True)
    for rank, item in enumerate(evaluated_candidates, 1):
        item["Rank"] = rank

    return req, evaluated_candidates

def get_screening_batches_list(limit: int = 50, db_path=None) -> list[dict]:
    """Retrieve historical batches with requisition information."""
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
        SELECT 
            sb.*,
            req.job_title,
            req.job_description,
            (SELECT COUNT(*) FROM screening_results sr WHERE sr.batch_id = sb.batch_id AND sr.screening_status = 'Meets Threshold') as shortlisted_count
        FROM screening_batches sb
        LEFT JOIN requisitions req ON sb.requisition_id = req.requisition_id
        ORDER BY sb.created_at DESC
        LIMIT ?;
        """, (limit,))
        return [dict(r) for r in cursor.fetchall()]

def get_batch_results(batch_id: str, db_path=None) -> list[dict]:
    """Retrieve all candidates from a specific historical batch."""
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
        SELECT 
            c.candidate_id,
            c.full_name as "Candidate Name",
            COALESCE(c.email, 'Not Found') as "Email ID",
            COALESCE(c.phone, 'Not Found') as "Contact Number",
            COALESCE(c.total_experience, 'Not Available') as "Total Experience",
            req.job_title as "Role Applied For",
            sr.match_score as "JD Match Score",
            sr.match_score as "Score (%)",
            sr.screening_status as "Screening Status",
            sr.screening_id as "Screening ID",
            r.original_filename as "Resume Name",
            r.extracted_text as "Resume Text",
            c.email as "Email",
            c.phone as "Phone"
        FROM screening_results sr
        JOIN candidates c ON sr.candidate_id = c.candidate_id
        JOIN resumes r ON sr.resume_id = r.resume_id
        JOIN requisitions req ON sr.requisition_id = req.requisition_id
        WHERE sr.batch_id = ?
        ORDER BY sr.match_score DESC;
        """, (batch_id,))
        results = [dict(r) for r in cursor.fetchall()]
        for rank, r in enumerate(results, 1):
            r["Rank"] = rank
        return results
