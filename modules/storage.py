import os
import re
import hashlib
import uuid

DEFAULT_STORAGE_DIR = os.path.join("storage", "resumes")
ALLOWED_EXTENSIONS = {".pdf", ".docx"}
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB

def get_configured_storage_dir():
    dir_path = os.getenv("STORAGE_DIR", "").strip() or DEFAULT_STORAGE_DIR
    os.makedirs(dir_path, exist_ok=True)
    return os.path.abspath(dir_path)

def calculate_file_hash(file_bytes: bytes) -> str:
    """Calculate SHA-256 hash of file content."""
    sha256 = hashlib.sha256()
    sha256.update(file_bytes)
    return sha256.hexdigest()

def sanitize_filename(filename: str) -> str:
    """Sanitize base filename and prevent path traversal."""
    base_name = os.path.basename(filename)
    safe_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', base_name)
    return safe_name or "resume_document"

def validate_uploaded_file(file_bytes: bytes, filename: str) -> tuple[bool, str]:
    """Validate file extension and file size."""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file format '{ext}'. Only PDF and DOCX files are supported."
    
    if len(file_bytes) == 0:
        return False, "File is empty (0 bytes)."
    
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        return False, f"File exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB."
        
    return True, ""

def save_resume_file(file_bytes: bytes, original_filename: str) -> tuple[str, str]:
    """
    Save resume bytes into the persistent storage directory with a secure unique filename.
    Returns (stored_relative_path, file_hash).
    """
    storage_root = get_configured_storage_dir()
    file_hash = calculate_file_hash(file_bytes)
    safe_name = sanitize_filename(original_filename)
    
    # Store with a hash-prefixed safe name to prevent collisions and path traversal
    unique_prefix = f"{file_hash[:12]}_{uuid.uuid4().hex[:6]}"
    stored_name = f"{unique_prefix}_{safe_name}"
    target_abs_path = os.path.join(storage_root, stored_name)
    
    # Ensure target path stays inside storage directory
    if not os.path.abspath(target_abs_path).startswith(storage_root):
        raise ValueError("Invalid file path detected (path traversal attempt).")
        
    with open(target_abs_path, "wb") as f:
        f.write(file_bytes)
        
    return target_abs_path, file_hash

def get_resume_bytes(stored_path: str) -> bytes | None:
    """Safely read binary resume data from disk."""
    if not stored_path or not os.path.exists(stored_path):
        return None
    with open(stored_path, "rb") as f:
        return f.read()

def delete_resume_file(stored_path: str) -> bool:
    """Delete a resume file from disk if it exists."""
    try:
        if stored_path and os.path.exists(stored_path):
            os.remove(stored_path)
            return True
    except OSError:
        pass
    return False
