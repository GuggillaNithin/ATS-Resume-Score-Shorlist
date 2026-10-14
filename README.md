# Maple Applicant Tracking System (ATS)

A Python-based Applicant Tracking System built with **Streamlit**, **Sentence Transformers**, **scikit-learn**, and persistent **SQLite** storage.

It enables bulk screening of approximately 50 resumes per batch, automatic candidate extraction, JD-to-resume semantic matching, centralized candidate database management, candidate deduplication, Excel report generation, and historical candidate search for new job requisitions.

---

## 🚀 Key Features

### 1. ⚡ Bulk Resume Screening & ML Scoring
- Upload 1 Job Description (text or PDF/DOCX) and up to ~50 resumes simultaneously.
- Uses **Sentence Transformers (`all-MiniLM-L6-v2`)** cosine similarity match score (0–100) as an objective mathematical baseline.
- Deterministic extraction of candidate information (Name, Email, Phone, Total Experience, Skills) with regex and date-interval heuristics.
- Clearly flags scanned or image-based PDFs requiring OCR text extraction.
- Individual processing status tracking (Pending, Processing, Completed, Failed).
- Assessment invitations via Gmail SMTP and webhook logging to Google Sheets.

### 2. 👥 Centralized Candidate Database & Deduplication
- Persistent relational database (SQLite default, modular architecture ready for PostgreSQL).
- Multi-tier deduplication:
  1. Exact normalized email match
  2. Exact normalized phone match (10+ digits)
  3. SHA-256 resume file hash match
- Profile management: view contact details, download stored resumes, inspect historical screening results, and perform manual HR corrections with verified flags.
- Candidate deletion with file cleanup adhering to data retention policies.

### 3. 🎯 Search Existing Candidates for a New JD
- Enter a new hiring requisition / Job Description.
- Automatically retrieves active existing candidate profiles and rescreens them using Sentence Transformers without re-parsing files.
- Preserves previous screening history while creating new screening records.
- Instant rank calculation, threshold filtering, and Excel shortlist export.

### 4. 📊 Professional Excel Export (`.xlsx`)
- **Standard 6-Column HR Report**:
  1. `Candidate Name`
  2. `Email ID`
  3. `Contact Number`
  4. `Total Experience`
  5. `Role Applied For`
  6. `JD Match Score` *(Strictly the final column)*
- Formatted with bold header row, frozen panes, auto-filters, and auto-adjusted column widths.
- Sorted descending by match score.

---

## 🛠️ Project Structure

```
Maple ATS system/
├── app.py                      # Streamlit application UI & navigation (Dashboard, Bulk Screening, Database, New JD Search, History)
├── requirements.txt            # Python dependencies (sentence-transformers, openpyxl, etc.)
├── .env                        # Local environment variables
├── .env.example                # Configuration template
├── modules/
│   ├── __init__.py
│   ├── database.py             # Schema initialization, SQLite connection manager, indexes
│   ├── storage.py              # Secure resume storage, SHA-256 hashing, path traversal protection
│   ├── resume_parser.py        # PDF & DOCX extraction, scanned OCR detection, detail heuristics
│   ├── jd_matcher.py           # Text cleaning, SentenceTransformer caching, cosine similarity scoring
│   ├── excel_export.py         # Standard 6-column ATS Excel report generator
│   ├── candidate_service.py    # Candidate CRUD, deduplication, search, and profile retrieval
│   └── screening_service.py    # Batch screening orchestrator, new JD rescreening
├── data/
│   └── ats_database.db         # Persistent SQLite database (auto-generated)
├── storage/
│   └── resumes/                # Secure resume file store (auto-generated)
└── tests/
    ├── __init__.py
    └── test_ats.py             # Core ATS unit & integration tests
```

---

## 📦 Installation & Setup

### 1. Prerequisites
- Python 3.10+ (Recommended Python 3.11, 3.12, or 3.13)
- Virtual environment

### 2. Environment Setup
```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Variables (`.env`)
Create a `.env` file based on `.env.example`:
```ini
# Notification & Webhook Integrations
EMAIL_USER=info@maplelearningsolutions.com
EMAIL_PASS=your_gmail_app_password
WEBHOOK_URL=https://script.google.com/macros/s/your_script_id/exec

# Database and Storage Paths
DB_PATH=data/ats_database.db
STORAGE_DIR=storage/resumes
```

### 4. Run Unit & Integration Tests
```powershell
.\.venv\Scripts\python.exe -m unittest discover tests
```

### 5. Launch the Streamlit Application
```powershell
.\.venv\Scripts\streamlit run app.py
```

---

## 🔒 Security & Privacy Notes
- **Resume Access:** Resumes are saved with SHA-256 hashed unique filenames in `storage/resumes/` to prevent directory traversal and filename collisions.
- **Zero External AI Transmissions:** Resume contents and candidate details are processed locally and on-premise without transmission to external AI APIs.
- **Deterministic Contact Safety:** Deterministic regex extraction of email addresses and phone numbers ensures zero hallucination of personal contact info.
