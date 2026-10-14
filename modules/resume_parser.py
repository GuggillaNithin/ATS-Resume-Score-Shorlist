import re
import os
from io import BytesIO
from datetime import datetime
from pdfminer.high_level import extract_text as extract_pdf_text

try:
    import docx
except ImportError:
    docx = None

COMMON_SKILLS = [
    "Python", "Java", "C++", "C#", "JavaScript", "TypeScript", "HTML", "CSS", "SQL", "NoSQL",
    "PostgreSQL", "MySQL", "MongoDB", "Redis", "React", "Angular", "Vue", "Node.js", "Django",
    "Flask", "FastAPI", "Spring Boot", ".NET", "AWS", "Azure", "GCP", "Docker", "Kubernetes",
    "Git", "GitHub", "CI/CD", "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
    "Data Analysis", "Pandas", "NumPy", "Scikit-Learn", "TensorFlow", "PyTorch", "Tableau",
    "Power BI", "Excel", "Project Management", "Agile", "Scrum", "Jira", "DevOps", "Linux",
    "Communication", "Leadership", "Teamwork", "Problem Solving"
]

MONTH_MAP = {
    'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3,
    'apr': 4, 'april': 4, 'may': 5, 'jun': 6, 'june': 6, 'jul': 7, 'july': 7,
    'aug': 8, 'august': 8, 'sep': 9, 'september': 9, 'oct': 10, 'october': 10,
    'nov': 11, 'november': 11, 'dec': 12, 'december': 12
}

def extract_resume_text(file_source, filename: str) -> tuple[str, str | None]:
    """
    Extract raw text from a PDF or DOCX file (accepts bytes or file-like object).
    Detects scanned or image-based files and returns appropriate error messages.
    """
    if hasattr(file_source, "read"):
        file_source.seek(0)
        file_bytes = file_source.read()
    elif isinstance(file_source, bytes):
        file_bytes = file_source
    else:
        return "", "Invalid file object provided."

    if not file_bytes:
        return "", "File is empty (0 bytes)."

    ext = os.path.splitext(filename)[1].lower()
    text = ""
    error = None

    try:
        if ext == ".pdf":
            try:
                text = extract_pdf_text(BytesIO(file_bytes))
            except Exception as e:
                return "", f"Failed to parse PDF document: {str(e)}"
        elif ext == ".docx":
            if docx is None:
                return "", "python-docx library is not installed."
            try:
                doc = docx.Document(BytesIO(file_bytes))
                full_text = []
                for para in doc.paragraphs:
                    if para.text.strip():
                        full_text.append(para.text.strip())
                for table in doc.tables:
                    for row in table.rows:
                        row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                        if row_text:
                            full_text.append(" | ".join(row_text))
                text = "\n".join(full_text)
            except Exception as e:
                return "", f"Failed to parse DOCX document: {str(e)}"
        else:
            return "", f"Unsupported file extension '{ext}'. Only PDF and DOCX are supported."
    except Exception as e:
        return "", f"Unexpected error while extracting text: {str(e)}"

    cleaned_sample = re.sub(r'\s+', '', text or '')
    if len(cleaned_sample) < 35:
        return "", "Scanned or image-based document detected (requires OCR text extraction)."

    return text.strip(), None

def extract_email(text: str) -> str | None:
    """Extract and validate the primary email address from text."""
    if not text:
        return None
    matches = re.findall(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', text)
    for email in matches:
        email = email.strip(" .-_,").lower()
        if len(email) >= 6 and not email.endswith(('.png', '.jpg', '.jpeg', '.gif')):
            return email
    return None

def extract_phone(text: str) -> str | None:
    """Extract and validate a phone number from text."""
    if not text:
        return None
    pattern = r'(?:(?:\+91|91|0)[-\s]?)?(?:\(?\d{3,5}\)?[-\s]?)?\d{3,5}[-\s]?\d{3,5}[-\s]?\d{3,5}'
    matches = re.finditer(pattern, text)
    for match in matches:
        raw_phone = match.group(0)
        digits = re.sub(r'\D', '', raw_phone)
        if 10 <= len(digits) <= 14:
            if len(digits) > 10 and digits.startswith(('91', '0')):
                digits = digits[-10:]
            return digits
    return None

def extract_candidate_name(text: str, filename: str = "") -> str:
    """
    Extract candidate's full name from resume text using layout heuristics,
    falling back to a sanitized filename.
    """
    disallowed_name_keywords = {
        "resume", "curriculum", "vitae", "cv", "profile", "summary", "contact", 
        "education", "experience", "skills", "projects", "objective", "certifications",
        "email", "phone", "address", "page", "bio", "linkedin", "github"
    }

    if text:
        lines = [re.sub(r'\s+', ' ', line).strip() for line in text.splitlines() if line.strip()]
        for line in lines[:15]:
            clean_line = line.strip(" ,.-:_|*•#")
            if not clean_line or len(clean_line) > 50:
                continue

            lower_line = clean_line.lower()
            if any(kw in lower_line for kw in disallowed_name_keywords):
                continue
            if "@" in clean_line or re.search(r'\d', clean_line) or "http" in lower_line or "/" in clean_line:
                continue

            words = clean_line.split()
            if 2 <= len(words) <= 4:
                if all(re.fullmatch(r"[A-Za-z][A-Za-z'.-]*", w) and len(w) > 1 for w in words):
                    return " ".join(w.capitalize() for w in words)

    if filename:
        raw_name = os.path.splitext(os.path.basename(filename))[0]
        raw_name = re.sub(r'[\-_]+', ' ', raw_name)
        raw_name = re.sub(r'\b(resume|cv|profile|final|updated|v\d+|\d+)\b', '', raw_name, flags=re.IGNORECASE)
        raw_name = re.sub(r'\s+', ' ', raw_name).strip(" ,.-:_")
        if raw_name:
            words = raw_name.split()
            if 1 <= len(words) <= 5:
                return " ".join(w.capitalize() for w in words)

    return "Candidate"

def _month_year_to_decimal(year: int, month: int = 1) -> float:
    return year + (month - 1) / 12.0

def extract_experience(text: str) -> str:
    """
    Estimate professional work experience from employment date ranges and career text.
    Handles overlapping roles to avoid double-counting.
    Returns e.g. '4.5 Years', '2 Years', or 'Not Available'.
    """
    if not text:
        return "Not Available"

    current_year = datetime.now().year
    current_month = datetime.now().month
    current_decimal = _month_year_to_decimal(current_year, current_month)

    # 1. Search for date ranges like: (Jan 2018 - Dec 2021), (2019 to Present), (06/2017 - 08/2020)
    month_names = r'(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)'
    
    # Pattern A: Month Year - Month Year/Present
    p_month_year = re.compile(
        rf'\b({month_names})?\s*[.,/-]?\s*(20\d\d|19\d\d)\s*(?:-|–|—|to|till)\s*({month_names})?\s*[.,/-]?\s*(20\d\d|19\d\d|present|current|now|date)\b',
        re.IGNORECASE
    )

    # Pattern B: Year - Year/Present (e.g. 2018 - 2022)
    p_year_year = re.compile(
        r'\b(20\d\d|19\d\d)\s*(?:-|–|—|to|till)\s*(20\d\d|19\d\d|present|current|now|date)\b',
        re.IGNORECASE
    )

    intervals = []

    for match in p_month_year.finditer(text):
        m1, y1, m2, y2 = match.groups()
        start_year = int(y1)
        if start_year < 1980 or start_year > current_year:
            continue
        start_month = MONTH_MAP.get(m1.lower() if m1 else '', 1)
        start_dec = _month_year_to_decimal(start_year, start_month)

        if y2.lower() in {'present', 'current', 'now', 'date'}:
            end_dec = current_decimal
        else:
            end_year = int(y2)
            if end_year < start_year or end_year > current_year + 1:
                continue
            end_month = MONTH_MAP.get(m2.lower() if m2 else '', 12)
            end_dec = _month_year_to_decimal(end_year, end_month)

        if end_dec > start_dec:
            intervals.append((start_dec, end_dec))

    if not intervals:
        for match in p_year_year.finditer(text):
            y1, y2 = match.groups()
            start_year = int(y1)
            if start_year < 1980 or start_year > current_year:
                continue
            start_dec = float(start_year)

            if y2.lower() in {'present', 'current', 'now', 'date'}:
                end_dec = current_decimal
            else:
                end_year = int(y2)
                if end_year < start_year or end_year > current_year + 1:
                    continue
                end_dec = float(end_year)

            if end_dec > start_dec:
                intervals.append((start_dec, end_dec))

    # Calculate total non-overlapping duration
    if intervals:
        intervals.sort(key=lambda x: x[0])
        merged = []
        for start, end in intervals:
            if not merged:
                merged.append([start, end])
            else:
                if start <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], end)
                else:
                    merged.append([start, end])

        total_years = sum(end - start for start, end in merged)
        if 0.3 <= total_years <= 45:
            rounded_years = round(total_years, 1)
            return f"{rounded_years:g} Years"

    # Fallback 2: Look for explicit experience statements
    exp_explicit = re.findall(
        r'\b(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)(?:\s*(?:of)?\s*(?:relevant|professional|work|total)?\s*experience)?\b',
        text,
        re.IGNORECASE
    )
    if exp_explicit:
        for val in exp_explicit:
            try:
                num = float(val)
                if 0.5 <= num <= 45:
                    return f"{round(num, 1):g} Years"
            except ValueError:
                pass

    return "Not Available"

def extract_skills(text: str) -> list[str]:
    """Extract matching technical and professional skills from text."""
    if not text:
        return []
    found_skills = []
    text_lower = f" {text.lower()} "
    for skill in COMMON_SKILLS:
        pattern = rf'(?<![a-zA-Z0-9]){re.escape(skill.lower())}(?![a-zA-Z0-9])'
        if re.search(pattern, text_lower):
            found_skills.append(skill)
    return found_skills
