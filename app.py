import os
import sys
import warnings
import logging

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import streamlit as st
import re
import smtplib
import requests
import numpy as np
import pandas as pd
from io import BytesIO
from email.message import EmailMessage
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from streamlit.errors import StreamlitSecretNotFoundError

# Import ATS Architecture Modules
from modules.database import init_db, get_db
from modules.storage import (
    validate_uploaded_file,
    save_resume_file,
    get_resume_bytes,
    delete_resume_file
)
from modules.resume_parser import (
    extract_resume_text,
    extract_candidate_name,
    extract_email,
    extract_phone,
    extract_experience,
    extract_skills
)
from modules.jd_matcher import (
    clean_text,
    calculate_jd_match_score,
    batch_calculate_match_scores,
    get_screening_outcome,
    get_sentence_transformer_model
)
from modules.excel_export import (
    generate_candidate_excel_report,
    generate_failed_resumes_report,
    get_report_filename,
    EXCEL_REPORT_COLUMNS
)
from modules.candidate_service import (
    get_or_create_candidate,
    add_resume_record,
    search_candidates,
    get_candidate_details,
    update_candidate_details,
    update_candidate_status,
    delete_candidate,
    get_ats_dashboard_stats
)
from modules.screening_service import (
    create_requisition,
    process_single_resume,
    run_batch_screening,
    rescreen_candidates_for_new_jd,
    get_screening_batches_list,
    get_batch_results
)

# --- Configuration & Styling ---
ST_PAGE_TITLE = "Maple ATS - Resume Screening & Candidate Management"
ST_PAGE_ICON = "🎯"

# Ensure Database is initialized on startup
init_db()

def get_streamlit_secret(key, default=None):
    """Safely read a Streamlit secret without crashing when secrets.toml is missing."""
    try:
        return st.secrets.get(key, default)
    except StreamlitSecretNotFoundError:
        return default

def get_webhook_url():
    """Resolve webhook URL from environment or Streamlit secrets at runtime."""
    return (os.getenv("WEBHOOK_URL") or get_streamlit_secret("WEBHOOK_URL") or "").strip()

@st.cache_resource
def load_cached_transformer():
    """Load and cache the SentenceTransformer model in memory."""
    return get_sentence_transformer_model('all-MiniLM-L6-v2')

def apply_custom_style():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .top-candidate-card {
        padding: 20px;
        border-radius: 12px;
        background: var(--secondary-background-color);
        border-left: 6px solid var(--primary-color);
        margin-bottom: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.06);
    }
    .rank-1 { border-left-color: #00DC82; }
    .rank-2 { border-left-color: #007BFF; }
    .rank-3 { border-left-color: #FFA500; }

    .badge-success {
        display: inline-block;
        padding: 3px 10px;
        font-size: 0.75rem;
        font-weight: 700;
        color: #065F46;
        background-color: #D1FAE5;
        border-radius: 9999px;
    }
    .badge-warning {
        display: inline-block;
        padding: 3px 10px;
        font-size: 0.75rem;
        font-weight: 700;
        color: #92400E;
        background-color: #FEF3C7;
        border-radius: 9999px;
    }
    .badge-danger {
        display: inline-block;
        padding: 3px 10px;
        font-size: 0.75rem;
        font-weight: 700;
        color: #991B1B;
        background-color: #FEE2E2;
        border-radius: 9999px;
    }
    .badge-info {
        display: inline-block;
        padding: 3px 10px;
        font-size: 0.75rem;
        font-weight: 700;
        color: #1E40AF;
        background-color: #DBEAFE;
        border-radius: 9999px;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.55rem 1.5rem;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
    }
    </style>
    """, unsafe_allow_html=True)

def send_email(to_email, subject, message_body):
    """Send an HTML email using Gmail SMTP credentials from environment variables."""
    email_user = os.getenv("EMAIL_USER")
    email_pass = os.getenv("EMAIL_PASS")

    if not email_user or not email_pass:
        return False, "Missing EMAIL_USER or EMAIL_PASS environment variables."

    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = email_user
        msg["To"] = to_email
        msg.set_content("This email contains HTML content. Please view it in an HTML-compatible email client.")
        msg.add_alternative(message_body, subtype="html")

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(email_user, email_pass)
            server.send_message(msg)

        return True, "Email sent successfully."
    except Exception as e:
        return False, str(e)

def log_to_google_sheet(name, email, phone, score, resume_name, status):
    """Log invitation activity to a Google Sheet webhook without interrupting the app."""
    webhook_url = get_webhook_url()
    if not webhook_url:
        return False, "Webhook URL not configured."

    payload = {
        "name": name,
        "candidate_name": name,
        "email": email,
        "phone": phone,
        "candidate_phone": phone,
        "score": score,
        "candidate_score": score,
        "resume": resume_name,
        "resume_name": resume_name,
        "resume_file_name": resume_name,
        "status": status,
    }

    try:
        response = requests.post(webhook_url, json=payload, headers={"Content-Type": "application/json"}, timeout=10)
        if response.ok:
            return True, "Data logged to Google Sheets."
        response = requests.post(webhook_url, data=payload, timeout=10)
        if response.ok:
            return True, "Data logged to Google Sheets."
        return False, f"Logging failed with status {response.status_code}."
    except Exception as e:
        return False, f"Logging failed: {e}"

def get_candidate_email_key(candidate):
    rank = candidate.get("Rank", "unknown")
    resume_name = candidate.get("Resume Name", "candidate")
    safe_resume_name = re.sub(r"[^a-zA-Z0-9_]+", "_", str(resume_name)).strip("_")
    return f"candidate_email_{rank}_{safe_resume_name}"

def get_candidate_name_key(candidate):
    rank = candidate.get("Rank", "unknown")
    resume_name = candidate.get("Resume Name", "candidate")
    safe_resume_name = re.sub(r"[^a-zA-Z0-9_]+", "_", str(resume_name)).strip("_")
    return f"candidate_name_{rank}_{safe_resume_name}"

def get_invitation_candidates(results_data):
    invitation_candidates = []
    for candidate in results_data:
        candidate_copy = dict(candidate)
        name_key = get_candidate_name_key(candidate_copy)
        email_key = get_candidate_email_key(candidate_copy)

        if name_key not in st.session_state:
            st.session_state[name_key] = (candidate_copy.get("Candidate Name") or "").strip()
        if email_key not in st.session_state:
            st.session_state[email_key] = (candidate_copy.get("Email") or "").strip()

        candidate_copy["Candidate Name"] = (st.session_state.get(name_key) or "").strip()
        candidate_copy["Name Key"] = name_key
        candidate_copy["Email"] = (st.session_state.get(email_key) or "").strip()
        candidate_copy["Email Key"] = email_key
        invitation_candidates.append(candidate_copy)
    return invitation_candidates

DEFAULT_EMAIL_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Interview Invitation</title>
</head>
<body style="margin:0; padding:0; font-family: Arial, sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" role="presentation">
    <tr>
      <td align="center">
        <table width="600" cellpadding="0" cellspacing="0" role="presentation">
          <tr>
            <td>
              <img src="https://maple.maplelearningsolutions.com/wp-content/uploads/2026/04/Purple-Gradient-Graphic-Designer-Email-Header.png" alt="Header Image" width="100%" style="display:block; border:0; outline:none; text-decoration:none;">
            </td>
          </tr>
          <tr>
            <td style="padding:20px; line-height:1.6;">
              <h2>Hi {name},</h2>
              <p>Congratulations! We're pleased to inform you that, based on your resume and profile, you have been shortlisted for the next stage of our hiring process.</p>
              <p>As a next step, we invite you to complete an assessment that will help us better understand your skills and suitability for the role. This assessment is an important part of our evaluation process.</p>
              <p><strong>Assessment Details:</strong></p>
              <ul>
                <li><strong>Access Link:</strong> <a href="{assessment_link}" style="text-decoration:none;">Assessment Details</a></li>
              </ul>
              <p>We recommend completing the assessment in one sitting in a quiet environment with a stable internet connection. Kindly ensure that you submit your responses within the given timeframe.</p>
              <p>If you have any questions or face any technical difficulties, feel free to reach out to us - we'll be happy to assist you.</p>
              <p>We wish you the very best and look forward to reviewing your submission.</p>
              <p>Best regards,<br>Hiring Team</p>
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>"""

# ==========================================
# PAGE VIEWS
# ==========================================

def render_dashboard():
    st.title("📊 ATS Overview & Analytics Dashboard")
    st.markdown("Monitor candidate screening pipelines, persistent database metrics, and recent matching activity.")

    stats = get_ats_dashboard_stats()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Candidates", stats["total_candidates"])
    with col2:
        st.metric("Active Candidates", stats["active_candidates"])
    with col3:
        st.metric("Screening Batches", stats["total_batches"])
    with col4:
        st.metric("Meets Threshold", stats["meets_threshold_count"])

    st.markdown("---")

    dash_col1, dash_col2 = st.columns([1, 1], gap="large")

    with dash_col1:
        st.subheader("🎯 Screening Outcomes Breakdown")
        total_screened = stats["total_screenings"]
        shortlisted = stats["meets_threshold_count"]
        below_threshold = total_screened - shortlisted

        if total_screened > 0:
            chart_df = pd.DataFrame({
                "Status": ["Meets Threshold", "Below Threshold"],
                "Candidates": [shortlisted, max(0, below_threshold)]
            })
            st.bar_chart(chart_df.set_index("Status"))
        else:
            st.info("No candidates screened yet. Upload your first batch under 'Bulk Resume Screening'.")

    with dash_col2:
        st.subheader("⚡ Recent Screening Activity")
        recent = stats["recent_activity"]
        if recent:
            for item in recent:
                status_color = "badge-success" if item["screening_status"] == "Meets Threshold" else "badge-warning"
                date_str = item["screened_at"][:16].replace("T", " ")
                st.markdown(f"""
                <div style="background: var(--secondary-background-color); padding: 12px 16px; border-radius: 8px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong>{item['full_name']}</strong> <span style="font-size:0.85rem; color:#6B7280;">({item['job_title']})</span><br>
                        <small style="color:#9CA3AF;">{date_str}</small>
                    </div>
                    <div style="text-align: right;">
                        <span style="font-weight: 700; font-size: 1.1rem; color: var(--primary-color);">{item['match_score']:.1f}%</span><br>
                        <span class="{status_color}">{item['screening_status']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Screening activity log will appear here after processing resumes.")


def render_bulk_screening():
    st.title(f"{ST_PAGE_ICON} Bulk Resume Screening & Matching")
    st.markdown("Screen up to **50 resumes** per batch using **Sentence-Transformers semantic similarity** matching against the Job Description.")
    st.divider()

    # Layout: JD and Resume Uploads
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 📄 Job Description & Requisition")
        job_title = st.text_input("Role / Job Title", value=st.session_state.get("active_job_title", "Software Engineer"), placeholder="e.g. Senior Python Developer")
        st.session_state.active_job_title = job_title

        jd_text = st.text_area(
            "Paste Job Description",
            height=200,
            value=st.session_state.get("active_jd_text", ""),
            placeholder="Key responsibilities, required skills, experience level..."
        )
        jd_file = st.file_uploader("Or upload JD Document (PDF/DOCX)", type=['pdf', 'docx'], key="jd_upload")
        if jd_file:
            with st.spinner("Extracting JD text..."):
                extracted_jd, jd_err = extract_resume_text(jd_file, jd_file.name)
                if extracted_jd and extracted_jd != st.session_state.get("active_jd_text"):
                    st.session_state.active_jd_text = extracted_jd
                    st.rerun()
                elif jd_err:
                    st.error(f"❌ {jd_err}")

    with col2:
        st.markdown("### 📂 Bulk Upload Resumes")
        resumes = st.file_uploader(
            "Drop resumes here (PDF or DOCX)",
            type=['pdf', 'docx'],
            accept_multiple_files=True,
            help="Supports approximately 50 resumes per batch"
        )
        if resumes:
            st.write(f"📊 **Total Resumes Uploaded:** {len(resumes)}")
            if len(resumes) > 60:
                st.warning("⚠️ Recommended limit is approximately 50 resumes per batch for optimal performance.")
            filenames = [f.name for f in resumes]
            st.caption(f"Files: {', '.join(filenames[:4])}" + (f" and {len(resumes)-4} more..." if len(resumes) > 4 else ""))

        threshold = st.slider("Matching Threshold (%)", min_value=0, max_value=100, value=50, step=1)

    st.markdown("<br>", unsafe_allow_html=True)

    # Action Buttons
    m_col1, m_col2, _ = st.columns([1, 1, 1])
    with m_col1:
        match_button = st.button("🚀 Process & Match Resumes", type="primary", use_container_width=True)
    with m_col2:
        if st.button("🔄 Clear Current Session", use_container_width=True):
            st.session_state.results_data = None
            st.session_state.failed_resumes = None
            st.session_state.last_batch_info = None
            st.rerun()

    # Batch Processing Trigger
    if match_button:
        if not jd_text or not jd_text.strip():
            st.error("Please provide or upload a Job Description.")
        elif not resumes:
            st.error("Please upload at least one resume file.")
        else:
            with st.spinner("🚀 Analyzing resumes with Sentence-Transformers matching..."):
                progress_bar = st.progress(0)
                status_placeholder = st.empty()

                def progress_cb(current, total, filename, status_label):
                    pct = min(1.0, current / max(1, total))
                    progress_bar.progress(pct)
                    status_placeholder.info(f"[{current}/{total}] {status_label}: {filename}")

                batch_info, successful, failed = run_batch_screening(
                    uploaded_files=resumes,
                    jd_text=jd_text,
                    job_title=job_title,
                    threshold=threshold,
                    progress_callback=progress_cb
                )

                st.session_state.results_data = successful
                st.session_state.failed_resumes = failed
                st.session_state.last_batch_info = batch_info

                status_placeholder.success(f"✅ Processing complete! {len(successful)} parsed successfully, {len(failed)} failed.")
                st.toast(f"Saved {len(successful)} candidates to persistent database!", icon="💾")
                st.rerun()

    # Failed Extractions Section
    if st.session_state.get("failed_resumes"):
        failed_list = st.session_state.failed_resumes
        st.divider()
        st.subheader(f"⚠️ Failed / Unsupported Documents ({len(failed_list)})")
        st.caption("Documents that could not be parsed as text (scanned image PDFs requiring OCR, corrupted files, unsupported formats).")
        
        failed_display = [{"Resume Name": f["Resume Name"], "Status": "Failed", "Reason": f["Reason"]} for f in failed_list]
        st.dataframe(pd.DataFrame(failed_display), use_container_width=True)

        failed_excel_bytes = generate_failed_resumes_report(failed_display)
        f_col1, _ = st.columns([1, 2])
        with f_col1:
            st.download_button(
                "📥 Download Failed Resumes Report",
                data=failed_excel_bytes,
                file_name=f"Failed_Resumes_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

    # Render Screening Results
    if st.session_state.get("results_data"):
        results = st.session_state.results_data
        st.divider()
        st.header("📊 Screening Results & Candidate Rankings")

        # Top 3 Candidates Showcase
        st.markdown("#### 🏆 Top Ranked Candidates")
        t_cols = st.columns(3)
        ranks = ["🥇 Rank 1", "🥈 Rank 2", "🥉 Rank 3"]
        rank_classes = ["rank-1", "rank-2", "rank-3"]

        for i in range(min(3, len(results))):
            cand = results[i]
            with t_cols[i]:
                st.markdown(f"""
                <div class="top-candidate-card {rank_classes[i]}">
                    <h4 style="margin:0; opacity: 0.8;">{ranks[i]}</h4>
                    <p style="font-weight:700; font-size:1.1rem; margin:8px 0 4px 0;">{cand['Candidate Name']}</p>
                    <p style="font-size:0.85rem; color:#6B7280; margin:0 0 6px 0;">{cand['Resume Name']} | {cand['Total Experience']}</p>
                    <h2 style="margin:0; color: var(--primary-color);">{cand['JD Match Score']:.1f}%</h2>
                    <p style="font-size:0.85rem; color:#374151; margin-top:8px;">Outcome: <strong>{cand['Screening Status']}</strong></p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Filter Controls
        st.markdown("#### 🔎 Filter & Search Candidates")
        f1, f2, f3 = st.columns(3)
        with f1:
            score_filter = st.slider("Min Match Score (%)", 0, 100, int(threshold), key="score_display_slider")
        with f2:
            status_filter = st.selectbox("Screening Outcome", ["All", "Meets Threshold", "Below Threshold"], index=0)
        with f3:
            search_text = st.text_input("Search Name / Skills", placeholder="e.g. Python, Alice")

        filtered_results = []
        for c in results:
            if c["JD Match Score"] < score_filter:
                continue
            if status_filter != "All" and c.get("Screening Status") != status_filter:
                continue
            if search_text:
                q = search_text.lower()
                name_match = q in c.get("Candidate Name", "").lower()
                skills_match = any(q in s.lower() for s in c.get("Skills", []))
                role_match = q in c.get("Role Applied For", "").lower()
                if not (name_match or skills_match or role_match):
                    continue
            filtered_results.append(c)

        st.caption(f"Showing **{len(filtered_results)}** of {len(results)} screened candidates")

        # Official 6-column Layout Table
        display_df = pd.DataFrame(filtered_results)[EXCEL_REPORT_COLUMNS]
        st.dataframe(
            display_df.style.format({"JD Match Score": "{:.1f}%"}).background_gradient(subset=["JD Match Score"], cmap="Blues"),
            use_container_width=True,
            hide_index=True
        )

        # Excel Export Section
        st.markdown("#### 📥 Download ATS Candidate Report")
        ex_col1, _ = st.columns([1, 1])
        with ex_col1:
            std_excel = generate_candidate_excel_report(filtered_results, default_role=job_title)
            st.download_button(
                label="📥 Download Candidate Screening Report (.xlsx)",
                data=std_excel,
                file_name=get_report_filename("ATS_Candidate_Screening"),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                use_container_width=True,
                help="Download official 6-column ATS report sorted by JD Match Score."
            )

        # Assessment Invitation & Webhook Section
        st.markdown("<br>", unsafe_allow_html=True)
        st.header("📧 Send Assessment Invitations")

        if st.session_state.get("logging_status_message"):
            if st.session_state.get("logging_status_type") == "success":
                st.success(st.session_state.logging_status_message)
            else:
                st.info(st.session_state.logging_status_message)

        invitation_threshold = st.slider("Minimum Score Threshold for Invitations", 0, 100, int(threshold), key="invite_threshold")
        assessment_link = st.text_input("Assessment Link", placeholder="https://assessment.maplelearningsolutions.com/test-id")
        email_subject = st.text_input("Subject", value="Assessment Invitation - Maple Learning Solutions")
        email_template = st.text_area("Message Template (HTML)", value=DEFAULT_EMAIL_TEMPLATE, height=200)

        invitation_candidates = get_invitation_candidates(results)
        invitation_df = pd.DataFrame(invitation_candidates)
        shortlisted_df = invitation_df[invitation_df["JD Match Score"] >= invitation_threshold].copy()
        valid_shortlisted_df = shortlisted_df[shortlisted_df["Email"].notna() & (shortlisted_df["Email"] != "")]
        bulk_disabled = valid_shortlisted_df.empty

        st.caption(f"📬 {len(valid_shortlisted_df)} eligible candidate(s) with valid email.")

        b_col1, b_col2 = st.columns([1, 2])
        with b_col1:
            if st.button("Send to All Shortlisted Candidates", disabled=bulk_disabled, use_container_width=True, key="bulk_send_button"):
                st.session_state.bulk_send_confirmed = True

        with b_col2:
            if st.session_state.get("bulk_send_confirmed"):
                st.info("Bulk send sends up to 20 emails per batch. Click Confirm to dispatch.")
                if st.button("Confirm Bulk Send", disabled=bulk_disabled, use_container_width=True, key="confirm_bulk_send_button"):
                    candidates_to_send = valid_shortlisted_df.head(20).to_dict("records")
                    sent_count = 0

                    with st.spinner("Dispatching emails..."):
                        for candidate in candidates_to_send:
                            cand_name = candidate.get("Candidate Name") or "Candidate"
                            personalized_msg = email_template.format(name=cand_name, assessment_link=assessment_link)
                            success, feedback = send_email(candidate["Email"], email_subject, personalized_msg)

                            status_str = "Sent" if success else "Failed"
                            if success:
                                sent_count += 1
                                st.success(f"✅ Sent to {candidate['Email']}")
                            else:
                                st.error(f"❌ Failed for {candidate['Email']}: {feedback}")

                            log_to_google_sheet(
                                cand_name, candidate["Email"], candidate.get("Contact Number") or "",
                                candidate["JD Match Score"], candidate["Resume Name"], status_str
                            )

                    st.info(f"{sent_count} emails dispatched.")
                    st.session_state.bulk_send_confirmed = False


def render_candidate_database():
    st.title("👥 Centralized Candidate Database")
    st.markdown("Search, retrieve, update, and manage persistent candidate profiles and resume records.")
    st.divider()

    s_col1, s_col2, s_col3 = st.columns([2, 1, 1])
    with s_col1:
        search_query = st.text_input("🔍 Search candidates by name, email, phone, or skills", placeholder="e.g. Python, Alice, john@example.com")
    with s_col2:
        status_filter = st.selectbox("Status", ["Active", "Archived", "All"], index=0)
    with s_col3:
        min_score_filter = st.number_input("Min Previous Score (%)", min_value=0, max_value=100, value=0, step=5)

    with st.expander("Advanced Filters", expanded=False):
        af1, af2, af3 = st.columns(3)
        with af1:
            skill_filter = st.text_input("Required Skill", placeholder="e.g. AWS, Docker, React")
        with af2:
            role_filter = st.text_input("Role / Job Title Screened For", placeholder="e.g. Developer")
        with af3:
            page_size = st.selectbox("Candidates Per Page", [10, 20, 50], index=0)

    candidates, total_found = search_candidates(
        search_query=search_query,
        skill_filter=skill_filter,
        role_filter=role_filter,
        min_score=min_score_filter if min_score_filter > 0 else None,
        status_filter=status_filter,
        limit=page_size or 10,
        offset=st.session_state.get("db_page_offset", 0)
    )

    st.markdown(f"**Found {total_found} candidate profile(s)**")

    if not candidates:
        st.info("No candidates found matching your criteria. Try uploading resumes via 'Bulk Resume Screening'.")
        return

    for cand in candidates:
        cand_id = cand["candidate_id"]
        status_badge = "badge-success" if cand["status"] == "Active" else "badge-warning"
        score_display = f"{cand['latest_score']:.1f}%" if cand.get("latest_score") is not None else "N/A"
        
        with st.container():
            c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
            with c1:
                st.markdown(f"**{cand['full_name']}** <span class='{status_badge}'>{cand['status']}</span>", unsafe_allow_html=True)
                st.caption(f"📧 {cand['email'] or 'Not Found'} | 📞 {cand['phone'] or 'Not Found'}")
            with c2:
                st.write(f"💼 **Exp:** {cand['total_experience']}")
                st.caption(f"🛠️ {cand['skills'][:40] + '...' if cand.get('skills') and len(cand['skills']) > 40 else (cand.get('skills') or 'None')}")
            with c3:
                st.write(f"🎯 **Latest Score:** {score_display}")
                st.caption(f"Role: {cand.get('latest_role') or 'General'}")
            with c4:
                btn_view = st.button("📄 View Profile", key=f"view_{cand_id}", use_container_width=True)

            if btn_view or st.session_state.get("selected_cand_id") == cand_id:
                st.session_state.selected_cand_id = cand_id
                details = get_candidate_details(cand_id)
                if details:
                    with st.expander(f"Candidate Profile: {details['full_name']}", expanded=True):
                        prof_tab1, prof_tab2, prof_tab3 = st.tabs(["Profile & Resumes", "Screening History", "Edit Details"])
                        
                        with prof_tab1:
                            p_col1, p_col2 = st.columns(2)
                            with p_col1:
                                st.write(f"**Candidate ID:** `{details['candidate_id']}`")
                                st.write(f"**Full Name:** {details['full_name']}")
                                st.write(f"**Email:** {details['email'] or 'Not Available'}")
                                st.write(f"**Phone:** {details['phone'] or 'Not Available'}")
                            with p_col2:
                                st.write(f"**Total Experience:** {details['total_experience']}")
                                st.write(f"**Skills:** {details['skills'] or 'None'}")
                                st.write(f"**Verified Record:** {'Yes (Manual HR Edit)' if details['is_verified'] else 'No (Auto-Extracted)'}")
                                st.write(f"**Created At:** {details['created_at']}")

                            st.markdown("##### 📁 Stored Resume Documents")
                            if details.get("resumes"):
                                for res in details["resumes"]:
                                    res_bytes = get_resume_bytes(res["stored_file_path"])
                                    r_col1, r_col2 = st.columns([3, 1])
                                    with r_col1:
                                        st.write(f"📄 **{res['original_filename']}** (Uploaded: {res['uploaded_at'][:10]})")
                                    with r_col2:
                                        if res_bytes:
                                            st.download_button(
                                                "Download Resume",
                                                data=res_bytes,
                                                file_name=res["original_filename"],
                                                key=f"dl_res_{res['resume_id']}",
                                                use_container_width=True
                                            )
                                        else:
                                            st.caption("File unavailable on disk")
                            else:
                                st.info("No stored resume documents found for this candidate.")

                        with prof_tab2:
                            screenings = details.get("screenings", [])
                            if screenings:
                                for s in screenings:
                                    st.markdown(f"**Role:** `{s['job_title']}` | **Match Score:** `{s['match_score']:.1f}%` | **Status:** `{s['screening_status']}` | **Date:** `{s['screened_at'][:10]}`")
                                    st.markdown("---")
                            else:
                                st.info("No historical screening records found.")

                        with prof_tab3:
                            st.markdown("##### ✏️ HR Manual Data Corrections")
                            with st.form(f"edit_form_{cand_id}"):
                                ed_name = st.text_input("Candidate Name", value=details["full_name"])
                                ed_email = st.text_input("Email", value=details["email"] or "")
                                ed_phone = st.text_input("Phone", value=details["phone"] or "")
                                ed_exp = st.text_input("Total Experience", value=details["total_experience"] or "")
                                ed_role = st.text_input("Current / Target Role", value=details.get("current_role") or "")
                                ed_skills = st.text_area("Skills", value=details.get("skills") or "")
                                ed_status = st.selectbox("Status", ["Active", "Archived", "Inactive"], index=["Active", "Archived", "Inactive"].index(details["status"]))

                                submit_edit = st.form_submit_button("Save Corrections & Mark Verified")
                                if submit_edit:
                                    update_candidate_details(cand_id, ed_name, ed_email, ed_phone, ed_exp, ed_role, ed_skills, ed_status)
                                    st.success("✅ Profile updated and verified!")
                                    st.rerun()

                            st.markdown("---")
                            d_col1, d_col2 = st.columns([3, 1])
                            with d_col1:
                                st.caption("Permanently delete candidate and remove their stored resumes from disk.")
                            with d_col2:
                                if st.button("🗑️ Delete Profile", key=f"del_{cand_id}", type="secondary", use_container_width=True):
                                    delete_candidate(cand_id)
                                    st.toast("Candidate profile deleted.", icon="🗑️")
                                    st.session_state.selected_cand_id = None
                                    st.rerun()

            st.divider()


def render_new_jd_matching():
    st.title("🎯 Search Existing Candidates for a New JD")
    st.markdown("Leverage your centralized candidate database. Enter a new hiring requisition and instantly recalculate semantic match scores against all active candidates in your pool.")
    st.divider()

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        new_title = st.text_input("New Requisition Job Title", placeholder="e.g. Senior Machine Learning Engineer")
        new_jd_text = st.text_area("Paste New Job Description", height=220, placeholder="Requirements, responsibilities, technical requirements...")
        new_jd_file = st.file_uploader("Or upload Job Description (PDF/DOCX)", type=['pdf', 'docx'], key="new_jd_upload")
        if new_jd_file:
            with st.spinner("Parsing JD file..."):
                txt, err = extract_resume_text(new_jd_file, new_jd_file.name)
                if txt:
                    new_jd_text = txt
                    st.success("✅ New JD text extracted!")
                elif err:
                    st.error(f"❌ {err}")

    with col2:
        st.markdown("### ⚙️ Screening Criteria")
        threshold = st.slider("New Match Threshold (%)", min_value=0, max_value=100, value=50, step=1, key="new_jd_threshold")
        save_db_toggle = st.checkbox("Save these screening scores to candidate history", value=True)

        stats = get_ats_dashboard_stats()
        st.info(f"💾 **Current Database Pool:** {stats['active_candidates']} active candidate profiles available for instant matching.")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Match Candidate Pool Against New JD", type="primary", use_container_width=True):
        if not new_title or not new_title.strip():
            st.error("Please provide a Job Title for the new requisition.")
        elif not new_jd_text or not new_jd_text.strip():
            st.error("Please provide the Job Description content.")
        else:
            with st.spinner("Evaluating candidate database against new JD..."):
                req, evaluated = rescreen_candidates_for_new_jd(
                    new_job_title=new_title,
                    new_jd_text=new_jd_text,
                    threshold=threshold,
                    save_to_database=save_db_toggle
                )

                st.session_state.new_jd_results = evaluated
                st.session_state.new_jd_role = new_title
                st.toast(f"Evaluated {len(evaluated)} candidates!", icon="🎯")
                st.rerun()

    if st.session_state.get("new_jd_results"):
        evaluated = st.session_state.new_jd_results
        role = st.session_state.get("new_jd_role", "Target Role")
        st.divider()
        st.subheader(f"📊 Ranked Candidates for: {role} ({len(evaluated)} Profiles)")

        r_col1, _ = st.columns([1, 1])
        with r_col1:
            excel_bytes = generate_candidate_excel_report(evaluated, default_role=role)
            st.download_button(
                "📥 Download Shortlisted Candidates (.xlsx)",
                data=excel_bytes,
                file_name=get_report_filename(f"ATS_Shortlist_{re.sub(r'[^a-zA-Z0-9]', '_', role)}"),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                use_container_width=True
            )

        res_df = pd.DataFrame(evaluated)[EXCEL_REPORT_COLUMNS]
        st.dataframe(
            res_df.style.format({"JD Match Score": "{:.1f}%"}).background_gradient(subset=["JD Match Score"], cmap="Blues"),
            use_container_width=True,
            hide_index=True
        )


def render_screening_history():
    st.title("📜 Screening History & Batch Reports")
    st.markdown("Review all previous screening batches, inspect historical scores, and re-download reports.")
    st.divider()

    batches = get_screening_batches_list(limit=50)
    if not batches:
        st.info("No historical batches found. Batches will appear here once you run screening under 'Bulk Resume Screening'.")
        return

    st.markdown(f"**Found {len(batches)} historical screening batches**")

    for batch in batches:
        batch_id = batch["batch_id"]
        created_date = batch["created_at"][:16].replace("T", " ")
        role_title = batch.get("job_title") or "Untitled Role"

        with st.expander(f"📦 Batch: {role_title} ({created_date}) — {batch['total_resumes']} Resumes", expanded=False):
            b_col1, b_col2, b_col3, b_col4 = st.columns(4)
            with b_col1:
                st.write(f"**Batch ID:** `{batch_id}`")
            with b_col2:
                st.write(f"**Total Resumes:** {batch['total_resumes']}")
            with b_col3:
                st.write(f"**Successful:** {batch['successful_count']}")
            with b_col4:
                st.write(f"**Failed / OCR:** {batch['failed_count']}")

            batch_candidates = get_batch_results(batch_id)
            if batch_candidates:
                st.markdown("##### Candidate Scores in this Batch")
                df_b = pd.DataFrame(batch_candidates)[EXCEL_REPORT_COLUMNS]
                st.dataframe(
                    df_b.style.format({"JD Match Score": "{:.1f}%"}),
                    use_container_width=True,
                    hide_index=True
                )

                h1, _ = st.columns([1, 1])
                with h1:
                    batch_excel = generate_candidate_excel_report(batch_candidates, default_role=role_title)
                    st.download_button(
                        "📥 Download Batch Report (.xlsx)",
                        data=batch_excel,
                        file_name=f"ATS_Batch_{batch_id}_{batch['created_at'][:10]}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"dl_std_{batch_id}",
                        use_container_width=True
                    )
            else:
                st.caption("No candidates recorded in this batch.")


# ==========================================
# MAIN APPLICATION CONTROLLER
# ==========================================

def main():
    st.set_page_config(page_title=ST_PAGE_TITLE, page_icon=ST_PAGE_ICON, layout="wide")
    apply_custom_style()

    # Preload SentenceTransformer model
    load_cached_transformer()

    # Sidebar Navigation
    with st.sidebar:
        st.title("🛡️ Maple ATS")
        st.caption("Enterprise Resume Screening System")
        st.markdown("---")

        nav_section = st.radio(
            "Navigation",
            [
                "📊 Dashboard",
                "⚡ Bulk Resume Screening",
                "👥 Candidate Database",
                "🎯 Search Candidates for a New JD",
                "📜 Screening History"
            ],
            index=1
        )

        st.markdown("---")
        st.markdown("**ML Engine:** `all-MiniLM-L6-v2`")
        st.markdown("**Matching:** Semantic Cosine Similarity")
        st.markdown("**Storage:** Persistent SQLite & Filesystem")
        st.caption("v3.0 - Clean & Fast Screening Engine")

    # Routing
    if nav_section == "📊 Dashboard":
        render_dashboard()
    elif nav_section == "⚡ Bulk Resume Screening":
        render_bulk_screening()
    elif nav_section == "👥 Candidate Database":
        render_candidate_database()
    elif nav_section == "🎯 Search Candidates for a New JD":
        render_new_jd_matching()
    elif nav_section == "📜 Screening History":
        render_screening_history()

if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is not None:
            main()
        else:
            from streamlit.web import cli as stcli
            sys.argv = ["streamlit", "run", os.path.abspath(__file__)]
            sys.exit(stcli.main())
    except Exception:
        main()
