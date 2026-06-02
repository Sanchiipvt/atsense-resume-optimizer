from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.main import (
    extract_text_from_file, preprocess_text, extract_skills,
    match_skills, generate_skill_suggestions, tfidf_similarity,
    final_ats_score, keyword_gap, keyword_suggestions,
    split_resume_sections, section_wise_scores, SKILLS
)

from backend.llm_optimizer import (
    generate_optimized_bullets,
    optimization_loop_summary
)

# 1. BRANDING & PAGE CONFIG
st.set_page_config(page_title="ATSense | AI Resume Optimizer", layout="wide")

# Safe CSS for Premium Font (Inter) that won't break the upload button
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        html, body, p, div, h1, h2, h3, h4, h5, h6, span {
            font-family: 'Inter', sans-serif !important;
        }
        
        div[data-testid="stMetricValue"] {
            color: #40ABD9 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Custom Title (Removed "Enterprise")
st.markdown("<h1 style='color: #40ABD9; margin-bottom: 0px;'>ATSense</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #A0B2C6; font-size: 1.2rem; margin-top: 0px;'>AI Resume Optimization</p>", unsafe_allow_html=True)
st.divider()

# --- UI Layout ---
c_left, c_right = st.columns([1, 1], gap="large")

with c_left:
    uploaded_resume = st.file_uploader("📂 Upload Resume (PDF/DOCX)", type=["pdf", "docx"])

with c_right:
    job_text = st.text_area("📝 Paste Job Description", height=150)

st.markdown("<br>", unsafe_allow_html=True)

# Main Action Button (AI Toggle Removed)
if st.button("Analyze Resume", use_container_width=True):
    if not uploaded_resume or not job_text.strip():
        st.warning("Please upload a resume and paste a job description to begin.")
    else:
        st.toast('File uploaded successfully. Analyzing data...')
        
        with st.spinner("Processing documents..."):
            # 1. Processing
            resume_text = extract_text_from_file(uploaded_resume)
            tfidf_score = tfidf_similarity(resume_text, job_text)

            resume_tokens = preprocess_text(resume_text)
            job_tokens = preprocess_text(job_text)

            resume_skills = extract_skills(resume_tokens, SKILLS)
            job_skills = extract_skills(job_tokens, SKILLS)

            skill_score, matched, missing = match_skills(resume_skills, job_skills)
            final_score = final_ats_score(skill_score, tfidf_score)

        # --- 2. Display Results (Top Banner) ---
        st.divider()
        st.markdown("<h2 style='text-align: center;'>ATS Analysis Results</h2>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Skill Match", f"{skill_score}%")
        c2.metric("TF-IDF Match", f"{tfidf_score}%")
        c3.metric("Final ATS Score", f"{final_score}%")
        
        st.markdown("<br>", unsafe_allow_html=True)

        # --- 3. Breakdown Sections (Tabbed Layout) ---
        tab1, tab2, tab3 = st.tabs(["Keyword Gaps", "Section Breakdown", "Quick Fixes"])
        
        with tab1:
            st.write("### Keyword & Skill Analysis")
            col_a, col_b = st.columns(2)
            with col_a:
                st.success("**Matched Skills:**\n\n" + (", ".join(sorted(matched)) if matched else "None"))
            with col_b:
                st.error("**Missing Skills:**\n\n" + (", ".join(sorted(missing)) if missing else "None"))
            
            st.markdown("<br>", unsafe_allow_html=True)
            gaps = keyword_suggestions(keyword_gap(resume_text, job_text))
            st.warning("**Additional Keyword Gaps:**\n\n" + (", ".join(gaps) if gaps else "None"))

        with tab2:
            st.write("### Section Scoring")
            sections = split_resume_sections(resume_text)
            sec_scores = section_wise_scores(sections, job_text)
            for sec, score in sec_scores.items():
                st.write(f"**{sec.capitalize()}**")
                st.progress(score / 100) 

        with tab3:
            st.write("### Immediate Improvement Suggestions")
            for s in generate_skill_suggestions(missing):
                st.info(f"• {s}")

        # ══════════════════════════════════════════════
        # AI OPTIMIZATION LOOP (Runs automatically now)
        # ══════════════════════════════════════════════
        st.divider()
        st.subheader("AI-Powered Optimization Engine")

        if not os.environ.get("GEMINI_API_KEY"):
            st.error(
                "GEMINI_API_KEY not found. "
                "Ensure it is set in your .env file or Streamlit secrets."
            )
        else:
            with st.spinner("Generating AI recommendations..."):
                diagnosis = optimization_loop_summary(
                    original_score=final_score,
                    skill_score=skill_score,
                    tfidf_score=tfidf_score,
                    missing_skills=list(missing),
                    matched_skills=list(matched)
                )

                st.info(f"**Score Diagnosis:** {diagnosis}")

                ai_result = generate_optimized_bullets(
                    resume_text=resume_text,
                    job_description=job_text,
                    missing_skills=list(missing),
                    keyword_gaps=gaps
                )

            st.subheader("Priority Skills to Address")
            priority = ai_result.get("priority_skills", [])
            if priority:
                cols = st.columns(len(priority))
                for i, skill in enumerate(priority):
                    cols[i].success(skill)

            st.subheader("Strategy Summary")
            st.write(ai_result.get("strategy_summary", ""))

            st.subheader("Suggested Rewritten Bullet Points")
            bullets = ai_result.get("rewritten_bullets", [])

            if bullets:
                for i, bullet in enumerate(bullets, 1):
                    with st.expander(
                        f"Bullet {i} — {bullet.get('section', 'Resume')} "
                        f"| Targets: {bullet.get('skill_addressed', '')}"
                    ):
                        st.caption(
                            f"**Replace a bullet like:** "
                            f"{bullet.get('original_hint', '')}"
                        )
                        st.success(
                            f"**Suggested Rewrite:**\n\n"
                            f"{bullet.get('rewritten', '')}"
                        )
            else:
                st.success(
                    "Your resume is already well-optimized for this job description."
                )

            st.divider()
            st.info(
                "**Optimization Loop:** Apply these suggestions to your "
                "resume, re-upload the document, and run the analysis again "
                "to measure your score improvement."
            )
