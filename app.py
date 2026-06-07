"""
ATSense — AI Resume Optimizer  |  app.py
=========================================
Streamlit UI layer integrated with modern frontend styling.
"""

from dotenv import load_dotenv
load_dotenv()

import os
import sys
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.main import (
    # Text extraction
    extract_text_from_file,
    # Pre-processing
    preprocess_text, extract_skills, SKILLS,
    # Core ML pipeline
    compute_similarity_score,
    weighted_match_skills,
    final_ats_score,
    # Analysis
    keyword_gap, keyword_suggestions,
    split_resume_sections, section_wise_scores,
    generate_skill_suggestions,
    # New: stuffing detection
    detect_keyword_stuffing,
)

from backend.llm_optimizer import (
    generate_optimized_bullets,
    optimization_loop_summary,
)

# ──────────────────────────────────────────────────────────────────────────────
# 1. PAGE CONFIG & BRANDING
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ATSense | AI Resume Optimizer",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Combine new layout CSS with your existing badge/metric CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Global Styles */
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif !important;
    }
    .reportview-container, .main {
        background: linear-gradient(180deg, #f5f8ff 0%, #ffffff 100%);
    }

    /* Your Custom Result Styles */
    div[data-testid="stMetricValue"] { color: #8b5cf6 !important; } /* Updated to purple from new theme */
    div[data-testid="stExpander"] details summary {
        font-size: 0.9rem;
        font-weight: 500;
    }
    .badge-critical {
        background: #fde8e8; color: #c0392b;
        padding: 2px 9px; border-radius: 4px;
        font-size: 0.75rem; font-weight: 600;
    }
    .badge-warn {
        background: #fef9e7; color: #b7770d;
        padding: 2px 9px; border-radius: 4px;
        font-size: 0.75rem; font-weight: 600;
    }
    .badge-ok {
        background: #e8f8f5; color: #1e8449;
        padding: 2px 9px; border-radius: 4px;
        font-size: 0.75rem; font-weight: 600;
    }
    
    /* Navigation Bar */
    .nav-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1.5rem 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    .nav-logo {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 1.25rem;
        font-weight: 700;
        color: #1e293b;
    }
    .nav-logo-icon {
        background: #a855f7;
        color: white;
        padding: 6px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .nav-links {
        display: flex;
        gap: 2rem;
        font-size: 0.875rem;
        font-weight: 500;
        color: #64748b;
    }

    /* Hero Section */
    .hero-container {
        text-align: center;
        max-width: 800px;
        margin: 3rem auto 2rem auto;
        padding: 0 1rem;
    }
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: white;
        border: 1px solid #e2e8f0;
        padding: 0.375rem 1rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        color: #475569;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
    }
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        line-height: 1.15;
        color: #1e293b;
        margin-bottom: 1.5rem;
    }
    .hero-gradient {
        background: linear-gradient(90deg, #3b82f6, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.125rem;
        color: #64748b;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* Feature Steps Cards */
    .steps-container {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1.5rem;
        max-width: 1200px;
        margin: 4rem auto 2rem auto;
        padding: 0 2rem;
    }
    .step-card {
        background: white;
        border: 1px solid #f1f5f9;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    .step-num { font-size: 0.875rem; font-weight: 700; margin-bottom: 0.5rem; display: block; }
    .step-1 { color: #3b82f6; }
    .step-2 { color: #a855f7; }
    .step-3 { color: #3b82f6; }
    .step-title { font-size: 1.1rem; font-weight: 700; color: #1e293b; margin-bottom: 0.5rem; }
    .step-desc { font-size: 0.875rem; color: #64748b; line-height: 1.5; }

    .custom-footer { text-align: center; padding: 3rem 0; font-size: 0.875rem; color: #94a3b8; }
    </style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# 2. HEADER & HERO
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
    <div class="nav-container">
        <div class="nav-logo">
            <div class="nav-logo-icon">
                <svg width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
            </div>
            <span>Atsense</span>
        </div>
        <div class="nav-links">
            <a href="#analyzer" style="text-decoration:none;">Analyzer</a>
        </div>
    </div>
    
    <div class="hero-container">
        <div class="badge">
            <span style="color: #a855f7;">✨</span> Powered by TF-IDF & cosine similarity
        </div>
        <h1 class="hero-title">
            Screen resumes like a<br>
            <span class="hero-gradient">human, at machine speed.</span>
        </h1>
        <p class="hero-subtitle">
            Upload a resume and paste a job description to instantly calculate the match score, surface matched skills, and spot the keywords you're missing.
        </p>
    </div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# 3. INPUT SECTION
# ──────────────────────────────────────────────────────────────────────────────

st.markdown('<div id="analyzer" style="max-width: 1200px; margin: 0 auto; padding: 0 2rem;">', unsafe_allow_html=True)

col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.markdown('<p style="font-weight: 600; color: #1e293b; margin-bottom: 0.5rem;">Resume</p>', unsafe_allow_html=True)
    uploaded_resume = st.file_uploader(
        "Upload Resume (PDF or DOCX)",
        type=["pdf", "docx"],
        label_visibility="collapsed",
        help="Digital (text-based) PDFs give the best results. Scanned PDFs trigger OCR fallback with reduced accuracy."
    )
    st.markdown('<p style="text-align: center; font-size: 0.75rem; color: #94a3b8; margin-top: -0.5rem;">PDF or DOCX • up to 10 MB</p>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div style="display: flex; justify-content: space-between;"><p style="font-weight: 600; color: #1e293b; margin-bottom: 0.5rem;">Job description</p></div>', unsafe_allow_html=True)
    job_text = st.text_area(
        "Paste Job Description",
        height=225,
        placeholder="e.g. We're hiring a Senior Frontend Engineer with strong experience in React, TypeScript...",
        label_visibility="collapsed",
        help="The more complete the JD, the more accurate the keyword importance scoring."
    )

st.markdown('</div>', unsafe_allow_html=True)

# Controls beneath inputs
st.markdown("<br>", unsafe_allow_html=True)
c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    use_ai = st.toggle(
        "Enable AI-Powered Bullet Rewriter (requires GEMINI_API_KEY)",
        value=False,
        help="Uses Google Gemini to generate contextual bullet-point rewrites targeting your specific missing skills."
    )
    analyze_btn = st.button("✨ Analyze Resume", use_container_width=True, type="primary")


# ──────────────────────────────────────────────────────────────────────────────
# 4. PIPELINE AND RESULTS
# ──────────────────────────────────────────────────────────────────────────────

if analyze_btn:
    if not uploaded_resume or not job_text.strip():
        st.warning("Please upload a resume and paste a job description to continue.")
        st.stop()

    st.toast("File received — running analysis pipeline...")

    # ── Step 1: Text Extraction ───────────────────────────────────────────────
    with st.spinner("Extracting text from document..."):
        extraction = extract_text_from_file(uploaded_resume)

    if not extraction.ok:
        st.error(f"**Extraction failed:** {extraction.error}")
        st.stop()

    if extraction.warning:
        st.warning(f"**Extraction notice:** {extraction.warning}")

    if extraction.method == "ocr":
        st.info("OCR mode active — text was extracted from a scanned PDF. Scores may be slightly lower.", icon="🔬")

    resume_text = extraction.text

    # ── Step 2: Core ML Pipeline ──────────────────────────────────────────────
    with st.spinner("Running semantic analysis and skill matching..."):
        sim_result     = compute_similarity_score(resume_text, job_text)
        similarity_score = sim_result["score"]
        sim_method       = sim_result["method"]

        resume_tokens  = preprocess_text(resume_text)
        job_tokens     = preprocess_text(job_text)
        resume_skills  = extract_skills(resume_tokens, SKILLS)
        job_skills     = extract_skills(job_tokens, SKILLS)

        skill_result   = weighted_match_skills(resume_skills, job_skills, job_text)
        skill_score    = skill_result["weighted_score"]
        matched        = skill_result["matched"]
        missing        = skill_result["missing"]
        dealbreakers   = skill_result["dealbreakers"]
        skill_weights  = skill_result["weights"]

        gaps           = keyword_suggestions(keyword_gap(resume_text, job_text))

        stuffing       = detect_keyword_stuffing(resume_text, job_text)
        stuffing_penalty = (30.0 if stuffing["severity"] == "confirmed" else 15.0 if stuffing["severity"] == "suspected" else 0.0)

        final_score = final_ats_score(skill_score, similarity_score, stuffing_penalty)
        sections     = split_resume_sections(resume_text)
        sec_scores   = section_wise_scores(sections, job_text)

    # ──────────────────────────────────────────────────────────────────────────
    # RESULTS RENDERING
    # ──────────────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("<h2 style='text-align:center; color:#1e293b;'>Analysis Results</h2>", unsafe_allow_html=True)
    
    method_label = "🧠 Semantic (bi-encoder)" if sim_method == "semantic" else "📊 TF-IDF + synonyms"
    st.caption(f"<div style='text-align:center;'>Similarity engine: **{method_label}**</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    score_color = "#27ae60" if final_score >= 75 else "#e67e22" if final_score >= 50 else "#e74c3c"
    
    m1.markdown(
        f"<div style='text-align:center; background:white; padding:1rem; border-radius:1rem; box-shadow:0 1px 3px rgba(0,0,0,0.02); border:1px solid #f1f5f9;'>"
        f"<div style='font-size:0.85rem; color:#64748b; margin-bottom:4px; font-weight:600;'>Final ATS Score</div>"
        f"<div style='font-size:2.8rem; font-weight:800; color:{score_color};'>{final_score}%</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    m2.metric("Weighted Skill Match", f"{skill_score}%", help="Skills weighted by JD importance signals (required > preferred)")
    m3.metric("Semantic Similarity", f"{similarity_score}%", help=f"Method: {method_label}")
    m4.metric("Skills Detected", f"{len(matched)}/{len(job_skills)} matched", delta=f"-{len(missing)} missing" if missing else "✓ All matched", delta_color="inverse")

    st.markdown("<br>", unsafe_allow_html=True)

    # Warnings
    if stuffing["is_stuffed"]:
        severity_label = stuffing["severity"].upper()
        penalty_msg = f"  Score penalty applied: **-{stuffing_penalty:.0f} points**."
        if stuffing["severity"] == "confirmed":
            st.error(f"🚨 **Keyword stuffing {severity_label}** — {stuffing['global_overlap']*100:.0f}% of JD vocabulary found verbatim in your resume (threshold: 72%).{penalty_msg} Real ATS parsers flag this as fraudulent and auto-reject. Use paraphrasing and action verbs instead of copying JD text.")
        else:
            st.warning(f"⚠️ **Stuffing {severity_label}** — unusually high JD token overlap ({stuffing['global_overlap']*100:.0f}%). Affected sections: **{', '.join(stuffing['stuffed_sections']) or 'global'}**.{penalty_msg}")

    if dealbreakers:
        db_list = ", ".join(f"`{s}`" for s in sorted(dealbreakers))
        st.error(f"🔴 **Dealbreaker alert:** {db_list} — flagged as non-negotiable requirements in this JD. Missing these skills may trigger automatic rejection regardless of overall score. Prioritise addressing these before applying.")

    if sim_method != "semantic":
        st.info("ℹ️ Running in TF-IDF + synonym mode. For full semantic scoring, install `sentence-transformers` and ensure the `all-MiniLM-L6-v2` model is accessible.", icon="💡")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Skill Analysis", "📊 Section Breakdown", "🔑 Keyword Gaps", "⚡ Quick Fixes"])

    with tab1:
        st.markdown("#### Skill Match Detail")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**✅ Matched Skills**")
            if matched:
                sorted_matched = sorted(matched, key=lambda s: skill_weights.get(s, 1.0), reverse=True)
                for skill in sorted_matched:
                    w = skill_weights.get(skill, 1.0)
                    badge = '<span class="badge-ok">required</span>' if w >= 2.5 else '<span class="badge-warn">preferred</span>' if w < 0.8 else ""
                    st.markdown(f"• {skill} {badge}", unsafe_allow_html=True)
            else:
                st.markdown("_No skills matched._")
        with col_b:
            st.markdown("**❌ Missing Skills**")
            if missing:
                sorted_missing = sorted(missing, key=lambda s: skill_weights.get(s, 1.0), reverse=True)
                for skill in sorted_missing:
                    w = skill_weights.get(skill, 1.0)
                    badge = '<span class="badge-critical">dealbreaker</span>' if w >= 2.5 else '<span class="badge-warn">important</span>' if w >= 1.5 else ""
                    st.markdown(f"• {skill} {badge}", unsafe_allow_html=True)
            else:
                st.success("All detected JD skills are present in your resume! 🎉")

        if sim_method == "semantic" and sim_result.get("semantic_score") is not None:
            st.markdown("<br>#### Similarity Score Breakdown", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Semantic Score", f"{sim_result['semantic_score']}%", help="Bi-encoder sentence embedding cosine similarity (max-pooled)")
            c2.metric("TF-IDF Score",   f"{sim_result['tfidf_score']}%", help="Synonym-augmented TF-IDF cosine similarity")
            c3.metric("Blended Score",  f"{sim_result['score']}%", help="70% semantic + 30% TF-IDF — rewards both meaning and exact keywords")

        if stuffing["is_stuffed"]:
            with st.expander("🔍 Stuffing Analysis Detail"):
                st.markdown(f"**Global overlap ratio:** {stuffing['global_overlap']*100:.1f}%\n\n**Severity:** {stuffing['severity'].title()}")
                if stuffing["stuffed_sections"]:
                    st.markdown(f"**Flagged sections:** {', '.join(stuffing['stuffed_sections'])}")
                st.markdown("**How to fix:** Replace verbatim JD phrases with action-verb bullet points that demonstrate how you applied each skill.")

    with tab2:
        st.markdown("#### Per-Section Relevance Scores")
        st.caption("Each section of your resume is scored independently against the JD. Low-scoring sections are where you should add JD-relevant content.")
        for sec_name, sec_score in sorted(sec_scores.items(), key=lambda x: x[1], reverse=True):
            sec_display = sec_name.replace("_", " ").title()
            col_label, col_bar = st.columns([1, 3])
            with col_label:
                colour = "#27ae60" if sec_score >= 60 else "#e67e22" if sec_score >= 30 else "#e74c3c"
                st.markdown(f"<span style='color:{colour}; font-weight:500;'>{sec_display}</span> &nbsp; <span style='color:#64748b; font-size:0.85rem;'>{sec_score:.1f}%</span>", unsafe_allow_html=True)
            with col_bar:
                st.progress(min(sec_score / 100, 1.0))
        st.info("💡 **Tip:** If your Experience section scores below 40%, rewrite your bullet points to use the same terminology as the JD while preserving your quantified achievements.", icon="💡")

    with tab3:
        st.markdown("#### High-Value Keywords Missing from Resume")
        st.caption("Ranked by TF-IDF weight in the JD — top entries are the most distinctive terms the recruiter and ATS are scanning for.")
        if gaps:
            pills_html = " ".join(f'<span style="background:#f3e8ff; color:#7e22ce; padding:3px 10px; border-radius:12px; font-size:0.85rem; margin:3px; display:inline-block; border:1px solid #e9d5ff;">{g}</span>' for g in gaps)
            st.markdown(pills_html, unsafe_allow_html=True)
        else:
            st.success("No significant keyword gaps detected.")

    with tab4:
        st.markdown("#### Immediate Improvement Actions")
        for suggestion in generate_skill_suggestions(missing):
            st.info(f"• {suggestion}")
        if dealbreakers:
            st.markdown("---\n#### 🔴 Dealbreaker Remediation (Priority 1)")
            for db_skill in sorted(dealbreakers):
                st.error(f"**{db_skill.title()}** — flagged as required/non-negotiable. If you have this experience, make it explicit in your Skills section and add a bullet point in Experience that demonstrates applied use.")

    # AI Optimizer
    if use_ai:
        st.divider()
        st.subheader("🤖 AI-Powered Bullet Rewriter")
        if not os.environ.get("GEMINI_API_KEY"):
            st.error("GEMINI_API_KEY not found in environment. Add it to your `.env` file: `GEMINI_API_KEY=your_key_here`")
        else:
            with st.spinner("Generating AI-powered rewrite recommendations..."):
                diagnosis = optimization_loop_summary(
                    original_score=final_score, skill_score=skill_score,
                    tfidf_score=sim_result["tfidf_score"], missing_skills=list(missing), matched_skills=list(matched)
                )
                ai_result = generate_optimized_bullets(
                    resume_text=resume_text, job_description=job_text,
                    missing_skills=list(missing), keyword_gaps=gaps
                )
            st.info(f"**Score Diagnosis:** {diagnosis}")

            priority = ai_result.get("priority_skills", [])
            if priority:
                st.markdown("**Priority Skills to Address:**")
                cols = st.columns(min(len(priority), 4))
                for i, skill in enumerate(priority[:4]):
                    cols[i].success(skill)

            strategy = ai_result.get("strategy_summary", "")
            if strategy:
                st.markdown("**Strategy Summary:**")
                st.write(strategy)

            bullets = ai_result.get("rewritten_bullets", [])
            if bullets:
                st.markdown("**Suggested Bullet Rewrites:**")
                for i, bullet in enumerate(bullets, 1):
                    section  = bullet.get("section", "Resume")
                    skill_addressed = bullet.get("skill_addressed", "")
                    original = bullet.get("original_hint", "")
                    rewrite  = bullet.get("rewritten", "")

                    with st.expander(f"Bullet {i} — {section}  |  targets: {skill_addressed}"):
                        if original:
                            st.markdown("**Original (weak version):**")
                            st.markdown(f"<div style='background:#fde8e8; padding:10px; border-radius:6px; color:#7b241c;'>{original}</div><br>", unsafe_allow_html=True)
                        st.markdown("**Rewritten (ATS-optimised):**")
                        st.markdown(f"<div style='background:#e8f8f5; padding:10px; border-radius:6px; color:#1e8449;'>{rewrite}</div>", unsafe_allow_html=True)
            else:
                st.success("Your resume is already well-optimised for this JD — no major rewrites suggested.")

            st.divider()
            st.info("**Optimization Loop:** Apply the suggestions above, re-upload your resume, and re-run the analysis to measure score improvement. Target: ≥75% final score before submitting.")

else:
    # ──────────────────────────────────────────────────────────────────────────
    # 5. DEFAULT STATE: HOW IT WORKS (Hides when results appear)
    # ──────────────────────────────────────────────────────────────────────────
    st.markdown("""
        <div id="how-it-works" class="steps-container">
            <div class="step-card">
                <span class="step-num step-1">01</span>
                <h4 class="step-title">Upload a resume</h4>
                <p class="step-desc">Drop in a PDF or DOCX. We parse it locally for a fast first pass.</p>
            </div>
            <div class="step-card">
                <span class="step-num step-2">02</span>
                <h4 class="step-title">Paste the job</h4>
                <p class="step-desc">Bring any job description. We extract the signal terms automatically.</p>
            </div>
            <div class="step-card">
                <span class="step-num step-3">03</span>
                <h4 class="step-title">Get a clear score</h4>
                <p class="step-desc">See match percentage, matched skills, and missing keywords in seconds.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# 6. FOOTER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
    <div class="custom-footer">
        © 2026 Atsense · Crafted for modern hiring teams
    </div>
""", unsafe_allow_html=True)
