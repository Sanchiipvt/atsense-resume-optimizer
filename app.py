"""
ATSense — AI Resume Optimizer  |  app.py
=========================================
Streamlit UI layer. Intentionally thin: all ML logic lives in backend/main.py.
This module is responsible for rendering, layout, and user feedback only.
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
# PAGE CONFIG & BRANDING
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ATSense | AI Resume Optimizer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif !important;
    }

    /* Brand colour on metric values */
    div[data-testid="stMetricValue"] { color: #40ABD9 !important; }

    /* Tighten up expander padding */
    div[data-testid="stExpander"] details summary {
        font-size: 0.9rem;
        font-weight: 500;
    }

    /* Severity badge helpers — rendered via st.markdown unsafe_allow_html */
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
    .score-ring {
        font-size: 3rem; font-weight: 700;
        color: #40ABD9; text-align: center;
        line-height: 1.1;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    "<h1 style='color:#40ABD9; margin-bottom:0;'>ATSense</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#A0B2C6; font-size:1.15rem; margin-top:0;'>"
    "Enterprise AI Resume Optimization</p>",
    unsafe_allow_html=True,
)
st.divider()


# ──────────────────────────────────────────────────────────────────────────────
# INPUT SECTION
# ──────────────────────────────────────────────────────────────────────────────

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    uploaded_resume = st.file_uploader(
        "Upload Resume (PDF or DOCX)",
        type=["pdf", "docx"],
        help="Digital (text-based) PDFs give the best results. "
             "Scanned PDFs trigger OCR fallback with reduced accuracy.",
    )

with col_right:
    job_text = st.text_area(
        "Paste Job Description",
        height=165,
        placeholder="Paste the full job description here...",
        help="The more complete the JD, the more accurate the keyword importance scoring.",
    )

st.markdown("<br>", unsafe_allow_html=True)

use_ai = st.toggle(
    "Enable AI-Powered Bullet Rewriter  (requires GEMINI_API_KEY)",
    value=False,
    help="Uses Google Gemini to generate contextual bullet-point rewrites "
         "targeting your specific missing skills.",
)

analyze_btn = st.button("Analyze Resume", use_container_width=True, type="primary")


# ──────────────────────────────────────────────────────────────────────────────
# ANALYSIS PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

if analyze_btn:
    if not uploaded_resume or not job_text.strip():
        st.warning("Please upload a resume and paste a job description to continue.")
        st.stop()

    st.toast("File received — running analysis pipeline...")

    # ── Step 1: Text Extraction ───────────────────────────────────────────────
    with st.spinner("Extracting text from document..."):
        extraction = extract_text_from_file(uploaded_resume)

    # Surface extraction warnings/errors immediately — before any ML work
    if not extraction.ok:
        st.error(f"**Extraction failed:** {extraction.error}")
        st.stop()

    if extraction.warning:
        st.warning(f"**Extraction notice:** {extraction.warning}")

    if extraction.method == "ocr":
        st.info(
            "OCR mode active — text was extracted from a scanned PDF. "
            "Scores may be slightly lower than with a digital resume.",
            icon="🔬",
        )

    resume_text = extraction.text

    # ── Step 2: Core ML Pipeline ──────────────────────────────────────────────
    with st.spinner("Running semantic analysis and skill matching..."):

        # Similarity scoring (semantic → TF-IDF fallback)
        sim_result     = compute_similarity_score(resume_text, job_text)
        similarity_score = sim_result["score"]
        sim_method       = sim_result["method"]

        # Tokenise and extract skills
        resume_tokens  = preprocess_text(resume_text)
        job_tokens     = preprocess_text(job_text)
        resume_skills  = extract_skills(resume_tokens, SKILLS)
        job_skills     = extract_skills(job_tokens, SKILLS)

        # Weighted skill matching with synonym expansion
        skill_result   = weighted_match_skills(resume_skills, job_skills, job_text)
        skill_score    = skill_result["weighted_score"]
        matched        = skill_result["matched"]
        missing        = skill_result["missing"]
        dealbreakers   = skill_result["dealbreakers"]
        skill_weights  = skill_result["weights"]

        # Keyword gap analysis
        gaps           = keyword_suggestions(keyword_gap(resume_text, job_text))

        # Stuffing detection
        stuffing       = detect_keyword_stuffing(resume_text, job_text)
        stuffing_penalty = (
            30.0 if stuffing["severity"] == "confirmed"
            else 15.0 if stuffing["severity"] == "suspected"
            else 0.0
        )

        # Final composite score
        final_score = final_ats_score(skill_score, similarity_score, stuffing_penalty)

        # Section breakdown
        sections     = split_resume_sections(resume_text)
        sec_scores   = section_wise_scores(sections, job_text)

    # ──────────────────────────────────────────────────────────────────────────
    # RESULTS — TOP BANNER
    # ──────────────────────────────────────────────────────────────────────────

    st.divider()
    st.markdown(
        "<h2 style='text-align:center; color:#40ABD9;'>Analysis Results</h2>",
        unsafe_allow_html=True,
    )

    # Similarity method badge
    method_label = (
        "🧠 Semantic (bi-encoder)" if sim_method == "semantic"
        else "📊 TF-IDF + synonyms"
    )
    st.caption(f"Similarity engine: **{method_label}**")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Score ring + key metrics ───────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)

    # Color-code the final score
    score_color = (
        "#27ae60" if final_score >= 75
        else "#e67e22" if final_score >= 50
        else "#e74c3c"
    )
    m1.markdown(
        f"<div style='text-align:center'>"
        f"<div style='font-size:0.85rem; color:#A0B2C6; margin-bottom:4px;'>Final ATS Score</div>"
        f"<div style='font-size:2.8rem; font-weight:700; color:{score_color};'>{final_score}%</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    m2.metric("Weighted Skill Match", f"{skill_score}%",
              help="Skills weighted by JD importance signals (required > preferred)")
    m3.metric("Semantic Similarity", f"{similarity_score}%",
              help=f"Method: {method_label}")
    m4.metric(
        "Skills Detected",
        f"{len(matched)}/{len(job_skills)} matched",
        delta=f"-{len(missing)} missing" if missing else "✓ All matched",
        delta_color="inverse",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Stuffing warning banner ────────────────────────────────────────────────
    if stuffing["is_stuffed"]:
        severity_label = stuffing["severity"].upper()
        penalty_msg = f"  Score penalty applied: **-{stuffing_penalty:.0f} points**."
        if stuffing["severity"] == "confirmed":
            st.error(
                f"🚨 **Keyword stuffing {severity_label}** — "
                f"{stuffing['global_overlap']*100:.0f}% of JD vocabulary found verbatim "
                f"in your resume (threshold: 72%).{penalty_msg} "
                "Real ATS parsers flag this as fraudulent and auto-reject. "
                "Use paraphrasing and action verbs instead of copying JD text."
            )
        else:
            st.warning(
                f"⚠️ **Stuffing {severity_label}** — unusually high JD token overlap "
                f"({stuffing['global_overlap']*100:.0f}%). "
                f"Affected sections: **{', '.join(stuffing['stuffed_sections']) or 'global'}**."
                f"{penalty_msg}"
            )

    # ── Dealbreaker warning ────────────────────────────────────────────────────
    if dealbreakers:
        db_list = ", ".join(f"`{s}`" for s in sorted(dealbreakers))
        st.error(
            f"🔴 **Dealbreaker alert:** {db_list} — "
            "flagged as non-negotiable requirements in this JD. "
            "Missing these skills may trigger automatic rejection regardless of overall score. "
            "Prioritise addressing these before applying."
        )

    # ── If semantic model is unavailable, explain why ─────────────────────────
    if sim_method != "semantic":
        st.info(
            "ℹ️ Running in TF-IDF + synonym mode. "
            "For full semantic scoring, install `sentence-transformers` and ensure "
            "the `all-MiniLM-L6-v2` model is accessible.",
            icon="💡",
        )

    # ──────────────────────────────────────────────────────────────────────────
    # TABBED BREAKDOWN
    # ──────────────────────────────────────────────────────────────────────────

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Skill Analysis",
        "📊 Section Breakdown",
        "🔑 Keyword Gaps",
        "⚡ Quick Fixes",
    ])

    # ── Tab 1: Skill Analysis ─────────────────────────────────────────────────
    with tab1:
        st.markdown("#### Skill Match Detail")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**✅ Matched Skills**")
            if matched:
                # Sort matched skills by JD weight descending so the most
                # important ones appear first — gives the user immediate signal
                sorted_matched = sorted(
                    matched,
                    key=lambda s: skill_weights.get(s, 1.0),
                    reverse=True,
                )
                for skill in sorted_matched:
                    w = skill_weights.get(skill, 1.0)
                    badge = (
                        '<span class="badge-ok">required</span>' if w >= 2.5
                        else '<span class="badge-warn">preferred</span>' if w < 0.8
                        else ""
                    )
                    st.markdown(f"• {skill} {badge}", unsafe_allow_html=True)
            else:
                st.markdown("_No skills matched._")

        with col_b:
            st.markdown("**❌ Missing Skills**")
            if missing:
                # Sort missing skills by weight descending — dealbreakers first
                sorted_missing = sorted(
                    missing,
                    key=lambda s: skill_weights.get(s, 1.0),
                    reverse=True,
                )
                for skill in sorted_missing:
                    w = skill_weights.get(skill, 1.0)
                    if w >= 2.5:
                        badge = '<span class="badge-critical">dealbreaker</span>'
                    elif w >= 1.5:
                        badge = '<span class="badge-warn">important</span>'
                    else:
                        badge = ""
                    st.markdown(f"• {skill} {badge}", unsafe_allow_html=True)
            else:
                st.success("All detected JD skills are present in your resume! 🎉")

        # Semantic score breakdown (only shown when semantic model was used)
        if sim_method == "semantic" and sim_result.get("semantic_score") is not None:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### Similarity Score Breakdown")
            c1, c2, c3 = st.columns(3)
            c1.metric("Semantic Score", f"{sim_result['semantic_score']}%",
                      help="Bi-encoder sentence embedding cosine similarity (max-pooled)")
            c2.metric("TF-IDF Score",   f"{sim_result['tfidf_score']}%",
                      help="Synonym-augmented TF-IDF cosine similarity")
            c3.metric("Blended Score",  f"{sim_result['score']}%",
                      help="70% semantic + 30% TF-IDF — rewards both meaning and exact keywords")

        # Stuffing detail expander
        if stuffing["is_stuffed"]:
            with st.expander("🔍 Stuffing Analysis Detail"):
                st.markdown(f"**Global overlap ratio:** {stuffing['global_overlap']*100:.1f}%")
                st.markdown(f"**Severity:** {stuffing['severity'].title()}")
                if stuffing["stuffed_sections"]:
                    st.markdown(
                        f"**Flagged sections:** {', '.join(stuffing['stuffed_sections'])}"
                    )
                st.markdown(
                    "**How to fix:** Replace verbatim JD phrases with action-verb bullet points "
                    "that demonstrate how you applied each skill. Example: instead of listing "
                    "'kubernetes container orchestration', write "
                    "'Deployed 12-service ML inference stack on Kubernetes (EKS), reducing "
                    "cold-start latency by 40%'."
                )

    # ── Tab 2: Section Breakdown ───────────────────────────────────────────────
    with tab2:
        st.markdown("#### Per-Section Relevance Scores")
        st.caption(
            "Each section of your resume is scored independently against the JD. "
            "Low-scoring sections are where you should add JD-relevant content."
        )

        for sec_name, sec_score in sorted(sec_scores.items(),
                                          key=lambda x: x[1], reverse=True):
            sec_display = sec_name.replace("_", " ").title()
            col_label, col_bar = st.columns([1, 3])
            with col_label:
                # Color the label by score band
                if sec_score >= 60:
                    colour = "#27ae60"
                elif sec_score >= 30:
                    colour = "#e67e22"
                else:
                    colour = "#e74c3c"
                st.markdown(
                    f"<span style='color:{colour}; font-weight:500;'>"
                    f"{sec_display}</span> &nbsp; "
                    f"<span style='color:#A0B2C6; font-size:0.85rem;'>"
                    f"{sec_score:.1f}%</span>",
                    unsafe_allow_html=True,
                )
            with col_bar:
                st.progress(min(sec_score / 100, 1.0))

        st.markdown("<br>", unsafe_allow_html=True)
        st.info(
            "💡 **Tip:** If your Experience section scores below 40%, "
            "rewrite your bullet points to use the same terminology as the JD "
            "while preserving your quantified achievements.",
            icon="💡",
        )

    # ── Tab 3: Keyword Gaps ────────────────────────────────────────────────────
    with tab3:
        st.markdown("#### High-Value Keywords Missing from Resume")
        st.caption(
            "Ranked by TF-IDF weight in the JD — top entries are the most "
            "distinctive terms the recruiter and ATS are scanning for."
        )

        if gaps:
            # Render as a pill grid rather than a flat list
            pills_html = " ".join(
                f'<span style="background:#eaf4fc; color:#2e86c1; padding:3px 10px; '
                f'border-radius:12px; font-size:0.85rem; margin:3px; display:inline-block;">'
                f'{g}</span>'
                for g in gaps
            )
            st.markdown(pills_html, unsafe_allow_html=True)
        else:
            st.success("No significant keyword gaps detected.")

    # ── Tab 4: Quick Fixes ─────────────────────────────────────────────────────
    with tab4:
        st.markdown("#### Immediate Improvement Actions")
        for suggestion in generate_skill_suggestions(missing):
            st.info(f"• {suggestion}")

        if dealbreakers:
            st.markdown("---")
            st.markdown("#### 🔴 Dealbreaker Remediation (Priority 1)")
            for db_skill in sorted(dealbreakers):
                st.error(
                    f"**{db_skill.title()}** — flagged as required/non-negotiable. "
                    f"If you have this experience, make it explicit in your Skills section "
                    f"and add a bullet point in Experience that demonstrates applied use."
                )

    # ──────────────────────────────────────────────────────────────────────────
    # AI OPTIMIZATION LOOP (conditional on toggle + API key)
    # ──────────────────────────────────────────────────────────────────────────

    if use_ai:
        st.divider()
        st.subheader("🤖 AI-Powered Bullet Rewriter")

        if not os.environ.get("GEMINI_API_KEY"):
            st.error(
                "GEMINI_API_KEY not found in environment. "
                "Add it to your `.env` file: `GEMINI_API_KEY=your_key_here`"
            )
        else:
            with st.spinner("Generating AI-powered rewrite recommendations..."):
                diagnosis = optimization_loop_summary(
                    original_score=final_score,
                    skill_score=skill_score,
                    tfidf_score=sim_result["tfidf_score"],
                    missing_skills=list(missing),
                    matched_skills=list(matched),
                )

                ai_result = generate_optimized_bullets(
                    resume_text=resume_text,
                    job_description=job_text,
                    missing_skills=list(missing),
                    keyword_gaps=gaps,
                )

            st.info(f"**Score Diagnosis:** {diagnosis}")

            # Priority skills grid
            priority = ai_result.get("priority_skills", [])
            if priority:
                st.markdown("**Priority Skills to Address:**")
                cols = st.columns(min(len(priority), 4))
                for i, skill in enumerate(priority[:4]):
                    cols[i].success(skill)

            # Strategy summary
            strategy = ai_result.get("strategy_summary", "")
            if strategy:
                st.markdown("**Strategy Summary:**")
                st.write(strategy)

            # Bullet rewrites in expanders — diff-style original vs rewrite
            bullets = ai_result.get("rewritten_bullets", [])
            if bullets:
                st.markdown("**Suggested Bullet Rewrites:**")
                for i, bullet in enumerate(bullets, 1):
                    section  = bullet.get("section", "Resume")
                    skill_addressed = bullet.get("skill_addressed", "")
                    original = bullet.get("original_hint", "")
                    rewrite  = bullet.get("rewritten", "")

                    with st.expander(
                        f"Bullet {i} — {section}  |  targets: {skill_addressed}"
                    ):
                        if original:
                            st.markdown("**Original (weak version):**")
                            st.markdown(
                                f"<div style='background:#fde8e8; padding:10px; "
                                f"border-radius:6px; color:#7b241c;'>{original}</div>",
                                unsafe_allow_html=True,
                            )
                            st.markdown("<br>", unsafe_allow_html=True)

                        st.markdown("**Rewritten (ATS-optimised):**")
                        st.markdown(
                            f"<div style='background:#e8f8f5; padding:10px; "
                            f"border-radius:6px; color:#1e8449;'>{rewrite}</div>",
                            unsafe_allow_html=True,
                        )
                        st.button(
                            "Copy",
                            key=f"copy_{i}",
                            help="Copy rewritten bullet to clipboard",
                        )
            else:
                st.success(
                    "Your resume is already well-optimised for this JD — "
                    "no major rewrites suggested."
                )

            st.divider()
            st.info(
                "**Optimization Loop:** Apply the suggestions above, re-upload your "
                "resume, and re-run the analysis to measure score improvement. "
                "Target: ≥75% final score before submitting."
            )
