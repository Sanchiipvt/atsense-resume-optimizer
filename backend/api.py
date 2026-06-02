from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from backend.main import (
    extract_text_from_file,
    preprocess_text,
    extract_skills,
    match_skills,
    generate_skill_suggestions,
    tfidf_similarity,
    final_ats_score,
    keyword_gap,
    keyword_suggestions,
    split_resume_sections,
    section_wise_scores,
    SKILLS
)
app = FastAPI()

# ADD these imports at the top of api.py
from backend.llm_optimizer import (
    generate_optimized_bullets,
    optimization_loop_summary
)

# Allow frontend connection later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(...)
):

    resume_text = extract_text_from_file(resume.file)

    tfidf_score = tfidf_similarity(resume_text, job_description)

    resume_tokens = preprocess_text(resume_text)
    job_tokens = preprocess_text(job_description)

    resume_skills = extract_skills(resume_tokens, SKILLS)
    job_skills = extract_skills(job_tokens, SKILLS)

    skill_score, matched, missing = match_skills(
        resume_skills, job_skills
    )

    final_score = final_ats_score(skill_score, tfidf_score)

    sections = split_resume_sections(resume_text)
    section_scores = section_wise_scores(sections, job_description)

    missing_keywords = keyword_gap(resume_text, job_description)
    suggested_keywords = keyword_suggestions(missing_keywords)

    suggestions = generate_skill_suggestions(missing)

    return {
        "skill_score": skill_score,
        "tfidf_score": tfidf_score,
        "final_score": final_score,
        "matched_skills": list(matched),
        "missing_skills": list(missing),
        "section_scores": section_scores,
        "keyword_gaps": suggested_keywords,
        "improvement_suggestions": suggestions
    }
    
# ADD this new endpoint at the bottom of api.py
@app.post("/optimize")
async def optimize_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(...),
    missing_skills: str = Form(default=""),   # comma-separated string
    keyword_gaps: str = Form(default="")      # comma-separated string
):
    """
    AI-powered optimization endpoint.
    Call this AFTER /analyze to get LLM-generated suggestions.
    """
    resume_text = extract_text_from_file(resume.file)

    # Parse comma-separated strings back into lists
    missing_list = [s.strip() for s in missing_skills.split(",") if s.strip()]
    gaps_list = [s.strip() for s in keyword_gaps.split(",") if s.strip()]

    # Run both AI functions
    bullets = generate_optimized_bullets(
        resume_text=resume_text,
        job_description=job_description,
        missing_skills=missing_list,
        keyword_gaps=gaps_list
    )

    # We need the original scores for the diagnosis
    # Re-compute them (stateless API design)
    tfidf_score = tfidf_similarity(resume_text, job_description)
    resume_tokens = preprocess_text(resume_text)
    job_tokens = preprocess_text(job_description)
    resume_skills = extract_skills(resume_tokens, SKILLS)
    job_skills = extract_skills(job_tokens, SKILLS)
    skill_score, matched, missing = match_skills(resume_skills, job_skills)
    final_score = final_ats_score(skill_score, tfidf_score)

    diagnosis = optimization_loop_summary(
        original_score=final_score,
        skill_score=skill_score,
        tfidf_score=tfidf_score,
        missing_skills=missing_list,
        matched_skills=list(matched)
    )

    return {
        "diagnosis": diagnosis,
        "priority_skills": bullets.get("priority_skills", []),
        "strategy_summary": bullets.get("strategy_summary", ""),
        "rewritten_bullets": bullets.get("rewritten_bullets", [])
    }