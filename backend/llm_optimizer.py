import os
import json
import google.generativeai as genai


# =========================
# LAZY MODEL LOADER
# =========================
# The model is NOT instantiated at import time.
# Previously, `genai.configure()` and `GenerativeModel()` ran on every import,
# meaning any session that imported this module — even with AI toggled OFF —
# would attempt to configure Gemini with a potentially None API key.
# On Streamlit Community Cloud this caused silent credential errors that
# surfaced as unhandled exceptions rather than the graceful st.error in app.py.
#
# Fix: _get_model() is called only inside the two public functions,
# and only after the API key has been validated as non-empty.

def _get_model():
    """
    Validate the API key and return a configured GenerativeModel.
    Raises ValueError with a clear message if the key is missing,
    so the caller's except block can return the graceful fallback dict.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not set. "
            "Add it to your .env file or Streamlit secrets."
        )
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")


# =========================
# CORE FUNCTION: BULLET REWRITER
# =========================
def generate_optimized_bullets(
    resume_text: str,
    job_description: str,
    missing_skills: list,
    keyword_gaps: list
) -> dict:

    if not missing_skills and not keyword_gaps:
        return {
            "rewritten_bullets": [],
            "strategy_summary": (
                "Your resume already covers the key skills and keywords for this job description. "
                "Focus on quantifying your existing bullet points with metrics."
            ),
            "priority_skills": []
        }

    missing_skills_str = ", ".join(missing_skills) if missing_skills else "None"
    keyword_gaps_str   = ", ".join(keyword_gaps)   if keyword_gaps   else "None"

    prompt = f"""
You are an expert resume coach and ATS optimization specialist.
A candidate needs help rewriting their resume to better match a job description.

## Candidate's Current Resume (partial):
{resume_text[:2000]}

## Target Job Description:
{job_description[:1500]}

## Skills Missing from Resume:
{missing_skills_str}

## Important Keywords Missing from Resume:
{keyword_gaps_str}

## Your Task:
Analyze the text and produce a valid JSON object matching this structure:
{{
  "priority_skills": ["skill1", "skill2", "skill3"],
  "strategy_summary": "A 2-3 sentence paragraph strategy explanation.",
  "rewritten_bullets": [
    {{
      "section": "Experience",
      "original_hint": "Brief description of existing bullet to replace",
      "rewritten": "Full rewritten bullet point incorporating missing items naturally",
      "skill_addressed": "The exact skill targeted"
    }}
  ]
}}
"""
    try:
        model = _get_model()   # validated and instantiated here, not at import
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json"
            )
        )
        return json.loads(response.text.strip())
    except Exception as e:
        return {
            "priority_skills": missing_skills[:3],
            "strategy_summary": (
                f"Focus on adding these key skills to your experience sections: "
                f"{missing_skills_str}."
            ),
            "rewritten_bullets": [
                {
                    "section": "Experience",
                    "original_hint": "Any technical bullet point",
                    "rewritten": (
                        f"Developed solutions leveraging {skill}, "
                        "improving foundational system metrics."
                    ),
                    "skill_addressed": skill
                }
                for skill in missing_skills[:3]
            ],
            "error": str(e)
        }


# =========================
# HELPER: DIAGNOSIS GENERATOR
# =========================
def optimization_loop_summary(
    original_score: float,
    skill_score: float,
    tfidf_score: float,
    missing_skills: list,
    matched_skills: list
) -> str:

    prompt = f"""
You are an ATS expert. A candidate received the following resume analysis scores.
Write a concise 3-sentence diagnosis paragraph (plain text only, no bullet points, no markdown headings).
Sentence 1: What the score means in plain English.
Sentence 2: The single most important reason the score is not higher.
Sentence 3: The single highest-leverage action to improve the score.

Scores:
- Final ATS Score: {original_score}%
- Skill Match Score: {skill_score}%
- Similarity Score: {tfidf_score}%
- Matched Skills: {", ".join(matched_skills) if matched_skills else "None"}
- Missing Skills: {", ".join(missing_skills) if missing_skills else "None"}
"""
    try:
        model = _get_model()   # same pattern — instantiated here, not at import
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return (
            f"Your resume scored {original_score}% overall. "
            "Your biggest leverage point is addressing the missing skill gaps."
        )
- Skill Match Score: {skill_score}%
- TF-IDF Semantic Score: {tfidf_score}%
- Matched Skills: {", ".join(matched_skills) if matched_skills else "None"}
- Missing Skills: {", ".join(missing_skills) if missing_skills else "None"}
"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Your resume scored {original_score}% overall. Your biggest leverage point is addressing the missing skill gaps."
