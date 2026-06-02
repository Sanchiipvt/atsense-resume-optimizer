import re
import pdfplumber 
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# FILE TEXT EXTRACTION
# =========================
def extract_text_from_file(uploaded_file):
    """
    Safely extract text from an uploaded Streamlit file object (PDF or DOCX).
    """
    if uploaded_file is None:
        return ""

    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        text = ""
        # Ensure we seek to the beginning of the file buffer
        uploaded_file.seek(0)
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
        return text.strip()

    elif filename.endswith(".docx"):
        uploaded_file.seek(0)
        doc = Document(uploaded_file)
        return " ".join(p.text for p in doc.paragraphs)

    return ""

# =========================
# TEXT PREPROCESSING
# =========================
STOPWORDS = {
    "the", "is", "and", "with", "for", "a", "an",
    "to", "of", "in", "on", "at", "by", "from"
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def preprocess_text(text):
    if not text:
        return []
    tokens = clean_text(text).split()
    return [t for t in tokens if t not in STOPWORDS]

def keyword_gap(resume_text, job_text):
    resume_tokens = set(preprocess_text(resume_text))
    job_tokens = set(preprocess_text(job_text))

    missing_keywords = job_tokens - resume_tokens
    missing_keywords = {word for word in missing_keywords if len(word) > 3}
    return sorted(missing_keywords)

def keyword_suggestions(missing_keywords, limit=8):
    return missing_keywords[:limit]

# =========================
# SKILLS & DICTIONARIES
# =========================
SKILLS = {
    "python", "java", "sql", "machine learning", "deep learning",
    "data analysis", "data visualization", "nlp", "tensorflow", "pandas", "numpy"
}

SKILL_ALIASES = {
    "machine learning": {"ml", "machine-learning"},
    "deep learning": {"dl", "deep-learning"},
    "data analysis": {"analytics", "data-analysis"},
    "python": {"py"},
    "nlp": {"natural language processing"}
}

SKILL_SUGGESTIONS = {
    "python": "Add Python projects demonstrating automation or data handling.",
    "java": "Include object-oriented or backend development projects.",
    "sql": "Show complex queries, joins, and database schema design.",
    "machine learning": "Add ML projects with model training and evaluation.",
    "deep learning": "Include neural networks or CNN/RNN implementations.",
    "data analysis": "Show data cleaning, EDA, and insights generation.",
    "data visualization": "Add dashboards using Matplotlib, Seaborn, or Power BI.",
    "nlp": "Mention NLP tasks like text classification or sentiment analysis.",
    "tensorflow": "Include hands-on TensorFlow model implementations.",
    "pandas": "Show real-world data manipulation pipelines.",
    "numpy": "Mention numerical computing or matrix-based operations."
}

def normalize_skill(skill):
    for main, aliases in SKILL_ALIASES.items():
        if skill == main or skill in aliases:
            return main
    return skill

def extract_skills(tokens, skill_set):
    found = set()
    for token in tokens:
        norm = normalize_skill(token)
        if norm in skill_set:
            found.add(norm)
    for i in range(len(tokens) - 1):
        phrase = normalize_skill(tokens[i] + " " + tokens[i + 1])
        if phrase in skill_set:
            found.add(phrase)
    return found

def match_skills(resume_skills, job_skills):
    if not job_skills:
        return 0, set(), set()
    matched = resume_skills & job_skills
    missing = job_skills - resume_skills
    score = int((len(matched) / len(job_skills)) * 100)
    return score, matched, missing

def generate_skill_suggestions(missing_skills):
    return [
        SKILL_SUGGESTIONS.get(skill, f"Consider adding experience related to {skill}.")
        for skill in missing_skills
    ]
    
def split_resume_sections(resume_text):
    sections = {"skills": "", "experience": "", "projects": "", "education": ""}
    current_section = None
    lines = resume_text.lower().split("\n")

    for line in lines:
        if "skill" in line:
            current_section = "skills"
        elif "experience" in line or "work" in line:
            current_section = "experience"
        elif "project" in line:
            current_section = "projects"
        elif "education" in line:
            current_section = "education"

        if current_section:
            sections[current_section] += line + " "
    return sections

def tfidf_similarity(resume_text, job_text):
    if not resume_text or not job_text:
        return 0.0
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    vectors = vectorizer.fit_transform([clean_text(resume_text), clean_text(job_text)])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    return round(similarity * 100, 2)

def final_ats_score(skill_score, tfidf_score, skill_weight=0.7):
    return round((skill_score * skill_weight) + (tfidf_score * (1 - skill_weight)), 2)

def section_wise_scores(resume_sections, job_text):
    scores = {}
    for section, text in resume_sections.items():
        if text.strip():
            scores[section] = tfidf_similarity(text, job_text)
        else:
            scores[section] = 0.0
    return scores