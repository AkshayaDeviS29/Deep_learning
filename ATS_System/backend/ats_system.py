import os, json
import pdfplumber
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()
client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
MODEL = "mistral-small-latest"

# PDF Parsing 
def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# LLM Call
def call_mistral(system_prompt: str, user_prompt: str, json_mode=True) -> str:
    response = client.chat.complete(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"} if json_mode else None,
        temperature=0.2
    )
    return response.choices[0].message.content

# Resume Analysis
ANALYSIS_SYSTEM = """You are an expert ATS analyzer.
Respond ONLY in valid JSON with this structure, no extra text:
{
  "match_percentage": <0-100>,
  "matching_skills": [...],
  "missing_skills": [...],
  "strengths": [...],
  "improvements": [...],
  "ai_suggestions": [...],
  "recommendation": "Selected" | "Shortlisted" | "Rejected"
}"""

def analyze_resume(resume_text: str, jd_text: str) -> dict:
    prompt = f"RESUME:\n{resume_text}\n\nJOB DESCRIPTION:\n{jd_text}\n\nReturn the JSON as specified."
    raw = call_mistral(ANALYSIS_SYSTEM, prompt, json_mode=True)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(raw.strip().strip("```json").strip("```"))

# Score Breakdown
SCORING_BREAKDOWN_SYSTEM = """You are an ATS scoring engine.
Based on the resume and job description, score the candidate on these
categories from 0-100 each. Respond ONLY in valid JSON, no extra text:
{
  "skill_match_score": <0-100>,
  "experience_score": <0-100>,
  "education_score": <0-100>,
  "semantic_match_score": <0-100>,
  "skill_breakdown": {
    "<skill_name>": <0-100 relevance/proficiency estimate>,
    ... (5-8 key skills from the JD, each scored)
  }
}"""

def get_scoring_breakdown(resume_text: str, jd_text: str) -> dict:
    prompt = f"RESUME:\n{resume_text}\n\nJOB DESCRIPTION:\n{jd_text}\n\nReturn the JSON as specified."
    raw = call_mistral(SCORING_BREAKDOWN_SYSTEM, prompt, json_mode=True)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(raw.strip().strip("```json").strip("```"))

SECTION_ELABORATION_SYSTEM = """You are an expert career coach and ATS consultant.
The user wants a detailed explanation of ONE category from their resume
analysis. Using the FULL analysis context provided (all categories), explain
the items in this category: why each matters for this job, how they relate
to the candidate's other strengths/gaps, and what the candidate should do
about them. Respond in plain text (4-7 short paragraphs or bullet points),
not JSON."""

def elaborate_section_full(category: str, items: list, full_analysis: dict, jd_text: str) -> str:
    items_text = "\n".join(f"- {item}" for item in items)
    context = f"""FULL ANALYSIS CONTEXT:
Match Percentage: {full_analysis.get('match_percentage')}%
Matching Skills: {full_analysis.get('matching_skills')}
Missing Skills: {full_analysis.get('missing_skills')}
Strengths: {full_analysis.get('strengths')}
Improvements: {full_analysis.get('improvements')}
AI Suggestions: {full_analysis.get('ai_suggestions')}
Recommendation: {full_analysis.get('recommendation')}

JOB DESCRIPTION:
{jd_text}"""

    prompt = f"""{context}

CATEGORY: {category}
ITEMS IN THIS CATEGORY:
{items_text}

Provide a detailed elaboration on this category's items, in context of the
full analysis above."""
    return call_mistral(SECTION_ELABORATION_SYSTEM, prompt, json_mode=False)