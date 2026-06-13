from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List
import tempfile, os

from ats_system import extract_text_from_pdf, analyze_resume, elaborate_section_full, get_scoring_breakdown
from rag import chunk_resume, add_chunks, chat_with_resumes
from graph import process_resumes_batch

app = FastAPI(title="Resume ATS API")

# Single resume analysis
@app.post("/analyze")
async def analyze(resume: UploadFile = File(...), jd_text: str = Form(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await resume.read())
        tmp_path = tmp.name

    resume_text = extract_text_from_pdf(tmp_path)
    os.unlink(tmp_path)

    result = analyze_resume(resume_text, jd_text)

    chunks = chunk_resume(resume_text, candidate_id=resume.filename)
    add_chunks(chunks)

    return JSONResponse(content=result)


# Single resume analysis analysis + score breakdown for charts
@app.post("/analyze-full")
async def analyze_full(resume: UploadFile = File(...), jd_text: str = Form(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await resume.read())
        tmp_path = tmp.name

    resume_text = extract_text_from_pdf(tmp_path)
    os.unlink(tmp_path)

    analysis = analyze_resume(resume_text, jd_text)
    breakdown = get_scoring_breakdown(resume_text, jd_text)

    chunks = chunk_resume(resume_text, candidate_id=resume.filename)
    add_chunks(chunks)

    return JSONResponse(content={"analysis": analysis, "breakdown": breakdown, "candidate_id": resume.filename})


# Batch resume analysis
@app.post("/analyze-batch")
async def analyze_batch(resumes: List[UploadFile] = File(...), jd_text: str = Form(...)):
    temp_paths = []
    for f in resumes:
        path = os.path.join(tempfile.gettempdir(), f.filename)
        with open(path, "wb") as out:
            out.write(await f.read())
        temp_paths.append(path)

    results = process_resumes_batch(temp_paths, jd_text)

    for path in temp_paths:
        os.remove(path)

    return JSONResponse(content=results)


#Compare resumes
@app.post("/compare-breakdown")
async def compare_breakdown(resumes: List[UploadFile] = File(...), jd_text: str = Form(...)):
    results = []
    for f in resumes:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await f.read())
            tmp_path = tmp.name

        resume_text = extract_text_from_pdf(tmp_path)
        os.unlink(tmp_path)

        analysis = analyze_resume(resume_text, jd_text)
        breakdown = get_scoring_breakdown(resume_text, jd_text)

        results.append({
            "candidate_id": f.filename,
            "analysis": analysis,
            "breakdown": breakdown
        })

    return JSONResponse(content={"results": results})


# Section elaboration
@app.post("/elaborate-section")
async def elaborate_section_endpoint(
    category: str = Form(...),
    items: str = Form(...),
    full_analysis: str = Form(...),
    jd_text: str = Form(...)
):
    import json
    items_list = json.loads(items)
    analysis_dict = json.loads(full_analysis)
    explanation = elaborate_section_full(category, items_list, analysis_dict, jd_text)
    return JSONResponse(content={"explanation": explanation})

# Chatbot
@app.post("/chat")
async def chat(query: str = Form(...), candidate_id: str = Form(None)):
    answer = chat_with_resumes(query, candidate_id=candidate_id)
    return JSONResponse(content={"answer": answer})