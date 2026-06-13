from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from ats_system import extract_text_from_pdf, analyze_resume
from rag import chunk_resume, add_chunks

# State definition
class ResumeState(TypedDict):
    file_path: str
    candidate_id: str
    jd_text: str
    resume_text: str
    analysis: dict
    status: str

# Node functions
def parse_node(state: ResumeState) -> ResumeState:
    text = extract_text_from_pdf(state["file_path"])
    state["resume_text"] = text
    return state

def analyze_node(state: ResumeState) -> ResumeState:
    result = analyze_resume(state["resume_text"], state["jd_text"])
    state["analysis"] = result
    state["status"] = result["recommendation"]
    return state

def embed_node(state: ResumeState) -> ResumeState:
    chunks = chunk_resume(state["resume_text"], state["candidate_id"])
    add_chunks(chunks)
    return state

# Build graph
def build_graph():
    workflow = StateGraph(ResumeState)

    workflow.add_node("parse", parse_node)
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("embed", embed_node)

    workflow.set_entry_point("parse")
    workflow.add_edge("parse", "analyze")
    workflow.add_edge("analyze", "embed")
    workflow.add_edge("embed", END)

    return workflow.compile()

# Run for one resume
def process_resume(file_path: str, candidate_id: str, jd_text: str) -> dict:
    graph = build_graph()
    result = graph.invoke({
        "file_path": file_path,
        "candidate_id": candidate_id,
        "jd_text": jd_text,
        "resume_text": "",
        "analysis": {},
        "status": ""
    })
    return result

# Run for many resumes
def process_resumes_batch(file_paths: List[str], jd_text: str) -> List[dict]:
    graph = build_graph()
    results = []
    for path in file_paths:
        candidate_id = path.split("/")[-1]
        result = graph.invoke({
            "file_path": path,
            "candidate_id": candidate_id,
            "jd_text": jd_text,
            "resume_text": "",
            "analysis": {},
            "status": ""
        })
        results.append({
            "candidate_id": candidate_id,
            "analysis": result["analysis"]
        })
    # Sort by match percentage
    results.sort(key=lambda x: x["analysis"]["match_percentage"], reverse=True)
    return results