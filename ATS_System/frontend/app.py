import streamlit as st
import requests
import json
import plotly.graph_objects as go

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="AI Resume Analyzer", layout="wide")
st.title("📄 AI Resume Analyzer (FastAPI + Mistral + RAG + LangGraph)")

tab1, tab2, tab3, tab4 = st.tabs(["Single Resume", "Batch Upload", "Chat with Resumes", "Compare Resumes"])


def render_gauge(match_pct: int, key: str): 
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=match_pct,
        title={"text": "Overall Match %"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#4CAF50" if match_pct >= 75 else "#FFC107" if match_pct >= 50 else "#F44336"},
            "steps": [
                {"range": [0, 50], "color": "#3a1f1f"},
                {"range": [50, 75], "color": "#3a331f"},
                {"range": [75, 100], "color": "#1f3a23"},
            ],
        }
    ))
    fig.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True, key=f"gauge_{key}") 

def render_skills_donut(matching: list, missing: list, key: str): # Added key parameter
    fig = go.Figure(data=[go.Pie(
        labels=["Matching Skills", "Missing Skills"],
        values=[len(matching), len(missing)],
        hole=0.5,
        marker_colors=["#4CAF50", "#F44336"]
    )])
    fig.update_layout(title="Skills Coverage", height=300)
    st.plotly_chart(fig, use_container_width=True, key=f"donut_{key}")
def render_section_batch(label, items, category, jd_text, analysis, cand_key):
    col_a, col_b = st.columns([5, 1])
    with col_a:
        st.markdown(f"**{label}**")
    with col_b:
        if st.button("View", key=f"btn_{cand_key}_{category}"):
            st.session_state[f"batch_active_{cand_key}"] = {
                "category": category,
                "items": items,
                "jd": jd_text,
                "analysis": analysis
            }
            st.session_state.pop(f"batch_explanation_{cand_key}", None)

    active = st.session_state.get(f"batch_active_{cand_key}")
    if active and active["category"] == category:
        for item in items:
            st.write(f"- {item}")

# TAB 1: SINGLE RESUME
with tab1:
    resume_file = st.file_uploader("Upload Resume (PDF)", type="pdf", key="single")
    jd_text = st.text_area("Paste Job Description", height=200, key="jd1")

    if st.button("Analyze") and resume_file and jd_text:
        with st.spinner("Analyzing..."):
            files = {"resume": (resume_file.name, resume_file.getvalue(), "application/pdf")}
            data = {"jd_text": jd_text}
            response = requests.post(f"{BACKEND_URL}/analyze-full", files=files, data=data)
            full = response.json()

        st.session_state["result"] = full["analysis"]
        st.session_state["jd_text"] = jd_text
        st.session_state.pop("active_category", None)
        st.session_state.pop("explanation_cache", None)

    if "result" in st.session_state:
        result = st.session_state["result"]
        jd_text_stored = st.session_state["jd_text"]

        main_col, side_col = st.columns([2, 1])

        with main_col:
            g1, g2 = st.columns(2)
            with g1:
                render_gauge(result["match_percentage"])
            with g2:
                render_skills_donut(result["matching_skills"], result["missing_skills"])

            st.divider()

            render_section("✅ Matching Skills", result["matching_skills"], "Matching Skills", jd_text_stored, result)
            st.divider()
            render_section("❌ Missing Skills", result["missing_skills"], "Missing Skills", jd_text_stored, result)
            st.divider()
            render_section("💪 Strengths", result["strengths"], "Strengths", jd_text_stored, result)
            st.divider()
            render_section("📈 Improvements", result["improvements"], "Improvements", jd_text_stored, result)
            st.divider()
            render_section("🤖 AI Suggestions", result["ai_suggestions"], "AI Suggestions", jd_text_stored, result)

            st.success(f"Recommendation: {result['recommendation']}")

        with side_col:
            st.subheader("🔍 Detail Panel")
            if "active_category" in st.session_state:
                category = st.session_state["active_category"]
                items = st.session_state["active_items"]
                jd = st.session_state["active_jd"]
                analysis = st.session_state["active_analysis"]

                st.markdown(f"**{category}**")

                if "explanation_cache" not in st.session_state:
                    with st.spinner("Getting AI explanation..."):
                        resp = requests.post(
                            f"{BACKEND_URL}/elaborate-section",
                            data={
                                "category": category,
                                "items": json.dumps(items),
                                "full_analysis": json.dumps(analysis),
                                "jd_text": jd
                            }
                        )
                        st.session_state["explanation_cache"] = resp.json()["explanation"]

                st.write(st.session_state["explanation_cache"])
            else:
                st.info("Click 'View' next to any category to see a detailed AI breakdown here.")

# TAB 2: BATCH UPLOAD 
with tab2:
    uploaded_files = st.file_uploader(
        "Upload Multiple Resumes (PDF)", type="pdf",
        accept_multiple_files=True, key="batch"
    )
    jd_text_batch = st.text_area("Paste Job Description", height=200, key="jd2")

    if st.button("Analyze All") and uploaded_files and jd_text_batch:
        with st.spinner(f"Processing {len(uploaded_files)} resumes..."):
            files = [("resumes", (f.name, f.getvalue(), "application/pdf")) for f in uploaded_files]
            data = {"jd_text": jd_text_batch}
            response = requests.post(f"{BACKEND_URL}/analyze-batch", files=files, data=data)
            results = response.json()

        st.session_state["batch_results"] = results
        st.session_state["batch_jd"] = jd_text_batch
        for key in list(st.session_state.keys()):
            if key.startswith("batch_active_") or key.startswith("batch_explanation_"):
                del st.session_state[key]

    if "batch_results" in st.session_state:
        results = st.session_state["batch_results"]
        jd_text_stored = st.session_state["batch_jd"]

        st.subheader("📊 Overview")

        names = [r["candidate_id"] for r in results]
        match_pcts = [r["analysis"]["match_percentage"] for r in results]
        recommendations = [r["analysis"]["recommendation"] for r in results]

        # Color each bar based on recommendation
        bar_colors = []
        for rec in recommendations:
            if rec == "Selected":
                bar_colors.append("#4CAF50")
            elif rec == "Shortlisted":
                bar_colors.append("#FFC107")
            else:
                bar_colors.append("#F44336")

        ov_col1, ov_col2 = st.columns([2, 1])

        with ov_col1:
            fig_bar = go.Figure(data=[go.Bar(
                x=names, y=match_pcts,
                marker_color=bar_colors,
                text=[f"{p}%" for p in match_pcts],
                textposition="outside"
            )])
            fig_bar.update_layout(
                title="Match % by Candidate",
                yaxis=dict(range=[0, 100], title="Match %"),
                xaxis=dict(title="Candidate"),
                height=400
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with ov_col2:
            selected_count = recommendations.count("Selected")
            shortlisted_count = recommendations.count("Shortlisted")
            rejected_count = recommendations.count("Rejected")
            total = len(recommendations)

            st.metric("Total Resumes", total)
            st.metric("✅ Selected", selected_count)
            st.metric("⚡ Shortlisted", shortlisted_count)
            st.metric("❌ Rejected", rejected_count)

            fig_donut = go.Figure(data=[go.Pie(
                labels=["Selected", "Shortlisted", "Rejected"],
                values=[selected_count, shortlisted_count, rejected_count],
                hole=0.5,
                marker_colors=["#4CAF50", "#FFC107", "#F44336"]
            )])
            fig_donut.update_layout(title="Recommendation Breakdown", height=300, showlegend=True)
            st.plotly_chart(fig_donut, use_container_width=True)

        st.divider()

        # PER-CANDIDATE DETAILS
        st.subheader("📋 Candidate Details")

        for i, r in enumerate(results, 1):
            a = r["analysis"]
            candidate_id = r["candidate_id"]
            cand_key = f"cand_{i}"

            with st.expander(f"#{i} — {candidate_id} — {a['match_percentage']}% — {a['recommendation']}", expanded=(i == 1)):
                main_col, side_col = st.columns([2, 1])

                with main_col:
                    g1, g2 = st.columns(2)
                    with g1:
                        render_gauge(a["match_percentage"], key=cand_key)
                    with g2:
                        render_skills_donut(a["matching_skills"], a["missing_skills"], key=cand_key)

                    st.divider()

                    render_section_batch("✅ Matching Skills", a["matching_skills"], "Matching Skills", jd_text_stored, a, cand_key)
                    st.divider()
                    render_section_batch("❌ Missing Skills", a["missing_skills"], "Missing Skills", jd_text_stored, a, cand_key)
                    st.divider()
                    render_section_batch("💪 Strengths", a["strengths"], "Strengths", jd_text_stored, a, cand_key)
                    st.divider()
                    render_section_batch("📈 Improvements", a["improvements"], "Improvements", jd_text_stored, a, cand_key)
                    st.divider()
                    render_section_batch("🤖 AI Suggestions", a["ai_suggestions"], "AI Suggestions", jd_text_stored, a, cand_key)

                    st.success(f"Recommendation: {a['recommendation']}")

                with side_col:
                    st.markdown("**🔍 Detail Panel**")
                    active_key = f"batch_active_{cand_key}"
                    if active_key in st.session_state:
                        category = st.session_state[active_key]["category"]
                        items = st.session_state[active_key]["items"]
                        jd = st.session_state[active_key]["jd"]
                        analysis = st.session_state[active_key]["analysis"]

                        st.markdown(f"**{category}**")

                        explanation_key = f"batch_explanation_{cand_key}"
                        if explanation_key not in st.session_state:
                            with st.spinner("Getting AI explanation..."):
                                resp = requests.post(
                                    f"{BACKEND_URL}/elaborate-section",
                                    data={
                                        "category": category,
                                        "items": json.dumps(items),
                                        "full_analysis": json.dumps(analysis),
                                        "jd_text": jd
                                    }
                                )
                                st.session_state[explanation_key] = resp.json()["explanation"]

                        st.write(st.session_state[explanation_key])
                    else:
                        st.info("Click 'View' next to any category.")


# TAB 3: CHATBOT
with tab3:
    query = st.text_input("Ask about the uploaded resumes")
    if st.button("Ask") and query:
        with st.spinner("Thinking..."):
            response = requests.post(f"{BACKEND_URL}/chat", data={"query": query})
            answer = response.json()["answer"]
        st.write(answer)


# TAB 4: COMPARE RESUMES
with tab4:
    st.write("Upload 2 or more resumes to compare against the same job description.")
    compare_files = st.file_uploader(
        "Upload Resumes to Compare (PDF)", type="pdf",
        accept_multiple_files=True, key="compare"
    )
    jd_text_compare = st.text_area("Paste Job Description", height=200, key="jd4")

    if st.button("Compare") and compare_files and len(compare_files) >= 2 and jd_text_compare:
        with st.spinner(f"Comparing {len(compare_files)} resumes..."):
            files = [("resumes", (f.name, f.getvalue(), "application/pdf")) for f in compare_files]
            data = {"jd_text": jd_text_compare}
            response = requests.post(f"{BACKEND_URL}/compare-breakdown", files=files, data=data)
            results = response.json()["results"]

        st.session_state["compare_results"] = results

    if "compare_results" in st.session_state:
        results = st.session_state["compare_results"]

        # Grouped bar: match % per candidate
        names = [r["candidate_id"] for r in results]
        match_pcts = [r["analysis"]["match_percentage"] for r in results]

        fig_match = px.bar(x=names, y=match_pcts, labels={"x": "Candidate", "y": "Match %"},
                            title="Overall Match % Comparison", color=match_pcts,
                            color_continuous_scale=["#F44336", "#FFC107", "#4CAF50"], range_color=[0, 100])
        fig_match.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_match, use_container_width=True)

        # Radar comparison: category scores across candidates
        categories = ["Skill Match", "Experience", "Education", "Semantic Match"]
        fig_radar = go.Figure()
        for r in results:
            b = r["breakdown"]
            values = [
                b.get("skill_match_score", 0),
                b.get("experience_score", 0),
                b.get("education_score", 0),
                b.get("semantic_match_score", 0),
            ]
            fig_radar.add_trace(go.Scatterpolar(
                r=values + [values[0]], theta=categories + [categories[0]],
                fill='toself', name=r["candidate_id"]
            ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                                 title="Category Comparison (Radar)", height=450)
        st.plotly_chart(fig_radar, use_container_width=True)

        # skill-level comparison
        all_skills = set()
        for r in results:
            all_skills.update(r["breakdown"].get("skill_breakdown", {}).keys())
        all_skills = sorted(all_skills)

        rows = []
        for r in results:
            sb = r["breakdown"].get("skill_breakdown", {})
            for skill in all_skills:
                rows.append({"Candidate": r["candidate_id"], "Skill": skill, "Score": sb.get(skill, 0)})

        if rows:
            df = pd.DataFrame(rows)
            fig_skills = px.bar(df, x="Skill", y="Score", color="Candidate", barmode="group",
                                 title="Skill-by-Skill Comparison", range_y=[0, 100])
            fig_skills.update_layout(height=400)
            st.plotly_chart(fig_skills, use_container_width=True)

        # Ranking table
        st.subheader("📋 Ranking Summary")
        summary = pd.DataFrame([{
            "Candidate": r["candidate_id"],
            "Match %": r["analysis"]["match_percentage"],
            "Recommendation": r["analysis"]["recommendation"],
            "Skill Match": r["breakdown"].get("skill_match_score", 0),
            "Experience": r["breakdown"].get("experience_score", 0),
            "Education": r["breakdown"].get("education_score", 0),
        } for r in results]).sort_values("Match %", ascending=False)
        st.dataframe(summary, use_container_width=True)