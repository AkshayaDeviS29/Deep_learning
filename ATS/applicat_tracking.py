import os
import google.generativeai as genai
from flask import Flask, request
from dotenv import load_dotenv
import PyPDF2 as pdf

app = Flask(__name__)

load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Generate Response Using Google Generative AI
def get_gemini_repsonse(resume_text, job_description):

    model = genai.GenerativeModel('gemini-pro')
    input_prompt = """
            Hey Act Like a skilled or very experience ATS(Application Tracking System)
            with a deep understanding of tech field,software engineering,data science ,data analyst
            and big data engineer. Your task is to evaluate the resume based on the given job description.
            You must consider the job market is very competitive and you should provide 
            best assistance for improving thr resumes. Assign the percentage Matching based 
            on Jd and
            the missing keywords with high accuracy
            resume:{resume_text}
            description:{job_description}

            I want the response in one single string having the structure
            {{"JD Match":"%","MissingKeywords:[]","Profile Summary":""}}
    """

    response = model.generate_content(input_prompt)

    return response.text


# Extract Text from PDF
def input_pdf_text(uploaded_file,job_description):
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page in range(len(reader.pages)):
        page = reader.pages[page]
        text += str(page.extract_text())
        response = get_gemini_repsonse(text,job_description)
    return response


