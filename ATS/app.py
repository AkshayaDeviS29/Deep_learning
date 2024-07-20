from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import io
import base64
import pdf2image
import google.generativeai as genai
from dotenv import load_dotenv

app = Flask(__name__)
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(user_input, pdf_content, prompt):
    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content([user_input, pdf_content[0], prompt])
    return response.text

def input_pdf_setup(uploaded_file):
    if uploaded_file:
        images = pdf2image.convert_from_bytes(uploaded_file.read())
        first_page = images[0]
        img_byte_arr = io.BytesIO()
        first_page.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        pdf_parts = [
            {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(img_byte_arr).decode()
            }
        ]
        return pdf_parts
    else:
        raise FileNotFoundError("No file uploaded")

input_prompt1 = """
You are an experienced Technical Human Resource Manager. Your task is to review the provided resume against the job description. 
Please share your professional evaluation on whether the candidate's profile aligns with the role. 
Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements.
"""

input_prompt3 = """
You are a skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality. 
Your task is to evaluate the resume against the provided job description. Provide the percentage match if the resume matches
the job description. First, the output should come as a percentage, then list missing keywords, and finally, give final thoughts.
"""

@app.route('/ats', methods=['POST'])
def applicant_tracking():
    try:
        # Debugging statement to check request files and form data
        print("Request files:", request.files)
        print("Request form:", request.form)

        resume_file = request.files.get('resume')
        job_description = request.form.get('job_description', '')

        if not resume_file or resume_file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if resume_file and resume_file.filename.endswith('.pdf'):
            filename = secure_filename(resume_file.filename)
            pdf_content = input_pdf_setup(resume_file)

            submit_type = request.form.get('submit_type')
            if submit_type == 'evaluation':
                response = get_gemini_response(input_prompt1, pdf_content, job_description)
            elif submit_type == 'percentage_match':
                response = get_gemini_response(input_prompt3, pdf_content, job_description)
            else:
                return jsonify({"error": "Invalid submit type"}), 400

            return jsonify({"response": response})
        else:
            return jsonify({"error": "Unsupported file type"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
