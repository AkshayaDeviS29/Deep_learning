from flask import Flask, request
from werkzeug.utils import secure_filename
from applicat_tracking import input_pdf_text
import json

app = Flask(__name__)


@app.route('/ats', methods=['POST'])
def applicant_tracking():
    print("****************  api called successfully ******************")

    resume_file = request.files['resume']
    print(" ******* To get the input file from user *******", resume_file)
    job_description = request.form.get('job_description', '')
    print(" ******* To get the job description from user *******", job_description)

    if resume_file and resume_file.filename.endswith('.pdf'):
        filename = secure_filename(resume_file.filename)
        print("####### file name #######", filename)
        pdf_content = input_pdf_text(resume_file,job_description)
        print("##### pdf content###", pdf_content)
        res = json.dumps(pdf_content)
        print("******************* res *******************", res)
        return res


# Runing  Flask
if __name__ == '__main__':
    app.run(debug=True)
