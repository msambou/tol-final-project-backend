from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import zipfile
import io
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()


app = FastAPI()


class LLMAnalyzer:
    def __init__(self, student_submissions):
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        self.student_submissions = student_submissions

    def getMisconceptionsResponse(self):
        # I want to loop through 
        system_message = (
        "You are an expert programming tutor that analyzes student code submissions. "
        )
        
        prompt = """The following are student codes:

        def fibonacci(n):
            a, b = 0, 1
            while a < n:
                a, b = b, a + b
            return b

        def fibonacci(n):
            a, b = 0, 1
            while a < n:
                a, b = b, a + b
            return b

        def fibonacci(n):
            return [i for i in range(n)]

        def fibonacci(n):
            if n <= 0:
                return []
            elif n == 1:
                return [0]
            elif n == 2:
                return [0, 1]
            else:
                a = 0
                b = 1
                result = [a, b]
                for i in range(n):
                    c = a + b
                    result.append(c)
                    a = b
                    b = c
                return result

        Please analyze their submissions and respond using the following structure:
        1. The goal of the coding assignment
        2. Count of Misconceptions
        3. Count of Coding Errors
        4. Count of Improvements
        5. Count of Strengths
        6. Overall Breakdown
        7. Overall misconceptions students have
        8. Overall coding errors students have
        9. Overall improvements students need
        10. Overall strengths students have
        11. Correct implementation
        """

        response = self.client.responses.create(
        model="gpt-4o",
        instructions=system_message,
        input=prompt,
        )

        return response

@app.get("/healthcheck")
async def healthcheck():
    return "Ok"

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Uploaded file must be a .zip archive.")

    try:
        # Read file bytes and open as zip
        contents = await file.read()
        zip_file = zipfile.ZipFile(io.BytesIO(contents))

        files_content = []

        for name in zip_file.namelist():
            with zip_file.open(name) as f:
                try:
                    text = f.read().decode('utf-8')  # assuming UTF-8 encoding
                    files_content.append({
                        "filename": name,
                        "content": text
                    })
                except UnicodeDecodeError:
                    # Skip non-text files
                    files_content.append({
                        "filename": name,
                        "content": "Could not decode file as text"
                    })

        llmAgent = LLMAnalyzer(student_submissions=files_content)
        return llmAgent.getMisconceptionsResponse()

    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid ZIP file.")
