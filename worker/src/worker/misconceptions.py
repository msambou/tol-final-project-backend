from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import zipfile
import io
from openai import OpenAI
import os
from dotenv import load_dotenv
import re

load_dotenv()


app = FastAPI()

class LLMAnalyzer:
    def __init__(self, student_submissions):
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        self.student_submissions = student_submissions

    def getMisconceptionsResponse(self):
        # Build a code block from all the student submissions
        student_code_snippets = "\n\n".join(
            f"# File: {submission['filename']}\n{submission['content']}"
            for submission in self.student_submissions
            if isinstance(submission['content'], str)
        )

        # System message tells the model what role to play
        system_message = (
            "You are an expert programming tutor that analyzes student code submissions."
        )

        # Prompt now dynamically includes the extracted code
        prompt = f"""The following are student codes:

        {student_code_snippets}

        Please analyze their submissions and respond using the following structure:
        1. The goal of the coding assignment  
        2. Overall Count of Misconceptions  
        3. Overall Count of Coding Errors  
        4. Overall Count of Improvements  
        5. Overall Count of Strengths  
        6. Overall Breakdown  
        7. Overall misconceptions students have  
        8. Overall coding errors students have  
        9. Overall improvements students need  
        10. Overall strengths students have  
        11. Correct implementation  
        """

        # Call to OpenAI model
        response = self.client.responses.create(
            model="gpt-4o",
            instructions=system_message,
            input=prompt,
        )

       
        return self.extract_analysis_data(response.output[0].content[0].text)


    def extract_analysis_data(self, text):
        data = {}

        # Define expected keys and their regex-friendly names
        keys = [
            "The Goal of the Coding Assignment",
            "Overall Count of Misconceptions",
            "Overall Count of Coding Errors",
            "Overall Count of Improvements",
            "Overall Count of Strengths",
            "Overall Breakdown",
            "Overall Misconceptions Students Have",
            "Overall Coding Errors Students Have",
            "Overall Improvements Students Need",
            "Overall Strengths Students Have",
            "Correct Implementation",
        ]

        # Build a regex pattern to split the sections
        pattern = r"\d+\.\s+\*\*(.*?)\*\*"

        # Find the matches (headers)
        matches = list(re.finditer(pattern, text))

        for i, match in enumerate(matches):
            key = match.group(1).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            value = text[start:end].strip()

            # Clean up value if needed
            value = re.sub(r"^\n+", "", value)  # Remove leading newlines
            data[key] = value

        return data


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
