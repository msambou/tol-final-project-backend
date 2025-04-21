from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import zipfile
import io
from openai import OpenAI
import os
from dotenv import load_dotenv
import re
import json
from fastapi.encoders import jsonable_encoder
from sqlalchemy.future import select

from worker.models.base import Base
from worker.database.database import engine

from worker.models.models import Analysis
from worker.database.database import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends

from collections import Counter
from fastapi.middleware.cors import CORSMiddleware



load_dotenv()


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["*"] to allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # allows all HTTP methods
    allow_headers=["*"],  # allows all headers
)

@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

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
        12. Topic for The Problem  
        13. Student Scores out of hundred as list in multiples of 10
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

        # Convert all keys to lowercase
        data = {k.lower(): v for k, v in data.items()}
        
        # Rename keys so client can easily read values
        # Replace spaces with underscores in all keys
        data = {k.replace(" ", "_"): v for k, v in data.items()}

        # convert scores to array of frequencies
        raw_scores = data["student_scores_out_of_hundred_as_list_in_multiples_of_10"]

        scores = re.findall(r"-\s*Student\s*\d+:\s*(\d+)", raw_scores)

        # Count frequency of each score
        score_counts = Counter(map(int, scores))

        # Convert to desired format
        result = [{"score": score, "count": count} for score, count in score_counts.items()]

        data["student_scores_out_of_hundred_as_list_in_multiples_of_10"] = result
        return data


@app.get("/healthcheck")
async def healthcheck():
    return "Ok"

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Uploaded file must be a .zip archive.")

    try:
        contents = await file.read()
        zip_file = zipfile.ZipFile(io.BytesIO(contents))

        files_content = []

        for name in zip_file.namelist():
            with zip_file.open(name) as f:
                try:
                    text = f.read().decode('utf-8')
                    files_content.append({"filename": name, "content": text})
                except UnicodeDecodeError:
                    files_content.append({"filename": name, "content": "Could not decode file as text"})

        llmAgent = LLMAnalyzer(student_submissions=files_content)
        analysis_data = llmAgent.getMisconceptionsResponse()

        # Use the 'goal' value as the topic
        topic = analysis_data.get("topic_for_the_problem", "Untitled Topic")

        # Save to database
        new_analysis = Analysis(topic=topic, response=json.dumps(analysis_data))
        db.add(new_analysis)
        await db.commit()
        await db.refresh(new_analysis)

        return {"analysis_id": new_analysis.id, "analysis_data": analysis_data}

    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid ZIP file.")


@app.get("/analyses")
async def get_all_analyses(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Analysis).order_by(Analysis.created_at.desc()))
    analyses = result.scalars().all()
    return jsonable_encoder(analyses)



@app.get("/analyses/latest")
async def get_latest_analysis(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Analysis).order_by(Analysis.created_at.desc()).limit(1)
    )
    latest = result.scalars().first()
    
    if not latest:
        raise HTTPException(status_code=404, detail="No analysis found.")
    
    return jsonable_encoder(latest)

@app.get("/analyses/{analysis_id}")
async def get_analysis_by_id(analysis_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Analysis).where(Analysis.id == analysis_id))
    analysis = result.scalars().first()

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found.")

    return jsonable_encoder(analysis)