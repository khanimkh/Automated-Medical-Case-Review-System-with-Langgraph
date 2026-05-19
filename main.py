
# Import necessary modules and types
from typing import TypedDict, Annotated, List
from case_review_workflow import case_review_workflow, CaseState, Case
# Automated Medical Case Review System - Backend
# Combines orchestrator-worker and reflection patterns using LangGraph
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app and Jinja2 templates
app = FastAPI()
templates = Jinja2Templates(directory="templates")
# Mount static files (CSS, JS, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Home page route: renders the upload form
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"request": request})

# Review cases route: handles CSV upload, runs workflow, and renders results
@app.post("/review", response_class=HTMLResponse)
def review_cases(request: Request, file: UploadFile = File(...)):
    import pandas as pd  # Local import for isolation
    # Read uploaded CSV file into DataFrame
    df = pd.read_csv(file.file)
    results = []
    # Iterate through each case in the CSV
    for _, row in df.iterrows():
        # Create a Case object from the row
        case = Case(**row)
        # Initialize state for the workflow
        state = {"case": case, "n": 0, "feedback": ""}
        # Run the case through the review workflow
        result = case_review_workflow.invoke(state)
        # Collect the review result
        results.append(result["review"])
    # Render results in the results.html template
    return templates.TemplateResponse(
        request=request,
        name="results.html",
        context={"request": request, "results": results}
    )


