
# Medical Case Review Workflow (LangGraph)
# This module defines the workflow for automated medical case review using LangGraph and LLMs.

from langgraph.graph import StateGraph, END, START  # Workflow graph utilities
from typing import TypedDict, Annotated, List       # Type hints
from pydantic import BaseModel, Field              # Data validation
import operator
from langchain_openai import ChatOpenAI            # LLM interface
from langchain_core.prompts import ChatPromptTemplate  # Prompt templates
from dotenv import load_dotenv                     # For environment variables
load_dotenv()  # Load .env file if present


# Initialize the language model (OpenAI GPT-4o-mini)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


# Data model for a medical case
class Case(BaseModel):
    patient_id: str
    age: int
    sex: str
    symptoms: str
    diagnosis: str
    lab_result: float
    doctor_notes: str


# Data model for the review result
class ReviewResult(BaseModel):
    patient_id: str
    summary: str
    risk_level: str
    recommendations: str


# State dictionary for workflow steps
class CaseState(TypedDict):
    case: Case
    review: ReviewResult
    feedback: str
    n: int


# Prompt templates for each LLM step
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a medical reviewer. Summarize the case and highlight key findings."),
    ("user", "Case: {case}")
])
risk_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a risk evaluator. Classify the risk as low, moderate, or high and explain why."),
    ("user", "Summary: {summary}\nDiagnosis: {diagnosis}\nLab: {lab_result}")
])
recommend_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a senior physician. Give recommendations for next steps in management."),
    ("user", "Summary: {summary}\nRisk: {risk_level}\nNotes: {doctor_notes}")
])


# LLM pipelines for each review step
summary_pipe = summary_prompt | llm
risk_pipe = risk_prompt | llm
recommend_pipe = recommend_prompt | llm


# Step 1: Review a single case using LLMs
def review_case(state: CaseState):
    case = state["case"]
    # Generate summary
    summary = summary_pipe.invoke({"case": case.dict()}).content
    # Evaluate risk
    risk = risk_pipe.invoke({"summary": summary, "diagnosis": case.diagnosis, "lab_result": case.lab_result}).content
    # Extract risk level from LLM output
    risk_level = "low" if "low" in risk.lower() else ("high" if "high" in risk.lower() else "moderate")
    # Generate recommendations
    recommendations = recommend_pipe.invoke({"summary": summary, "risk_level": risk_level, "doctor_notes": case.doctor_notes}).content
    # Return review result
    return {"review": ReviewResult(patient_id=case.patient_id, summary=summary, risk_level=risk_level, recommendations=recommendations)}


# Step 2: Feedback loop (could be extended for real feedback)
def feedback_loop(state: CaseState):
    # For demonstration, just increment n and echo feedback
    return {"n": state.get("n", 0) + 1, "feedback": state.get("feedback", "")}


# Step 3: Route review based on feedback and risk
def route_review(state: CaseState, iteration_limit: int = 2):
    # Accept after 2 iterations or if risk is not high
    if state.get("n", 0) >= iteration_limit or state["review"].risk_level != "high":
        return "Accepted"
    else:
        return "Needs Review"


# Build the workflow graph
graph = StateGraph(CaseState)
graph.add_node("review_case", review_case)         # Add review node
graph.add_node("feedback_loop", feedback_loop)     # Add feedback node
graph.add_edge(START, "review_case")               # Start -> review
graph.add_edge("review_case", "feedback_loop")     # Review -> feedback
graph.add_conditional_edges(
    "feedback_loop",
    lambda state: route_review(state),
    {"Accepted": END, "Needs Review": "review_case"}  # Route based on review
)
# Compile the workflow for execution
case_review_workflow = graph.compile()


# Utility: Run the workflow on all cases in a CSV file
def run_review_on_csv(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    results = []
    for _, row in df.iterrows():
        case = Case(**row)
        state = {"case": case, "n": 0, "feedback": ""}
        result = case_review_workflow.invoke(state)
        results.append(result["review"])
    return results


# Utility: Generate synthetic cases and save to CSV
def read_synthetic_cases(num_cases=20, path="data/synthetic_cases.csv"):
    fake = Faker()
    data = []
    for _ in range(num_cases):
        case = {
            "patient_id": fake.uuid4(),
            "age": random.randint(1, 99),
            "sex": random.choice(["M", "F"]),
            "symptoms": fake.sentence(nb_words=6),
            "diagnosis": random.choice(["Flu", "COVID-19", "Diabetes", "Hypertension", "Asthma"]),
            "lab_result": round(random.uniform(0.5, 10.0), 2),
            "doctor_notes": fake.text(max_nb_chars=100)
        }
        data.append(case)
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    return path


# Main block for standalone testing
if __name__ == "__main__":
    # Test with synthetic data in data/ folder
    csv_path = read_synthetic_cases(path="data/synthetic_cases.csv")
    reviews = run_review_on_csv(csv_path)
    for r in reviews[:3]:
        print(r)
