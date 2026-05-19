
# Automated Medical Case Review System

This project is an automated medical case review web application powered by LangGraph and OpenAI LLMs. It allows users to upload CSV files of medical cases and receive automated reviews, risk assessments, and recommendations for each case.

## Features
- Upload CSV files containing medical case data
- Automated review, risk classification, and recommendations using LLMs
- Simple web interface built with FastAPI and Jinja2 templates
- Synthetic data generation for testing
- Dockerized for easy deployment

## Project Structure
```
medical_case_review-app/
├── case_review_workflow.py   # Workflow logic and LLM integration
├── main.py                  # FastAPI web server
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker image definition
├── docker-compose.yml       # Docker Compose setup
├── data/
│   └── synthetic_cases.csv  # Example synthetic data
├── static/
│   └── style.css            # CSS styles
├── templates/
│   ├── index.html           # Upload form
│   └── results.html         # Results page
└── README.md                # Project documentation
```

## Getting Started

### Prerequisites
- Docker and Docker Compose installed

### Build and Run with Docker
1. Clone this repository.
2. Open a terminal in the project directory.
3. Run:
   ```
   docker-compose up --build
   ```
4. Open your browser and go to [http://localhost:8000](http://localhost:8000)

### Usage
- Upload a CSV file with medical cases or use the provided synthetic data.
- Click **Review Cases** to receive automated reviews and recommendations.

## CSV Format
The CSV file should have the following columns:
- `patient_id`
- `age`
- `sex`
- `symptoms`
- `diagnosis`
- `lab_result`
- `doctor_notes`

## Development
- Python 3.11 recommended
- See `requirements.txt` for dependencies
- Main logic in `case_review_workflow.py` and `main.py`

## Architecture & Flow
- The backend uses a LangGraph workflow to process each case:
  1. Summarize the case
  2. Evaluate risk
  3. Generate recommendations
  4. Feedback loop for high-risk cases
- The frontend allows file upload and synthetic data download.

