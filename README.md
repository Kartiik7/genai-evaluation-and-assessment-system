# GenAI Evaluation & Assessment System

A Python research project for automated short-answer grading using classical NLP, machine learning, and GenAI models.

## Project Structure
```text
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   └── __init__.py
├── results/
├── prompts/
├── .env
├── .gitignore
├── README.md
└── requirements.txt
```

## Virtual Environment Setup

### 1. Create Virtual Environment
Using `venv` in standard Python:

**Windows (PowerShell / Command Prompt):**
```powershell
python -m venv venv
```

**macOS / Linux:**
```bash
python3 -m venv venv
```

### 2. Activate Virtual Environment

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
.\venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
