# Project Setup Instructions

This document outlines the steps to get the project up and running locally.

## Prerequisites

- Python
- PostgreSQL server installed and running
- Git
- [Gemini API key](https://developers.gemini.com/) 

## 1. Create and Activate Virtual Environment

```bash
# Create a new virtual environment named .venv
python3 -m venv .venv

# Activate the environment (Linux/macOS)
source .venv/bin/activate

# On Windows (PowerShell)
.venv\Scripts\Activate.ps1
```  

## 2. Install Dependencies

With the virtual environment activated, install all required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Setup PostgreSQL Database

1. Update the `db.py` file with your database connection details, for example:
   ```python
   DATABASE_URL = "postgresql://username:password@localhost:5432/your_database_name"
   ```

## 4. Configure Environment Variables

1. Its preferred that you add your own Gemini API key, mine is still there. Update `vector_search.py` with your gemini api, note that `gemini.py` is just an example file. Run this to test the Gemini API.
   ```ini
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

## 5. Run the Application

Start the FastAPI server with live reload:

```bash
uvicorn main:app --reload
```

The server will run at:  
```
http://localhost:8000
```

## 6. Initialize or Reindex the Qdrant Collection

> **Important:** Qdrant is running in memory. Each time you stop the server, the index is cleared. You must reindex after every restart.

1. Open the interactive docs:
   ```
   http://localhost:8000/docs#/
   ```
2. Find the **`/product/reindex/`** endpoint (POST).
3. Click **`Try it out`** then **`Execute`**.  
   No request body is needed.

## 7. Search via Frontend

Once reindexed, you can perform searches using the `search.html` file:

1. Open `search.html` in your browser.
2. Enter a query and submit to view results.

---

Follow these steps each time you clone the repo or restart the server to ensure your local environment and search index are up-to-date.

