
---

# Project Setup Instructions

This document outlines the steps to set up and run the project locally.

---

## Prerequisites

- Python installed
- PostgreSQL server installed and running
- Redis server installed and running
- A local embedding generation model  
  *(We used [BAAI/bge-small-en](https://huggingface.co/BAAI/bge-small-en))*
- Git installed

---

## 1. Create and Activate Virtual Environment

```bash
# Create a new virtual environment
python3 -m venv .venv

# Activate the environment (Linux/macOS)
source .venv/bin/activate

# Activate on Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

---

## 2. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Setup PostgreSQL Database

```bash
# Access PostgreSQL
psql postgres

# Create a new database and user
CREATE DATABASE your_db_name;
CREATE USER your_user_name WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE your_db_name TO your_user_name;
```

Update your `.env` file with the correct database URL.

Example:
```ini
DATABASE_URL = "postgresql://your_user_name:your_password@localhost:5432/your_db_name"
```

---

## 4. Setup Redis Server

```bash
# Start Redis server (Linux/macOS)
redis-server

```

Update your `.env` file:

```ini
REDIS_URL = "redis://localhost:6379"
```

---

## 5. Setup Local Embedding Model

- Download the [BAAI/bge-small-en](https://huggingface.co/BAAI/bge-small-en) model.
- Save it locally (for example: `/home/user/models/bge-small-en`).
- Update your `.env` file:

```ini
MODEL_PATH = "/path/to/your/downloaded/model"
```

---

## 6. Configure Environment Variables

Create or update a `.env` file at the project root:

```ini
# Example .env file

DATABASE_URL = "postgresql://your_user_name:your_password@localhost:5432/your_db_name"
REDIS_URL = "redis://localhost:6379"
MODEL_PATH = "/path/to/your/downloaded/model"
```

---

## 7. Run the Application

Start the FastAPI server with live reload:

```bash
uvicorn main:app --reload
```

The server will be accessible at:  
[http://localhost:8000](http://localhost:8000)

---


## 8. Initialize or Reindex the Qdrant Collection

> **Important:** You must reindex after each server restart.

Steps:
1. Open the FastAPI interactive docs at:  
   [http://localhost:8000/docs#/](http://localhost:8000/docs#/)
   
2. Find the `POST /product/reindex/` endpoint.

3. Click **Try it out** ➔ **Execute** (no request body needed).

---

## 9. Search via Frontend

1. Open `search.html` in your browser.
2. Enter a query and submit to view search results.

---



