@echo off
echo Starting RAG Backend...
start "RAG Backend" cmd /k "cd backend && .\.venv\Scripts\activate.bat && python -m uvicorn app.main:app --reload --port 8000"

echo Starting RAG Frontend...
start "RAG Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo Both servers are starting in new windows!
echo - Backend: http://127.0.0.1:8000
echo - Frontend: http://localhost:5173
echo.
pause