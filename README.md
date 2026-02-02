# DSA Visualizer

DSA Visualizer is an interactive, canvas-first knowledge map for data structures and algorithms. Instead of presenting a linear list of topics, it renders patterns and relationships as a navigable graph and generates roadmap sequences of real problems that reinforce those patterns.

## Motivation
- Most DSA study resources are list-based and hide relationships between concepts.
- This project makes dependencies and thematic groupings visible on a single canvas.
- Roadmaps are generated from a curated problem database and are meant to guide, not enforce, a learning path.

## Architecture (high level)
- Frontend: React + TypeScript (Vite) with an embedded Excalidraw canvas and Tailwind-based UI.
- Backend: FastAPI + SQLAlchemy with a SQLite database for problems and pattern statistics.
- Storage: SQLite for curated problem metadata; browser localStorage for client-side progress and view state.

## How to run (minimal)
Backend (FastAPI):
- `cd backend`
- `python -m venv .venv && .venv\Scripts\activate` (or your preferred venv)
- `pip install -r requirements.txt`
- `uvicorn app.main:app --reload --port 3001`

Frontend (Vite):
- `cd frontend`
- `npm install`
- `npm run dev`

The frontend proxies `/api/*` to `http://localhost:3001` via `frontend/vite.config.ts`.

## Project status
Active development. Expect schema and UI changes, and note that some legacy or exploratory files remain in the repository (e.g., older DB helpers, planning docs). The codebase is being cleaned up progressively as functionality stabilizes.
