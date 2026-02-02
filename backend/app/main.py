from fastapi import FastAPI
from .config import settings
from .routes import roadmap_routes

app = FastAPI(title="DSA Backend (FastAPI)", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok", "env": settings.env}


app.include_router(roadmap_routes.router, prefix="/api")
