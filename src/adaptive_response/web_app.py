from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .mission_control import MissionControlSession

WEB_DIR = Path(__file__).with_name("web")
app = FastAPI(title="Grab the Crab — Adaptive First-Response Mission Control")
session = MissionControlSession()
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


class ResetRequest(BaseModel):
    seed: int | None = None
    case_id: str | None = None


class DeployRequest(BaseModel):
    site_id: str
    effort: int


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/state")
def state() -> dict:
    return session.snapshot()


@app.get("/api/cases")
def cases() -> dict:
    return session.case_library()


@app.post("/api/reset")
def reset(request: ResetRequest) -> dict:
    return _call(
        lambda: session.reset(
            seed=request.seed,
            case_id=request.case_id,
        )
    )


@app.post("/api/deploy")
def deploy(request: DeployRequest) -> dict:
    return _call(
        lambda: session.deploy(
            site_id=request.site_id,
            effort=request.effort,
        )
    )


# Backwards-compatible rehearsal endpoints. The interactive UI uses /api/deploy.
@app.post("/api/plan")
def plan() -> dict:
    return _call(session.plan)


@app.post("/api/execute")
def execute() -> dict:
    return _call(session.execute)


@app.post("/api/reveal")
def reveal() -> dict:
    return _call(session.reveal)


def _call(fn):
    try:
        return fn()
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def main() -> None:
    import uvicorn

    uvicorn.run(
        "adaptive_response.web_app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
