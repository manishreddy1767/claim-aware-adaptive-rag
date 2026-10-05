"""HTTP API and static frontend (Starlette).

Routes (JSON unless noted):
    GET    /api/health                  model status (no login needed)
    GET    /api/auth/config             whether sign-up is open
    POST   /api/auth/register           {username, password} -> signs in
    POST   /api/auth/login              {username, password}
    POST   /api/auth/logout
    GET    /api/auth/me
    GET    /api/documents
    POST   /api/documents               multipart: file
    DELETE /api/documents/{id}
    POST   /api/ask                     {question}
    POST   /api/verify                  {text}
    GET    /api/history
    DELETE /api/history
    GET    /                            the single-page frontend
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from ..config import RAGConfig
from ..models import resolve_device
from .auth import SESSION_COOKIE, SESSION_TTL_S, Accounts, AuthError
from .store import Store
from .workspace import MAX_UPLOAD_BYTES, Workspaces, WorkspaceError

logger = logging.getLogger(__name__)

STATIC = Path(__file__).resolve().parent / "static"


@dataclass
class Settings:
    data_dir: Path
    allow_signup: bool = True
    warm_up: bool = True               # load models in the background at start-up
    config: RAGConfig = field(default_factory=RAGConfig)


def _error(message: str, status: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


async def _json(request: Request) -> dict:
    try:
        data = await request.json()
    except ValueError:
        data = None
    if not isinstance(data, dict):
        raise WorkspaceError("Expected a JSON object.")
    return data


class SecurityMiddleware(BaseHTTPMiddleware):
    """Rejects cross-site state-changing requests and sets protective headers."""

    async def dispatch(self, request: Request, call_next):
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin")
            if origin and urlsplit(origin).netloc != request.headers.get("host"):
                return _error("Cross-site requests are not allowed.", 403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; "
            "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response


def create_app(settings: Settings) -> Starlette:
    data_dir = Path(settings.data_dir)
    store = Store(data_dir / "carag.db")
    accounts = Accounts(store)
    workspaces = Workspaces(store, data_dir, settings.config)

    def signed_in(request: Request):
        return accounts.user_for(request.cookies.get(SESSION_COOKIE))

    def with_session(response: JSONResponse, token: str) -> JSONResponse:
        response.set_cookie(SESSION_COOKIE, token, max_age=int(SESSION_TTL_S), httponly=True,
                            samesite="strict", path="/")
        return response

    def requires_user(handler):
        async def wrapped(request: Request):
            user = signed_in(request)
            if user is None:
                return _error("Please sign in.", 401)
            try:
                return await handler(request, user)
            except WorkspaceError as exc:
                return _error(str(exc), 400)
            except Exception:
                logger.exception("Request failed: %s %s", request.method, request.url.path)
                return _error("Something went wrong while processing the request. "
                              "See the server log for details.", 500)
        return wrapped

    # -- public ----------------------------------------------------------------------
    async def health(request: Request):
        ready = workspaces.models_ready.is_set()
        return JSONResponse({"status": "ok", "models_ready": ready, "model_error": workspaces.model_error,
                             "device": resolve_device(settings.config.models.device)})

    async def auth_config(request: Request):
        return JSONResponse({"allow_signup": settings.allow_signup or store.user_count() == 0,
                             "has_users": store.user_count() > 0})

    async def register(request: Request):
        # Sign-up can be closed with --no-signup, but the first account can always be created.
        if not settings.allow_signup and store.user_count() > 0:
            return _error("Creating new accounts is disabled on this installation.", 403)
        try:
            body = await _json(request)
            user = accounts.register(str(body.get("username", "")), str(body.get("password", "")))
        except (AuthError, WorkspaceError) as exc:
            return _error(str(exc), 400)
        return with_session(JSONResponse({"username": user.username}, status_code=201), accounts.start_session(user))

    async def login(request: Request):
        try:
            body = await _json(request)
            user = accounts.login(str(body.get("username", "")), str(body.get("password", "")))
        except (AuthError, WorkspaceError) as exc:
            return _error(str(exc), 401 if isinstance(exc, AuthError) else 400)
        return with_session(JSONResponse({"username": user.username}), accounts.start_session(user))

    async def logout(request: Request):
        user = signed_in(request)
        accounts.end_session(request.cookies.get(SESSION_COOKIE))
        if user is not None:
            workspaces.forget(user.id)
        response = JSONResponse({"ok": True})
        response.delete_cookie(SESSION_COOKIE, path="/")
        return response

    # -- signed in -------------------------------------------------------------------
    @requires_user
    async def me(request: Request, user):
        return JSONResponse({"username": user.username})

    @requires_user
    async def list_documents(request: Request, user):
        return JSONResponse({"documents": workspaces.documents(user.id)})

    @requires_user
    async def upload_document(request: Request, user):
        length = int(request.headers.get("content-length") or 0)
        if length > MAX_UPLOAD_BYTES + 64 * 1024:
            return _error(f"Files can be at most {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.", 413)
        form = await request.form()
        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            return _error("Choose a file to upload.", 400)
        data = await upload.read()
        doc = await _run(workspaces.add_document, user.id, upload.filename or "", data)
        return JSONResponse({"document": doc}, status_code=201)

    @requires_user
    async def delete_document(request: Request, user):
        try:
            await _run(workspaces.delete_document, user.id, request.path_params["doc_id"])
        except KeyError:
            return _error("Document not found.", 404)
        return JSONResponse({"ok": True})

    @requires_user
    async def ask(request: Request, user):
        body = await _json(request)
        return JSONResponse(await _run(workspaces.ask, user.id, str(body.get("question", ""))))

    @requires_user
    async def verify(request: Request, user):
        body = await _json(request)
        return JSONResponse(await _run(workspaces.verify, user.id, str(body.get("text", ""))))

    @requires_user
    async def history(request: Request, user):
        return JSONResponse({"history": store.history(user.id)})

    @requires_user
    async def clear_history(request: Request, user):
        store.clear_history(user.id)
        return JSONResponse({"ok": True})

    async def api_not_found(request: Request):
        return _error("Not found.", 404)

    routes = [
        Route("/api/health", health),
        Route("/api/auth/config", auth_config),
        Route("/api/auth/register", register, methods=["POST"]),
        Route("/api/auth/login", login, methods=["POST"]),
        Route("/api/auth/logout", logout, methods=["POST"]),
        Route("/api/auth/me", me),
        Route("/api/documents", list_documents),
        Route("/api/documents", upload_document, methods=["POST"]),
        Route("/api/documents/{doc_id}", delete_document, methods=["DELETE"]),
        Route("/api/ask", ask, methods=["POST"]),
        Route("/api/verify", verify, methods=["POST"]),
        Route("/api/history", history),
        Route("/api/history", clear_history, methods=["DELETE"]),
        Route("/api/{rest:path}", api_not_found, methods=["GET", "POST", "DELETE", "PUT", "PATCH"]),
        Mount("/", StaticFiles(directory=STATIC, html=True), name="static"),
    ]
    app = Starlette(routes=routes, middleware=[Middleware(SecurityMiddleware)])
    app.state.workspaces = workspaces
    app.state.store = store
    if settings.warm_up:
        threading.Thread(target=workspaces.warm_up, name="model-warm-up", daemon=True).start()
    else:
        workspaces.models_ready.set()
    return app


async def _run(func, *args):
    """Run blocking pipeline work in a worker thread so the server stays responsive."""
    return await run_in_threadpool(func, *args)
