"""HTTP API and static frontend (Starlette).

Routes (JSON unless noted):
    GET    /api/health                  model status (no login needed)
    GET    /api/auth/config             storage mode and whether sign-up is open
    POST   /api/auth/register           {username, password} -> signs in
    POST   /api/auth/login              {username, password}
    POST   /api/auth/logout
    GET    /api/auth/me                 username, storage mode, whether local files can be added
    GET    /api/documents
    POST   /api/documents               multipart: file
    POST   /api/documents/url           {url}
    POST   /api/documents/path          {path}   local mode, from this computer only
    DELETE /api/documents/{id}
    POST   /api/ask                     {question, style: "quotes" | "ai"}
    GET    /api/ai                      local LLM settings and Ollama status
    PUT    /api/ai                      {enabled, model}   local mode only
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
from starlette.formparsers import MultiPartParser
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from ..config import RAGConfig
from ..models import resolve_device
from .auth import SESSION_COOKIE, SESSION_TTL_S, Accounts, AuthError, token_hash
from .store import Store
from .workspace import MAX_UPLOAD_BYTES, Workspaces, WorkspaceError

logger = logging.getLogger(__name__)

STATIC = Path(__file__).resolve().parent / "static"
_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


@dataclass
class Settings:
    data_dir: Path
    # "local": the installed application, documents saved on this computer.
    # "web": a hosted website, documents kept in memory for one sign-in session only.
    mode: str = "local"
    allow_signup: bool = True
    warm_up: bool = True               # load models in the background at start-up
    idle_timeout_s: float = 2 * 3600   # web mode: drop a session's documents after this idle time
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


async def _run(func, *args):
    """Run blocking pipeline work in a worker thread so the server stays responsive."""
    return await run_in_threadpool(func, *args)


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
    if settings.mode not in ("local", "web"):
        raise ValueError(f"Unknown mode '{settings.mode}'; use 'local' or 'web'.")
    web = settings.mode == "web"
    if web:
        # Keep uploads in memory: by default the multipart parser spools files over 1 MB
        # to a temporary file on disk.
        MultiPartParser.spool_max_size = MAX_UPLOAD_BYTES + 1024 * 1024
    data_dir = Path(settings.data_dir)
    store = Store(data_dir / "carag.db")
    accounts = Accounts(store)
    workspaces = Workspaces(store, data_dir, settings.config, settings.mode, settings.idle_timeout_s)

    def signed_in(request: Request):
        return accounts.user_for(request.cookies.get(SESSION_COOKIE))

    def owner_of(request: Request, user):
        """Whose workspace a request uses: the account (local) or this sign-in session (web)."""
        return token_hash(request.cookies[SESSION_COOKIE]) if web else user.id

    def local_files_allowed(request: Request) -> bool:
        """Files can be added by path only in local mode, and only by someone at this computer."""
        return not web and request.client is not None and request.client.host in _LOOPBACK

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
                return await handler(request, user, owner_of(request, user))
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
                             "has_users": store.user_count() > 0, "mode": settings.mode})

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
        if user is not None:
            await _run(workspaces.forget, owner_of(request, user))
        accounts.end_session(request.cookies.get(SESSION_COOKIE))
        response = JSONResponse({"ok": True})
        response.delete_cookie(SESSION_COOKIE, path="/")
        return response

    # -- signed in -------------------------------------------------------------------
    @requires_user
    async def me(request: Request, user, owner):
        return JSONResponse({"username": user.username, "mode": settings.mode,
                             "local_files": local_files_allowed(request)})

    @requires_user
    async def list_documents(request: Request, user, owner):
        return JSONResponse({"documents": workspaces.documents(owner)})

    @requires_user
    async def upload_document(request: Request, user, owner):
        length = int(request.headers.get("content-length") or 0)
        if length > MAX_UPLOAD_BYTES + 64 * 1024:
            return _error(f"Files can be at most {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.", 413)
        form = await request.form()
        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            return _error("Choose a file to upload.", 400)
        data = await upload.read()
        await upload.close()
        doc = await _run(workspaces.add_document, owner, upload.filename or "", data)
        return JSONResponse({"document": doc}, status_code=201)

    @requires_user
    async def add_url(request: Request, user, owner):
        body = await _json(request)
        doc = await _run(workspaces.add_url, owner, str(body.get("url", "")))
        return JSONResponse({"document": doc}, status_code=201)

    @requires_user
    async def add_path(request: Request, user, owner):
        if not local_files_allowed(request):
            return _error("Files on this computer can only be added in the local application, "
                          "from the computer it runs on.", 403)
        body = await _json(request)
        return JSONResponse(await _run(workspaces.add_path, owner, str(body.get("path", ""))), status_code=201)

    @requires_user
    async def delete_document(request: Request, user, owner):
        try:
            await _run(workspaces.delete_document, owner, request.path_params["doc_id"])
        except KeyError:
            return _error("Document not found.", 404)
        return JSONResponse({"ok": True})

    @requires_user
    async def ask(request: Request, user, owner):
        body = await _json(request)
        question = str(body.get("question", ""))
        if body.get("style") == "ai":
            return JSONResponse(await _run(workspaces.ask_ai, owner, question))
        return JSONResponse(await _run(workspaces.ask, owner, question))

    @requires_user
    async def ai_settings(request: Request, user, owner):
        status = await _run(workspaces.ai_status)
        return JSONResponse({**status, "editable": not web})

    @requires_user
    async def save_ai_settings(request: Request, user, owner):
        if web:
            return _error("AI settings are managed by the website's operator.", 403)
        body = await _json(request)
        status = await _run(workspaces.save_ai_settings, bool(body.get("enabled")), str(body.get("model", "")))
        return JSONResponse({**status, "editable": True})

    @requires_user
    async def verify(request: Request, user, owner):
        body = await _json(request)
        return JSONResponse(await _run(workspaces.verify, owner, str(body.get("text", ""))))

    @requires_user
    async def history(request: Request, user, owner):
        return JSONResponse({"history": workspaces.history(owner)})

    @requires_user
    async def clear_history(request: Request, user, owner):
        workspaces.clear_history(owner)
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
        Route("/api/documents/url", add_url, methods=["POST"]),
        Route("/api/documents/path", add_path, methods=["POST"]),
        Route("/api/documents/{doc_id}", delete_document, methods=["DELETE"]),
        Route("/api/ask", ask, methods=["POST"]),
        Route("/api/ai", ai_settings),
        Route("/api/ai", save_ai_settings, methods=["PUT"]),
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
