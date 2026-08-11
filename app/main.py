from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.chat import ChatService
from app.config import Settings, get_settings
from app.errors import AppError, error_envelope
from app.llm import build_llm
from app.repository import Repository, create_client
from app.routes import router

STATIC_DIR = Path(__file__).parent / "static"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        client = create_client(settings.mongo_uri)
        repository = Repository(client, settings.mongo_db)
        await repository.ensure_indexes()

        app.state.repository = repository
        app.state.llm = build_llm(settings)
        app.state.chat_service = ChatService(app.state.repository, app.state.llm)

        yield

        await repository.close()

    app = FastAPI(title="chatbot-fastapi-mongo", lifespan=lifespan)

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request_id = request.headers.get("x-request-id", str(uuid4()))
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        return response

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        return JSONResponse(status_code=exc.http_status, content=error_envelope(exc, request_id))

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        # exc.errors() only — str(exc) includes an internal traceback with
        # file paths and line numbers, which has no business in a response body.
        details = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": details,
                    "request_id": request_id,
                    "retry_after_seconds": None,
                }
            },
        )

    app.include_router(router)
    # Mounted last: API routes always take precedence over static files —
    # same container, same origin, so no CORS and no second service.
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


app = create_app()
