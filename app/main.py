import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.chat import ChatService
from app.config import Settings, get_settings
from app.errors import AppError, error_envelope
from app.llm import build_llm
from app.repository import Repository, create_client
from app.routes import router

# Uvicorn only configures its own loggers, so without this the app's lines
# reach stderr through logging's last-resort handler, bare and level-less,
# easy to mistake for a stray print. The format mirrors uvicorn's so the two
# read as one log stream.
logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(message)s")

logger = logging.getLogger("app")


def _log_provider(llm, settings: Settings) -> None:
    """First thing in the logs on `docker compose up`. Somebody who brings this
    stack up without a key should learn that from the very first screenful,
    not from a reply that sounds oddly canned ten minutes later. Never logs
    the key itself, only whether one is present.
    """
    if llm.name == "demo":
        logger.warning(
            "Offline demo provider active, no real model is being called. Set "
            "LLM_API_KEY for OpenRouter, or LLM_PROVIDER=ollama for a local model."
        )
    elif not settings.llm_configured:
        logger.warning(
            "LLM_PROVIDER=openrouter but LLM_API_KEY is unset, so every message will "
            "fail with llm_not_configured. Set the key, or use LLM_PROVIDER=demo."
        )
    elif llm.name == "ollama":
        # Names the endpoint: "connection refused" against a URL you can see is
        # a much shorter debugging session than against one you can't.
        logger.info("Using Ollama at %s with model %s.", settings.ollama_base_url, llm.model)
    else:
        logger.info("Using %s with model %s.", llm.name, llm.model)


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
        _log_provider(app.state.llm, settings)

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
        # exc.errors() only: str(exc) includes an internal traceback with
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
    # No static mount: the UI is streamlit_app.py, a separate process that
    # talks to this API over HTTP. This app now serves the API only:
    # `/` 404s, `/docs` is FastAPI's own Swagger UI.
    return app


app = create_app()
