"""One error taxonomy for the whole app. Every member carries its own
http_status, so there's a single source of truth per error — no separate
CODE->HTTP lookup table to keep in sync. Every message names the fix; none
ever contains the API key or the raw upstream response body.
"""


class AppError(Exception):
    code: str = "internal_error"
    http_status: int = 500

    def __init__(self, message: str, *, retry_after_seconds: int | None = None):
        super().__init__(message)
        self.message = message
        self.retry_after_seconds = retry_after_seconds


class LLMNotConfigured(AppError):
    code = "llm_not_configured"
    http_status = 503

    def __init__(self):
        super().__init__(
            "No AI provider is configured. Set LLM_API_KEY in .env (see README), "
            "or set LLM_PROVIDER=demo to use the offline provider."
        )


class InvalidApiKey(AppError):
    code = "invalid_key"
    http_status = 502

    def __init__(self):
        super().__init__(
            "The configured LLM_API_KEY was rejected by the provider. "
            "Check the key at https://openrouter.ai/keys."
        )


class ProviderUnreachable(AppError):
    code = "provider_unreachable"
    http_status = 502

    def __init__(self, base_url: str, provider: str):
        fix = (
            "Is Ollama running? Start it with `ollama serve`."
            if provider == "ollama"
            else "Check your network connection and LLM_BASE_URL."
        )
        super().__init__(f"Could not reach the AI provider at {base_url}. {fix}")


class ModelNotAvailable(AppError):
    code = "model_not_available"
    http_status = 502

    def __init__(self, model: str, provider: str):
        fix = (
            f"Run `ollama pull {model}`, or set OLLAMA_MODEL to a model you have."
            if provider == "ollama"
            else "Check LLM_MODEL against https://openrouter.ai/models."
        )
        super().__init__(f"The provider has no model named '{model}'. {fix}")


class RateLimited(AppError):
    code = "rate_limited"
    http_status = 429

    def __init__(self, retry_after_seconds: int | None = None):
        wait = (
            f" Retry after {retry_after_seconds}s."
            if retry_after_seconds is not None
            else " Retry shortly."
        )
        super().__init__(
            "The AI provider's free-tier rate limit was reached." + wait,
            retry_after_seconds=retry_after_seconds,
        )


class UpstreamError(AppError):
    code = "upstream_error"
    http_status = 502

    def __init__(self, detail: str = ""):
        suffix = f" ({detail})" if detail else ""
        super().__init__(f"The AI provider returned an error{suffix}. Try again in a moment.")


class ConversationNotFound(AppError):
    code = "conversation_not_found"
    http_status = 404

    def __init__(self, conversation_id: str):
        super().__init__(f"No conversation found with id '{conversation_id}'.")


class NothingToRetry(AppError):
    code = "nothing_to_retry"
    http_status = 400

    def __init__(self, conversation_id: str):
        super().__init__(
            f"Conversation '{conversation_id}' has no failed or interrupted reply to retry."
        )


def error_envelope(exc: AppError, request_id: str | None = None) -> dict:
    return {
        "error": {
            "code": exc.code,
            "message": exc.message,
            "request_id": request_id,
            "retry_after_seconds": exc.retry_after_seconds,
        }
    }
