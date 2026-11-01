"""Error translation.

Every failure leaves the API as an RFC 9457 problem document with a stable machine
code. Two rules matter:

* An AgriError carries its own code and HTTP status, so the mapping is data, not
  a chain of isinstance checks that drifts from the error taxonomy.
* An unexpected exception becomes a 500 with a generic message. The detail is
  logged with the request id; it is never returned to the caller, because
  internal messages leak file paths, credentials and query fragments.
"""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from agri.common.errors import AgriError
from agri.common.logging import get_logger

_log = get_logger("api.errors")


def problem(
    *,
    status_code: int,
    title: str,
    detail: str,
    code: str,
    request_id: str | None = None,
    context: dict | None = None,
) -> dict:
    """Build an RFC 9457 problem document."""
    body = {
        "type": "about:blank",
        "title": title,
        "status": status_code,
        "code": code,
        "detail": detail,
    }
    if request_id:
        body["request_id"] = request_id
    if context:
        # Never echo anything that looks like a secret.
        body["context"] = {
            key: value
            for key, value in context.items()
            if not any(token in key.lower() for token in ("key", "secret", "token", "password"))
        }
    return body


def install_error_handlers(app: FastAPI) -> None:
    """Attach the handlers. Called once during app construction."""

    @app.exception_handler(AgriError)
    async def handle_agri_error(request: Request, exc: AgriError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        _log.warning(
            "agri error",
            extra={"code": exc.code, "path": request.url.path, "status": exc.http_status},
        )
        return JSONResponse(
            status_code=exc.http_status,
            content=problem(
                status_code=exc.http_status,
                title=exc.code.replace("_", " ").title(),
                detail=exc.message,
                code=exc.code,
                request_id=request_id,
                context=exc.context,
            ),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        errors = []
        for error in exc.errors():
            location = ".".join(str(part) for part in error.get("loc", ()) if part != "body")
            errors.append({"field": location or "body", "message": error.get("msg", "invalid")})
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=problem(
                status_code=422,
                title="Unprocessable Entity",
                detail="The request body failed validation.",
                code="request_validation_failed",
                request_id=request_id,
                context={"errors": errors},
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        detail = exc.detail
        if isinstance(detail, dict):
            # Raised by our own security layer, already a problem document.
            return JSONResponse(
                status_code=exc.status_code, content=detail, headers=getattr(exc, "headers", None)
            )
        return JSONResponse(
            status_code=exc.status_code,
            content=problem(
                status_code=exc.status_code,
                title=str(detail),
                detail=str(detail),
                code=f"http_{exc.status_code}",
                request_id=request_id,
            ),
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        # Log the real detail; return a generic one.
        _log.exception(
            "unhandled exception",
            extra={"path": request.url.path, "exception_type": type(exc).__name__},
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=problem(
                status_code=500,
                title="Internal Server Error",
                detail="An unexpected error occurred. Quote the request id when reporting it.",
                code="internal_error",
                request_id=request_id,
            ),
        )
