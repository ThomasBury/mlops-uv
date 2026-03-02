"""Request/response logging middleware utilities."""

import json
import logging
import time
from typing import Any, Callable
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import Message


class RouterLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware that records request/response metadata for each route call."""

    def __init__(self, app: FastAPI, *, logger: logging.Logger) -> None:
        """Initialize middleware with a target logger.

        Args:
            app: FastAPI application instance.
            logger: Logger used to write structured event payloads.
        """
        self._logger = logger
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process the request and emit a structured log event.

        Args:
            request: Incoming request object.
            call_next: Callback that invokes the next middleware/route.

        Returns:
            Response: Outgoing response from the wrapped application.
        """
        request_id: str = str(uuid4())
        logging_dict = {"X-API-REQUEST-ID": request_id}

        await self.set_body(request)
        response, response_dict = await self._log_response(
            call_next, request, request_id
        )
        request_dict = await self._log_request(request)
        logging_dict["request"] = request_dict
        logging_dict["response"] = response_dict

        self._logger.info(logging_dict)

        return response

    async def set_body(self, request: Request) -> None:
        """Store request body bytes so body consumers can read them multiple times.

        Args:
            request: Incoming request object.
        """
        receive_ = await request._receive()

        async def receive() -> Message:
            return receive_

        request._receive = receive

    async def _log_request(self, request: Request) -> dict[str, Any]:
        """Extract request fields for structured logging.

        Args:
            request: Incoming request object.

        Returns:
            dict[str, Any]: Request metadata including method, path, and optional
            JSON body.
        """
        path = request.url.path
        if request.query_params:
            path += f"?{request.query_params}"

        request_logging: dict[str, Any] = {
            "method": request.method,
            "path": path,
            "ip": request.client.host,
        }

        try:
            body = await request.json()
            request_logging["body"] = body
        except ValueError as exc:
            request_logging["error"] = f"Invalid JSON: {exc}"
        except Exception as exc:
            request_logging["error"] = f"Unexpected error: {exc}"

        return request_logging

    async def _log_response(
        self, call_next: Callable, request: Request, request_id: str
    ) -> tuple[Response, dict[str, Any]]:
        """Capture response metrics and body payload for logging.

        Args:
            call_next: Callback that invokes the next middleware/route.
            request: Incoming request object.
            request_id: Request identifier propagated in response headers.

        Returns:
            tuple[Response, dict[str, Any]]: Response object and logging payload.
        """
        start_time = time.perf_counter()
        response = await self._execute_request(call_next, request, request_id)
        finish_time = time.perf_counter()

        overall_status = "successful" if response.status_code < 400 else "failed"
        execution_time = finish_time - start_time

        response_logging: dict[str, Any] = {
            "status": overall_status,
            "status_code": response.status_code,
            "time_taken": f"{execution_time:0.4f}s",
        }

        resp_body = [section async for section in response.__dict__["body_iterator"]]
        response.__setattr__("body_iterator", AsyncIteratorWrapper(resp_body))

        try:
            parsed_body = json.loads(resp_body[0].decode())
        except (ValueError, IndexError, AttributeError) as exc:
            response_logging["error"] = f"Error processing response body: {exc}"
            parsed_body = str(resp_body)
        except Exception as exc:
            response_logging["error"] = f"Unexpected error: {exc}"
            parsed_body = str(resp_body)

        response_logging["body"] = parsed_body

        return response, response_logging

    async def _execute_request(
        self, call_next: Callable, request: Request, request_id: str
    ) -> Response:
        """Execute the downstream request handler and inject request ID header.

        Args:
            call_next: Callback that invokes the next middleware/route.
            request: Incoming request object.
            request_id: Correlation ID for this request.

        Returns:
            Response: Response returned by the downstream handler.

        Raises:
            Exception: Propagates any error raised by the downstream handler.
        """
        try:
            response: Response = await call_next(request)
            response.headers["X-API-Request-ID"] = request_id
            return response
        except Exception as exc:
            self._logger.exception(
                {"path": request.url.path, "method": request.method, "reason": exc}
            )
            raise


class AsyncIteratorWrapper:
    """Adapt a synchronous iterable into an asynchronous iterator interface."""

    def __init__(self, obj):
        """Initialize wrapper around an iterable object.

        Args:
            obj: Iterable object consumed one item at a time.
        """
        self._it = iter(obj)

    def __aiter__(self):
        """Return the asynchronous iterator instance."""
        return self

    async def __anext__(self):
        """Return next item from the wrapped iterable.

        Returns:
            Any: Next item from the wrapped iterable.

        Raises:
            StopAsyncIteration: When the wrapped iterable is exhausted.
        """
        try:
            value = next(self._it)
        except StopIteration as exc:
            raise StopAsyncIteration from exc
        return value
