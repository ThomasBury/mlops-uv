import logging
import time
import json
from typing import Callable
from uuid import uuid4
from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import Message


class RouterLoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI, *, logger: logging.Logger) -> None:
        self._logger = logger
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id: str = str(uuid4())
        logging_dict = {
            "X-API-REQUEST-ID": request_id  # X-API-REQUEST-ID maps each request-response to a unique ID
        }

        await self.set_body(request)
        response, response_dict = await self._log_response(
            call_next, request, request_id
        )
        request_dict = await self._log_request(request)
        logging_dict["request"] = request_dict
        logging_dict["response"] = response_dict

        self._logger.info(logging_dict)

        return response

    async def set_body(self, request: Request):
        """Avails the response body to be logged within a middleware as,
        it is generally not a standard practice.

           Arguments:
           - request: Request
           Returns:
           - receive_: Receive
        """
        receive_ = await request._receive()

        async def receive() -> Message:
            return receive_

        request._receive = receive

    async def _log_request(self, request: Request) -> str:
        """Logs request part
         Arguments:
        - request: Request

        """

        path = request.url.path
        if request.query_params:
            path += f"?{request.query_params}"

        request_logging = {
            "method": request.method,
            "path": path,
            "ip": request.client.host,
        }

        try:
            body = await request.json()
            request_logging["body"] = body
        except ValueError as e:
            # Handle case where the request body is not valid JSON
            body = None
            request_logging["error"] = f"Invalid JSON: {str(e)}"
        except Exception as e:
            # Catch other unexpected exceptions
            body = None
            request_logging["error"] = f"Unexpected error: {str(e)}"

        return request_logging

    async def _log_response(
        self, call_next: Callable, request: Request, request_id: str
    ) -> Response:
        """Logs response part

        Arguments:
        - call_next: Callable (To execute the actual path function and get response back)
        - request: Request
        - request_id: str (uuid)
        Returns:
        - response: Response
        - response_logging: str
        """

        start_time = time.perf_counter()
        response = await self._execute_request(call_next, request, request_id)
        finish_time = time.perf_counter()

        overall_status = "successful" if response.status_code < 400 else "failed"
        execution_time = finish_time - start_time

        response_logging = {
            "status": overall_status,
            "status_code": response.status_code,
            "time_taken": f"{execution_time:0.4f}s",
        }

        resp_body = [section async for section in response.__dict__["body_iterator"]]
        response.__setattr__("body_iterator", AsyncIteratorWrapper(resp_body))

        try:
            resp_body = json.loads(resp_body[0].decode())
        except (ValueError, IndexError, AttributeError) as e:
            # Handle specific exceptions for decoding and accessing `resp_body`
            response_logging["error"] = f"Error processing response body: {str(e)}"
            resp_body = str(resp_body)
        except Exception as e:
            # Catch any unexpected exceptions
            response_logging["error"] = f"Unexpected error: {str(e)}"
            resp_body = str(resp_body)

        response_logging["body"] = resp_body

        return response, response_logging

    async def _execute_request(
        self, call_next: Callable, request: Request, request_id: str
    ) -> Response:
        """Executes the actual path function using call_next.
        It also injects "X-API-Request-ID" header to the response.

        Arguments:
        - call_next: Callable (To execute the actual path function
                     and get response back)
        - request: Request
        - request_id: str (uuid)
        Returns:
        - response: Response
        """
        try:
            response: Response = await call_next(request)

            # Kickback X-Request-ID
            response.headers["X-API-Request-ID"] = request_id
            return response

        except Exception as e:
            self._logger.exception(
                {"path": request.url.path, "method": request.method, "reason": e}
            )


class AsyncIteratorWrapper:
    """The following is a utility class that transforms a
    regular iterable to an asynchronous one.

    link: https://www.python.org/dev/peps/pep-0492/#example-2
    """

    def __init__(self, obj):
        self._it = iter(obj)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            value = next(self._it)
        except StopIteration:
            raise StopAsyncIteration
        return value


# import logging
# from fastapi import Request, FastAPI, HTTPException
# from fastapi.middleware.base import BaseHTTPMiddleware
# from datetime import datetime, timedelta
# from collections import defaultdict
# from typing import Callable, Any
# from functools import wraps

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# )


# # Define a decorator function to log user activity
# def log_user_activity(func: Callable) -> Callable:
#     """
#     Decorator to log user activity and execution time for a route function.

#     Parameters
#     ----------
#     func : Callable
#         The route function to be wrapped.

#     Returns
#     -------
#     Callable
#         The wrapped route function.
#     """

#     @wraps(func)
#     async def wrapper(*args, **kwargs):
#         start_time = datetime.now()

#         # Call the route function
#         response = await func(*args, **kwargs)

#         end_time = datetime.now()
#         elapsed_time = (end_time - start_time).total_seconds()

#         # Log user activity
#         route_name = func.__name__
#         request: Request = args[1]

#         username = (
#             request.user.username
#             if hasattr(request.user, "username")
#             else "Anonymous"
#         )

#         logging.info(f"User '{username}' accessed route '{route_name}'")
#         logging.info(f"Execution time: {elapsed_time:.2f} seconds")
#         return response

#     return wrapper


# # A defaultdict to keep track of user activity
# user_activity = defaultdict(list)

# # Define a middleware to log user requests
# class UserActivityMiddleware(BaseHTTPMiddleware):
#     async def dispatch(self, request: Request, call_next: Callable) -> Any:
#         start_time = datetime.now()
#         response = await call_next(request)
#         end_time = datetime.now()
#         elapsed_time = (end_time - start_time).total_seconds()

#         # Log user request
#         username = (
#             request.state.user.username
#             if hasattr(request.state.user, "username")
#             else "Anonymous"
#         )
#         route_name = request.url.path
#         user_activity[username].append((route_name, elapsed_time))

#         return response

# # Log user predictions
# def log_prediction(
#     username: str, p1_name: str, p2_name: str, date: str, prediction: dict
# ) -> None:
#     """
#     Log a user's prediction details.

#     Parameters
#     ----------
#     username : str
#         The username of the user making the prediction.

#     p1_name : str
#         The name of player 1.

#     p2_name : str
#         The name of player 2.

#     date : str
#         The date of the prediction.

#     prediction : dict
#         The prediction details.
#     """
#     logging.info(f"User '{username}' made a prediction:")
#     logging.info(f"Player 1: {p1_name}")
#     logging.info(f"Player 2: {p2_name}")
#     logging.info(f"Date: {date}")
#     logging.info(f"Prediction: {prediction}")


# def clear_old_activity(threshold_hours: int) -> None:
#     """
#     Clear user activity history older than a specified time.

#     Parameters
#     ----------
#     threshold_hours : int
#         The threshold time in hours.

#     Returns
#     -------
#     None
#     """
#     threshold_time = datetime.now() - timedelta(hours=threshold_hours)
#     for user, activity in user_activity.items():
#         user_activity[user] = [
#             (route, elapsed_time)
#             for route, elapsed_time in activity
#             if elapsed_time >= threshold_time
#         ]
