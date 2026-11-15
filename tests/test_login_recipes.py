"""Exercise the login recipes through curl with shell and form metacharacters."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
from threading import Thread
from urllib.parse import parse_qs

import pytest


@pytest.mark.api
def test_login_recipes_preserve_configured_credentials():
    if shutil.which("just") is None:
        pytest.skip("requires just")
    credentials = {
        "username": "user+'\"&= space",
        "password": "correct+horse&battery'\"$`\\\nmore",
    }
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            received.append(parse_qs(body.decode()))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"access_token": "recipe-token"}).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        for recipe in ("token", "token-json"):
            result = subprocess.run(
                [
                    "just",
                    "--set",
                    "base_url",
                    f"http://127.0.0.1:{server.server_port}",
                    recipe,
                ],
                cwd=Path(__file__).resolve().parents[1],
                env={
                    **os.environ,
                    "ACEBET_DEMO_USERNAME": credentials["username"],
                    "ACEBET_DEMO_PASSWORD": credentials["password"],
                    "UV_NO_SYNC": "1",
                },
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
            if recipe == "token":
                assert result.stdout.strip() == "recipe-token"
            else:
                assert json.loads(result.stdout)["access_token"] == "recipe-token"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert received == [{key: [value] for key, value in credentials.items()}] * 2


@pytest.mark.parametrize("port", ["8000", "18443"])
@pytest.mark.parametrize(
    "recipe", ["docker-run", "docker-run-with-assets", "docker-smoke"]
)
def test_docker_recipes_publish_only_on_loopback(recipe, port):
    if shutil.which("just") is None:
        pytest.skip("requires just")
    result = subprocess.run(
        ["just", "--dry-run", recipe],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "ACEBET_DOCKER_PORT": port},
        check=True,
        capture_output=True,
        text=True,
    )
    commands = result.stdout + result.stderr
    if recipe == "docker-smoke":
        assert '--publish "127.0.0.1:${port}:8000"' in commands
    else:
        assert f'-p "127.0.0.1:{port}:8000"' in commands
