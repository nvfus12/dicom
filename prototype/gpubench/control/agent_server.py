"""HTTP front of the lab agent. Thin: parses requests, checks the token, calls AgentCore."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .agent_core import AgentCore, AgentError
from .protocol import TOKEN_HEADER


def make_server(core: AgentCore, host: str, port: int, token: str) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, body: dict) -> None:
            self._send(status, json.dumps(body, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        def _authorised(self) -> bool:
            if self.headers.get(TOKEN_HEADER) == token:
                return True
            self._json(401, {"error": "Sai token"})
            return False

        def _run_id(self) -> str:
            values = parse_qs(urlparse(self.path).query).get("run_id")
            if not values:
                raise AgentError(400, "Thiếu run_id")
            return values[0]

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise AgentError(400, "Nội dung yêu cầu không phải JSON") from None
            if not isinstance(body, dict):
                raise AgentError(400, "Nội dung yêu cầu phải là một object JSON")
            return body

        def _dispatch(self, routes: dict) -> None:
            if not self._authorised():
                return
            route = routes.get(urlparse(self.path).path)
            if route is None:
                self._json(404, {"error": "Không có đường dẫn này"})
                return
            try:
                route()
            except AgentError as e:
                self._json(e.status, {"error": str(e)})

        def do_GET(self) -> None:
            def results():
                self._send(200, core.results_zip(self._run_id()), "application/zip")
            self._dispatch({
                "/info": lambda: self._json(200, core.info()),
                "/status": lambda: self._json(200, core.status(self._run_id())),
                "/results": results,
            })

        def do_POST(self) -> None:
            def stop():
                body = self._body()
                if "run_id" not in body:
                    raise AgentError(400, "Thiếu run_id")
                self._json(200, core.stop(body["run_id"], body.get("reason", ""), by="controller"))
            self._dispatch({
                "/request": lambda: self._json(200, core.request(self._body())),
                "/start": lambda: self._json(200, core.start(self._body())),
                "/stop": stop,
            })

        def log_message(self, fmt: str, *args) -> None:
            pass  # the agent window shows what matters

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server
