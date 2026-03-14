"""Micro HTTP server for the interactive dashboard.

Serves the dashboard HTML on ``/`` and exposes a small API so the UI can
read entity Markdown files and mutate relations without a page reload.

Routes
------
GET  /                        -> dashboard HTML (generated on the fly)
GET  /api/entity/<id>/raw     -> raw Markdown content of the entity file
POST /api/relation            -> add a relation  (JSON body: {from, to, type})
DELETE /api/relation           -> delete a relation (JSON body: {from, to, type})

Start with ``start_server(config, port=9077)``.  ``Ctrl-C`` stops it.
"""

from __future__ import annotations

import json
import logging
import signal
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import unquote

from src.core.config import Config
from src.core.models import GraphRelation
from src.memory.graph import add_relation, load_graph, remove_relation, save_graph
from src.pipeline.dashboard import generate_dashboard

logger = logging.getLogger(__name__)


class DashboardHandler(BaseHTTPRequestHandler):
    """Lightweight request handler — stdlib only."""

    # Attached by ``start_server`` before the server loop starts.
    config: Config

    # Suppress default stderr logging per request.
    def log_message(self, fmt, *args):  # noqa: D401
        logger.debug(fmt, *args)

    # -- routing helpers ---------------------------------------------------

    def _send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text: str, status: int = 200, content_type: str = "text/plain; charset=utf-8") -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length)

    # -- GET ---------------------------------------------------------------

    def do_GET(self):  # noqa: N802
        if self.path == "/" or self.path == "":
            self._serve_dashboard()
        elif self.path.startswith("/api/entity/") and self.path.endswith("/raw"):
            self._serve_entity_raw()
        else:
            self._send_text("Not found", 404)

    def _serve_dashboard(self) -> None:
        html_path = generate_dashboard(self.config)
        html = html_path.read_text(encoding="utf-8")
        self._send_text(html, content_type="text/html; charset=utf-8")

    def _serve_entity_raw(self) -> None:
        # /api/entity/<entity_id>/raw
        parts = self.path.split("/")
        # ['', 'api', 'entity', '<id>', 'raw']
        if len(parts) < 5:
            self._send_json({"error": "invalid path"}, 400)
            return
        entity_id = unquote(parts[3])

        config = self.config
        graph = load_graph(config.memory_path)
        entity = graph.entities.get(entity_id)
        if not entity:
            self._send_json({"error": "entity not found"}, 404)
            return

        md_path = config.memory_path / entity.file
        if not md_path.exists():
            self._send_json({"error": "file not found"}, 404)
            return

        # Guard against path traversal
        try:
            md_path.resolve().relative_to(config.memory_path.resolve())
        except ValueError:
            self._send_json({"error": "forbidden"}, 403)
            return

        raw = md_path.read_text(encoding="utf-8")
        self._send_json({"entity_id": entity_id, "content": raw})

    # -- POST --------------------------------------------------------------

    def do_POST(self):  # noqa: N802
        if self.path == "/api/relation":
            self._add_relation()
        else:
            self._send_text("Not found", 404)

    def _add_relation(self) -> None:
        try:
            payload = json.loads(self._read_body())
        except (json.JSONDecodeError, ValueError):
            self._send_json({"error": "invalid JSON"}, 400)
            return

        from_id = payload.get("from", "").strip()
        to_id = payload.get("to", "").strip()
        rel_type = payload.get("type", "").strip()

        if not from_id or not to_id or not rel_type:
            self._send_json({"error": "from, to, type are required"}, 400)
            return

        config = self.config
        graph = load_graph(config.memory_path)

        if from_id not in graph.entities:
            self._send_json({"error": f"entity '{from_id}' not found"}, 404)
            return
        if to_id not in graph.entities:
            self._send_json({"error": f"entity '{to_id}' not found"}, 404)
            return

        try:
            rel = GraphRelation(from_entity=from_id, to_entity=to_id, type=rel_type)
        except Exception as exc:
            self._send_json({"error": str(exc)}, 400)
            return

        add_relation(graph, rel)
        save_graph(config.memory_path, graph)

        self._send_json({"ok": True, "from": from_id, "to": to_id, "type": rel_type})

    # -- DELETE ------------------------------------------------------------

    def do_DELETE(self):  # noqa: N802
        if self.path == "/api/relation":
            self._delete_relation()
        else:
            self._send_text("Not found", 404)

    def _delete_relation(self) -> None:
        try:
            payload = json.loads(self._read_body())
        except (json.JSONDecodeError, ValueError):
            self._send_json({"error": "invalid JSON"}, 400)
            return

        from_id = payload.get("from", "").strip()
        to_id = payload.get("to", "").strip()
        rel_type = payload.get("type", "").strip()

        if not from_id or not to_id or not rel_type:
            self._send_json({"error": "from, to, type are required"}, 400)
            return

        config = self.config
        graph = load_graph(config.memory_path)
        removed = remove_relation(graph, from_id, to_id, rel_type)

        if not removed:
            self._send_json({"error": "relation not found"}, 404)
            return

        save_graph(config.memory_path, graph)
        self._send_json({"ok": True, "from": from_id, "to": to_id, "type": rel_type})


def start_server(config: Config, port: int = 9077) -> None:
    """Start the dashboard HTTP server and open the browser."""
    DashboardHandler.config = config

    server = HTTPServer(("127.0.0.1", port), DashboardHandler)
    url = f"http://localhost:{port}"

    # Graceful shutdown on Ctrl-C
    def _shutdown(sig, frame):
        print("\nShutting down dashboard server...")
        server.shutdown()

    signal.signal(signal.SIGINT, _shutdown)

    print(f"Dashboard server running at {url}")
    print("Press Ctrl+C to stop.")
    webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
