"""Local web dashboard server for MLB odds."""

from __future__ import annotations

import json
import mimetypes
import socket
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from mlb_cache import DASHBOARD_CACHE, dashboard_cache_key
from mlb_data import fetch_dashboard_data
from sbr_client import SBRClientError

DASHBOARD_DIR = Path(__file__).resolve().parent / "dashboard"
DEFAULT_PORT = 8765
API_ROUTES = {"/api/odds", "/api/games"}


@dataclass
class DashboardConfig:
    league: str = "mlb"
    source: str = "espn"
    fixture: str | None = None
    retries: int = 3
    retry_delay: float = 1.0
    verify_ssl: bool = True
    view_filter: str = "Spread|MoneyLine|Total"
    include_odds: bool = True
    include_enrichment: bool = True


def _log(message: str) -> None:
    print(message, flush=True)


def find_available_port(host: str, start_port: int, attempts: int = 10) -> int:
    for offset in range(attempts):
        port = start_port + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise OSError(f"No free port found in range {start_port}-{start_port + attempts - 1}")


def create_handler(config: DashboardConfig) -> type[BaseHTTPRequestHandler]:
    class DashboardHandler(BaseHTTPRequestHandler):
        server_version = "SportsDashboard/1.0"

        def log_message(self, format: str, *args: Any) -> None:
            print(f"[dashboard] {self.address_string()} {format % args}")

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, file_path: Path) -> None:
            if not file_path.is_file():
                self.send_error(404, "File not found")
                return

            content = file_path.read_bytes()
            content_type, _ = mimetypes.guess_type(str(file_path))
            self.send_response(200)
            self.send_header("Content-Type", content_type or "application/octet-stream")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def _send_api_error(self, message: str, status: int = 502) -> None:
            self._send_json({"error": message}, status=status)

        def _handle_api(self, query: dict[str, list[str]]) -> None:
            date_value = query.get("date", [None])[0]
            league = query.get("league", [config.league])[0] or config.league
            view_filter = query.get("view", [config.view_filter])[0] or config.view_filter
            force_refresh = query.get("force", ["0"])[0] in {"1", "true", "yes"}
            cache_key = dashboard_cache_key(
                league=league,
                date_value=date_value,
                view_filter=view_filter,
                source=config.source,
                fixture=config.fixture,
                include_odds=config.include_odds,
                include_enrichment=config.include_enrichment,
            )

            if not force_refresh:
                cached_payload = DASHBOARD_CACHE.get(cache_key)
                if cached_payload is not None:
                    age = DASHBOARD_CACHE.get_age_seconds(cache_key)
                    response = dict(cached_payload)
                    response["fromCache"] = True
                    response["cacheAgeSeconds"] = round(age or 0, 1)
                    self._send_json(response)
                    return

            try:
                payload = fetch_dashboard_data(
                    league=league,
                    date=date_value,
                    source=config.source,
                    fixture=config.fixture,
                    view_filter=view_filter,
                    include_odds=config.include_odds,
                    include_enrichment=config.include_enrichment,
                    retries=config.retries,
                    retry_delay=config.retry_delay,
                    verify_ssl=config.verify_ssl,
                )
                payload["fromCache"] = False
                payload["cacheAgeSeconds"] = 0
                DASHBOARD_CACHE.set(cache_key, payload)
                self._send_json(payload)
            except SBRClientError as exc:
                cached_payload = DASHBOARD_CACHE.get(cache_key)
                if cached_payload is not None:
                    age = DASHBOARD_CACHE.get_age_seconds(cache_key)
                    response = dict(cached_payload)
                    response["fromCache"] = True
                    response["stale"] = True
                    response["cacheAgeSeconds"] = round(age or 0, 1)
                    response["refreshError"] = str(exc)
                    self._send_json(response)
                    return
                self._send_api_error(str(exc))
            except Exception as exc:
                cached_payload = DASHBOARD_CACHE.get(cache_key)
                if cached_payload is not None:
                    age = DASHBOARD_CACHE.get_age_seconds(cache_key)
                    response = dict(cached_payload)
                    response["fromCache"] = True
                    response["stale"] = True
                    response["cacheAgeSeconds"] = round(age or 0, 1)
                    response["refreshError"] = str(exc)
                    self._send_json(response)
                    return
                self._send_api_error(f"Unexpected server error: {exc}")

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            route = parsed.path.rstrip("/") or "/"
            query = parse_qs(parsed.query)

            if route == "/api/health":
                self._send_json(
                    {
                        "status": "ok",
                        "league": config.league,
                        "source": config.source,
                        "mode": "fixture" if config.fixture else "live",
                        "api": sorted(API_ROUTES),
                    }
                )
                return

            if route in API_ROUTES:
                self._handle_api(query)
                return

            if route.startswith("/api/"):
                self._send_api_error(f"Unknown API route: {route}", status=404)
                return

            if route == "/":
                self._send_file(DASHBOARD_DIR / "index.html")
                return

            asset_path = DASHBOARD_DIR / route.lstrip("/")
            if asset_path.resolve().is_relative_to(DASHBOARD_DIR.resolve()):
                self._send_file(asset_path)
                return

            self.send_error(404, "Not found")

    return DashboardHandler


def run_dashboard(
    *,
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    open_browser: bool = True,
    config: DashboardConfig | None = None,
    strict_port: bool = False,
) -> None:
    config = config or DashboardConfig()
    handler = create_handler(config)

    if not strict_port:
        try:
            port = find_available_port(host, port)
        except OSError as exc:
            _log(f"Could not start dashboard server: {exc}")
            _log("Close any other dashboard window and try again.")
            raise SystemExit(1) from exc

    server = ThreadingHTTPServer((host, port), handler)
    local_url = f"http://127.0.0.1:{port}/"
    localhost_url = f"http://localhost:{port}/"

    _log("")
    _log("=" * 56)
    _log(" Sports Dashboard is running")
    _log("=" * 56)
    if host == "0.0.0.0":
        _log(f" Listening on:          0.0.0.0:{port}")
    else:
        _log(f" Open in your browser: {local_url}")
        _log(f" Alternate URL:         {localhost_url}")
    _log("")
    _log(" Keep this window open while using the dashboard.")
    _log(" Press Ctrl+C here to stop the server.")
    _log("")

    if config.fixture:
        _log(f" Data source: fixture ({config.fixture})")
    else:
        _log(f" League: {config.league.upper()}")
        _log(f" Schedule source: {config.source.upper()}")
        if config.include_odds:
            _log(" Odds source: SBR (MLB) or ESPN (World Cup/AFL)")
        if not config.verify_ssl:
            _log(" TLS verification: disabled (--insecure)")

    if open_browser:
        webbrowser.open(local_url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _log("\nShutting down dashboard.")
    finally:
        server.server_close()
