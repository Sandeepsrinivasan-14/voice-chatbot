"""
Tests for how the two browser UIs are served (src/web_common.py): static
asset delivery and MIME types, path-traversal safety, the Content-Security-
Policy and companion headers -- and a guard that the front ends stay fully
offline and CSP-clean (no CDN URLs, no inline script/style/handlers).

These read the real files under src/web_chat and src/web_recorder, so adding
a page that references a missing asset, or an ES module that imports a
missing sibling, fails here rather than as a blank page in a browser.
"""

from __future__ import annotations

import os
import re

import pytest

import chat_web
import record_web
from web_common import CONTENT_SECURITY_POLICY, SECURITY_HEADERS

SRC_DIR = os.path.dirname(os.path.abspath(chat_web.__file__))
CHAT_DIR = os.path.join(SRC_DIR, "web_chat")
RECORDER_DIR = os.path.join(SRC_DIR, "web_recorder")

JS_TYPES = {"text/javascript", "application/javascript"}


@pytest.fixture
def recorder_client():
    record_web.app.config["TESTING"] = True
    with record_web.app.test_client() as test_client:
        yield test_client


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _walk(directory: str, suffixes: tuple[str, ...]) -> list[str]:
    found = []
    for base, _dirs, files in os.walk(directory):
        for name in files:
            if name.endswith(suffixes):
                found.append(os.path.join(base, name))
    return sorted(found)


def _local_refs(html: str) -> list[str]:
    """Every href/src in the page that points at a file served by the app."""
    refs = re.findall(r'(?:href|src)="([^"#][^"]*)"', html)
    return [r for r in refs if not r.startswith(("http:", "https:", "data:", "#"))]


class TestChatUiAssets:
    def test_index_is_served_as_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.mimetype == "text/html"
        assert b"Local Voice Chatbot" in resp.data

    def test_every_asset_referenced_by_the_page_is_served(self, client):
        refs = _local_refs(_read(os.path.join(CHAT_DIR, "index.html")))
        assert "js/main.js" in refs and "css/app.css" in refs
        for ref in refs:
            resp = client.get("/" + ref)
            assert resp.status_code == 200, ref

    def test_es_module_imports_all_resolve(self, client):
        # Static `import ... from "./x.js"` and dynamic `import("./x.js")`.
        pattern = re.compile(r"""(?:from\s+|import\()\s*["'](\./[^"']+)["']""")
        modules = _walk(os.path.join(CHAT_DIR, "js"), (".js",))
        assert len(modules) >= 8
        for module in modules:
            for target in pattern.findall(_read(module)):
                resolved = os.path.normpath(os.path.join(os.path.dirname(module), target))
                assert os.path.isfile(resolved), (
                    f"{os.path.basename(module)} imports missing {target}"
                )
                rel = os.path.relpath(resolved, CHAT_DIR).replace(os.sep, "/")
                assert client.get("/" + rel).status_code == 200, rel

    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("/js/main.js", JS_TYPES),
            ("/js/orb.js", JS_TYPES),
            ("/css/app.css", {"text/css"}),
            ("/css/theme.css", {"text/css"}),
            ("/favicon.svg", {"image/svg+xml"}),
        ],
    )
    def test_mime_types_are_correct(self, client, path, expected):
        # nosniff + a wrong type (Windows registries map .js to text/plain)
        # would make the browser refuse to run the ES modules.
        assert client.get(path).mimetype in expected

    def test_unknown_path_is_a_json_404(self, client):
        resp = client.get("/js/does-not-exist.js")
        assert resp.status_code == 404
        assert resp.get_json() == {"error": "Not found."}

    @pytest.mark.parametrize(
        "path", ["/../config.py", "/%2e%2e/config.py", "/js/../../config.py", "/js/", "/css"]
    )
    def test_cannot_escape_or_list_the_static_directory(self, client, path):
        resp = client.get(path)
        assert resp.status_code == 404
        assert b"CONFIG" not in resp.data

    def test_api_routes_still_win_over_the_static_catch_all(self, client):
        assert client.get("/api/health").get_json()["ok"] is True


class TestSecurityHeaders:
    @pytest.mark.parametrize("path", ["/", "/js/main.js", "/api/health", "/nope"])
    def test_every_response_carries_the_hardening_headers(self, client, path):
        resp = client.get(path)
        for name, value in SECURITY_HEADERS.items():
            assert resp.headers[name] == value, f"{name} missing on {path}"

    def test_csp_is_strict(self):
        policy = CONTENT_SECURITY_POLICY
        assert "default-src 'none'" in policy
        assert "script-src 'self'" in policy
        assert "connect-src 'self'" in policy
        assert "frame-ancestors 'none'" in policy
        assert "object-src 'none'" in policy
        for forbidden in ("'unsafe-inline'", "'unsafe-eval'", "http:", "https:", " *"):
            assert forbidden not in policy

    def test_microphone_is_allowed_only_for_this_origin(self, client):
        assert "microphone=(self)" in client.get("/").headers["Permissions-Policy"]

    def test_api_responses_are_not_cacheable(self, client):
        assert client.get("/api/health").headers["Cache-Control"] == "no-store"

    def test_recorder_ui_is_served_with_the_same_policy(self, recorder_client):
        resp = recorder_client.get("/")
        assert resp.status_code == 200
        assert resp.headers["Content-Security-Policy"] == CONTENT_SECURITY_POLICY
        assert recorder_client.get("/recorder.js").mimetype in JS_TYPES
        assert recorder_client.get("/recorder.css").mimetype == "text/css"


class TestFrontEndsStayOfflineAndCspClean:
    @pytest.mark.parametrize("directory", [CHAT_DIR, RECORDER_DIR], ids=["chat", "recorder"])
    def test_no_external_urls(self, directory):
        # The app is fully offline: nothing may be fetched from a CDN. The only
        # permitted URLs are XML namespace identifiers (never requested).
        offenders = []
        for path in _walk(directory, (".html", ".css", ".js", ".svg")):
            for url in re.findall(
                r"""(?:https?:)?//[A-Za-z0-9.-]+\.[a-z]{2,}[^\s"'`)]*""", _read(path)
            ):
                if "www.w3.org" in url:
                    continue
                offenders.append(f"{os.path.relpath(path, directory)}: {url}")
        assert not offenders, offenders

    @pytest.mark.parametrize("directory", [CHAT_DIR, RECORDER_DIR], ids=["chat", "recorder"])
    def test_html_has_no_inline_script_style_or_handlers(self, directory):
        # A strict CSP (script-src/style-src 'self') silently blocks all of these.
        for path in _walk(directory, (".html",)):
            html = _read(path)
            name = os.path.relpath(path, directory)
            assert not re.search(r"<script(?![^>]*\bsrc=)[^>]*>", html), f"{name}: inline <script>"
            assert not re.search(r"<style\b", html), f"{name}: inline <style>"
            assert not re.search(r"\sstyle=", html), f"{name}: style= attribute"
            assert not re.search(r"\son[a-z]+\s*=", html), f"{name}: inline event handler"

    def test_javascript_never_uses_eval_or_inline_style_attributes(self):
        for path in _walk(CHAT_DIR, (".js",)) + _walk(RECORDER_DIR, (".js",)):
            code = _read(path)
            name = os.path.relpath(path, SRC_DIR)
            assert not re.search(r"\beval\(|new Function\(|\.innerHTML\s*=\s*(?!['\"\s])", code), (
                name
            )
            assert 'setAttribute("style"' not in code and "setAttribute('style'" not in code, name
            assert "cssText" not in code, name
