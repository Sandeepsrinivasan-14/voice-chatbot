"""
Shared Flask setup for the two local web front ends (src/chat_web.py,
src/record_web.py): consistent JSON error responses instead of Flask's
default HTML error pages, static-file serving with correct MIME types,
a strict Content-Security-Policy plus the other browser hardening headers,
and a dev-vs-production serving switch.
"""

from __future__ import annotations

import mimetypes

from flask import Flask, Response, jsonify, request, send_from_directory

from config import CONFIG

# `mimetypes` reads the Windows registry, where .js is often registered as
# text/plain. Combined with `X-Content-Type-Options: nosniff` (below) the
# browser would then refuse to run our ES modules, so pin the types we serve
# rather than trusting the host's configuration.
for _ext, _mime in (
    (".js", "text/javascript"),
    (".mjs", "text/javascript"),
    (".css", "text/css"),
    (".svg", "image/svg+xml"),
    (".json", "application/json"),
    (".woff2", "font/woff2"),
):
    mimetypes.add_type(_mime, _ext)

# The UIs are fully offline and load nothing from any other origin, so the
# policy is a strict allow-list of "this origin only": no inline script or
# style, no eval, no framing, no plugins. `blob:` is for recorded and
# synthesized audio, `data:` for the few inline SVG backgrounds in the CSS.
CONTENT_SECURITY_POLICY = "; ".join(
    [
        "default-src 'none'",
        "script-src 'self'",
        "style-src 'self'",
        "img-src 'self' data: blob:",
        "media-src 'self' blob:",
        "font-src 'self'",
        "connect-src 'self'",
        "manifest-src 'self'",
        "base-uri 'none'",
        "form-action 'self'",
        "frame-ancestors 'none'",
        "object-src 'none'",
    ]
)

SECURITY_HEADERS = {
    "Content-Security-Policy": CONTENT_SECURITY_POLICY,
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
    # The microphone is the one powerful feature these pages exist for.
    "Permissions-Policy": "microphone=(self), camera=(), geolocation=(), payment=(), usb=()",
}


def register_security_headers(app: Flask) -> None:
    """Attach the headers above to every response, and mark API responses
    (transcripts, replies, audio) as uncacheable."""

    @app.after_request
    def _add_headers(resp: Response) -> Response:
        for name, value in SECURITY_HEADERS.items():
            resp.headers.setdefault(name, value)
        if request.path.startswith("/api/"):
            resp.headers.setdefault("Cache-Control", "no-store")
        return resp


def register_static_routes(app: Flask, static_dir: str) -> None:
    """Serve `static_dir` at the site root: `/` is index.html, everything
    else is looked up inside the directory (send_from_directory rejects any
    path that escapes it). Registered last-resort, so /api/* routes win."""

    @app.route("/")
    def index():
        return send_from_directory(static_dir, "index.html")

    @app.route("/<path:filename>")
    def static_files(filename):
        return send_from_directory(static_dir, filename)


def register_error_handlers(app: Flask) -> None:
    """Every error -- ours or Flask's own (404, 413, an uncaught 500) --
    comes back as {"error": "..."} JSON. Matters the moment this is
    driven by anything other than a developer reading the response by eye.
    """

    @app.errorhandler(400)
    def _bad_request(e):
        return jsonify({"error": getattr(e, "description", None) or "Bad request."}), 400

    @app.errorhandler(404)
    def _not_found(e):
        return jsonify({"error": "Not found."}), 404

    @app.errorhandler(413)
    def _too_large(e):
        limit_mb = CONFIG.max_upload_bytes // (1024 * 1024)
        return jsonify({"error": f"Upload too large (max {limit_mb}MB)."}), 413

    @app.errorhandler(500)
    def _server_error(e):
        app.logger.exception("Unhandled server error")
        return jsonify({"error": "Internal server error."}), 500


def run_app(app: Flask, *, host: str, port: int, name: str) -> None:
    """Serve via waitress (a real production WSGI server, and Windows-
    compatible -- gunicorn isn't) when CONFIG.production is set (env var
    PRODUCTION=1); Flask's own dev server otherwise. `name` is only used
    in the startup log line.
    """
    if CONFIG.production:
        from waitress import serve

        print(f"[{name}] PRODUCTION mode -- serving via waitress on http://{host}:{port}")
        serve(app, host=host, port=port)
    else:
        print(f"[{name}] Dev mode -- serving via Flask's development server on http://{host}:{port}")
        print(f"[{name}] Set PRODUCTION=1 to serve via waitress instead (see README section 7).")
        app.run(host=host, port=port, debug=False, threaded=True)
