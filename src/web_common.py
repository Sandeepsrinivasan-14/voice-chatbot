"""
Shared Flask setup for the two local web front ends (src/chat_web.py,
src/record_web.py): consistent JSON error responses instead of Flask's
default HTML error pages, and a dev-vs-production serving switch.
"""

from __future__ import annotations

from flask import Flask, jsonify

from config import CONFIG


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
