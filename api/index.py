from __future__ import annotations

from app.api.main import app as pricing_app


class StripApiPrefixMiddleware:
    """
    Vercel puede entregar la request con path '/api/...'
    o ya normalizada. Este wrapper soporta ambos casos.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] in {"http", "websocket"}:
            path = scope.get("path", "")

            if path == "/api":
                scope = dict(scope)
                scope["path"] = "/"
                scope["root_path"] = "/api"
            elif path.startswith("/api/"):
                scope = dict(scope)
                scope["path"] = path[len("/api"):] or "/"
                root_path = scope.get("root_path", "")
                scope["root_path"] = f"{root_path.rstrip('/')}/api" if root_path else "/api"

        await self.app(scope, receive, send)


app = StripApiPrefixMiddleware(pricing_app)
