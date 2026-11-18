"""Password login for when the app is hosted on the internet.

Enabled when AEL_PASSWORD is set. When AEL_PUBLIC=1 (set by the hosting
config) and no password is configured, the app refuses every request rather
than run open to the world, since anyone could otherwise upload files and
spend the API key's credit.

Sessions are an HMAC-signed cookie (no server-side state).
"""
from __future__ import annotations

import hashlib
import hmac
import html
import os
import secrets
import time
from collections import defaultdict

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

COOKIE = "ael_session"
SESSION_DAYS = 14
OPEN_PATHS = {"/login", "/logout", "/healthz", "/static/styles.css"}

_failures: dict[str, list[float]] = defaultdict(list)
_FAIL_WINDOW, _FAIL_MAX = 900, 8  # 8 wrong passwords per 15 minutes per IP


def password() -> str | None:
    return os.environ.get("AEL_PASSWORD") or None


def is_public() -> bool:
    return os.environ.get("AEL_PUBLIC", "0") == "1"


def enabled() -> bool:
    return bool(password())


def _secret() -> bytes:
    key = os.environ.get("AEL_SECRET_KEY")
    if key:
        return key.encode()
    # Derive from the password so changing the password logs everyone out.
    return hashlib.sha256(b"ael-session:" + (password() or "").encode()).digest()


def _sign(payload: str) -> str:
    return hmac.new(_secret(), payload.encode(), hashlib.sha256).hexdigest()


def make_token() -> str:
    exp = int(time.time()) + SESSION_DAYS * 86400
    payload = f"{exp}.{secrets.token_hex(8)}"
    return f"{payload}.{_sign(payload)}"


def valid_token(token: str | None) -> bool:
    if not token or token.count(".") != 2:
        return False
    exp, nonce, sig = token.split(".")
    if not hmac.compare_digest(sig, _sign(f"{exp}.{nonce}")):
        return False
    return exp.isdigit() and int(exp) > time.time()


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "?")


async def middleware(request: Request, call_next) -> Response:
    path = request.url.path
    if path in OPEN_PATHS:
        return await call_next(request)
    if not enabled():
        if is_public():
            return HTMLResponse("<h1>AI Evidence Lab is not configured</h1><p>The server is running in public "
                                "mode without a password. Set the AEL_PASSWORD environment variable on the "
                                "host and restart.</p>", status_code=503)
        return await call_next(request)
    if valid_token(request.cookies.get(COOKIE)):
        return await call_next(request)
    if path.startswith("/api/"):
        return JSONResponse({"detail": "Not logged in"}, status_code=401)
    return RedirectResponse("/login", status_code=303)


def login_page(error: str = "") -> HTMLResponse:
    err = f'<p class="err">{html.escape(error)}</p>' if error else ""
    return HTMLResponse(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Sign in - AI Evidence Lab</title>
<link rel="stylesheet" href="/static/styles.css">
<style>.login{{max-width:380px;margin:12vh auto;padding:0 16px}} .login form{{display:flex;flex-direction:column;gap:12px}}
.err{{color:var(--bad);margin:0}}</style></head><body>
<div class="login panel"><h1>AI Evidence Lab</h1><p class="muted small">Enter the password to continue.</p>{err}
<form method="post" action="/login"><input type="password" name="password" placeholder="Password" autofocus required
autocomplete="current-password"><button class="btn" type="submit">Sign in</button></form></div></body></html>""")


async def handle_login(request: Request) -> Response:
    ip = _client_ip(request)
    now = time.time()
    _failures[ip] = [t for t in _failures[ip] if now - t < _FAIL_WINDOW]
    if len(_failures[ip]) >= _FAIL_MAX:
        return login_page("Too many attempts. Try again in 15 minutes.")
    form = await request.form()
    given = str(form.get("password", ""))
    if not enabled() or not hmac.compare_digest(given.encode(), (password() or "").encode()):
        _failures[ip].append(now)
        return login_page("Wrong password.")
    _failures.pop(ip, None)
    resp = RedirectResponse("/", status_code=303)
    secure = is_public() or request.headers.get("x-forwarded-proto") == "https" or request.url.scheme == "https"
    resp.set_cookie(COOKIE, make_token(), max_age=SESSION_DAYS * 86400, httponly=True, secure=secure,
                    samesite="lax")
    return resp


def handle_logout() -> Response:
    resp = RedirectResponse("/login" if enabled() else "/", status_code=303)
    resp.delete_cookie(COOKIE)
    return resp
