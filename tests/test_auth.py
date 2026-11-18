"""Login protection for hosted deployments."""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("AEL_DATA_DIR", tempfile.mkdtemp(prefix="ael-auth-"))

from fastapi.testclient import TestClient  # noqa: E402

from app import auth  # noqa: E402
from app.main import app  # noqa: E402


def test_open_when_no_password_locally(monkeypatch):
    monkeypatch.delenv("AEL_PASSWORD", raising=False)
    monkeypatch.delenv("AEL_PUBLIC", raising=False)
    c = TestClient(app)
    assert c.get("/api/cases").status_code == 200


def test_public_mode_without_password_refuses(monkeypatch):
    monkeypatch.delenv("AEL_PASSWORD", raising=False)
    monkeypatch.setenv("AEL_PUBLIC", "1")
    c = TestClient(app)
    assert c.get("/api/cases").status_code == 503
    assert c.get("/").status_code == 503
    assert c.get("/healthz").status_code == 200


def test_password_flow(monkeypatch):
    monkeypatch.setenv("AEL_PASSWORD", "correct horse")
    monkeypatch.setenv("AEL_PUBLIC", "1")
    auth._failures.clear()
    c = TestClient(app)
    assert c.get("/api/cases").status_code == 401
    r = c.get("/", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/login"
    assert c.get("/static/app.js", follow_redirects=False).status_code == 303
    assert "Wrong password" in c.post("/login", data={"password": "nope"}).text
    r = c.post("/login", data={"password": "correct horse"}, follow_redirects=False)
    assert r.status_code == 303
    token = r.cookies.get(auth.COOKIE) or c.cookies.get(auth.COOKIE)
    assert auth.valid_token(token)
    c.cookies.set(auth.COOKIE, token)
    assert c.get("/api/cases").status_code == 200
    # tampered token rejected
    assert not auth.valid_token(token[:-1] + ("0" if token[-1] != "0" else "1"))


def test_login_rate_limit(monkeypatch):
    monkeypatch.setenv("AEL_PASSWORD", "pw")
    auth._failures.clear()
    c = TestClient(app)
    for _ in range(auth._FAIL_MAX):
        c.post("/login", data={"password": "x"})
    assert "Too many attempts" in c.post("/login", data={"password": "pw"}).text
    auth._failures.clear()
