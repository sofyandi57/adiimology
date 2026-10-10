"""Smoke test UI Streamlit dengan requests.get dipalsukan."""
import adimology as ad
import requests
from streamlit.testing.v1 import AppTest
from fakes import mcas

APP = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "app.py")
EMAIL, PW = "user@example.com", "rahasia-tes"

class R:
    status_code = 200
    def __init__(s, j): s.j = j
    def raise_for_status(s): pass
    def json(s): return s.j


def login_env(monkeypatch):
    monkeypatch.setenv("APP_LOGIN_EMAIL", EMAIL)
    monkeypatch.setenv("APP_LOGIN_PASSWORD", PW)


def open_app(monkeypatch, email=EMAIL, pw=PW):
    login_env(monkeypatch)
    at = AppTest.from_file(APP, default_timeout=60).run()
    [t for t in at.text_input if t.label == "Email"][0].set_value(email)
    [t for t in at.text_input if t.label == "Password"][0].set_value(pw)
    [b for b in at.button if b.label == "Masuk"][0].click().run()
    return at


def test_login_required(monkeypatch):
    login_env(monkeypatch)
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert [t.label for t in at.text_input] == ["Email", "Password"]
    assert not [b for b in at.button if b.label == "Hitung"]


def test_login_wrong_password(monkeypatch):
    at = open_app(monkeypatch, pw="salah")
    assert any("salah" in e.value for e in at.error)
    assert not [b for b in at.button if b.label == "Hitung"]

def test_app_runs_and_computes(monkeypatch):
    fc = mcas()
    def fake_get(url, headers=None, params=None, timeout=None):
        path = url.replace(ad.BASE_URL, "")
        return R(fc.get(path, params))
    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setenv("INVEZGO_API_KEY", "dummy")
    at = open_app(monkeypatch)
    assert not at.exception
    [b for b in at.button if b.label == "Hitung"][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("TIDAK ADA SINYAL" in w.value for w in at.warning)
    labels = [m.label for m in at.metric]
    assert "Target realistis" in labels and "Target max" in labels
    vals = {m.label: m.value for m in at.metric}
    assert vals["Target realistis"] == "292" and vals["Target max"] == "296"


def test_multi_batch(monkeypatch):
    fc = mcas()
    monkeypatch.setattr(requests, "get", lambda url, headers=None, params=None, timeout=None: R(fc.get(url.replace(ad.BASE_URL, ""), params)))
    monkeypatch.setenv("INVEZGO_API_KEY", "dummy")
    at = open_app(monkeypatch)
    at.text_area[0].set_value("MCAS").run()
    [b for b in at.button if b.label == "Jalankan batch"][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    df = at.session_state["multi_df"]
    assert df.loc[0, "Kode"] == "MCAS" and df.loc[0, "Verdict"] == "TIDAK ADA SINYAL"
