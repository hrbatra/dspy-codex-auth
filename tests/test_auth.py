from __future__ import annotations

import base64
import json
from pathlib import Path
import tomllib

from dspy_codex_auth.auth import (
    AuthStorage,
    build_openai_codex_authorization_url,
    extract_chatgpt_account_id,
    normalize_provider_id,
    parse_authorization_input,
)


def _b64url(data: dict) -> str:
    raw = json.dumps(data, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _fake_jwt(account_id: str = "acct_test") -> str:
    return ".".join(
        [
            _b64url({"alg": "none", "typ": "JWT"}),
            _b64url(
                {"https://api.openai.com/auth": {"chatgpt_account_id": account_id}}
            ),
            "signature",
        ]
    )


def test_auth_storage_uses_codex_aliases(tmp_path):
    storage = AuthStorage(tmp_path / "auth.json")
    storage.set("codex", {"type": "api_key", "key": "token"})

    assert normalize_provider_id("chatgpt") == "openai-codex"
    assert storage.get_api_key("openai-codex") == "token"
    assert storage.get_api_key("chatgpt") == "token"


def test_extract_chatgpt_account_id_from_jwt():
    assert extract_chatgpt_account_id(_fake_jwt("acct_live")) == "acct_live"


def test_parse_authorization_input_accepts_url_query_and_hash_pair():
    url_code, url_state = parse_authorization_input(
        "http://localhost:1455/auth/callback?code=abc&state=xyz"
    )
    hash_code, hash_state = parse_authorization_input("abc#xyz")

    assert (url_code, url_state) == ("abc", "xyz")
    assert (hash_code, hash_state) == ("abc", "xyz")


def test_authorization_url_uses_package_originator():
    url = build_openai_codex_authorization_url(state="state", challenge="challenge")

    assert "originator=dspy_codex_auth" in url
    assert "client_id=app_EMoamEEZ73f0CkXaXp7hrann" in url


def test_project_has_no_dspy_lm_auth_runtime_dependency():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]

    assert all("dspy-lm-auth" not in dependency for dependency in dependencies)
