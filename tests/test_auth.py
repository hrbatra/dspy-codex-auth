"""The auth layer lives in openai-codex-auth; this package only re-exports the
user-facing helpers."""

import base64
import importlib
import json
import time
import tomllib
from pathlib import Path

import openai_codex_auth
import pytest
from packaging.requirements import Requirement

import dspy_codex_auth


def test_user_facing_auth_helpers_are_re_exported():
    for name in ("CodexAuth", "getauthtoken"):
        assert getattr(dspy_codex_auth, name) is getattr(openai_codex_auth, name)


def test_project_depends_on_dspy_and_published_openai_codex_auth():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    project = tomllib.loads(pyproject_path.read_text())["project"]
    requirements = [Requirement(value) for value in project["dependencies"]]
    dependencies = {requirement.name: requirement for requirement in requirements}

    assert {"dspy", "openai-codex-auth"} <= dependencies.keys()
    assert dependencies["openai-codex-auth"].url is None
    assert "dspy-lm-auth" not in dependencies


def test_auth_dependency_is_locked_to_published_registry_package():
    lock_path = Path(__file__).resolve().parents[1] / "uv.lock"
    packages = tomllib.loads(lock_path.read_text())["package"]
    auth_package = next(
        package for package in packages if package["name"] == "openai-codex-auth"
    )

    assert auth_package["source"] == {"registry": "https://pypi.org/simple"}


def test_legacy_auth_module_is_removed():
    with pytest.raises(ModuleNotFoundError) as exc_info:
        importlib.import_module("dspy_codex_auth.auth")

    assert exc_info.value.name == "dspy_codex_auth.auth"


def test_legacy_auth_apis_are_removed():
    for name in (
        "AuthStorage",
        "ApiKeyCredential",
        "OAuthCredential",
        "OpenAICodexOAuthProvider",
        "login",
        "logout",
        "login_openai_codex",
        "register_oauth_provider",
        "get_default_auth_storage",
        "set_default_auth_storage",
    ):
        assert not hasattr(dspy_codex_auth, name), name


def test_lm_rejects_pi_credentials_without_modifying_the_file(tmp_path):
    auth_path = tmp_path / "auth.json"
    original = json.dumps(
        {
            "openai-codex": {
                "type": "oauth",
                "access": "synthetic-pi-access-token",
                "refresh": "synthetic-pi-refresh-token",
                "expires": int(time.time() * 1000) + 3_600_000,
                "accountId": "acct_pi",
            }
        }
    )
    auth_path.write_text(original)

    with pytest.raises(openai_codex_auth.CodexAuthError, match="codex login"):
        dspy_codex_auth.LM("codex/gpt-5.5", auth_storage=auth_path, cache=False)

    assert auth_path.read_text() == original


@pytest.mark.parametrize("path_type", (str, Path))
def test_lm_reads_codex_cli_credentials_from_explicit_path(tmp_path, path_type):
    claims = {
        "exp": int(time.time()) + 3600,
        "https://api.openai.com/auth": {"chatgpt_account_id": "acct_codex"},
    }
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=")
    token = f"e30.{payload.decode()}.synthetic-signature"
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": token,
                    "refresh_token": "synthetic-refresh-token",
                    "account_id": "acct_codex",
                },
            }
        )
    )

    lm = dspy_codex_auth.LM(
        "codex/gpt-5.5", auth_storage=path_type(auth_path), cache=False
    )

    assert lm.kwargs["api_key"] == token
    assert lm.kwargs["headers"]["chatgpt-account-id"] == "acct_codex"
