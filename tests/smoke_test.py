import dspy
import dspy_codex_auth


def main() -> None:
    assert dspy_codex_auth.CodexTransport is not None
    assert dspy_codex_auth.DEFAULT_CODEX_WEBSOCKET_CONNECT_TIMEOUT == 10.0
    assert dspy_codex_auth.DEFAULT_CODEX_WEBSOCKET_IDLE_TIMEOUT == 300.0

    dspy_codex_auth.install()
    assert dspy.LM is dspy_codex_auth.LM
    lm = dspy.LM("openai/test", api_key="dummy", api_base="http://example.invalid")
    assert isinstance(lm, dspy_codex_auth.LM)


if __name__ == "__main__":
    main()
