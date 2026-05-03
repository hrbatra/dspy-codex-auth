import dspy
import dspy_codex_auth


def main() -> None:
    dspy_codex_auth.install()
    assert dspy.LM is dspy_codex_auth.LM
    lm = dspy.LM("openai/test", api_key="dummy", api_base="http://example.invalid")
    assert isinstance(lm, dspy_codex_auth.LM)


if __name__ == "__main__":
    main()
