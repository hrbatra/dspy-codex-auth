from __future__ import annotations

import base64
import json
import time
from types import SimpleNamespace

import dspy

import dspy_codex_auth
from dspy_codex_auth.auth import AuthStorage
import dspy_codex_auth.lm as codex_lm


def _b64url(data: dict) -> str:
    raw = json.dumps(data, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def make_fake_jwt(account_id: str = "acct_test") -> str:
    header = _b64url({"alg": "none", "typ": "JWT"})
    payload = _b64url(
        {"https://api.openai.com/auth": {"chatgpt_account_id": account_id}}
    )
    return f"{header}.{payload}.signature"


def make_auth_storage(tmp_path, account_id: str = "acct_test") -> AuthStorage:
    storage = AuthStorage(tmp_path / "auth.json")
    storage.set(
        "openai-codex",
        {
            "type": "oauth",
            "access": make_fake_jwt(account_id),
            "refresh": "refresh-token",
            "expires": int(time.time() * 1000) + 60_000,
            "accountId": account_id,
        },
    )
    return storage


class FakeResponsesStream:
    def __init__(self, events: list[SimpleNamespace], response: SimpleNamespace):
        self._events = events
        self.completed_response = SimpleNamespace(response=response)

    def __iter__(self):
        return iter(self._events)


def make_response(output=None) -> SimpleNamespace:
    return SimpleNamespace(
        output=output or [], model="gpt-5.5", usage={}, _hidden_params={}
    )


def test_install_patches_dspy_lm(tmp_path):
    storage = make_auth_storage(tmp_path)
    original_lm = dspy.LM

    try:
        dspy_codex_auth.install(auth_storage=storage)
        assert dspy.LM is dspy_codex_auth.LM
        lm = dspy.LM("codex/gpt-5.5", cache=False)
        assert isinstance(lm, dspy_codex_auth.LM)
        assert lm.model == "openai/gpt-5.5"
    finally:
        dspy_codex_auth.uninstall()
        assert dspy.LM is original_lm


def test_explicit_codex_auth_provider_sets_codex_originator(tmp_path):
    storage = make_auth_storage(tmp_path)

    lm = dspy_codex_auth.LM(
        "openai/gpt-5.5",
        auth_provider="codex",
        auth_storage=storage,
        cache=False,
    )

    assert lm._uses_codex_route is True
    assert lm.kwargs["headers"]["originator"] == "dspy_codex_auth"


def test_codex_request_strips_token_caps_and_accepts_reasoning_summary():
    request = codex_lm._build_codex_request(
        {
            "model": "openai/gpt-5.5",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 100,
            "max_output_tokens": 100,
            "max_completion_tokens": 100,
            "reasoning_effort": "medium",
            "reasoning_summary": "detailed",
        }
    )

    assert "max_tokens" not in request
    assert "max_output_tokens" not in request
    assert "max_completion_tokens" not in request
    assert request["reasoning"] == {"effort": "medium", "summary": "detailed"}


def test_codex_request_normalizes_fast_service_tier_to_priority():
    request = codex_lm._build_codex_request(
        {
            "model": "openai/gpt-5.4",
            "messages": [{"role": "user", "content": "hello"}],
            "service_tier": "fast",
        }
    )

    assert request["service_tier"] == "priority"


def test_codex_request_accepts_codex_config_reasoning_aliases():
    request = codex_lm._build_codex_request(
        {
            "model": "openai/gpt-5.4",
            "messages": [{"role": "user", "content": "hello"}],
            "model_reasoning_effort": "low",
            "model_reasoning_summary": "concise",
        }
    )

    assert "model_reasoning_effort" not in request
    assert "model_reasoning_summary" not in request
    assert request["reasoning"] == {"effort": "low", "summary": "concise"}


def test_codex_request_encodes_assistant_messages_as_output_text():
    request = codex_lm._build_codex_request(
        {
            "model": "openai/gpt-5.5",
            "messages": [
                {"role": "system", "content": "Follow the schema."},
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "question"}],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "answer"},
                        {"type": "input_text", "text": "more answer"},
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "output_text", "text": "followup"}],
                },
            ],
        }
    )

    assert request["instructions"] == "Follow the schema."
    assert request["input"][0] == {
        "role": "user",
        "content": [{"type": "input_text", "text": "question"}],
    }
    assert request["input"][1] == {
        "role": "assistant",
        "content": [
            {"type": "output_text", "text": "answer"},
            {"type": "output_text", "text": "more answer"},
        ],
    }
    assert request["input"][2] == {
        "role": "user",
        "content": [{"type": "input_text", "text": "followup"}],
    }


def test_labeled_fewshot_demos_build_valid_codex_responses_request():
    from dspy.adapters import ChatAdapter
    from dspy.teleprompt import LabeledFewShot

    class QA(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    trainset = [
        dspy.Example(question="2+2?", answer="4").with_inputs("question"),
    ]
    compiled = LabeledFewShot(k=1).compile(
        dspy.Predict(QA),
        trainset=trainset,
        sample=False,
    )
    predictor = compiled.predictors()[0]
    messages = ChatAdapter().format(
        predictor.signature,
        predictor.demos,
        {"question": "3+3?"},
    )

    assert any(message["role"] == "assistant" for message in messages)

    request = codex_lm._build_codex_request(
        {"model": "openai/gpt-5.5", "messages": messages}
    )

    assistant_messages = [
        message for message in request["input"] if message["role"] == "assistant"
    ]
    assert assistant_messages
    for message in assistant_messages:
        assert message["content"]
        assert all(block["type"] != "input_text" for block in message["content"])


def test_gepa_reflection_lm_prompt_path_uses_codex_adapter(monkeypatch):
    from dspy.teleprompt.gepa.gepa_utils import DspyAdapter

    captured_request = {}

    def fake_responses(**kwargs):
        captured_request.update(kwargs)
        return FakeResponsesStream(
            events=[
                SimpleNamespace(
                    type="response.output_text.done",
                    output_index=0,
                    content_index=0,
                    text="Use a tighter instruction.",
                )
            ],
            response=make_response(),
        )

    monkeypatch.setattr(codex_lm.litellm, "responses", fake_responses)

    lm = dspy_codex_auth.LM(
        "openai/gpt-5.5",
        auth_provider="codex",
        api_key="dummy",
        chatgpt_account_id="acct_test",
        cache=False,
    )
    adapter = DspyAdapter(
        student_module=dspy.Predict("question -> answer"),
        metric_fn=lambda *args: 1.0,
        feedback_map={},
        reflection_lm=lm,
    )

    assert adapter.stripped_lm_call("Reflect on this trajectory.") == [
        "Use a tighter instruction."
    ]
    assert captured_request["input"] == [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "Reflect on this trajectory."}],
        }
    ]


def test_gepa_compile_smoke_uses_codex_lm_without_extra_patches(monkeypatch):
    from dspy.teleprompt import GEPA

    captured_inputs = []

    def fake_responses(**kwargs):
        captured_inputs.append(kwargs["input"])
        input_text = "\n".join(
            block.get("text", "")
            for message in kwargs["input"]
            for block in message["content"]
            if isinstance(block, dict)
        )
        if "Your task is to write a new instruction" in input_text:
            text = (
                "```Given the fields `question`, produce the fields `answer`. "
                "Return exactly the expected answer.```"
            )
        else:
            text = "[[ ## answer ## ]]\n4\n\n[[ ## completed ## ]]"

        return FakeResponsesStream(
            events=[
                SimpleNamespace(
                    type="response.output_text.done",
                    output_index=0,
                    content_index=0,
                    text=text,
                )
            ],
            response=make_response(),
        )

    monkeypatch.setattr(codex_lm.litellm, "responses", fake_responses)

    lm = dspy_codex_auth.LM(
        "openai/gpt-5.5",
        auth_provider="codex",
        api_key="dummy",
        chatgpt_account_id="acct_test",
        cache=False,
    )
    student = dspy.Predict("question -> answer")
    trainset = [dspy.Example(question="2+2?", answer="4").with_inputs("question")]

    def metric(example, pred, trace=None, pred_name=None, pred_trace=None):
        return 1.0 if pred.answer == example.answer else 0.0

    optimizer = GEPA(
        metric=metric,
        max_metric_calls=2,
        reflection_lm=lm,
        use_merge=False,
        skip_perfect_score=False,
        reflection_minibatch_size=1,
        add_format_failure_as_feedback=True,
        num_threads=1,
    )

    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        compiled = optimizer.compile(student, trainset=trainset, valset=trainset)

    assert compiled.signature.instructions
    assert captured_inputs
    assert all(
        block["type"] == "input_text"
        for input_messages in captured_inputs
        for message in input_messages
        for block in message["content"]
        if "text" in block
    )


def test_stream_reconstructs_message_from_output_item_done():
    stream = FakeResponsesStream(
        events=[
            SimpleNamespace(
                type="response.output_item.done",
                output_index=0,
                item=SimpleNamespace(
                    type="message",
                    content=[{"type": "output_text", "text": "hello"}],
                ),
            )
        ],
        response=make_response(),
    )

    response = codex_lm._consume_codex_response_stream(stream)

    assert response.output[0].type == "message"
    assert response.output[0].content[0].text == "hello"


def test_stream_reconstructs_message_from_text_done_when_output_item_is_empty():
    stream = FakeResponsesStream(
        events=[
            SimpleNamespace(
                type="response.output_item.done",
                output_index=0,
                item=SimpleNamespace(type="message", content=[]),
            ),
            SimpleNamespace(
                type="response.output_text.done",
                output_index=0,
                content_index=0,
                text="hello from done",
            ),
        ],
        response=make_response(),
    )

    response = codex_lm._consume_codex_response_stream(stream)

    assert response.output[0].content[0].text == "hello from done"


def test_stream_reconstructs_message_from_deltas():
    stream = FakeResponsesStream(
        events=[
            SimpleNamespace(
                type="response.output_text.delta",
                output_index=0,
                content_index=0,
                delta="hel",
            ),
            SimpleNamespace(
                type="response.output_text.delta",
                output_index=0,
                content_index=0,
                delta="lo",
            ),
        ],
        response=make_response(),
    )

    response = codex_lm._consume_codex_response_stream(stream)

    assert response.output[0].content[0].text == "hello"


def test_stream_preserves_reasoning_summary():
    stream = FakeResponsesStream(
        events=[
            SimpleNamespace(
                type="response.reasoning_summary_text.done",
                output_index=0,
                summary_index=0,
                text="Used the normal CDF difference.",
            ),
            SimpleNamespace(
                type="response.output_text.done",
                output_index=1,
                content_index=0,
                text="0.4332",
            ),
        ],
        response=make_response(),
    )

    response = codex_lm._consume_codex_response_stream(stream)
    lm = dspy_codex_auth.LM(
        "openai/gpt-5.5", api_key="dummy", api_base="http://example.invalid"
    )
    lm.model_type = "responses"

    outputs = lm._process_lm_response(response, prompt="x", messages=None)

    assert outputs == [
        {
            "reasoning_content": "Used the normal CDF difference.",
            "text": "0.4332",
        }
    ]


def test_non_codex_routes_fall_through(monkeypatch):
    called = {}

    def fake_forward(self, prompt=None, messages=None, **kwargs):
        called["prompt"] = prompt
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="ok", tool_calls=None),
                    logprobs=None,
                )
            ],
            model="fake",
            usage={},
        )

    monkeypatch.setattr(codex_lm._DSPY_LM, "forward", fake_forward)

    lm = dspy_codex_auth.LM(
        "openai/test", api_key="dummy", api_base="http://example.invalid"
    )
    assert lm("hello") == ["ok"]
    assert called["prompt"] == "hello"
