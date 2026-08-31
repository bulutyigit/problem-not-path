from __future__ import annotations

from collections import UserDict
from types import SimpleNamespace

import pytest

from reasonbench.config import ModelConfig
from reasonbench.exceptions import InstrumentationError
from reasonbench.generation.engine import InstrumentedGenerator, _prepare_inputs
from reasonbench.generation.segments import segment_generated_tokens, segment_token_texts
from reasonbench.instrumentation.adapters.gemma4 import Gemma4Adapter
from reasonbench.instrumentation.adapters.ministral3 import Ministral3Adapter
from reasonbench.instrumentation.recorder import TokenSignal, TokenSignalRecorder
from reasonbench.verification.extract import split_reasoning_and_answer


class TinyTokenizer:
    def __init__(self) -> None:
        self.mapping = {
            "</think>": [90],
            "[/THINK]": [93],
            "<channel|>": [94],
            "<|channel>thought": [95],
            "\\boxed{": [91],
            "\\fbox{": [92],
        }
        self.reverse_mapping = {
            ids[0]: text for text, ids in self.mapping.items()
        }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return self.mapping.get(text, [])

    def decode(self, token_ids, **kwargs) -> str:
        del kwargs
        return "".join(
            self.reverse_mapping.get(token_id, f"t{token_id} ") for token_id in token_ids
        )


class FakeTensor:
    shape = (1, 4)

    def to(self, device):
        del device
        return self


class BatchLike(UserDict):
    """Minimal non-dict mapping matching BatchFeature/BatchEncoding behavior."""

    def to(self, device):
        del device
        return self


class BatchReturningProcessor:
    def __init__(self, batch) -> None:
        self.batch = batch

    def apply_chat_template(self, messages, **kwargs):
        del messages, kwargs
        return self.batch


class TorchBatchProcessor:
    def apply_chat_template(self, messages, **kwargs):
        del messages, kwargs
        import torch

        return {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones((1, 2))}


class ContinuationTokenizer(TinyTokenizer):
    pad_token_id = 0
    eos_token_id = 9


class CapturingGenerateModel:
    device = "cpu"

    def __init__(self) -> None:
        self.received = None

    def generate(self, **kwargs):
        import torch

        self.received = kwargs
        return torch.cat([kwargs["input_ids"], torch.tensor([[8, 9]])], dim=-1)


def _signal(token_id: int) -> TokenSignal:
    return TokenSignal(
        token_index=0,
        token_id=token_id,
        token_text=str(token_id),
        entropy=1.0,
        normalized_entropy=0.1,
        top1_top2_logit_margin=1.0,
        top1_top2_probability_margin=0.2,
        top1_probability=0.45,
        top5_probability_mass=0.75,
        probability_tail_mass=0.25,
        effective_vocabulary_size=12.0,
        sampled_logprob=-0.5,
        sampled_token_regret=0.1,
        surprisal=0.5,
        successive_kl_divergence=None,
        successive_js_divergence=None,
        hidden_norm=2.0,
        relative_l2_step=None,
        cosine_drift=None,
    )


def test_reasoning_and_final_answer_segmentation() -> None:
    segments = segment_generated_tokens(
        [1, 2, 90, 3, 91, 4, 5],
        TinyTokenizer(),
        mode="reasoning",
    )
    assert segments == [
        "thinking",
        "thinking",
        "special",
        "solution",
        "final_answer",
        "final_answer",
        "final_answer",
    ]


def test_prepare_inputs_preserves_non_dict_mapping_payload() -> None:
    input_ids = FakeTensor()
    attention_mask = FakeTensor()
    batch = BatchLike(
        {"input_ids": input_ids, "attention_mask": attention_mask}
    )
    assert not isinstance(batch, dict)
    bundle = SimpleNamespace(
        adapter=Gemma4Adapter(),
        tokenizer=TinyTokenizer(),
        processor=BatchReturningProcessor(batch),
        model=SimpleNamespace(device="cuda:0"),
        model_config=SimpleNamespace(max_prompt_tokens=16),
    )
    prepared = _prepare_inputs(bundle, "What is 2 + 2?", "reasoning")
    assert isinstance(prepared, dict)
    assert prepared["input_ids"] is input_ids
    assert prepared["attention_mask"] is attention_mask


def test_breakthrough_continuation_replays_exact_generated_prefix() -> None:
    pytest.importorskip("torch")
    model = CapturingGenerateModel()
    config = ModelConfig(
        key="test",
        model_id="org/test",
        adapter="gemma4",
        max_new_tokens=16,
        capture_hidden_states=False,
    )
    bundle = SimpleNamespace(
        adapter=Gemma4Adapter(),
        tokenizer=ContinuationTokenizer(),
        processor=TorchBatchProcessor(),
        model=model,
        model_config=config,
    )

    result = InstrumentedGenerator(bundle).continue_from_generated_prefix(
        "What is 2 + 2?", [3, 4, 5], max_total_generated_tokens=10
    )

    assert model.received["input_ids"].tolist() == [[1, 2, 3, 4, 5]]
    assert model.received["max_new_tokens"] == 7
    assert result.generated_token_ids == [3, 4, 5, 8, 9]
    assert result.continuation_token_ids == [8, 9]
    assert result.finish_reason == "eos"


def test_recorder_alignment_detects_mismatch() -> None:
    recorder = TokenSignalRecorder(TinyTokenizer())
    recorder.records = [_signal(1), _signal(2)]
    recorder.validate_alignment([1, 2])
    with pytest.raises(InstrumentationError, match="alignment failed"):
        recorder.validate_alignment([1, 3])


def test_recorder_captures_distribution_and_transition_signals() -> None:
    torch = pytest.importorskip("torch")
    recorder = TokenSignalRecorder(TinyTokenizer(), capture_hidden_states=False)

    def record(logits, token_id: int) -> None:
        recorder._pending = {
            "hidden_norm": 2.0,
            "relative_l2_step": None,
            "cosine_drift": None,
        }
        recorder._capture_logits_output(None, (), torch.tensor([logits], dtype=torch.float32))
        recorder.finalize_sampled_token(token_id)

    record([4.0, 1.0, 0.0, -1.0, -2.0, -3.0], 0)
    record([0.0, 4.0, 1.0, -1.0, -2.0, -3.0], 1)

    first, second = recorder.records
    assert first.successive_kl_divergence is None
    assert first.successive_js_divergence is None
    assert 0.0 < first.top1_probability < 1.0
    assert 0.0 < first.top5_probability_mass <= 1.0
    assert first.probability_tail_mass > 0.0
    assert first.effective_vocabulary_size > 1.0
    assert first.sampled_token_regret == 0.0
    assert second.successive_kl_divergence is not None
    assert second.successive_kl_divergence > 0.0
    assert second.successive_js_divergence is not None
    assert 0.0 < second.successive_js_divergence <= 0.6932


def test_ministral_reasoning_markers_are_segmented() -> None:
    adapter = Ministral3Adapter()
    assert adapter.reasoning_close_markers() == ("[/THINK]",)
    assert adapter.reasoning_open_markers() == ("[THINK]",)
    segments = segment_generated_tokens(
        [1, 2, 93, 3, 91, 4],
        TinyTokenizer(),
        mode="reasoning",
        reasoning_close_markers=adapter.reasoning_close_markers(),
        reasoning_open_markers=adapter.reasoning_open_markers(),
    )
    assert segments == [
        "thinking",
        "thinking",
        "special",
        "solution",
        "final_answer",
        "final_answer",
    ]
    reasoning, final, status = split_reasoning_and_answer("[THINK]Work[/THINK] Result: \\boxed{4}")
    assert reasoning == "Work"
    assert final == "Result: \\boxed{4}"
    assert status == "mistral_think_tag"


def test_segmentation_matches_markers_split_across_tokens() -> None:
    segments = segment_token_texts(
        ["Okay, ", "[/TH", "INK]", " Result: ", "\\boxed{", "4}"],
        mode="reasoning",
        reasoning_close_markers=("[/THINK]",),
        reasoning_open_markers=("[THINK]",),
    )
    assert segments == [
        "thinking",
        "special",
        "special",
        "solution",
        "final_answer",
        "final_answer",
    ]


def test_segmentation_matches_boxed_answer_split_across_tokens() -> None:
    segments = segment_token_texts(
        ["reason ", "</think>", "so ", "\\box", "ed{4} done"],
        mode="reasoning",
        reasoning_close_markers=("</think>",),
        reasoning_open_markers=("<think>",),
    )
    assert segments == [
        "thinking",
        "special",
        "solution",
        "final_answer",
        "final_answer",
    ]


def test_ministral_adapter_uses_structured_official_prompt() -> None:
    adapter = Ministral3Adapter()
    adapter.configure_system_prompt(
        "Before [THINK]Example reasoning[/THINK] After",
        source="fixture",
    )
    plan = adapter.build_prompt("What is 2 + 2?", "reasoning")
    assert plan.messages[0]["role"] == "system"
    content = plan.messages[0]["content"]
    assert isinstance(content, list)
    assert content[1]["type"] == "thinking"
    assert content[1]["closed"] is True


def test_gemma4_adapter_uses_official_thinking_toggle_and_boundary() -> None:
    adapter = Gemma4Adapter()
    reasoning = adapter.build_prompt("What is 2 + 2?", "reasoning")
    non_reasoning = adapter.build_prompt("What is 2 + 2?", "non_reasoning")
    assert reasoning.template_kwargs == {"enable_thinking": True}
    assert non_reasoning.template_kwargs == {"enable_thinking": False}
    assert adapter.reasoning_close_markers() == ("<channel|>",)
    assert adapter.forced_reasoning_close_text() == "<channel|>"


def test_gemma4_thought_channel_is_segmented_and_parsed() -> None:
    segments = segment_generated_tokens(
        [95, 1, 2, 94, 3, 91, 4],
        TinyTokenizer(),
        mode="reasoning",
        reasoning_close_markers=("<channel|>",),
        reasoning_open_markers=("<|channel>thought",),
    )
    assert segments == [
        "special",
        "thinking",
        "thinking",
        "special",
        "solution",
        "final_answer",
        "final_answer",
    ]
    reasoning, final, status = split_reasoning_and_answer(
        "<|channel>thought\nWork through it.<channel|>Result: \\boxed{4}"
    )
    assert reasoning == "Work through it."
    assert final == "Result: \\boxed{4}"
    assert status == "gemma_thought_channel"
