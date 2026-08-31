"""Capture scalar logits and final-normalized hidden-state dynamics during generation."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from reasonbench.exceptions import InstrumentationError

TOKEN_METRIC_SCHEMA_VERSION = "v2_distribution_transitions"


@dataclass
class TokenSignal:
    """Signals aligned to one sampled token."""

    token_index: int
    token_id: int
    token_text: str
    entropy: float
    normalized_entropy: float
    top1_top2_logit_margin: float
    top1_top2_probability_margin: float
    top1_probability: float
    top5_probability_mass: float
    probability_tail_mass: float
    effective_vocabulary_size: float
    sampled_logprob: float
    sampled_token_regret: float
    surprisal: float
    successive_kl_divergence: float | None
    successive_js_divergence: float | None
    hidden_norm: float
    relative_l2_step: float | None
    cosine_drift: float | None
    segment: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TokenSignalRecorder:
    """Attach to a model output head and summarize one decoding step at a time."""

    def __init__(
        self,
        tokenizer: Any,
        hidden_state_stride: int = 8,
        capture_hidden_states: bool = True,
    ) -> None:
        if hidden_state_stride <= 0:
            raise ValueError("hidden_state_stride must be positive")
        self.tokenizer = tokenizer
        self.hidden_state_stride = hidden_state_stride
        self.capture_hidden_states = capture_hidden_states
        self.records: list[TokenSignal] = []
        self.hidden_state_indices: list[int] = []
        self.hidden_states: list[Any] = []
        self._pending: dict[str, Any] | None = None
        self._previous_hidden: Any | None = None
        # Keeping only one previous vocabulary-sized vector lets us measure how
        # the predictive distribution changes between two generated tokens.  It
        # is deliberately updated after a sampled token is committed, so the
        # first generated token has no artificial comparison to prompt logits.
        self._previous_sample_log_probabilities: Any | None = None
        self._handles: list[Any] = []

    def reset(self) -> None:
        """Clear all captured state before a generation call."""

        self.records.clear()
        self.hidden_state_indices.clear()
        self.hidden_states.clear()
        self._pending = None
        self._previous_hidden = None
        self._previous_sample_log_probabilities = None

    def attach(self, model: Any) -> None:
        """Attach hooks to the language-model output head."""

        self.detach()
        output_head = model.get_output_embeddings()
        if output_head is None:
            raise InstrumentationError("model.get_output_embeddings() returned None")
        self._handles = [
            output_head.register_forward_pre_hook(self._capture_hidden_input),
            output_head.register_forward_hook(self._capture_logits_output),
        ]

    def detach(self) -> None:
        """Remove active hooks."""

        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _capture_hidden_input(self, module: Any, args: tuple[Any, ...]) -> None:
        del module
        if not args:
            raise InstrumentationError("The language-model head received no positional input")
        hidden = args[0]
        if isinstance(hidden, (tuple, list)):
            hidden = hidden[0]
        if hidden.ndim < 2:
            raise InstrumentationError(
                f"Expected hidden state with at least two dimensions, got {hidden.shape}"
            )
        vector = hidden.reshape(-1, hidden.shape[-1])[-1].detach()
        vector_float = vector.float()
        hidden_norm_tensor = vector_float.norm(p=2)
        relative_step: float | None = None
        cosine_drift: float | None = None
        if self._previous_hidden is not None:
            previous = self._previous_hidden
            difference = (vector_float - previous).norm(p=2)
            relative_step = float((difference / previous.norm(p=2).clamp_min(1e-12)).item())
            cosine = (vector_float * previous).sum() / (
                vector_float.norm(p=2).clamp_min(1e-12) * previous.norm(p=2).clamp_min(1e-12)
            )
            cosine_drift = float((1.0 - cosine.clamp(-1.0, 1.0)).item())
        self._previous_hidden = vector_float
        token_index = len(self.records)
        if self.capture_hidden_states and token_index % self.hidden_state_stride == 0:
            self.hidden_state_indices.append(token_index)
            self.hidden_states.append(vector.to(dtype=vector.dtype, device="cpu"))
        self._pending = {
            "hidden_norm": float(hidden_norm_tensor.item()),
            "relative_l2_step": relative_step,
            "cosine_drift": cosine_drift,
        }

    def _capture_logits_output(self, module: Any, args: tuple[Any, ...], output: Any) -> None:
        import torch

        del module, args
        if self._pending is None:
            raise InstrumentationError("Logits were produced before the hidden-state hook fired")
        logits = output[0] if isinstance(output, (tuple, list)) else output
        vector = logits.reshape(-1, logits.shape[-1])[-1].detach().float()
        logsumexp = vector.logsumexp(dim=-1)
        log_probabilities = vector - logsumexp
        probabilities = log_probabilities.exp()
        entropy = -(probabilities * log_probabilities).sum()
        top_logits, top_indices = vector.topk(k=2, dim=-1)
        top_probabilities = probabilities[top_indices]
        top5_probability_mass = probabilities.topk(
            k=min(5, vector.shape[-1]), dim=-1
        ).values.sum()
        previous = self._previous_sample_log_probabilities
        if previous is None:
            successive_kl: float | None = None
            successive_js: float | None = None
        else:
            # KL is directional (current distribution relative to the prior
            # token); Jensen--Shannon is bounded and symmetric.  Their pairing
            # distinguishes a large one-way redistribution from a general
            # change in the model's local predictive beliefs.
            kl = (probabilities * (log_probabilities - previous)).sum()
            mixture_log_probabilities = torch.logaddexp(
                log_probabilities, previous
            ) - math.log(2.0)
            mixture_kl_current = (
                probabilities * (log_probabilities - mixture_log_probabilities)
            ).sum()
            previous_probabilities = previous.exp()
            mixture_kl_previous = (
                previous_probabilities * (previous - mixture_log_probabilities)
            ).sum()
            successive_kl = float(kl.item())
            successive_js = float((0.5 * (mixture_kl_current + mixture_kl_previous)).item())
        vocabulary_size = vector.shape[-1]
        self._pending.update(
            {
                "entropy": float(entropy.item()),
                "normalized_entropy": float((entropy / math.log(vocabulary_size)).item()),
                "top1_top2_logit_margin": float((top_logits[0] - top_logits[1]).item()),
                "top1_top2_probability_margin": float(
                    (top_probabilities[0] - top_probabilities[1]).item()
                ),
                "top1_probability": float(top_probabilities[0].item()),
                "top5_probability_mass": float(top5_probability_mass.item()),
                "probability_tail_mass": float((1.0 - top5_probability_mass).item()),
                "effective_vocabulary_size": float(math.exp(min(float(entropy.item()), 80.0))),
                "successive_kl_divergence": successive_kl,
                "successive_js_divergence": successive_js,
                "logits": vector,
                "logsumexp": logsumexp,
                "log_probabilities": log_probabilities,
            }
        )

    def finalize_sampled_token(self, token_id: int) -> None:
        """Align the latest forward pass to the token sampled by generate()."""

        if self._pending is None or "logits" not in self._pending:
            raise InstrumentationError("No complete pending model step is available")
        logits = self._pending.pop("logits")
        logsumexp = self._pending.pop("logsumexp")
        log_probabilities = self._pending.pop("log_probabilities")
        sampled_logprob = float((logits[token_id] - logsumexp).item())
        token_text = self.tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        self.records.append(
            TokenSignal(
                token_index=len(self.records),
                token_id=int(token_id),
                token_text=token_text,
                sampled_logprob=sampled_logprob,
                surprisal=-sampled_logprob,
                sampled_token_regret=float((logits.max() - logits[token_id]).item()),
                **self._pending,
            )
        )
        self._previous_sample_log_probabilities = log_probabilities
        self._pending = None

    def set_segments(self, segments: list[str]) -> None:
        """Assign one segment label per sampled token."""

        if len(segments) != len(self.records):
            raise InstrumentationError(
                f"Segment count {len(segments)} does not match record count {len(self.records)}"
            )
        for record, segment in zip(self.records, segments, strict=True):
            record.segment = segment

    def validate_alignment(self, generated_token_ids: list[int]) -> None:
        """Require exact alignment between generated tokens and captured steps."""

        recorded = [record.token_id for record in self.records]
        if recorded != generated_token_ids:
            mismatch = next(
                (
                    index
                    for index, pair in enumerate(zip(recorded, generated_token_ids, strict=False))
                    if pair[0] != pair[1]
                ),
                min(len(recorded), len(generated_token_ids)),
            )
            raise InstrumentationError(
                "Token/signal alignment failed at index "
                f"{mismatch}: recorded={len(recorded)}, generated={len(generated_token_ids)}"
            )


class RecorderStoppingCriteria:
    """Finalize recorder state after each sampled token and optionally stop on a suffix."""

    def __init__(
        self,
        recorder: TokenSignalRecorder,
        stop_sequences: list[list[int]] | None = None,
        on_token: Callable[[int], None] | None = None,
    ) -> None:
        self.recorder = recorder
        self.stop_sequences = [sequence for sequence in (stop_sequences or []) if sequence]
        self.on_token = on_token

    def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
        del scores, kwargs
        import torch

        if input_ids.shape[0] != 1:
            raise InstrumentationError("Instrumentation currently requires batch size 1")
        token_id = int(input_ids[0, -1].item())
        self.recorder.finalize_sampled_token(token_id)
        if self.on_token is not None:
            self.on_token(len(self.recorder.records))
        tokens = input_ids[0].tolist()
        should_stop = any(
            len(tokens) >= len(sequence) and tokens[-len(sequence) :] == sequence
            for sequence in self.stop_sequences
        )
        return torch.tensor([should_stop], dtype=torch.bool, device=input_ids.device)
