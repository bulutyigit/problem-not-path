"""MLX-native token instrumentation with the CUDA recorder's exact schema."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from reasonbench.exceptions import InstrumentationError
from reasonbench.instrumentation.recorder import TokenSignal


class MLXTokenSignalRecorder:
    """Summarize MLX log-probabilities and final-normalized hidden states.

    All vocabulary-sized calculations stay on the Metal device.  Only scalar
    summaries and strided hidden vectors cross to CPU, which is important for
    Qwen's large vocabulary and long 16K-token trajectories.
    """

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
        self.hidden_states: list[np.ndarray] = []
        self._previous_hidden: Any | None = None
        self._previous_log_probabilities: Any | None = None

    def finalize_sampled_token(
        self,
        token_id: int,
        log_probabilities: Any,
        hidden_state: Any,
    ) -> None:
        """Commit one sampled token and its predictive state."""

        import mlx.core as mx

        log_probabilities = log_probabilities.reshape(-1).astype(mx.float32)
        hidden = hidden_state.reshape(-1, hidden_state.shape[-1])[-1].astype(mx.float32)
        vocabulary_size = int(log_probabilities.shape[-1])
        if vocabulary_size < 2:
            raise InstrumentationError("MLX logits must contain at least two vocabulary items")

        probabilities = mx.exp(log_probabilities)
        entropy = -mx.sum(probabilities * log_probabilities)
        top5 = mx.sort(mx.topk(log_probabilities, k=min(5, vocabulary_size)))
        top1 = top5[-1]
        top2 = top5[-2]
        top1_probability = mx.exp(top1)
        top2_probability = mx.exp(top2)
        top5_probability_mass = mx.sum(mx.exp(top5))
        sampled_logprob = log_probabilities[int(token_id)]

        hidden_norm = mx.linalg.norm(hidden)
        relative_step = None
        cosine_drift = None
        if self._previous_hidden is not None:
            previous_hidden = self._previous_hidden
            previous_norm = mx.maximum(mx.linalg.norm(previous_hidden), 1e-12)
            relative_step = mx.linalg.norm(hidden - previous_hidden) / previous_norm
            cosine = mx.sum(hidden * previous_hidden) / (
                mx.maximum(hidden_norm, 1e-12) * previous_norm
            )
            cosine_drift = 1.0 - mx.clip(cosine, -1.0, 1.0)

        successive_kl = None
        successive_js = None
        if self._previous_log_probabilities is not None:
            previous = self._previous_log_probabilities
            successive_kl = mx.sum(probabilities * (log_probabilities - previous))
            mixture = mx.logaddexp(log_probabilities, previous) - math.log(2.0)
            successive_js = 0.5 * (
                mx.sum(probabilities * (log_probabilities - mixture))
                + mx.sum(mx.exp(previous) * (previous - mixture))
            )

        scalar_arrays = [
            entropy,
            top1,
            top2,
            top1_probability,
            top2_probability,
            top5_probability_mass,
            sampled_logprob,
            hidden_norm,
        ]
        optional_arrays = [
            value
            for value in (relative_step, cosine_drift, successive_kl, successive_js)
            if value is not None
        ]
        mx.eval(*(scalar_arrays + optional_arrays))

        token_index = len(self.records)
        if self.capture_hidden_states and token_index % self.hidden_state_stride == 0:
            hidden_cpu = np.asarray(hidden.astype(mx.float16))
            self.hidden_state_indices.append(token_index)
            self.hidden_states.append(hidden_cpu)

        entropy_value = float(entropy.item())
        sampled_logprob_value = float(sampled_logprob.item())
        self.records.append(
            TokenSignal(
                token_index=token_index,
                token_id=int(token_id),
                token_text=self.tokenizer.decode(
                    [int(token_id)],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                entropy=entropy_value,
                normalized_entropy=entropy_value / math.log(vocabulary_size),
                top1_top2_logit_margin=float((top1 - top2).item()),
                top1_top2_probability_margin=float(
                    (top1_probability - top2_probability).item()
                ),
                top1_probability=float(top1_probability.item()),
                top5_probability_mass=float(top5_probability_mass.item()),
                probability_tail_mass=float(1.0 - top5_probability_mass.item()),
                effective_vocabulary_size=math.exp(min(entropy_value, 80.0)),
                sampled_logprob=sampled_logprob_value,
                sampled_token_regret=float((top1 - sampled_logprob).item()),
                surprisal=-sampled_logprob_value,
                successive_kl_divergence=(
                    None if successive_kl is None else float(successive_kl.item())
                ),
                successive_js_divergence=(
                    None if successive_js is None else float(successive_js.item())
                ),
                hidden_norm=float(hidden_norm.item()),
                relative_l2_step=(
                    None if relative_step is None else float(relative_step.item())
                ),
                cosine_drift=(
                    None if cosine_drift is None else float(cosine_drift.item())
                ),
            )
        )
        self._previous_hidden = hidden
        self._previous_log_probabilities = log_probabilities

    def set_segments(self, segments: list[str]) -> None:
        if len(segments) != len(self.records):
            raise InstrumentationError(
                f"Segment count {len(segments)} does not match record count {len(self.records)}"
            )
        for record, segment in zip(self.records, segments, strict=True):
            record.segment = segment

    def validate_alignment(self, generated_token_ids: list[int]) -> None:
        recorded = [record.token_id for record in self.records]
        if recorded != generated_token_ids:
            raise InstrumentationError(
                "MLX token/signal alignment failed: "
                f"recorded={len(recorded)}, generated={len(generated_token_ids)}"
            )
