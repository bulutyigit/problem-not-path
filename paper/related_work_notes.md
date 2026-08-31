# Related-work sweep notes (2026-08-28)

Five parallel web sweeps over: (1) truncation/probing, (2) aha-moment +
overthinking, (3) test-time compute, (4) critical tokens + PRM value curves,
(5) outcome prediction from internal signals. Every entry verified to exist
(link fetched). Threat = does it preempt one of our claims.

Our claims, for reference:
- C1 (instrument): restart-controlled, budget-indexed, censoring-aware
  truncation probe; T_F (budget fit) vs T_V (prefix value) decomposition.
- C2 (artifact finding): naive breakthrough labels are mostly budget
  artifacts; 1/178 genuine prefix-limited cell.
- C3 (dose-response): restart curves separate compute-starved vs
  capability-limited failure modes per model.
- C4 (compression): at matched total compute own-prefix beats restart 11/13,
  large-budget restart catches up 11/13 — compression, not unlock.
- C5 (stochastic band): ~1/3 of ambiguous states genuinely intermediate
  (0.3–0.6 at n=8).
- C6 (prediction negative): pre-registered, difficulty-controlled null for
  early internal signals at every forecast point 128–2048.

## Area 1 — truncation / resampling probes (agent verdict)

**Verdict: no published paper implements the restart control (empty-prefix
from-scratch R(C) at matched TOTAL compute separating prefix value from
budget fit). The truncate-and-resample probe itself is crowded. Three papers
graze the restart axis; Re² is the single biggest must-cite.**

| Paper | What | Diff vs ours | Threat |
|---|---|---|---|
| Thought Anchors (Bogdan, Macar, Nanda, Conmy 2025, arXiv:2506.19143) | Sentence-level counterfactual importance via replacement resampling + continuation; attention analyses | Importance of sentences, not p̂(t) solve-rate curves; no restart control, no compute matching | MEDIUM |
| The Point of No Return (Merrill & Srivastava 2026, arXiv:2605.17113) | Prefix-fixed resampling, 94M continuations, localizes where a (deceptive) outcome "locks in" | Closest probe shape; outcome = deception not math solve rate; no restart control, no censoring; the term "point of no return" is taken | MEDIUM (HIGH on framing, LOW on restart) |
| Re² (Wang et al. 2026, arXiv:2603.07197) | Truncates incorrect traces at 20–80%, continues, measures accuracy drop "vs reasoning from scratch"; RL method teaching models to restart | Closest continue-vs-scratch comparison, but motivational pilot, incorrect prefixes only, no matched-total-compute, no event curves, inverse claim (bad prefixes trap) | **MEDIUM — must cite & differentiate** |
| Fractured Sampling (Liao et al. 2025, arXiv:2505.12992) | Sweeps #trajectories × truncation depth × solutions per trajectory; Pass@k vs token budget | Efficiency recipe; implicitly trades restarts vs prefix depth but no per-anchor value, no margin test | MEDIUM |
| Ballon et al. 2026 (arXiv:2601.23163) | Truncate at percentiles, read induced answer distribution via next-token probs; monotone commitment; length-matched content controls | No sampled continuations/budget; controls are content ablations, not from-scratch generation | MEDIUM |
| Wolf, Wies, Shashua 2026 (arXiv:2607.06720) | Theory: when reflection can't localize errors, continuing ≈ independent restarts | Whole-attempt-level theory; our C4 is an empirical instance of their "no benefit" regime — cite as theoretical frame | MEDIUM (conceptual) |
| Thought Branches (Macar et al. 2025, arXiv:2510.27484) | Position paper: resampling is required for CoT interpretation | Establishes the primitive as standard; no value curves/restart | MEDIUM |
| Forking Paths (Bigelow et al., ICLR 2025, arXiv:2412.07961) | Token-level resampling; sharp "forking tokens" | Outcome-shift analysis, no budget, no restart | LOW |
| Lanham et al. 2023 (arXiv:2307.13702) | Early-answering truncation curves for faithfulness | Forces immediate answer; canonical ancestor | LOW |
| Prefix Consistency (Iwase et al. 2026, arXiv:2605.07654) | Truncate-resample self-agreement as answer-selection signal | Selection/uncertainty, not value measurement | LOW |
| Math-Shepherd (Wang et al. 2023, arXiv:2312.08935) + ProcessThinker (arXiv:2606.11209) | Per-step rollout success rate as PRM training labels | Same estimator, different purpose; no restart, no censoring | LOW |
| The Illusion of Insight (d'Aliberti & Horta Ribeiro 2026, arXiv:2601.00514) | 1M+ traces: mid-reasoning strategy shifts rare, don't improve accuracy | Conclusion-adjacent to C2/C4 via trace statistics, not probes — cite as convergent evidence | LOW |
| Also: Answer Convergence (arXiv:2506.02536), ES-CoT (arXiv:2509.14004), Forking Fast (arXiv:2608.19611) | Early-stopping / efficiency cousins | No probes/restart | LOW |

Positioning implications (area 1):
- Cite Thought Anchors/Thought Branches as the resampling-probe tradition we
  instrument differently (value curves + events, not importance).
- Re² gets its own differentiation sentence: they show bad prefixes trap
  models and train restarting away; we measure, per anchor and at matched
  total compute, whether good prefixes carry value a restart cannot buy.
- Wolf et al. gives us a theory hook: C4 is empirical evidence for their
  "no asymptotic benefit" regime in small models.
- Avoid "point of no return" as terminology; keep "breakthrough / T_V".

## Area 2 — aha-moment claims + overthinking (agent verdict)

**Verdict: "aha moments are measurement artifacts" is NOT published in our
operational sense (no per-anchor resample + compute-matched restart control;
nobody frames breakthroughs as *budget* artifacts). BUT the narrative is
partially taken: Illusion of Insight (Jan 2026) says "insight is illusion"
correlationally; Ghosal et al. (NeurIPS 2025) shows parallel-at-same-budget
beats sequential at benchmark level. Pitch novelty on the counterfactual
compute-matched operationalization + budget-artifact decomposition.**

| Paper | What | Diff vs ours | Threat |
|---|---|---|---|
| DeepSeek-R1 (Guo et al. 2025, arXiv:2501.12948; Nature) | Origin of the "aha moment" claim; anecdotal trace + reflection-token surge | We are the direct stress test of this claim | LOW |
| There May Not be Aha Moment / Dr. GRPO (Liu et al. 2025, arXiv:2503.20783, COLM; sail.sea.com blog) | Aha keywords exist at epoch 0; "superficial self-reflection"; GRPO length bias | Debunks across training checkpoints via keywords; no within-trace counterfactual | MEDIUM |
| **The Illusion of Insight (d'Aliberti & Horta Ribeiro 2026, arXiv:2601.00514)** | 1M+ traces: shifts in ~6.3%, associated with LOWER accuracy; formal ahas ~1.8%; "unstable inference, not self-correction" | Closest published claim but correlational (shift vs no-shift traces); cannot separate prefix value from budget fit — exactly our contribution | **HIGH (narrative), methodologically distinct** |
| Ballon et al. 2026 (arXiv:2601.23163) | Truncation percentiles → answer distributions; gradual commitment, no jumps | Convergent "gradual" result; probability probing, no restart | MEDIUM |
| Can Aha Moments Be Fake? (Zhao et al. 2025, arXiv:2510.24941) | True Thinking Score via activation steering; many "aha" steps decorative | Internals-based step causality; complements our behavioral test | MEDIUM |
| First Try Matters (Kang et al. 2025, arXiv:2510.08308) | Reflections after first candidate answer rarely flip it | Convergent "no unlock from reflection"; anchor = first answer only | MEDIUM |
| Entropy Dynamics of CoT (Xu et al. 2026, arXiv:2606.02020) | Sharp two-phase entropy transition (CUSUM change points) | Could be cited AGAINST us; their "sharp commit" is entropy convergence, no restart control — our result says such commits are reachable from scratch | MEDIUM |
| Do NOT Think That Much (Chen et al., ICML 2025, arXiv:2412.21187) | Overthinking on trivial problems; efficiency metrics | Background for compute-compression view | LOW |
| Danger of Overthinking (Cuadron et al. 2025, arXiv:2502.08235) | Agentic overthinking correlates with worse performance | Background | LOW |
| Stop Overthinking survey (Sui et al. 2025, arXiv:2503.16419, TMLR) | Survey of efficient reasoning | Positioning only | LOW |
| **Does Thinking More Always Help? (Ghosal et al., NeurIPS 2025, arXiv:2506.04210)** | Non-monotone accuracy in thinking length; parallel independent samples at same budget beat sequential extension by up to 20% | Closest to our restart control but whole-benchmark inference strategy; we make it a per-anchor counterfactual test | **MEDIUM (HIGH on the sub-claim; must cite & differentiate)** |
| Inverse Scaling in Test-Time Compute (Gema et al. 2025, arXiv:2507.14417, TMLR) | Constructed tasks where longer reasoning lowers accuracy; five failure modes | Task-level, adversarial tasks; our non-monotone R(C) is on natural math | MEDIUM |

Second tier: Understanding Aha Moments (2504.02956); The Illusion of
Thinking, Apple (2506.06941); When More Thinking Hurts (2604.10739);
**Temporal Predictors of Outcome in Reasoning LMs (2511.14773) — linear
probes on early prefixes predict correctness with a difficulty-selection
caveat; in tension with our C6, must be addressed in the prediction
section**; Wait, We Don't Need to "Wait" (2506.08343); CyclicReflex
(2506.11077); VLM aha replications (2503.05132, 2506.17417); Not Thinking
Straight (2507.00711).

## Area 4 — critical tokens + PRM rollout-value lineage (agent verdict)

**Verdict: clear, with two must-engage neighbors. Nobody combines value
curves as phenomenon + censoring + restart control + intermediate-band
finding. But the Bigelow line (Forking Paths → Forking Fast, Aug 2026)
analyzes per-position resampled outcome distributions AS a phenomenon with a
"noise is a sampling artifact" claim that rhymes with C2 — engage in detail.
The MC estimator itself is standard (Math-Shepherd lineage): novelty must
rest on phenomenon claims and controls, not the estimator.**

| Paper | What | Diff vs ours | Threat |
|---|---|---|---|
| Math-Shepherd (Wang et al., ACL 2024, arXiv:2312.08935) | Step value = fraction of N=8 completions reaching correct answer; trains PRM | Identical MC primitive, different purpose; their 0.5 threshold erases exactly the band C5 studies | MEDIUM |
| Let's Verify Step by Step (Lightman et al., ICLR 2024, arXiv:2305.20050) | Human step labels, PRM800K | Background only | LOW |
| Critical Tokens Matter (Lin et al. 2024/25, arXiv:2411.19943) | Rollout-flipping culprit tokens in wrong traces → token-level cDPO | Single-token importance for training; no curves/timing/restart | MEDIUM |
| 80/20 high-entropy tokens (Wang et al., NeurIPS 2025, arXiv:2506.01939) | Entropy "forking tokens" drive RLVR | Model-entropy importance, not ground-truth value | LOW-MED |
| OmegaPRM (Luo et al. 2024, arXiv:2406.06592) | MCTS binary search for first error via per-position MC values | **Assumes monotone single-crossing value curve — our band/artifact findings undermine that assumption; useful foil** | MEDIUM |
| **Forking Paths (Bigelow et al., ICLR 2025, arXiv:2412.07961)** | Per-token resampled outcome distributions + Bayesian change-point detection; sudden "forking tokens" | Closest methodological neighbor as phenomenon; outcome/uncertainty of one path, not solve-rate value; no band, no censoring, no restart | **HIGH** |
| **Forking Fast (Bigelow et al., Aug 2026, arXiv:2608.19611)** | Cheap estimation of uncertainty dynamics; "apparent per-token noise is largely a sampling artifact" | Nearest relative of our artifact claim — but sampling-variance artifacts of uncertainty trajectories vs our budget/censoring artifacts of value crossings | **HIGH** |
| Prefix Consistency (Iwase et al. 2026, arXiv:2605.07654) | Truncate-resample answer-consistency as reliability weight | Primitive overlap, purpose differs | MED-HIGH |
| Phi-4 Pivotal Token Search (Abdin et al. 2024, arXiv:2412.08905) | p(success) along solution with recursive jump detection → DPO pairs | A value curve with jump detection as a data-mining subroutine; no phenomenon claims/band/restart/censoring | MED-HIGH |
| Rewarding Progress (Setlur et al., ICLR 2025, arXiv:2410.08146) | Process reward = change in success likelihood (prover-policy theory) | Formalizes the derivative of our curve as reward design; no measurement of when value accumulates | MEDIUM |
| Commitment Boundary (Scalena et al. 2026, arXiv:2606.13603) | Sharp guess→stable-answer transition via early-exit/attention probes; post-boundary epiphenomenal | Answer-probability commitment, not rollout value; their "sharp commitment" is what our budget/censoring analysis qualifies | MEDIUM |
| Failed Traces Tell What Is Fixable (Islah et al. 2026, arXiv:2606.05145) | Distributional signatures separate recoverable (unlucky) vs structural failures | Independent support for C5 at problem level — corroborating citation | MEDIUM |

Second tier: Max Out GRPO Signal (2607.07674 — κ(ρ) conditional success vs
prefix length, "near-monotone with ~6% violations", cite on monotonicity);
PRM Lessons (2501.07301 — MC step labels noisy/policy-dependent); SCAN
(2509.16548) and Noise-aware PRM (2601.12748) — treat MC value noise as
nuisance where we treat the band as signal; Answer Convergence (2506.02536);
Value-Guided Search (2505.17373); Tracing Uncertainty (2605.07776 —
uncertainty-trace shapes predict correctness AUROC ~0.80, address in
prediction section); Snowball Errors theory (2501.15602 — monotone error
accumulation, foil for non-monotone empirics).

## Area 3 — test-time compute: sequential vs parallel, continue vs restart (agent verdict)

**Verdict: our measurement is NOT taken. No paper runs per-problem,
per-trajectory continue-own-truncated-prefix vs restart-from-scratch at
matched TOTAL compute with restart dose-response R(C). Closest neighbors
condition on COMPLETE previous answers (Snell; Sequential Edge), compare at
aggregate level (Ghosal), or verifier-gate re-attempts. "Compression, not
reachability" and the dose-response diagnosis appear unclaimed. Reviewers
will raise Sequential Edge and Snell first.**

| Paper | What | Diff vs ours | Threat |
|---|---|---|---|
| Snell et al. 2024 (arXiv:2408.03314) | Revision-finetuned model conditions on ≤4 previous COMPLETE answers; compute-optimal sequential/parallel ratios per difficulty bin | Full answers, not truncated in-flight prefixes; bin-level not per-problem; needs revision finetuning | MEDIUM |
| Sequential Edge (Sharma & Chopra 2025, arXiv:2511.02309) | Matched-compute: sequential refinement beats parallel self-consistency in 95.6% configs; inverse-entropy voting | Closest matched-compute precedent, aggregate level, complete-answer conditioning | MEDIUM |
| Performance Gap Parallel vs Sequential (2026, arXiv:2604.05868) | Why parallel beats sequential (reduced exploration / induction-head copying) | OPPOSITE sign of our 11/13 — engage: their sequential arm is answer-conditioned re-solving, not prefix continuation | MEDIUM |
| Think Again or Think Longer? (2026, arXiv:2606.19808) | Verification-triggered re-attempt vs longer initial solve as budget allocation | Verifier-gated re-solve, no truncation anchors, no dose-response | MEDIUM |
| Why Retrying Fails (Yang 2026, arXiv:2605.08563) | Context contamination makes agent retries non-IID; budget-allocation theorem | Agent pipelines; published quantitative restart-value evidence — cite | MEDIUM |
| Ghosal et al. NeurIPS 2025 (arXiv:2506.04210) | Extend-vs-parallel at matched budget; extension helps then hurts | Aggregate; wait-forcing extension, not prefix-vs-restart probes | MEDIUM |
| Fractured Sampling (arXiv:2505.12992) | Truncation depth as efficiency knob in Pareto frontiers | Never compares continue vs fresh restart at matched compute | MEDIUM |
| Improvement Operators (Madaan et al. 2025, arXiv:2510.01123) | Parallel-Distill-Refine; sequential refinement beats one long CoT | Conditions on distilled summaries; method paper | MEDIUM |
| When More Thinking Hurts (2026, arXiv:2604.10739) | Accuracy-vs-budget curves, difficulty-dependent optima | Capped single traces, difficulty-level grain; overlaps C3 coarsely | MEDIUM |
| Large Language Monkeys (Brown et al. 2024, arXiv:2407.21787) | Coverage scales log-linearly in # independent restarts | Scales restart COUNT, not budget C; canonical citation for restart axis | LOW |
| s1 (Muennighoff et al., EMNLP 2025, arXiv:2501.19393) | Budget forcing; sequential scaling slope | Supplies truncation/extension mechanics; no restart comparison | LOW |
| Long CoT Worth Exponentially Many Short (Mirtaheri et al. 2025, arXiv:2505.21825) | Theory: sequential exponentially beats parallel on constructed tasks | Our MATH empirics are the counterpoint (compression, not exponential reachability) | LOW |

Peripheral: ParaThinker (2509.04475 — "tunnel vision" = mechanism for when
restart should win); Reasoning Relay (2512.20647 — different model continues
truncated traces); Art of Scaling TTC (2512.02008); Inference Regimes
vocabulary (2608.04001); Token Budget Saturation (2607.21433 — bimodal
converged/failed, adjacent to capability-limited diagnosis); Detection-
Extraction Gap (2604.06613 — 10% prefixes often already carry the answer, no
restart comparison).

Coverage note: queries for "value of a reasoning prefix in tokens",
"compute exchange rate", "per-problem restart dose-response" found nothing —
that instrument appears ours to claim.

## Area 5 — outcome prediction from internal signals (agent verdict)

**Verdict: no scoop. No published difficulty-CONTROLLED test at early
forecast points, and no published negative like C6. The field is dominated
by uncontrolled positives (probe AUROC 0.79–0.95, no question-only
baseline). Two partial-control papers must be engaged head-on (Yuan 2026;
Lugoloobi 2026). Two 2026 geometry papers independently SUPPORT our null.**

| Paper | What | Diff vs ours | Threat |
|---|---|---|---|
| Reasoning Models Know When They're Right (Zhang et al., COLM 2025, arXiv:2504.05419) | Linear probes at intermediate-answer positions, AUROC >0.9; early-exit verifier | No difficulty/question-only control; answer-position (mid/late), not early window; unstable transfer hints at difficulty entanglement | MEDIUM |
| **Hidden Error Awareness: Diagnostic, Not Causal (Yuan et al. 2026, arXiv:2605.09502)** | Raw-state probe 0.95 AUROC full-trace, 0.787 first-step; within-problem control on FULL-trace only; all causal interventions fail | The one paper a reviewer could wave at us: its controlled result is end-of-trace, its early positive is UNCONTROLLED (and within-problem d as low as 0.13) — argue precisely | **HIGH (misreadable)** |
| LLMs Encode Their Failures (Lugoloobi et al. 2026, arXiv:2602.09924) | PRE-generation activation probes beat TF-IDF/question-length baselines (0.79 vs 0.68); AUROC falls as budget rises | Never probes the reasoning window; "internals encode model-specific difficulty" + declining signal SUPPORTS our reading. Limitation to add: a pre-generation activation probe is a stronger difficulty baseline than text stats | MEDIUM |
| Temporal Predictors of Outcome (David 2025, arXiv:2511.14773) | Our exact forecast-point design, ~0.84 AUROC after 4 tokens — with NO baseline; author concedes selection artifact | The uncontrolled version of our design; its "positive" is the confound we demonstrate — best motivating citation | MEDIUM |
| No Answer Needed (Moreno Cencerrado et al., ICLR26 WS, arXiv:2509.10625) | Question-only activation probes predict correctness before generation | Evidence the question alone carries the signal (our baseline's strength is expected); falters on math reasoning | MEDIUM |
| Certaindex/Dynasor (Fu et al. 2024/25, arXiv:2412.20993) | Mid-reasoning certainty for compute scheduling | Systems paper; no beyond-difficulty claim; gains consistent with certainty≈difficulty | LOW |
| DeepConf (Fu et al. 2025, arXiv:2508.15260) | Token-confidence filtering of parallel traces | Within-problem trace selection, deep into generation | LOW |
| Answer Convergence (arXiv:2506.02536) + ES-CoT (arXiv:2509.14004) | Convergence at ~60% of trace; early stopping | Signals appear mid-to-late — consistent with early-window null | LOW |
| Trajectories geometry (Sun et al. 2026, arXiv:2604.05655) | Correct/incorrect diverge LATE; mid-trace 0.87 AUROC (uncontrolled) | Late-divergence finding corroborates us; cite both ways | MEDIUM |
| Move Differently (Gjølbye et al. 2026, arXiv:2605.15454) | After length-residualizing, trajectory geometry encodes DIFFICULTY | The mechanism behind our null delta (geometry duplicates difficulty); validates length-confound concern | LOW (supportive) |
| Hidden States as Early Signals (Liang et al. 2026, arXiv:2601.09093) | Step-scorer prunes parallel traces | Within-problem ranking for systems, no baseline | LOW |
| Semantic entropy (Farquhar et al., Nature 2024) + Kadavath et al. 2022 | Canonical uncertainty-predicts-correctness lineage | Short-form QA; uncontrolled; background for feature family | LOW |

Peripheral: THOUGHTTERMINATOR (2504.13367); Prompt Difficulty Prediction
(2511.03808); Real-Time Progress (2506.23274); entropy-trajectory papers
(2603.18940, 2606.02020 — informative transition is late); Tell-Tale Trace
(2608.03291). Threat sweeps for a published "adds nothing beyond difficulty"
negative returned nothing.

## Synthesis (all five areas)

**Bottom line: not scooped.** The four load-bearing novelties survive:
1. The restart-controlled instrument (C1) — unclaimed anywhere.
2. Budget-artifact decomposition of breakthroughs (C2) — unclaimed.
3. Per-problem restart dose-response diagnosis (C3) and the
   compression-not-reachability quantification (C4) — unclaimed.
4. The difficulty-controlled early-signal negative (C6) — unpublished;
   the field's positives are uncontrolled.

**Narrative collisions to manage (cite-and-differentiate obligations):**
- Illusion of Insight (2601.00514): "insight is illusion" headline exists —
  correlational; we provide the counterfactual, compute-matched version.
- Ghosal NeurIPS 2025 (2506.04210) + Sequential Edge (2511.02309) + Snell:
  matched-compute sequential/parallel comparisons exist at aggregate level
  with complete-answer conditioning; ours is per-anchor, own-prefix,
  per-problem.
- Re² (2603.07197): continue-vs-scratch exists as RL motivation on incorrect
  prefixes; inverse claim, no compute matching.
- Bigelow line (2412.07961, 2608.19611): per-position resampling as
  phenomenon + "sampling artifact" claim; uncertainty of one path, not
  budget-controlled value.
- Yuan 2026 (2605.09502): must pre-empt the "difficulty-controlled positive"
  misreading (their control is full-trace only).

**Supportive convergent evidence to harvest:** Ballon 2026 (gradual
commitment), First Try Matters, Illusion of Insight, Wolf theory (C4 as
their "no benefit" regime), Sun 2026 late divergence, Gjølbye 2026 geometry-
encodes-difficulty, Failed Traces 2026 (stochastic band cousin), Token
Budget Saturation (capability-limited cousin).

**Timing pressure:** five near-miss papers appeared Jan–Aug 2026. The
restart-controlled instrument is unclaimed TODAY; write now.

**Limitations to add from the sweep:** (a) pre-generation activation probes
(Lugoloobi) are a stronger difficulty baseline than text stats — ours is
text-level; (b) our models are non-reasoning-tuned small models; the aha
literature's claims target RL-trained reasoners (the planned R1-distill run
addresses exactly this).

## Addendum (2026-08-31): prior-critique check on the probe-positive family

Question: has anyone already hit Zhang-et-al.-style correctness probes?
Verdict: partial critiques exist on OTHER axes; the difficulty/within-problem
axis is untouched. Verified:
- PAIR (arXiv:2605.17877): stress test where correctness and prefix-coherence
  are anti-correlated → hidden-state probes fall to/below chance; probes
  track belief-consistency, not grounded correctness. Multi-turn agents,
  coherence confound — different axis, same genre. Must-cite.
- Yuan et al. (2605.09502): causal axis — steering/patching/selection all
  fail ("diagnostic, not causal"). Already in notes.
- Thinking Out Loud (2504.06564, EMNLP 2025): CHECKED FULL TEXT — verbalized
  confidence / "reasoning tax"; no difficulty adjustment, no probes. A
  search snippet suggesting difficulty controls was wrong; not a prior hit.
- OpenReview reviews of Zhang (forum O6I0Av7683): behind human verification,
  not read; whether reviewers raised a question-only baseline is unknown.
Supportive mechanism papers (difficulty IS decodable from states):
- The LLM Already Knows: perceived difficulty from hidden representations
  (2509.12886); Geometric Signatures of Reasoning: spectral task hardness,
  AUC ~0.93 (2607.01571); code-domain pre-generation positive (2606.14530).
Narrative for the paper: three independent legs now — probes are causally
inert (Yuan), coherence-driven under contamination (PAIR), and
difficulty-driven in pooled evaluation (ours). Ours is the only one with a
question-only baseline and a within-problem design.

## Addendum 2 (2026-08-31): Zhang et al. COLM review file read in full

Read all 16 replies on OpenReview forum O6I0Av7683 (3 reviews, decision,
full rebuttal cycle). Findings:
- **No reviewer, and not the program chairs, ever raised a question-
  difficulty confound, a question-only baseline, or within- vs
  across-problem evaluation.** Objections raised instead: novelty vs
  logit-lens/hallucination probing (KxZk, 5iSN), missing verifier baselines
  (KxZk — authors added deductive-verifier comparison: probe 0.84 vs 0.53),
  verbalized-confidence baseline (Etou — probe wins), non-math data (GPQA
  added: 0.765–0.82), segmentation-LLM sensitivity (5iSN).
- Every baseline discussed in the whole thread CONSUMES the trace; nobody
  asked what the question alone predicts.
- Program chairs' decision even lists "Lack of deeper analysis on why
  hidden states are predictive of reasoning correctness" as a Con — the
  question our difficulty-encoding account answers.
- Fairness nuance for our paper: Zhang's labels are per INTERMEDIATE answer
  (~8 per trace), so their design has within-trace variance and could in
  principle support a within-problem analysis — they simply never report
  one. Our claim therefore stays: unverified and burden-shifted, not
  refuted; their data could run our control (state this as the falsifiable
  prediction).
