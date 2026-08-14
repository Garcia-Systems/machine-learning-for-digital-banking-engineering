# Chapter 23 — Bias and Fairness

[Previous: Chapter 22 — Explainability](chapter-22-explainability.md) · [Next: Chapter 24 — Human-in-the-Loop Systems](chapter-24-human-in-the-loop-systems.md) · [Back to Part VI](README.md) · [Complete contents](../../CONTENTS.md)

Harbor Federal Credit Union needs to know whether an advisory model behaves differently across the technical environments in which it runs. A single aggregate metric can conceal weak behavior for one fictional vendor, endpoint, or channel. This chapter evaluates those operational slices without introducing demographic attributes or turning a technical comparison into a judgment about people.

## Learning objectives

By the end, you can calculate slice counts, base rates, precision, recall, false-positive rates, and false-negative rates; label low-support results; distinguish undefined rates from zero; and explain why technical slice analysis is a diagnostic prompt rather than a policy or decision.

## Scope and responsible boundary

The executable uses only established synthetic integration observations and the safe contexts `vendor` and `endpoint`. `channel` is also allowed when a compatible fixture explicitly provides it. These are operational contexts. They are not demographic proxies, protected-class analyses, measurements of member value, or evidence that any member did something wrong.

This narrow laboratory cannot establish that a model is fair to people. A production assessment would require a lawful purpose, governance, appropriate domain expertise, carefully justified data, privacy controls, and analysis of the actual decision context. Adding demographic data merely to make an educational example look comprehensive would violate this repository's data-minimization boundary.

## Metrics from one confusion table

For each technical slice, Chapter 23 counts true positives (`TP`), true negatives (`TN`), false positives (`FP`), and false negatives (`FN`) at the documented threshold. It then reports:

| Metric | Calculation | Question answered |
|---|---:|---|
| count | `TP + TN + FP + FN` | How much evidence supports this row? |
| base rate | `(TP + FN) / count` | How often did the historical label occur? |
| precision | `TP / (TP + FP)` | Among positive predictions, how often was the label positive? |
| recall | `TP / (TP + FN)` | Among positive labels, how often did the model flag them? |
| FPR | `FP / (FP + TN)` | Among negative labels, how often did the model flag them? |
| FNR | `FN / (FN + TP)` | Among positive labels, how often did the model miss them? |

When a denominator is zero, the rate is **undefined**, not zero. When support is below the declared minimum, the row is labeled **LOW SUPPORT**. Successful calculation does not make a noisy estimate reliable.

## Interpreting differences

A difference between vendors may reflect sample size, differing historical base rates, fixture construction, telemetry quality, threshold behavior, changing integrations, or model error. The table identifies where to investigate. It does not prove discrimination or causality, and it does not justify changing authentication, authorization, transaction validation, credit eligibility, or punitive member action.

Likewise, selecting a threshold is policy work informed by error costs, capacity, governance, and validation. A prediction is not a policy; a policy is not a decision; and a decision is not an action.

## Executable laboratory

From the repository root, run:

```bash
python examples/chapter_23_bias_and_fairness.py
pytest -q tests/test_fairness_slices.py
```

The example trains the existing integration-failure pipeline, scores the committed synthetic fixture, and prints vendor and endpoint slices. It introduces no new model or algorithm. The tests verify the rate calculations, explicit undefined values, low-support labeling, and the context allowlist.

## Review checklist

Before sharing a slice report, verify:

1. the slice has an operationally meaningful, lawful purpose;
2. counts appear beside every rate;
3. base-rate differences remain visible;
4. undefined values are not replaced with zero;
5. low support is labeled rather than hidden;
6. model version, policy threshold, fixture period, and label definition are known;
7. the report describes model behavior rather than member worth or wrongdoing; and
8. disparities prompt investigation, not automatic retraining, deployment, or action.

## Key takeaways

1. Aggregate quality can conceal technical-context failures.
2. Counts and base rates are necessary context for precision, recall, FPR, and FNR.
3. Undefined and low-support results must be communicated honestly.
4. Vendor, endpoint, and channel analysis is not a comprehensive assessment of fairness to people.
5. Slice evidence informs investigation; it does not establish causality or authorize a banking decision.

## What comes next: Chapter 24 — Human-in-the-Loop Systems

Chapter 24 keeps the historical model prediction separate from the reviewer's later outcome, and records model version, policy version, reason code, state transitions, and audit events.

[Previous: Chapter 22 — Explainability](chapter-22-explainability.md) · [Next: Chapter 24 — Human-in-the-Loop Systems](chapter-24-human-in-the-loop-systems.md) · [Back to Part VI](README.md) · [Complete contents](../../CONTENTS.md)
