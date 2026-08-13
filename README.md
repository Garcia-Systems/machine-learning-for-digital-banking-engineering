# Machine Learning for Digital Banking Engineering

**An Executable Textbook for Full-Stack Developers**

This textbook teaches experienced application developers how to use machine learning as an engineering tool for understanding, troubleshooting, operating, and improving digital banking systems. It connects operational telemetry, APIs, databases, automated tests, and production reasoning to progressively introduced ML techniques. It is not primarily a data-science or mathematics textbook.

## Who this book is for

The material is for full-stack, backend, platform, and operations-minded developers who already understand application code and want to learn where ML can—and cannot—help. Examples emphasize questions such as detecting unusual service behavior, prioritizing incident investigation, forecasting latency and demand, understanding digital journeys, and monitoring models deployed behind APIs.

## The Harbor setting

The book follows the engineering team at **Harbor Federal Credit Union**, a hypothetical organization with web and mobile applications, a digital banking application, databases, internal services, core-banking integrations, and external fintech providers. This shared setting lets each chapter build on a coherent system rather than isolated toy problems.

> [!IMPORTANT]
> **Fictional-use disclaimer:** Harbor Federal Credit Union—and every member, vendor, system, incident, transaction, and dataset associated with it—is fictional and designed solely for education. The repository contains no real financial data, credentials, or proprietary banking information.

## Why combine full-stack engineering and ML?

Deterministic code remains the right tool for security, authorization, accounting, transaction processing, and other explicit business rules. ML becomes useful when telemetry contains relationships and variation that cannot reasonably be covered by a few hand-written conditions. A model can produce an anomaly score, prediction, or classification to inform an engineer; it does not replace diagnosis, judgment, or financial controls.

The book begins with a transparent threshold-based detector, then moves toward trained models, evaluation, API serving, application integration, engineering dashboards, responsible use, and production monitoring. See the [complete planned contents](CONTENTS.md).

## Executable examples

Examples target Python 3.11+ and favor small, typed, deterministic components. Chapter 0 uses only the Python standard library; tests use `pytest`.

Install the dependencies, then run the implemented Part I, Part II, and Chapter 8 examples and tests from the repository root:

```bash
python -m pip install -r requirements-dev.txt
python examples/chapter_00_thresholds.py
python examples/chapter_01_telemetry.py
python examples/chapter_02_problem_framing.py
python examples/chapter_03_request_failure_model.py
python examples/chapter_04_anomaly_detection.py
python examples/chapter_05_incident_classification.py
python examples/chapter_06_feature_analysis.py
python examples/chapter_07_integration_failures.py
python examples/chapter_08_member_behavior.py
pytest
```

Chapters 0–2 deliberately use traditional programming and descriptive statistics. Chapter 3 introduces scikit-learn narrowly to train the book's first small model; Chapters 4–7 apply ML to production troubleshooting. Chapter 8 starts Part III by turning behavioral events into deterministic sessions, journeys, summaries, and funnel counts—descriptive analytics rather than an abandonment model. All fixtures and results are fictional educational material, not production banking evidence.

## Repository map

- [`CONTENTS.md`](CONTENTS.md) — complete roadmap for the book.
- [`book/`](book/) — chapter narrative; Parts I and II plus Chapter 8 are implemented.
- [`src/harbor_ml/`](src/harbor_ml/) — reusable, typed Python components.
- [`examples/`](examples/) — command-line examples.
- [`tests/`](tests/) — automated checks for executable material.
- [`data/`](data/) — small synthetic educational data fixtures and their documentation.
- `docs/` — reserved for supporting documentation as the book grows.
