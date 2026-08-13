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

Install the dependencies, then run the implemented Chapters 0–22 and 24 examples and tests from the repository root:

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
python examples/chapter_09_journey_abandonment.py
python examples/chapter_10_conversion_prediction.py
python examples/chapter_11_member_segmentation.py
python examples/chapter_12_transaction_anomaly_detection.py
python examples/chapter_13_classification_and_risk_signals.py
python examples/chapter_14_predicting_system_demand.py
python examples/chapter_15_database_performance.py
python examples/chapter_16_training_workflow.py
python examples/chapter_17_model_evaluation.py
python examples/chapter_18_model_api.py
python examples/chapter_20_monitoring_dashboard.py
python examples/chapter_21_data_security.py
python examples/chapter_22_explainability.py
python examples/chapter_24_human_in_the_loop.py
python scripts/train_integration_failure_model.py
python scripts/audit_ml_data.py
pytest
```

Chapter 19 adds a focused PHP 8.2+ package:

```bash
cd php
composer install
composer test
composer lint
php examples/chapter_19_ml_client.php
```

Chapters 0–2 deliberately use traditional programming and descriptive statistics. Chapter 3 introduces scikit-learn narrowly to train the book's first small model; Chapters 4–7 apply ML to production troubleshooting. Chapters 8–11 cover privacy-minimized digital experience analysis. Chapter 12 starts Part IV with mixed-feature, unsupervised transaction anomaly detection and explicitly separates unusualness from wrongdoing. Chapter 13 adds supervised historical review routing, threshold analysis, and careful label semantics. Chapter 14 introduces leakage-safe, time-aware regression for near-future system demand and capacity context. Chapter 15 completes Part IV with pre-execution database-duration regression as an observability aid. Chapter 16 starts Part V with a validated training command, trusted full-pipeline artifact, SHA-256 dataset fingerprint, and generated metadata under gitignored `artifacts/integration-failure/`. Chapter 17 evaluates its held-out probabilities against the actual target distribution and baseline, then examines thresholds, ranking, confident errors, technical slices, and calibration bins. Chapter 18 serves the trusted artifact through a typed, versioned FastAPI boundary with health, validation, controlled errors, and in-process tests. Chapter 19 consumes that API through a typed PHP/Guzzle adapter, explicit unavailability, dependency injection, and advisory-only observability. Chapter 20 completes Part V with a small FastAPI/Jinja2 engineering dashboard that separates direct telemetry, model suggestions, and deterministic investigation guidance while preserving unavailable and stale states. Chapter 21 begins Part VI with allowlist-first ML contracts, strict schema drift checks, safe logging context, artifact hashing, and a limited committed-dataset header audit. Chapter 22 adds exact, model-versioned logistic-regression contributions, fitted feature-name mapping, held-out permutation importance, and explicitly non-causal sensitivity experiments. Chapter 24 adds explicit model/policy/human boundaries, deterministic review routing, capacity analysis, state transitions, overrides, and privacy-minimized audit events. Chapters 23 and 25 remain planned. All fixtures and results are fictional educational material, not production banking evidence.

## Repository map

- [`CONTENTS.md`](CONTENTS.md) — complete roadmap for the book.
- [`book/`](book/) — chapter narrative; Parts I–V and Chapters 21–22 and 24 in Part VI are implemented.
- [`php/`](php/) — Chapter 19 PHP adapter, laboratory, and PHPUnit tests.
- [`src/harbor_ml/`](src/harbor_ml/) — reusable, typed Python components.
- [`examples/`](examples/) — command-line examples.
- [`tests/`](tests/) — automated checks for executable material.
- [`data/`](data/) — small synthetic educational data fixtures and their documentation.
- `docs/` — reserved for supporting documentation as the book grows.
