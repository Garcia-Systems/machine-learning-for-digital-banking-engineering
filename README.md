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

Across seven parts and 34 numbered chapters (0–33), the book moves from a transparent threshold detector through trained models, evaluation, API serving, PHP integration, engineering dashboards, responsible use, and operation of the complete capstone. See the [contents](CONTENTS.md).

## Executable examples

Examples target Python 3.11+ and favor small, typed, deterministic components. Chapter 0 uses only the Python standard library; tests use `pytest`.

Install dependencies and run the Python validation from the repository root:

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
python examples/chapter_25_model_monitoring.py
python examples/chapter_26_harbor_incident.py
python examples/chapter_27_building_telemetry_dataset.py
python examples/chapter_28_training_anomaly_detector.py
python examples/chapter_29_training_incident_classifier.py
python examples/chapter_30_capstone_ml_service.py
python examples/chapter_32_engineering_dashboard.py
python examples/chapter_33_operating_harbor.py
python scripts/train_integration_failure_model.py
python scripts/train_capstone_anomaly.py
python scripts/train_capstone_incident_classifier.py
python scripts/audit_ml_data.py
pytest
```

Chapter 19 and Chapter 31 use a focused PHP 8.2+ package:

```bash
cd php
composer install
composer test
composer lint
php examples/chapter_19_ml_client.php
php examples/chapter_31_banking_application_integration.php
```

## Complete capstone laboratory

Chapter 33 is the final operating exam. Its in-process command validates the capstone data; trains and inventories all three independently versioned artifacts; exercises service health and prediction contracts; replays the incident and dashboard; contains partial and complete ML outages; recognizes stale output; runs monitoring, human review, explanation, and vendor/endpoint slice checks; rejects a sensitive field; and restores a retained model version:

```bash
python examples/chapter_27_building_telemetry_dataset.py
python scripts/train_capstone_anomaly.py
python scripts/train_capstone_incident_classifier.py
python scripts/train_integration_failure_model.py
python examples/chapter_33_operating_harbor.py
```

Start the artifact-backed ML service and evidence-oriented dashboard locally with:

```bash
PYTHONPATH=src uvicorn harbor_ml.service.app:app --reload
PYTHONPATH=src uvicorn harbor_ml.dashboard.run:app --reload --port 8001
```

The master lab does not need either server running; it uses in-process application boundaries so automated validation remains deterministic and network-independent.

## Responsible-use boundaries

Machine learning supplies anomaly, classification, and failure-risk signals. It does not authenticate or authorize a member, validate a transaction, prove fraud or root cause, replace traces and database evidence, or make a final human-review decision. Unavailability is never converted to a low-risk score. Model, API, and policy versions remain separate; stale results and ambiguity remain visible; monitoring prompts investigation rather than automatic deployment.

All fixtures and results are fictional educational material. Completing the textbook or obtaining its example metrics does not establish production readiness, legal or regulatory compliance, fitness for a real banking decision, or validity on a real population.

## Repository map

- [`CONTENTS.md`](CONTENTS.md) — complete roadmap for the book.
- [`book/`](book/) — seven-part narrative, ending with Chapter 33's complete capstone operating exam (Chapter 23 remains a roadmap entry).
- [`php/`](php/) — Chapter 19 PHP adapter, laboratory, and PHPUnit tests.
- [`src/harbor_ml/`](src/harbor_ml/) — reusable, typed Python components.
- [`examples/`](examples/) — command-line examples.
- [`tests/`](tests/) — automated checks for executable material.
- [`data/`](data/) — small synthetic educational data fixtures and their documentation.
- `docs/` — reserved for supporting documentation as the book grows.
