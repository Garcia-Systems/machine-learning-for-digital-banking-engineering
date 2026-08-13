# Chapter 15 — Database Performance Prediction

A Harbor developer is troubleshooting intermittent latency in an account-history endpoint:

```text
GET /api/accounts/history
        │
        ├── authentication      14 ms
        ├── application logic   31 ms
        └── database query    1840 ms
```

The diagnosis for this completed request is obvious: **the database query consumed most of the response time.** The engineering team, however, wants to become proactive. Given database and request context available before a query finishes, can Harbor estimate how long that query is likely to take?

```text
QUERY REQUEST
    │
    ▼
known query + system context
    │
    ▼
REGRESSION MODEL
    │
    ▼
predicted query duration
    │
    ▼
developer / observability signal
```

The purpose is not to replace profiling or query optimization. Regression can help developers identify queries or operating conditions likely to become slow, so they know where to investigate.

> [!IMPORTANT]
> **Fictional-use disclaimer:** Every observation and scenario in this chapter is deterministic synthetic educational material for the fictional **Harbor Federal Credit Union**. It contains no production SQL, schema, member data, or measured credit-union performance and does not represent a real banking workload.

## Learning objectives

By the end, you should be able to frame latency prediction as regression; define the prediction point; reject post-query leakage; represent query shape safely; combine shape and current load; build and train a preprocessing pipeline; compare a model with a baseline using MAE, RMSE, and median absolute error; inspect residuals; recognize nonlinear interactions; and distinguish prediction from diagnosis, optimization, or database control.

## The prediction-time contract

> **The model predicts immediately before the query is executed.**

That single sentence defines what inputs are legitimate:

```text
BEFORE EXECUTION
      │
      ├── query shape
      ├── estimated workload
      └── current DB/system state
      │
      ▼
PREDICTION
      │
      ▼
QUERY EXECUTES
      │
      ▼
actual duration
```

Valid inputs include `query_family`, `rows_expected_band`, `join_count`, `filter_count`, `uses_sort`, `uses_aggregation`, `uses_grouping`, `current_db_connections`, `current_queue_depth`, `recent_db_latency_ms`, and `requests_per_minute`. All are available at that prediction point.

Invalid inputs include `actual_query_duration_ms`, `rows_actually_examined`, `rows_actually_returned`, `lock_wait_ms_observed_after_execution`, `actual_temp_tables_created`, and actual execution-plan runtime statistics when they become known only during or after execution. Historical `query_duration_ms` is the training target, never a feature. A derived `slow_query = query_duration_ms > 1000` would leak the answer too.

The model's output is an estimate of the eventual `query_duration_ms`; it is not known truth renamed as `expected_query_duration_ms`.

## Safe, structured query context

Arbitrary SQL may contain literals or implementation details, is difficult to group, and is unnecessary for this lesson. Harbor instead records normalized fictional families:

```text
account_summary        transaction_history
member_search          statement_lookup
transfer_history       verification_audit
```

These names do not disclose a real schema or proprietary SQL. Query shape uses intentionally simplified proxies: join and filter counts; sort, aggregation, and grouping flags; and an estimated row band. Row bands are `tiny`, `small`, `medium`, `large`, and `very_large`. They acknowledge uncertainty rather than pretending Harbor has a perfect cardinality estimate. A production optimizer has much richer statistics.

Shape alone is insufficient:

```text
QUERY CHARACTERISTICS
          +
CURRENT SYSTEM STATE
          │
          ▼
likely performance
```

`current_db_connections`, `current_queue_depth`, `recent_db_latency_ms`, and `requests_per_minute` describe pressure immediately before execution. The lesson is that query performance depends on both query shape and current operating conditions.

## The synthetic dataset

`data/harbor_query_performance.csv` contains 1,800 strictly chronological observations. `scripts/generate_query_performance.py` uses seed 1515 and combines family base cost, row-band cost, joins, filters, sort/group/aggregation, connections, queues, recent latency, request volume, controlled noise, and interactions. For example, a sort over `very_large` estimated rows under connection pressure costs more than a simple additive rule. The target is therefore not perfectly linear.

The columns are:

```text
timestamp, query_family, rows_expected_band, join_count, filter_count,
uses_sort, uses_aggregation, uses_grouping, current_db_connections,
current_queue_depth, recent_db_latency_ms, requests_per_minute,
query_duration_ms
```

The generator is an educational mechanism, not a performance simulator. It deliberately omits many causes found in real databases.

## Chronological evaluation and a baseline

Earlier observations form the 80% training set; later observations form the 20% test set. This prevents training on the future and then reporting performance on the past. Relationships may change over time even in operational regression.

For every test query, the baseline predicts the **median training duration for that query family**. If a family was absent from training, it falls back to the overall training median. Both medians use training data only. This asks whether ML adds value beyond knowing the typical duration of a family.

## Why a random forest?

Chapter 14 used Ridge, a linear-regression-style model that learns weighted numerical relationships. Chapter 15 uses `RandomForestRegressor`, a tree ensemble that can learn piecewise rules and interactions:

```text
query/system features
       │
       ├── Tree 1 ──► 780 ms
       ├── Tree 2 ──► 850 ms
       ├── Tree 3 ──► 810 ms
       └── ...
              │
              ▼
       combined prediction
```

Many decision trees each estimate duration; their predictions are combined. A tree can split differently when queue depth is high or the row band is large. The forest can therefore represent `join_count = 3` as acceptable under low load but costly when combined with large rows and a high queue. This is useful intuition—not a complete treatment of forest theory.

## Preprocessing and implementation

Categories must become numbers while meaningful numeric magnitudes pass through:

```text
raw query context
      │
      ▼
ColumnTransformer
      │
      ├── query_family, rows_expected_band → one-hot
      └── numeric and boolean values       → passthrough
      │
      ▼
RandomForestRegressor
      │
      ▼
predicted query duration
```

`OneHotEncoder(handle_unknown="ignore")` prevents a transform-time failure if the encoder sees a category absent during fitting. Production code should still monitor and govern new families. Boolean values pass through as zero/one-like values. Trees split on thresholds, so numerical scaling is not required; unlike Ridge, a forest is not comparing coefficient penalties across feature units.

The executable implementation uses frozen `QueryContext`, `QueryObservation`, `QuerySplit`, and metric dataclasses. Loading validates types, finite nonnegative measurements, approved families and bands, and chronological order. The feature tuple explicitly excludes the target. The pipeline fixes `random_state=1515` and `n_jobs=1` for repeatability.

```python
model = train_database_performance_model(
    build_database_performance_pipeline(), split.train
)
predicted = predict_query_duration(model, [item.context for item in split.test])
```

All reported model predictions come from that fitted pipeline—not from the generator formula.

## Measuring numerical error

For actual value \(y_i\) and prediction \(\hat y_i\), the residual is:

```text
residual = actual - predicted
```

A positive residual means underprediction. MAE averages absolute errors in milliseconds. RMSE squares errors before averaging, so a few very large misses matter more. Median absolute error describes the typical middle miss and is less dominated by extreme slow queries. None alone proves operational fitness; compare the forest to the baseline on the same held-out targets.

Run the laboratory from the repository root:

```bash
python examples/chapter_15_database_performance.py
```

A deterministic run reports:

```text
Historical query observations: 1800
Training observations: 1440
Test observations: 360

Query-family median baseline
MAE: 231.37 ms
RMSE: 310.11 ms
Median absolute error: 173.43 ms

Random forest model
MAE: 87.64 ms
RMSE: 112.67 ms
Median absolute error: 71.93 ms
```

Your installed compatible scikit-learn version should reproduce the fixture and deterministic run; small cross-version floating-point differences would not change the engineering conclusion. The example also prints actual test rows with `actual_ms`, fitted `predicted_ms`, and `residual_ms`.

## Residual investigation: error reveals observability gaps

The laboratory sorts actual held-out residuals by absolute magnitude and prints its three largest misses. These are useful debugging targets. Ask what the model could not observe:

- indexes and the actual optimizer plan;
- exact cache state and data skew;
- lock contention;
- network or storage behavior;
- background jobs and concurrent work.

> **Model error can reveal missing observability.**

A miss does not mean the database behaved irrationally. The relevant signal may be missing, changed, or unknowable at prediction time.

## Fictional pre-execution scenarios

The example builds four validated `QueryContext` values and obtains each displayed estimate from the fitted model:

1. **Simple account summary:** few joins, `small` rows, and low pressure.
2. **Large transaction history:** `large` rows, sorting, moderate joins and pressure.
3. **Query under connection pressure:** an ordinary shape but high connections, queue, and recent latency.
4. **Complex query during high load:** multiple shape and load factors together.

This demonstrates the interaction:

```text
feature A
+
feature B
+
system context
→ combined behavior
```

Do not interpret scenario differences as controlled causal experiments. They are predictions learned from synthetic correlations.

## Keep prediction separate from policy

A team may use a deterministic observability threshold:

```text
REGRESSION OUTPUT
predicted duration = 1240 ms
        │
        ▼
DETERMINISTIC POLICY
if predicted_duration > 1000:
    annotate request
```

This remains regression. The forest predicts a continuous duration; ordinary code decides whether to add a trace annotation or dashboard warning. Threshold ownership, monitoring, and response remain engineering policy.

## Prediction is not `EXPLAIN`

```text
ML PREDICTION
“This query is likely to take around 1200 ms.”

NOT
“This query is slow because index X is missing.”
```

Root-cause investigation still needs database `EXPLAIN`, query plans, slow-query logs, lock and connection metrics, index analysis, traces, and profiler output. The model can prioritize investigation. It cannot identify an unrepresented index or prove why a plan was selected.

Nor is the model a query optimizer:

```text
prediction
   │
   ▼
observability signal
   │
   ▼
developer investigates
```

Do not automatically rewrite SQL or create indexes because a model predicts slowness. Those actions require database expertise, workload testing, review, and safe change management.

## Feature importance: useful, not causal

A fitted forest exposes impurity-based feature importance: how much transformed features helped its trees reduce prediction error internally. That is not causation. One-hot encoding also expands each category into several columns, so printing a long list beside unexpanded numeric features can mislead. A responsible extension would aggregate encoded columns back to original groups such as `query_family` and `rows_expected_band`, validate the mapping, and label the result as model usage. This laboratory skips the display rather than presenting opaque importance as explanation. We also avoid SHAP and partial dependence to keep scope and dependencies controlled.

## Limits, change, and drift

The feature set cannot directly observe missing indexes, the actual optimizer plan, exact cache state, unrecorded locks, storage/network anomalies, plan regressions, schema changes, or distribution shifts. A new index, schema migration, query rewrite, database upgrade, or traffic-pattern change can alter learned relationships. A model trained before the change may become stale. Monitor residuals by time and family, compare to the baseline, validate after changes, and define retraining and rollback criteria.

## Chapter 14 and Chapter 15 are both regression

| Chapter 14 | Chapter 15 |
| --- | --- |
| predict system demand | predict query duration |
| time-series lag features | query shape + current DB state |
| Ridge/linear regression | nonlinear tree regression |
| capacity planning | database observability |
| future traffic | individual query performance |

Their targets and feature structures differ, but both estimate continuous numbers, protect a prediction-time boundary, use chronological evaluation, and compare numerical errors with a simple baseline.

## Exercises

### Exercise 1 — Feature or leakage?
Classify `query_family`, `join_count`, `current_db_connections`, `actual_query_duration_ms`, `uses_sort`, `rows_actually_examined`, and `recent_db_latency_ms` under the immediate-pre-execution contract. The actual duration and actually examined rows are leakage; the others are legitimate if captured before execution.

### Exercise 2 — Regression or classification?
Is predicting `query_duration_ms` regression or classification? Compare it with `will_query_be_slow`. The first is regression; the second could be classification but discards useful numerical magnitude and binds training to a threshold.

### Exercise 3 — Baseline
Why is the median duration for this query family useful? Explain its robustness, its engineering meaning, its training-only calculation, and unseen-family fallback.

### Exercise 4 — Residual
Given `actual = 1800 ms` and `predicted = 1100 ms`, calculate residual and absolute error. Both magnitude calculations are 700 ms, with a positive residual indicating substantial underprediction. Conclude that missing or changed conditions deserve investigation—not that the model proves the database is broken.

### Exercise 5 — Root cause
Why does a high predicted duration reveal neither the responsible index nor the chosen query plan? Name the diagnostic tools you would use next.

### Coding exercise — Add pagination
Add the prediction-time feature `has_pagination`: extend the generator and CSV schema; validate the context; update preprocessing; regenerate deterministically; retrain; compare MAE and RMSE with the original feature set; check whether the fitted model used useful signal; inspect the largest residuals again; and avoid causal claims.

## Key takeaways

1. Query-duration prediction is regression.
2. Prediction-time discipline prevents post-query leakage.
3. Query shape and current system state both affect performance.
4. A simple baseline makes evaluation meaningful.
5. MAE and RMSE quantify numerical prediction error; median absolute error adds robust context.
6. Large residuals can reveal missing observability.
7. Nonlinear models can capture interactions among query and load features.
8. Predicted latency is not root-cause diagnosis.
9. ML does not replace `EXPLAIN`, slow-query logs, traces, or database expertise.
10. Predictions should support proactive engineering investigation, not automatic database control.

## Part IV conclusion

```text
PART IV — MACHINE LEARNING FOR BANKING OPERATIONS

Chapter 12
Which transaction-like observations are unusual?
        │
        ▼
Chapter 13
Which observations resemble historically reviewed cases?
        │
        ▼
Chapter 14
How much system demand is coming?
        │
        ▼
Chapter 15
How long is this database operation likely to take?
```

Parts I–IV have focused primarily on **training and using models**. The next major challenge is: **How do we put a model inside a real application architecture?**

## What comes next: Part V — Putting Machine Learning Inside the Full Stack

Chapter 16, **Training a Model in Python**, will consolidate earlier educational training into a more production-shaped, repeatable workflow:

```text
data → validation → feature pipeline → training → evaluation → versioned model artifact
```

It introduces controlled model artifacts and repeatable training jobs.

[Previous: Chapter 14 — Predicting System Demand](chapter-14-predicting-system-demand.md) · [Back to Part IV](README.md) · [Next: Chapter 16 — Training a Model in Python](../part-05-ml-inside-the-full-stack/chapter-16-training-a-model-in-python.md) · [Complete contents](../../CONTENTS.md)
