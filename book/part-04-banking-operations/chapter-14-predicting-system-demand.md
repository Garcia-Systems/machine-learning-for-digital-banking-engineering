# Chapter 14 — Predicting System Demand

> Can Harbor use recent system behavior and time context to estimate near-future digital request volume?

This chapter introduces **regression** through capacity awareness, traffic forecasting, performance planning, and proactive monitoring. It does not forecast financial markets or individual member behavior. All observations are deterministic synthetic teaching data—not real traffic or a claim about actual credit-union systems.

## A recurring operational pattern

**Harbor Federal Credit Union** often sees a morning like this:

```text
08:00  420 requests/min
09:00  610 requests/min
10:00  780 requests/min
11:00  840 requests/min
12:00  760 requests/min
```

Payday activity, campaigns, app notifications, outage recovery, unusual traffic, maintenance ending, and seasonal patterns can make a day different. Engineers ask: **Based on what Harbor knows now, what will request volume look like shortly in the future?** We make “shortly” testable: **10 minutes ahead**.

```text
current telemetry
      │
      ▼
recent history
      │
      ▼
regression model
      │
      ▼
predicted requests/minute
```

## Learning objectives

By the end, you should be able to distinguish regression from classification; define a numerical target and horizon; construct lags; separate present observations from future targets; preserve temporal ordering; avoid future-data leakage; train and interpret a regression; calculate MAE, RMSE, and residuals; compare a naive baseline; recognize one-step forecasting limits; and explain why forecasts support—but do not become—capacity policy.

## Classification versus regression

```text
CLASSIFICATION                 REGRESSION
input → model → category       input → model → continuous numerical value

failure / success              future requests per minute
incident type
manual review / no review
```

Here the target is `y = 842.6`, not `y = traffic_spike`. A spike label discards magnitude that matters to capacity planning.

## Define the target before the model

The target is `future_requests_per_minute`, aligned exactly 10 minutes after prediction time `t`:

```text
TIME t
known: requests now, recent averages, latency, error rate, hour, day of week
        │
        ▼
      MODEL
        │
        ▼
TIME t + 10 minutes
future_requests_per_minute
```

Changing the horizon changes the question and requires rebuilding examples. A 10-minute model is not a next-day model.

## Lags and rolling features

A **lag** carries an earlier value into the current row:

```text
10:00  700
10:05  730
10:10  760
10:15  790

At 10:15:
requests_now = 790
requests_5m_ago = 760
requests_10m_ago = 730
```

To predict 10:25, `target = requests_at_t_plus_10m`. Lags turn recent history into ordinary supervised-learning inputs.

Deterministic summaries can include `recent_average_requests`, `recent_min_requests`, `recent_max_requests`, and `recent_growth_rate`. This implementation averages `t-15m` through `t` and defines growth as requests at `t` minus requests at `t-15m`. An average from `t` through `t+15m` is invalid because it looks forward.

## Temporal leakage: the central hazard

```text
PAST ---------------- PRESENT ---------------- FUTURE
         features          │             target
                           │
                     prediction time
```

Valid at `t`:

```text
requests_per_minute_now
requests_per_minute_5m_ago
requests_per_minute_10m_ago
recent_request_average
recent_request_growth
api_latency_ms
error_rate
hour_of_day
day_of_week
```

Invalid:

```text
requests_per_minute_5m_in_future
future_average
future_peak
future_error_rate
future_requests_per_minute
```

Including the answer—or a summary calculated using it—creates impressive but unusable offline metrics. The implementation checks fixed five-minute spacing, uses `t-3` through `t` for features, and `t+2` only for the target. Tests independently verify exact alignment and that the target is absent from `DEMAND_FEATURES`.

## Fictional dataset and alignment

[`harbor_system_demand.csv`](../../data/harbor_system_demand.csv) contains 20 synthetic days at five-minute intervals with `timestamp`, `requests_per_minute`, `api_latency_ms`, `error_rate`, and `queue_depth`. A fixed-seed generator combines base traffic, smooth time-of-day patterns, weekday/weekend effects, controlled noise, and a few bounded event effects. It includes lower overnight traffic, a morning rise, midday peak, and evening decline without making prediction perfect.

The derived row contains:

```text
timestamp
requests_now
requests_5m_ago
requests_10m_ago
requests_15m_ago
recent_average_requests
recent_growth
api_latency_ms
error_rate
queue_depth
hour_of_day
day_of_week
future_requests_per_minute
```

The timestamp is `t`; the target comes from exactly `t + 10 minutes`.

## Split time chronologically

The primary experiment does **not** shuffle:

```text
EARLIER DATA                  LATER DATA
      │                            │
      ▼                            ▼
TRAINING SET                    TEST SET
first 80%                       last 20%
```

A random split lets later regimes influence training while earlier rows appear in testing, leaking information about the future distribution. A chronological split asks whether training on earlier observations generalizes later. It does not eliminate feature leakage by itself; construction must also be causal.

## Ridge regression pipeline

Correlated lag columns make modest regularization useful:

```text
features → StandardScaler → Ridge regression → future requests/min
```

Scaling makes differently sized fields comparable before Ridge regularization. `hour_of_day` and `day_of_week` remain raw integers for simplicity. This is imperfect: hours 23 and 0 are adjacent but numerically far apart, and weekdays are not truly continuous. Cyclic hour encoding and categorical weekday encoding are possible refinements.

Fitted coefficients describe regression relationships; large weights do not prove causal traffic drivers, especially with correlated, scaled features.

## Persistence baseline

```text
baseline prediction = current requests_per_minute
```

This says, “Assume the near future will look like now.” Slowly changing traffic makes it a credible baseline. Both baseline and Ridge use exactly the same test targets. Complexity must earn its place; if Ridge loses, report it honestly.

## MAE, RMSE, and residuals

**Mean Absolute Error (MAE)** is the average absolute difference between actual and predicted values. If actual is 800 and predicted is 750, absolute error is 50 requests/minute—the natural unit.

**Root Mean Squared Error (RMSE)** penalizes large misses more strongly:

```text
errors → square → average → square root
```

A **residual** preserves direction:

```text
residual = actual - predicted
actual = 900, predicted = 820, residual = +80 (underpredicted by 80)
residual = -60                               (overpredicted by 60)
```

Residual samples can expose missed structure, but do not identify causes.

## Executable laboratory

Run:

```bash
python examples/chapter_14_predicting_system_demand.py
```

It prints Harbor Federal Credit Union’s 10-minute horizon, raw and supervised row counts, chronological periods, all features, and actual computed results:

```text
Persistence baseline
MAE: ...
RMSE: ...

Regression model
MAE: ...
RMSE: ...

timestamp                  actual  predicted  residual
2026-...                   ...     ...        ...
```

It also constructs a fictional current state—820 requests now; lags 790, 755, and 730; 310 ms latency; 0.008 error rate; queue depth 28; hour 10; weekday 2—and sends it through the fitted model. The prediction is not hard-coded.

## Capacity interpretation: forecast is not policy

```text
forecast demand → capacity context → engineering judgment

MODEL predicts 970 requests/min
              ≠
ACTION scale immediately

prediction → capacity policy → engineering action
```

If this fictional environment has comfortably handled 1,000 requests/minute, a forecast near it could prompt closer monitoring, capacity review, cache readiness checks, queue observation, and vendor dependency monitoring. A separate deterministic policy may consider safety margins, sustained forecasts, health, cooldowns, cost, and approval. ML informs policy; it does not replace it.

## Uncertainty, spikes, and drift

> A forecast of 850 requests/min does not mean exactly 850 requests/min will occur.

Actual demand might be 780, 850, or 940. A point prediction hides uncertainty; this chapter does not implement prediction intervals. Use held-out errors and operational margins rather than treating decimals as guarantees.

```text
unexpected app notification
      ↓
instant traffic surge
```

A lag model cannot anticipate a new event absent from inputs or history. One-step forecasts also cannot simply be repeated into a trustworthy long-range forecast. Patterns drift with product adoption, app redesign, member growth, integrations, notifications, seasons, and business hours. Monitor errors and define retraining criteria.

## Exercises

### Exercise 1 — Classification or regression?
Classify: “Will this request fail?”, “What incident type is occurring?”, “How many requests/minute will Harbor receive?”, and “Will this observation require review?”

### Exercise 2 — Lag features
Given `10:00 = 600`, `10:05 = 640`, `10:10 = 690`, and `10:15 = 720`, identify `requests_now`, `requests_5m_ago`, and `requests_10m_ago` at 10:15.

### Exercise 3 — Leakage
Why is `requests_5m_in_future` invalid when predicting future demand?

### Exercise 4 — MAE
Given absolute errors 20, 40, and 60, calculate MAE.

### Exercise 5 — Residual
Given `actual = 900` and `predicted = 840`, calculate residual and explain over- or underprediction.

### Exercise 6 — Baseline
Why is `future demand = current demand` useful?

### Coding exercise — Add a longer lag
Add `requests_20m_ago`; update construction; retrain; compare MAE and RMSE with the original model and persistence; explain usefulness; avoid causal claims and future-looking windows.

## Key takeaways

1. Regression predicts continuous numerical values.
2. Forecast horizon must be explicit.
3. Lags represent recent history.
4. Features use only information available at prediction time.
5. Chronological splitting suits time-ordered forecasting.
6. MAE is average absolute error in natural units.
7. RMSE penalizes large misses more heavily.
8. Compare with persistence.
9. Forecasts are estimates, not guarantees.
10. Predictions inform capacity policy; they do not replace engineering decisions.

## What comes next: Chapter 15 — Database Performance Prediction

Chapter 14 asks, **How much traffic is coming?** Chapter 15 will ask, **Given request and database telemetry, can Harbor estimate whether a database operation is likely to become slow or contribute to application latency?**

```text
query characteristics
 database load
 connection pressure
 estimated row band
 recent latency
        │
        ▼
      model
        │
        ▼
predicted query latency (future_query_duration_ms)
```

Chapter 15 is now implemented as the final chapter of Part IV.

[Previous: Chapter 13 — Classification and Risk Signals](chapter-13-classification-and-risk-signals.md) · [Back to Part IV](README.md) · [Complete contents](../../CONTENTS.md)
