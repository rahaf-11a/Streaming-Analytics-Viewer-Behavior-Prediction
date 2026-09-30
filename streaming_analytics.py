"""Reproducible streaming analytics demo using entirely synthetic viewing events.

Run: python streaming_analytics.py
Creates synthetic events, visualizations, a held-out forecast check, and a
30-day example forecast inside ./outputs. Not based on real customer records.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

OUTPUT = Path("outputs")
SEED = 42


def generate_demo_events(days=180, start="2025-01-01", seed=SEED):
    """Generate fictional events with a built-in weekly pattern for demonstration."""
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.date_range(start, periods=days, freq="D")
    content = {"Movie": ["Movie A", "Movie B", "Movie C"],
               "Series": ["Series A", "Series B", "Series C"]}
    for i, day in enumerate(dates):
        weekend = day.dayofweek >= 5
        sessions = rng.poisson(42 + (12 if weekend else 0))
        for _ in range(sessions):
            category = rng.choice(["Movie", "Series"], p=[0.42, 0.58])
            hd_probability = 0.65 if category == "Movie" else 0.47
            rows.append({
                "date": day,
                "viewer_id": f"demo_viewer_{rng.integers(1, 450):04d}",
                "content_title": rng.choice(content[category]),
                "content_category": category,
                "quality": rng.choice(["HD", "SD"], p=[hd_probability, 1-hd_probability]),
                "watch_minutes": max(3, round(rng.normal(88 if category == "Movie" else 43, 15), 1)),
            })
    return pd.DataFrame(rows)


def summarize(events):
    by_content = (events.groupby(["content_title", "content_category"])
                  .agg(views=("viewer_id", "size"), unique_viewers=("viewer_id", "nunique"),
                       watch_hours=("watch_minutes", lambda x: x.sum()/60))
                  .sort_values("watch_hours", ascending=False))
    by_quality = pd.crosstab(events["content_category"], events["quality"], normalize="index") * 100
    daily = (events.groupby("date")["watch_minutes"].sum()/60).asfreq("D", fill_value=0)
    return by_content, by_quality, daily


def plot_descriptives(by_content, by_quality):
    ax = by_content["watch_hours"].head(6).sort_values().plot.barh(figsize=(9, 5))
    ax.set(title="Most-watched fictional content", xlabel="Total watch time (hours)", ylabel="")
    plt.tight_layout()
    plt.savefig(OUTPUT / "top_content.png", dpi=160)
    plt.close()

    ax = by_quality.reindex(columns=["HD", "SD"], fill_value=0).plot.bar(figsize=(7, 5))
    ax.set(title="Streaming quality by content category (synthetic)", ylabel="Share of views (%)", xlabel="")
    ax.legend(title="Quality")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT / "quality_by_category.png", dpi=160)
    plt.close()


def fit_forecast(series, test_days=21, future_days=30):
    """Time-ordered evaluation; forecast interval is model-based, not a guarantee."""
    train, test = series.iloc[:-test_days], series.iloc[-test_days:]
    params = dict(order=(1, 0, 1), seasonal_order=(1, 0, 0, 7),
                  enforce_stationarity=False, enforce_invertibility=False)
    model = SARIMAX(train, **params).fit(disp=False)
    predicted = model.get_forecast(steps=test_days).predicted_mean
    mae = float((test - predicted).abs().mean())
    print(f"Synthetic-data test MAE: {mae:.2f} watch-hours/day ({test_days} held-out days)")

    final_model = SARIMAX(series, **params).fit(disp=False)
    result = final_model.get_forecast(steps=future_days)
    future = pd.DataFrame({
        "forecast_watch_hours": result.predicted_mean.clip(lower=0),
        "lower_95": result.conf_int().iloc[:, 0].clip(lower=0),
        "upper_95": result.conf_int().iloc[:, 1].clip(lower=0),
    })
    future.index.name = "date"
    future.to_csv(OUTPUT / "synthetic_forecast.csv")

    ax = series.tail(60).plot(figsize=(10, 5), label="Historical synthetic data")
    future["forecast_watch_hours"].plot(ax=ax, label="Example forecast")
    ax.fill_between(future.index, future["lower_95"], future["upper_95"], alpha=0.15,
                    label="95% model interval")
    ax.set(title="Synthetic streaming watch-time forecast", xlabel="Date", ylabel="Watch time (hours/day)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT / "watch_time_forecast.png", dpi=160)
    plt.close()


def main():
    OUTPUT.mkdir(exist_ok=True)
    events = generate_demo_events()
    events.to_csv(OUTPUT / "synthetic_viewing_events.csv", index=False)
    by_content, by_quality, daily = summarize(events)
    by_content.to_csv(OUTPUT / "content_summary.csv")
    by_quality.to_csv(OUTPUT / "quality_summary.csv")
    plot_descriptives(by_content, by_quality)
    fit_forecast(daily)
    print(f"Generated {len(events):,} fictional events; results saved in {OUTPUT}/")
    print("All generated values are synthetic and must not be interpreted as actual platform insights.")


if __name__ == "__main__":
    main()
