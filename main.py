# make.fun polymarket-nvidia-ceo-forecast-01 entry (Python 3.13, stdlib only).
# For each case file in the input directory, forecast five quantiles of
# every market's Yes price for the minutes after the checkpoint.
#
# Model: the median stays at the market's last price (persistence), and the
# spread is per-market, estimated from how much that market actually moved
# lately: the mean absolute 1-minute move over the last 360 minutes, scaled
# to a normal sigma, floored at 0.0005 and capped at 0.005. Spreads widen
# with the square root of the horizon, as a random walk's do. Deterministic:
# the same input bytes always produce the same output bytes.
import json
import math
import os
from pathlib import Path

Z_BY_LEVEL = {0.05: -1.644854, 0.25: -0.67449, 0.5: 0.0, 0.75: 0.67449, 0.95: 1.644854}
WINDOW = 360
SIGMA_FLOOR = 0.0005
SIGMA_CAP = 0.005
FALLBACK_SIGMA = 0.0025


def price6(x):
    """Clamp to [0, 1], round to 6 decimals, never -0."""
    return round(min(1.0, max(0.0, x)) * 1e6) / 1e6 + 0.0


def estimate_sigma(prices):
    diffs = [b - a for a, b in zip(prices, prices[1:])][-WINDOW:]
    if not diffs:
        return FALLBACK_SIGMA
    sigma = math.sqrt(math.pi / 2) * sum(abs(d) for d in diffs) / len(diffs)
    return min(SIGMA_CAP, max(SIGMA_FLOOR, sigma))


def forecast_market(prices, horizon, z_values):
    x = prices[-1] if prices else 0.5
    sigma = estimate_sigma(prices)
    rows = []
    for h in range(1, horizon + 1):
        spread = sigma * math.sqrt(h)
        row = [price6(x + z * spread) for z in z_values]
        for i in range(1, len(row)):
            row[i] = max(row[i], row[i - 1])
        rows.append(row)
    return rows


def validate_output(payload, case):
    """Check the forecast against the gate's shape rules before writing:
    one row per market, horizon rows of five finite, in-range, non-decreasing
    quantiles each, and the whole file under the 1 MiB cap. An invalid case
    scores as all-0.5 and takes no prize, so fail loudly instead of shipping
    a bad file."""
    forecasts = payload["forecasts"]
    if len(forecasts) != len(case["markets"]):
        raise SystemExit("forecast markets do not match the case")
    nq = len(case["quantiles"])
    for row in forecasts:
        if len(row["q"]) != case["horizon_minutes"]:
            raise SystemExit(f"market {row['id']}: wrong row count")
        for q in row["q"]:
            if len(q) != nq or not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in q):
                raise SystemExit(f"market {row['id']}: bad quantile row")
            if any(q[i] > q[i + 1] for i in range(nq - 1)):
                raise SystemExit(f"market {row['id']}: quantiles not ordered")
    blob = json.dumps(payload, separators=(",", ":"))
    if len(blob.encode("utf-8")) > 1_048_576:
        raise SystemExit("forecast file over the 1 MiB cap")
    return blob


def main():
    in_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    out_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for case_file in sorted(in_dir.glob("*.in")):
        case = json.loads(case_file.read_text(encoding="utf-8"))
        z_values = [Z_BY_LEVEL[q] for q in case["quantiles"]]
        horizon = case["horizon_minutes"]
        forecasts = [
            {"id": m["id"], "q": forecast_market(m["prices"], horizon, z_values)}
            for m in case["markets"]
        ]
        blob = validate_output({"forecasts": forecasts}, case)
        out_path = out_dir / (case_file.stem + ".out")
        out_path.write_text(blob, encoding="utf-8")


if __name__ == "__main__":
    main()
