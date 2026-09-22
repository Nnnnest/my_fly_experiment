import os
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "tictactoe")
WINDOW = 100  # matches this project's existing "last-100" convention

TIER1_AGENTS = ["qlearning", "mlp", "connectome"]
TIER1OPT_AGENTS = ["qlearning_sym", "mlp_sym", "connectome_sym_cache"]
TIER2_AGENTS = ["connectome_t2_error", "connectome_t2_visit"]
OPPONENTS = ["random", "heuristic", "selfplay"]


def _load(agent, opponent):
    path = os.path.join(RESULTS_DIR, f"tictactoe_{agent}_{opponent}.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def _outcome_col(df):
    return "outcome" if "outcome" in df.columns else "outcome_a1"


def summarize(df):
    col = _outcome_col(df)
    wins = (df[col] == 1).astype(float)
    overall = wins.mean()
    last_w = wins.iloc[-WINDOW:].mean()
    roll = wins.rolling(WINDOW, min_periods=WINDOW).mean()
    peak = roll.max() if roll.notna().any() else float("nan")
    return dict(n_episodes=len(wins), overall_win=round(overall, 3),
                last100_win=round(last_w, 3),
                peak100_win=(round(peak, 3) if pd.notna(peak) else None),
                collapsed=bool(pd.notna(peak) and peak >= 0.5 and (peak - last_w) > 0.3))


def build_table(agents, label):
    rows = []
    for agent in agents:
        for opp in OPPONENTS:
            df = _load(agent, opp)
            if df is None:
                continue
            row = {"tier": label, "agent": agent, "opponent": opp}
            row.update(summarize(df))
            dcol = "drift_pct" if "drift_pct" in df.columns else \
                   ("drift_pct_a1" if "drift_pct_a1" in df.columns else None)
            if dcol:
                row["final_drift_pct"] = round(float(df[dcol].iloc[-1]), 3)
            rows.append(row)
    return rows


def tier3_table():
    path = os.path.join(RESULTS_DIR, "tier3_imitation_summary.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def main():
    rows = build_table(TIER1_AGENTS, "tier1_fair") \
         + build_table(TIER1OPT_AGENTS, "tier1_opt_sym_cache") \
         + build_table(TIER2_AGENTS, "tier2_assisted")
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    if len(out):
        for tier in out["tier"].unique():
            print(f"\n=== {tier} ===")
            print(out[out["tier"] == tier].drop(columns=["tier"]).to_string(index=False))
    else:
        print(f"No Tier 1/1-opt/2 CSVs found under {RESULTS_DIR}")

    t3 = tier3_table()
    print("\n=== tier3_imitation (different paradigm: full-coverage accuracy, not win rate -- do not read row-for-row against Tiers 1/2) ===")
    print(t3.to_string(index=False) if t3 is not None else
          f"none found -- run eval_tier3_imitation.py first")


if __name__ == "__main__":
    main()
