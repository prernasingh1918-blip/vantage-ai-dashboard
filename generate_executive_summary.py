"""
Vantage — AI Executive Summary Generator
------------------------------------------
Companion automation script for the Vantage Power BI dashboard
(ABC Consumer Products Pvt. Ltd. — AI-Powered Management Decision &
Performance Dashboard).

WHAT THIS DOES
This script reads the same transaction-level sales data used by the
Power BI model, independently recomputes the H1-vs-H2 profitability
bridge (the exact same logic as the dashboard's DAX measures), and
asks an AI model to turn those verified figures into a short,
plain-English executive summary — a written briefing a manager could
read in 15 seconds instead of clicking through the dashboard.

DESIGN PRINCIPLE
The AI never calculates anything. Every number in the output is
computed by this script first, in plain Python, using the same
arithmetic as the dashboard. The model's only job is to turn correct,
pre-computed figures into fluent prose — it cannot invent or alter a
number, because it is never asked to calculate one.

USAGE
    pip install -r requirements.txt
    export GEMINI_API_KEY="your-key-here"        (macOS/Linux)
    set GEMINI_API_KEY=your-key-here              (Windows cmd)
    python generate_executive_summary.py

Get a free Gemini API key (no billing required) at:
    https://aistudio.google.com
"""

import os
import pandas as pd
# Note: google.generativeai is imported inside generate_summary() rather than
# here at the top, so the data-loading and calculation functions can be
# tested/reused independently without requiring that package to be installed.

# ----------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------
DATA_PATH = "ABC_Consumer_Products_Sales_Dataset_FY2025-26.xlsx"
SHEET_NAME = "Transactions"
FY_HALF_CUTOFF = "2025-10-01"   # Apr-Sep = H1, Oct-Mar = H2
MODEL_NAME = "gemini-2.5-flash"


# ----------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    """Read the transactions sheet and tag each row with its fiscal half."""
    df = pd.read_excel(path, sheet_name=SHEET_NAME)
    df["Date"] = pd.to_datetime(df["Date"])
    df["FY_Half"] = df["Date"].apply(
        lambda d: "H1" if d < pd.Timestamp(FY_HALF_CUTOFF) else "H2"
    )
    return df


# ----------------------------------------------------------------
# 2. RECOMPUTE THE PROFIT BRIDGE (same logic as the Power BI DAX measures)
# ----------------------------------------------------------------
def compute_bridge(df: pd.DataFrame) -> dict:
    h1 = df[df["FY_Half"] == "H1"]
    h2 = df[df["FY_Half"] == "H2"]

    net_sales_h1, net_sales_h2 = h1["Net Sales"].sum(), h2["Net Sales"].sum()
    gross_profit_h1, gross_profit_h2 = h1["Gross Profit"].sum(), h2["Gross Profit"].sum()

    return {
        "net_sales_h1": net_sales_h1,
        "net_sales_h2": net_sales_h2,
        "revenue_growth_pct": (net_sales_h2 - net_sales_h1) / net_sales_h1,
        "gross_profit_h1": gross_profit_h1,
        "gross_profit_h2": gross_profit_h2,
        "margin_h1": gross_profit_h1 / net_sales_h1,
        "margin_h2": gross_profit_h2 / net_sales_h2,
        "volume_price_impact": h2["Gross Sales"].sum() - h1["Gross Sales"].sum(),
        "discount_impact": -1 * (h2["Discount Amount"].sum() - h1["Discount Amount"].sum()),
        "product_cost_impact": -1 * (h2["Product Cost"].sum() - h1["Product Cost"].sum()),
        "logistics_impact": -1 * (h2["Logistics Cost"].sum() - h1["Logistics Cost"].sum()),
    }


# ----------------------------------------------------------------
# 3. REGION / CATEGORY DIAGNOSTICS
# ----------------------------------------------------------------
def compute_diagnostics(df: pd.DataFrame) -> dict:
    region = df.groupby("Region").agg(
        net_sales=("Net Sales", "sum"),
        avg_discount=("Discount %", "mean"),
        gross_profit=("Gross Profit", "sum"),
    )
    region["logistics_pct"] = df.groupby("Region")[["Logistics Cost", "Gross Sales"]].apply(
        lambda x: x["Logistics Cost"].sum() / x["Gross Sales"].sum()
    )

    category = df.groupby("Category").agg(
        net_sales=("Net Sales", "sum"),
        gross_profit=("Gross Profit", "sum"),
    )
    category["margin"] = category["gross_profit"] / category["net_sales"]
    category["cost_pct"] = df.groupby("Category")[["Product Cost", "Net Sales"]].apply(
        lambda x: x["Product Cost"].sum() / x["Net Sales"].sum()
    )

    return {
        "worst_discount_region": region["avg_discount"].idxmax(),
        "worst_discount_value": region["avg_discount"].max(),
        "worst_logistics_region": region["logistics_pct"].idxmax(),
        "worst_logistics_value": region["logistics_pct"].max(),
        "worst_margin_category": category["margin"].idxmin(),
        "worst_margin_value": category["margin"].min(),
        "worst_cost_value": category["cost_pct"].max(),
    }


# ----------------------------------------------------------------
# 4. BUILD THE PROMPT
# ----------------------------------------------------------------
def build_prompt(bridge: dict, diag: dict) -> str:
    lines = [
        "You are writing a short executive briefing for the management team",
        "of ABC Consumer Products Pvt. Ltd., a mid-sized Indian FMCG company.",
        "Use only the figures below. Do not invent numbers. Write 4-6 sentences,",
        "plain English, no bullet points. Lead with the headline tension",
        "(revenue vs margin), name the specific drivers with rupee or percentage",
        "figures, then close with a one sentence action implication.",
        "",
        "DATA (FY 2025-26, H1 = Apr-Sep, H2 = Oct-Mar):",
        f"- Net Sales: H1 Rs.{bridge['net_sales_h1']:,.0f} -> H2 Rs.{bridge['net_sales_h2']:,.0f} "
        f"({bridge['revenue_growth_pct']:.1%} growth)",
        f"- Gross Margin %: H1 {bridge['margin_h1']:.1%} -> H2 {bridge['margin_h2']:.1%}",
        f"- Gross Profit: H1 Rs.{bridge['gross_profit_h1']:,.0f} -> H2 Rs.{bridge['gross_profit_h2']:,.0f}",
    ]

    bridge_line = "- Profit bridge: Volume/Price Rs." + f"{bridge['volume_price_impact']:+,.0f}"
    bridge_line += ", Discount Rs." + f"{bridge['discount_impact']:+,.0f}"
    bridge_line += ", Product Cost Rs." + f"{bridge['product_cost_impact']:+,.0f}"
    bridge_line += ", Logistics Rs." + f"{bridge['logistics_impact']:+,.0f}"
    lines.append(bridge_line)

    lines.append(f"- Highest-discount region: {diag['worst_discount_region']} "
                  f"({diag['worst_discount_value']:.1%})")
    lines.append(f"- Highest-logistics-cost region: {diag['worst_logistics_region']} "
                  f"({diag['worst_logistics_value']:.1%})")

    weakest_line = "- Weakest-margin category: " + diag["worst_margin_category"]
    weakest_line += " (" + f"{diag['worst_margin_value']:.1%}" + "), cost at "
    weakest_line += f"{diag['worst_cost_value']:.1%}" + " of net sales"
    lines.append(weakest_line)

    return "\n".join(lines)


# ----------------------------------------------------------------
# 5. CALL THE AI MODEL
# ----------------------------------------------------------------
def generate_summary(prompt: str) -> str:
    """
    Calls Google's Gemini API (free tier, no billing required).
    Requires GEMINI_API_KEY to be set as an environment variable,
    or available via Colab's userdata secrets.
    """
    import google.generativeai as genai

    api_key = os.environ.get("GEMINI_API_KEY")

    if api_key is None:
        try:
            from google.colab import userdata
            api_key = userdata.get("GEMINI_API_KEY")
        except ImportError:
            pass

    if api_key is None:
        raise RuntimeError(
            "No GEMINI_API_KEY found. Set it as an environment variable, "
            "or as a Colab secret if running in Google Colab."
        )

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(prompt)
    return response.text


# ----------------------------------------------------------------
# 6. MAIN
# ----------------------------------------------------------------
def main():
    print("Loading data...")
    df = load_data(DATA_PATH)

    print("Computing profit bridge and diagnostics...")
    bridge = compute_bridge(df)
    diag = compute_diagnostics(df)
    prompt = build_prompt(bridge, diag)

    print("\n" + "=" * 60)
    print("PROMPT SENT TO MODEL")
    print("=" * 60)
    print(prompt)

    print("\nCalling Gemini...")
    summary = generate_summary(prompt)

    print("\n" + "=" * 60)
    print("AI-GENERATED EXECUTIVE SUMMARY")
    print("=" * 60)
    print(summary)

    out_path = "vantage_executive_summary.txt"
    with open(out_path, "w") as f:
        f.write(summary)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
