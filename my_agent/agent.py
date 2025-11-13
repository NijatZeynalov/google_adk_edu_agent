# ===============================================================
#   EDUCATION ANALYTICS AGENT (ADK) — SINGLE PY FILE
#   Dataset: nijatzeynalov/az-school-graduate-enrollment
# ===============================================================

import pandas as pd
from datasets import load_dataset
from google.adk.agents.llm_agent import Agent
from typing import Optional


# ===============================================================
# Load dataset from HuggingFace
# ===============================================================

print("Loading HF dataset…")
dataset = load_dataset("nijatzeynalov/az-school-graduate-enrollment")
df = dataset["train"].to_pandas()
print("Dataset loaded:", df.shape)


# ===============================================================
# ---------------------- CORE TOOLS ------------------------------
# ===============================================================

def get_years() -> dict:
    years = sorted(df["year"].dropna().unique().tolist())
    return {"years": years}


def get_schools(region: Optional[str] = None) -> dict:
    filtered = df[df["region"] == region] if region else df
    schools = sorted(filtered["school_name"].dropna().unique().tolist())
    return {"region": region, "schools": schools}


def school_year_stats(school_code: int, year: int) -> dict:
    row = df[(df.school_code == school_code) & (df.year == year)]
    if row.empty:
        return {"error": "No matching school/year."}
    return row.to_dict(orient="records")[0]


def compare_schools(school_code_a: int, school_code_b: int, year: int) -> dict:
    a = df[(df.school_code == school_code_a) & (df.year == year)]
    b = df[(df.school_code == school_code_b) & (df.year == year)]
    if a.empty or b.empty:
        return {"error": "Missing data for comparison."}

    return {
        "year": year,
        "school_a": a.to_dict(orient="records")[0],
        "school_b": b.to_dict(orient="records")[0],
    }


def trend(school_code: int, metric: str) -> dict:
    data = df[df.school_code == school_code][["year", metric]].dropna()
    return {
        "school_code": school_code,
        "metric": metric,
        "trend": data.to_dict(orient="records")
    }


def region_summary(region: str) -> dict:
    data = df[df.region == region]
    summary = data.groupby("year").agg({
        "rating_b": "mean",
        "rating_g": "mean",
        "accepted_b": "sum",
        "accepted_g": "sum",
    }).reset_index()
    
    return {"region": region, "summary": summary.to_dict(orient="records")}


# ===============================================================
# -------- 10 NEW ADVANCED TOOLS (Analytics + ML Ready) ----------
# ===============================================================

def ranking_by_acceptance(year: int) -> dict:
    data = df[df.year == year].copy()
    data["total"] = data["accepted_b"] + data["accepted_g"]
    data = data.sort_values("total", ascending=False)

    return {
        "year": year,
        "ranking": data[["school_name", "school_code", "total"]].to_dict(orient="records")
    }


def gender_gap(school_code: int) -> dict:
    data = df[df.school_code == school_code][["year", "rating_b", "rating_g"]].dropna()
    data["gap"] = data["rating_g"] - data["rating_b"]
    return {"school_code": school_code, "gap": data.to_dict(orient="records")}


def zero_acceptance_years(school_code: int) -> dict:
    data = df[df.school_code == school_code]
    filtered = data[(data.accepted_b + data.accepted_g) == 0]
    return {
        "school_code": school_code,
        "years": filtered["year"].tolist(),
        "records": filtered.to_dict(orient="records")
    }


def best_year(school_code: int) -> dict:
    data = df[df.school_code == school_code].copy()
    data["total"] = data.accepted_b + data.accepted_g
    row = data.loc[data["total"].idxmax()]
    return {
        "school_code": school_code,
        "year": int(row["year"]),
        "total": float(row["total"]),
        "details": row.to_dict(),
    }


def worst_year(school_code: int) -> dict:
    data = df[df.school_code == school_code].copy()
    data["total"] = data.accepted_b + data.accepted_g
    row = data.loc[data["total"].idxmin()]
    return {
        "school_code": school_code,
        "year": int(row["year"]),
        "total": float(row["total"]),
        "details": row.to_dict(),
    }


def region_acceptance_trend(region: str) -> dict:
    data = df[df.region == region].copy()
    summary = data.groupby("year").agg({"accepted_b": "sum", "accepted_g": "sum"}).reset_index()
    return {
        "region": region,
        "trend": summary.to_dict(orient="records")
    }


def score_trend(school_code: int) -> dict:
    data = df[df.school_code == school_code]
    summary = data[["year", "attendance_mean_points_b", "attendance_mean_points_g"]]
    return {"school_code": school_code, "trend": summary.to_dict(orient="records")}


def top_schools_gender(year: int, gender: str) -> dict:
    column = "accepted_b" if gender == "male" else "accepted_g"
    data = df[df.year == year].sort_values(column, ascending=False)
    return {
        "year": year,
        "gender": gender,
        "ranking": data[["school_name", "school_code", column]].to_dict(orient="records")
    }


def improvement_rate(school_code: int, metric: str) -> dict:
    data = df[df.school_code == school_code][["year", metric]].dropna()
    data["diff"] = data[metric].diff()
    return {
        "school_code": school_code,
        "metric": metric,
        "changes": data.to_dict(orient="records")
    }


def anomaly_detection(school_code: int) -> dict:
    data = df[df.school_code == school_code][["year", "accepted_b", "accepted_g"]].copy()
    data["total"] = data.accepted_b + data.accepted_g
    data["diff"] = data.total.diff()

    anomalies = data[data["diff"] < -20]
    return {
        "school_code": school_code,
        "anomalies": anomalies.to_dict(orient="records"),
        "threshold": -20
    }


# ===============================================================
#                     REGISTER ROOT AGENT
# ===============================================================

all_tools = [
    get_years,
    get_schools,
    school_year_stats,
    compare_schools,
    trend,
    region_summary,
    ranking_by_acceptance,
    gender_gap,
    zero_acceptance_years,
    best_year,
    worst_year,
    region_acceptance_trend,
    score_trend,
    top_schools_gender,
    improvement_rate,
    anomaly_detection,
]

root_agent = Agent(
    model="gemini-2.5-flash",
    name="education_insights_agent",
    description="Analyzes Azerbaijani school graduate data (1995–2023).",
    instruction="""
You are an analytical agent specializing in Azerbaijani school graduate data.
Use tools for comparisons, trends, rankings, anomalies, and gender gaps.

If unsure, ask follow-up questions.
""",
    tools=all_tools,
)

print("Root Agent loaded successfully.")
