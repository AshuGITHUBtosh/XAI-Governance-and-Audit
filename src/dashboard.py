"""Minimal Streamlit dashboard to display trajectory analysis results."""
from __future__ import annotations
from pathlib import Path
import json
from typing import Any

import pandas as pd

try:
    import streamlit as st
except Exception:
    raise


ROOT = Path(__file__).resolve().parents[0]
REPORTS = Path('reports')


def load_trajectory() -> dict[str, Any]:
    p = REPORTS / 'trajectory_summary.json'
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding='utf-8'))


def render_psi(psi: dict[str, float]):
    if not psi:
        st.info('No PSI data available')
        return
    s = pd.Series(psi).sort_values(ascending=False)
    st.bar_chart(s)


def render_bucket_perf(bucket_perf: list[dict]):
    if not bucket_perf:
        st.info('No bucket performance data')
        return
    df = pd.DataFrame(bucket_perf)
    # ensure columns
    if 'age_bucket' not in df.columns and 'index' in df.columns:
        df = df.rename(columns={'index': 'age_bucket'})
    st.dataframe(df)


def render_cohort(cohort_records: list[dict]):
    if not cohort_records:
        st.info('No cohort data available')
        return
    df = pd.DataFrame(cohort_records)
    if df.empty:
        st.info('No cohort records')
        return
    # pivot to have groups as columns
    df['period_start'] = pd.to_datetime(df['period_start'])
    pivot = df.pivot_table(index='period_start', columns='group', values='accuracy')
    st.line_chart(pivot.fillna(method='ffill'))


def main():
    st.title('Explainable AI — Trajectory Dashboard')
    tr = load_trajectory()
    st.header('Population Stability (PSI)')
    render_psi(tr.get('psi', {}))

    st.header('Bucket Performance')
    render_bucket_perf(tr.get('bucket_performance', []))

    st.header('Cohort Performance Over Time')
    render_cohort(tr.get('cohort_over_time', []))


if __name__ == '__main__':
    main()

