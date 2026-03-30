"""GlobalNews Pipeline — Interactive Dashboard (Multi-Period).

Launch:
    streamlit run dashboard.py

Reads Parquet/JSONL/SQLite outputs produced by the 8-stage analysis pipeline.
Supports daily, monthly, quarterly, and yearly aggregation via sidebar controls.

Tabs: Overview, Topics, Sentiment & Emotions, Time Series, Word Cloud,
      Article Explorer.
"""

from __future__ import annotations

import datetime
import json
import re
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from wordcloud import WordCloud

# ---------------------------------------------------------------------------
# Base paths
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"

# Sub-directory names that contain date-partitioned outputs
_DATE_PARTITIONED_DIRS = ("raw", "processed", "features", "analysis", "output")

# ---------------------------------------------------------------------------
# Date discovery
# ---------------------------------------------------------------------------


@st.cache_data(ttl=600)
def discover_dates() -> list[str]:
    """Scan data/raw/ for valid YYYY-MM-DD subdirectories and return sorted."""
    raw_dir = DATA_DIR / "raw"
    if not raw_dir.exists():
        return []
    dates: list[str] = []
    for p in sorted(raw_dir.iterdir()):
        if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.name):
            dates.append(p.name)
    return dates


def dates_for_period(
    all_dates: list[str], period: str, ref_date: str,
) -> list[str]:
    """Return the subset of *all_dates* that fall within the selected period.

    Parameters
    ----------
    all_dates : available date strings (YYYY-MM-DD), sorted ascending.
    period : "Daily" | "Monthly" | "Quarterly" | "Yearly"
    ref_date : reference date string chosen in the sidebar.
    """
    ref = datetime.date.fromisoformat(ref_date)

    if period == "Daily":
        return [ref_date] if ref_date in all_dates else []

    if period == "Monthly":
        return [d for d in all_dates
                if d[:7] == ref_date[:7]]  # same YYYY-MM

    if period == "Quarterly":
        q_start_month = ((ref.month - 1) // 3) * 3 + 1
        q_start = datetime.date(ref.year, q_start_month, 1)
        q_end_month = q_start_month + 2
        if q_end_month == 12:
            q_end = datetime.date(ref.year, 12, 31)
        else:
            q_end = datetime.date(ref.year, q_end_month + 1, 1) - datetime.timedelta(days=1)
        return [d for d in all_dates if q_start <= datetime.date.fromisoformat(d) <= q_end]

    if period == "Yearly":
        return [d for d in all_dates if d[:4] == ref_date[:4]]

    return [ref_date] if ref_date in all_dates else []


# ---------------------------------------------------------------------------
# Multi-date loaders
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600)
def load_multi_parquet(
    sub_dir: str, filename: str, dates: tuple[str, ...],
) -> pd.DataFrame | None:
    """Load and concatenate a parquet file from multiple date directories."""
    frames: list[pd.DataFrame] = []
    for d in dates:
        p = DATA_DIR / sub_dir / d / filename
        if p.exists():
            df = pd.read_parquet(str(p))
            df["_data_date"] = d
            frames.append(df)
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    # Deduplicate across dates if article_id column exists
    if "article_id" in combined.columns:
        combined = combined.drop_duplicates(subset=["article_id"], keep="last")
    return combined


@st.cache_data(ttl=3600)
def load_multi_jsonl(dates: tuple[str, ...]) -> pd.DataFrame | None:
    """Load and concatenate raw JSONL files from multiple date directories."""
    frames: list[pd.DataFrame] = []
    for d in dates:
        p = DATA_DIR / "raw" / d / "all_articles.jsonl"
        if not p.exists():
            continue
        records = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if records:
            df = pd.DataFrame(records)
            df["_data_date"] = d
            frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def format_number(n: int | float) -> str:
    if isinstance(n, float):
        return f"{n:,.1f}"
    return f"{n:,}"


# Source -> Group mapping: derived from SOT (data/config/sources.yaml)
_VALID_GROUPS = frozenset("ABCDEFGHIJ")


def _load_source_groups() -> dict[str, str]:
    """Load site->group mapping from sources.yaml (SOT).

    P1: validates group values and minimum site count to detect parse failures.
    """
    import yaml

    sources_path = DATA_DIR / "config" / "sources.yaml"
    if not sources_path.exists():
        return {}

    with open(sources_path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    groups: dict[str, str] = {}
    for site_id, site_cfg in config.get("sources", {}).items():
        if isinstance(site_cfg, dict):
            g = site_cfg.get("group", "?")
            if g not in _VALID_GROUPS:
                g = "?"
            groups[site_id] = g

    # P1: parse failure detection — 116 sites expected, 50 is safe minimum
    if 0 < len(groups) < 50:
        import logging
        logging.getLogger(__name__).error(
            "source_groups_parse_suspect loaded=%d expected=100+", len(groups),
        )

    return groups


SOURCE_GROUPS = _load_source_groups()

GROUP_NAMES = {
    "A": "Korean Major",
    "B": "Korean Tech/Biz",
    "C": "Korean Specialty",
    "D": "Korean Tech",
    "E": "English Major",
    "F": "Asia-Pacific",
    "G": "Europe/ME",
    "H": "Africa",
    "I": "Latin America",
    "J": "Russia/Central Asia",
}

LANG_NAMES = {
    "ko": "Korean", "en": "English", "fr": "French", "de": "German",
    "ja": "Japanese", "ru": "Russian", "es": "Spanish", "it": "Italian",
    "pt": "Portuguese", "no": "Norwegian", "cs": "Czech", "sv": "Swedish",
    "pl": "Polish", "mn": "Mongolian",
}

# ---------------------------------------------------------------------------
# Word Cloud helpers
# ---------------------------------------------------------------------------

_KO_STOPWORDS = {
    "것", "수", "등", "이", "그", "저", "때", "중", "년", "월", "일",
    "위", "곳", "바", "뉴스", "기자", "연합뉴스", "서울", "제공",
    "사진", "대한", "관련", "이후", "올해", "현재", "경우", "이상",
    "이번", "지난", "전체", "가장", "오늘", "지금", "우리", "모든",
    "뉴스1", "한편", "또한", "기사", "무단", "전재", "배포", "금지",
    "특파원", "통신", "보도", "데일리", "저작권", "재배포", "헤럴드",
}

_EN_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "shall", "may", "might", "can", "must", "need",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "about", "between", "through", "after", "before", "during",
    "above", "below", "up", "down", "out", "off", "over", "under",
    "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
    "neither", "each", "every", "all", "any", "few", "more", "most",
    "other", "some", "such", "no", "only", "same", "than", "too",
    "very", "just", "also", "now", "then", "here", "there", "when",
    "where", "why", "how", "what", "which", "who", "whom", "this",
    "that", "these", "those", "it", "its", "he", "she", "they", "them",
    "his", "her", "their", "our", "my", "your", "we", "you", "i", "me",
    "us", "him", "if", "while", "because", "since", "until", "unless",
    "although", "though", "even", "still", "already", "never", "always",
    "often", "much", "many", "well", "however", "said", "says", "new",
    "like", "one", "two", "first", "last", "get", "got", "make", "made",
    "going", "come", "take", "know", "think", "see", "look", "want",
    "give", "use", "find", "tell", "ask", "work", "call", "try", "keep",
    "let", "put", "say", "go", "people", "time", "year", "day", "way",
    "man", "world", "life", "part", "back", "long", "great", "right",
    "old", "big", "high", "different", "small", "large", "next", "early",
    "young", "important", "public", "bad", "according", "reuters", "ap",
    "per", "set", "don", "didn", "won", "isn", "aren", "wasn", "weren",
    "haven", "hasn", "hadn", "doesn", "couldn", "shouldn", "wouldn",
}


@st.cache_data(ttl=3600)
def extract_word_frequencies(
    texts: list[str], languages: list[str],
) -> dict[str, int]:
    """Extract word frequencies using kiwipiepy (Korean) + regex (English)."""
    ko_texts = [t for t, lang in zip(texts, languages) if lang == "ko" and t]
    en_texts = [t for t, lang in zip(texts, languages) if lang != "ko" and t]

    word_freq: dict[str, int] = {}

    if ko_texts:
        from kiwipiepy import Kiwi
        kiwi = Kiwi()
        for text in ko_texts:
            tokens = kiwi.tokenize(text)
            for token in tokens:
                if token.tag in ("NNG", "NNP") and len(token.form) >= 2:
                    w = token.form
                    if w not in _KO_STOPWORDS:
                        word_freq[w] = word_freq.get(w, 0) + 1

    if en_texts:
        pattern = re.compile(r"[a-zA-Z]{3,}")
        for text in en_texts:
            for m in pattern.finditer(text.lower()):
                w = m.group()
                if w not in _EN_STOPWORDS:
                    word_freq[w] = word_freq.get(w, 0) + 1

    return word_freq


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="GlobalNews Dashboard",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar — Period selector
# ---------------------------------------------------------------------------

all_dates = discover_dates()

with st.sidebar:
    st.header("Period Selection")

    if not all_dates:
        st.warning("No data found in data/raw/")
        st.stop()

    period = st.selectbox(
        "Analysis Period",
        ["Daily", "Monthly", "Quarterly", "Yearly"],
        index=0,
    )

    if period == "Daily":
        selected_ref = st.selectbox("Date", all_dates, index=len(all_dates) - 1)
    elif period == "Monthly":
        months = sorted(set(d[:7] for d in all_dates))
        selected_month = st.selectbox("Month", months, index=len(months) - 1)
        selected_ref = selected_month + "-01"
    elif period == "Quarterly":
        quarters: list[str] = []
        seen: set[str] = set()
        for d in all_dates:
            dt = datetime.date.fromisoformat(d)
            q = (dt.month - 1) // 3 + 1
            label = f"{dt.year} Q{q}"
            if label not in seen:
                seen.add(label)
                quarters.append(label)
        selected_q = st.selectbox("Quarter", quarters, index=len(quarters) - 1)
        # Parse back to a ref date
        q_year, q_num = selected_q.split(" Q")
        q_month = (int(q_num) - 1) * 3 + 1
        selected_ref = f"{q_year}-{q_month:02d}-01"
    else:  # Yearly
        years = sorted(set(d[:4] for d in all_dates))
        selected_year = st.selectbox("Year", years, index=len(years) - 1)
        selected_ref = f"{selected_year}-01-01"

    active_dates = dates_for_period(all_dates, period, selected_ref)

    if not active_dates:
        st.warning("No data for the selected period.")
        st.stop()

    st.info(f"**{len(active_dates)}** day(s) selected: {active_dates[0]} — {active_dates[-1]}"
            if len(active_dates) > 1
            else f"**1** day: {active_dates[0]}")

    st.markdown("---")

# Convert to tuple for caching
_dates_key = tuple(active_dates)

# ---------------------------------------------------------------------------
# Load data for the selected period
# ---------------------------------------------------------------------------

raw_df = load_multi_jsonl(_dates_key)
articles_df = load_multi_parquet("processed", "articles.parquet", _dates_key)
analysis_df = load_multi_parquet("analysis", "article_analysis.parquet", _dates_key)
topics_df = load_multi_parquet("analysis", "topics.parquet", _dates_key)
timeseries_df = load_multi_parquet("analysis", "timeseries.parquet", _dates_key)
cross_df = load_multi_parquet("analysis", "cross_analysis.parquet", _dates_key)
networks_df = load_multi_parquet("analysis", "networks.parquet", _dates_key)
mood_df = load_multi_parquet("analysis", "mood_trajectory.parquet", _dates_key)
output_df = load_multi_parquet("output", "analysis.parquet", _dates_key)

# Merge articles + analysis + topics for unified view
if articles_df is not None and analysis_df is not None and topics_df is not None:
    _topic_cols = ["article_id", "topic_id", "topic_label", "topic_probability"]
    _topic_cols = [c for c in _topic_cols if c in topics_df.columns]
    merged_df = (
        articles_df
        .merge(analysis_df, on="article_id", how="left", suffixes=("", "_analysis"))
        .merge(topics_df[_topic_cols], on="article_id", how="left", suffixes=("", "_topic"))
    )
else:
    merged_df = articles_df

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------

_period_label = (
    f"{active_dates[0]}" if period == "Daily"
    else f"{active_dates[0]} ~ {active_dates[-1]} ({len(active_dates)} days)"
)
st.title("🌐 GlobalNews Pipeline Dashboard")
st.caption(f"Period: **{period}** | {_period_label}")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_overview, tab_topics, tab_sentiment, tab_timeseries, tab_wordcloud, tab_explorer = st.tabs([
    "📊 Overview",
    "🏷️ Topics",
    "😊 Sentiment & Emotions",
    "📈 Time Series",
    "☁️ Word Cloud",
    "🔍 Article Explorer",
])

# ========================= TAB 1: OVERVIEW =================================

with tab_overview:
    st.header("Crawling & Pipeline Overview")

    if raw_df is not None:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Articles Crawled", format_number(len(raw_df)))
        col2.metric("Unique Sources", format_number(raw_df["source_id"].nunique()))
        col3.metric("Languages", format_number(raw_df["language"].nunique()))
        if articles_df is not None:
            col4.metric("After Dedup/Processing", format_number(len(articles_df)))
        else:
            col4.metric("After Processing", "N/A")

        # Days covered
        if len(active_dates) > 1:
            col_d1, col_d2 = st.columns(2)
            col_d1.metric("Days in Period", len(active_dates))
            avg_per_day = len(raw_df) / len(active_dates)
            col_d2.metric("Avg Articles/Day", format_number(avg_per_day))

    st.subheader("Articles by Source")

    if raw_df is not None:
        source_counts = (
            raw_df.groupby("source_id")
            .size()
            .reset_index(name="articles")
            .sort_values("articles", ascending=False)
        )
        source_counts["group"] = source_counts["source_id"].map(SOURCE_GROUPS).fillna("?")
        source_counts["group_name"] = source_counts["group"].map(GROUP_NAMES).fillna("Unknown")

        fig_src = px.bar(
            source_counts,
            x="source_id",
            y="articles",
            color="group_name",
            title="Articles per Source (colored by Group)",
            labels={"source_id": "Source", "articles": "Articles", "group_name": "Group"},
            text_auto=True,
            height=450,
        )
        fig_src.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_src, use_container_width=True)

        col_left, col_right = st.columns(2)

        with col_left:
            group_counts = (
                source_counts.groupby(["group", "group_name"])["articles"]
                .sum()
                .reset_index()
                .sort_values("group")
            )
            fig_grp = px.pie(
                group_counts,
                values="articles",
                names="group_name",
                title="Articles by Group",
                hole=0.4,
            )
            st.plotly_chart(fig_grp, use_container_width=True)

        with col_right:
            lang_counts = raw_df["language"].value_counts().reset_index()
            lang_counts.columns = ["language", "count"]
            lang_counts["lang_name"] = lang_counts["language"].map(LANG_NAMES).fillna(lang_counts["language"])
            fig_lang = px.pie(
                lang_counts,
                values="count",
                names="lang_name",
                title="Articles by Language",
                hole=0.4,
            )
            st.plotly_chart(fig_lang, use_container_width=True)

        # Daily trend (useful for multi-day periods)
        if len(active_dates) > 1 and "_data_date" in raw_df.columns:
            st.subheader("Daily Article Volume")
            daily_vol = raw_df.groupby("_data_date").size().reset_index(name="articles")
            fig_daily = px.bar(
                daily_vol, x="_data_date", y="articles",
                title="Articles Collected per Day",
                labels={"_data_date": "Date", "articles": "Articles"},
                text_auto=True,
            )
            st.plotly_chart(fig_daily, use_container_width=True)

    # Pipeline stage summary
    st.subheader("Pipeline Output Files")
    file_info = []
    _file_defs = [
        ("Raw JSONL", "raw", "all_articles.jsonl"),
        ("Processed Articles", "processed", "articles.parquet"),
        ("Article Analysis", "analysis", "article_analysis.parquet"),
        ("Topics", "analysis", "topics.parquet"),
        ("Time Series", "analysis", "timeseries.parquet"),
        ("Cross Analysis", "analysis", "cross_analysis.parquet"),
        ("Networks", "analysis", "networks.parquet"),
        ("Output Analysis", "output", "analysis.parquet"),
        ("Signals", "output", "signals.parquet"),
    ]
    for label, sub, fname in _file_defs:
        total_size = 0.0
        found = 0
        for d in active_dates:
            p = DATA_DIR / sub / d / fname
            if p.exists():
                total_size += p.stat().st_size / (1024 * 1024)
                found += 1
        status = f"✅ ({found}/{len(active_dates)})" if found > 0 else "❌"
        file_info.append({
            "File": label,
            "Name": fname,
            "Total Size (MB)": round(total_size, 2),
            "Days Found": found,
            "Status": status,
        })
    st.dataframe(pd.DataFrame(file_info), use_container_width=True, hide_index=True)


# ========================= TAB 2: TOPICS ====================================

with tab_topics:
    st.header("Topic Analysis")

    if topics_df is not None:
        topic_counts = (
            topics_df[topics_df["topic_id"] != -1]
            .groupby(["topic_id", "topic_label"])
            .size()
            .reset_index(name="articles")
            .sort_values("articles", ascending=False)
        )

        col_info1, col_info2, col_info3 = st.columns(3)
        n_topics = topics_df[topics_df["topic_id"] != -1]["topic_id"].nunique()
        n_outliers = (topics_df["topic_id"] == -1).sum()
        outlier_pct = n_outliers / len(topics_df) * 100 if len(topics_df) > 0 else 0
        col_info1.metric("Topics Discovered", n_topics)
        col_info2.metric("Outlier Articles", f"{n_outliers} ({outlier_pct:.1f}%)")
        col_info3.metric("Total Articles", format_number(len(topics_df)))

        topic_counts["short_label"] = topic_counts["topic_label"].str[:40]

        fig_topics = px.bar(
            topic_counts.head(20),
            x="articles",
            y="short_label",
            orientation="h",
            title="Top 20 Topics by Article Count",
            labels={"short_label": "Topic", "articles": "Articles"},
            text_auto=True,
            height=600,
        )
        fig_topics.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_topics, use_container_width=True)

        st.subheader("Topic Assignment Confidence")
        fig_prob = px.histogram(
            topics_df[topics_df["topic_id"] != -1],
            x="topic_probability",
            nbins=50,
            title="Distribution of Topic Assignment Probability",
            labels={"topic_probability": "Probability"},
        )
        st.plotly_chart(fig_prob, use_container_width=True)

        if merged_df is not None and "steeps_category" in merged_df.columns:
            st.subheader("STEEPS Classification")
            steeps = merged_df["steeps_category"].value_counts().reset_index()
            steeps.columns = ["category", "count"]
            fig_steeps = px.bar(
                steeps,
                x="category",
                y="count",
                title="Articles by STEEPS Category",
                color="category",
                text_auto=True,
            )
            st.plotly_chart(fig_steeps, use_container_width=True)

        # Topic evolution across days (multi-day periods)
        if len(active_dates) > 1 and "_data_date" in topics_df.columns:
            st.subheader("Topic Trends Across Days")
            top5_topics = topic_counts.head(5)["topic_id"].tolist()
            topic_daily = (
                topics_df[topics_df["topic_id"].isin(top5_topics)]
                .groupby(["_data_date", "topic_label"])
                .size()
                .reset_index(name="articles")
            )
            if len(topic_daily) > 0:
                fig_topic_trend = px.line(
                    topic_daily, x="_data_date", y="articles", color="topic_label",
                    title="Top 5 Topics — Daily Trend",
                    labels={"_data_date": "Date", "articles": "Articles", "topic_label": "Topic"},
                )
                st.plotly_chart(fig_topic_trend, use_container_width=True)
    else:
        st.warning("Topics data not available.")


# ========================= TAB 3: SENTIMENT & EMOTIONS =====================

with tab_sentiment:
    st.header("Sentiment & Emotion Analysis")

    if analysis_df is not None:
        col_s1, col_s2 = st.columns(2)

        with col_s1:
            sent_counts = analysis_df["sentiment_label"].value_counts().reset_index()
            sent_counts.columns = ["label", "count"]
            color_map = {"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"}
            fig_sent = px.pie(
                sent_counts,
                values="count",
                names="label",
                title="Sentiment Distribution",
                color="label",
                color_discrete_map=color_map,
                hole=0.4,
            )
            st.plotly_chart(fig_sent, use_container_width=True)

        with col_s2:
            fig_score = px.histogram(
                analysis_df,
                x="sentiment_score",
                nbins=50,
                title="Sentiment Score Distribution",
                labels={"sentiment_score": "Score (-1 to 1)"},
                color_discrete_sequence=["#3498db"],
            )
            st.plotly_chart(fig_score, use_container_width=True)

        # Sentiment trend across days
        if len(active_dates) > 1 and "_data_date" in analysis_df.columns:
            st.subheader("Sentiment Trend Across Days")
            sent_daily = (
                analysis_df.groupby(["_data_date", "sentiment_label"])
                .size()
                .reset_index(name="count")
            )
            fig_sent_trend = px.bar(
                sent_daily, x="_data_date", y="count", color="sentiment_label",
                title="Sentiment Distribution by Day",
                labels={"_data_date": "Date", "count": "Articles", "sentiment_label": "Sentiment"},
                color_discrete_map=color_map,
                barmode="stack",
            )
            st.plotly_chart(fig_sent_trend, use_container_width=True)

        # Emotion radar
        st.subheader("Emotion Profile (Average Across All Articles)")
        emotion_cols = [c for c in analysis_df.columns if c.startswith("emotion_")]
        if emotion_cols:
            avg_emotions = analysis_df[emotion_cols].mean()
            emotion_labels = [c.replace("emotion_", "").title() for c in emotion_cols]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=avg_emotions.values.tolist() + [avg_emotions.values[0]],
                theta=emotion_labels + [emotion_labels[0]],
                fill="toself",
                name="Average",
                line_color="#3498db",
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title="Emotion Radar — Global Average",
                height=500,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            if merged_df is not None and "source" in merged_df.columns:
                st.subheader("Emotion Heatmap by Source")
                _emo_cols_merged = [c for c in merged_df.columns if c.startswith("emotion_")]
                if _emo_cols_merged:
                    emo_by_source = (
                        merged_df.groupby("source")[_emo_cols_merged]
                        .mean()
                        .sort_index()
                    )
                    emo_labels = [c.replace("emotion_", "").title() for c in _emo_cols_merged]
                    emo_by_source.columns = emo_labels

                    fig_heat = px.imshow(
                        emo_by_source.values,
                        x=emo_by_source.columns.tolist(),
                        y=emo_by_source.index.tolist(),
                        color_continuous_scale="RdYlBu_r",
                        title="Average Emotion Scores by News Source",
                        labels=dict(color="Score"),
                        aspect="auto",
                        height=max(400, len(emo_by_source) * 25),
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)

        # Mood trajectory
        if mood_df is not None and len(mood_df) > 0:
            st.subheader("Mood Trajectory")
            fig_mood = px.line(
                mood_df,
                x="date",
                y="mood_index",
                color="source",
                title="Mood Index Over Time",
                labels={"mood_index": "Mood Index", "date": "Date"},
            )
            st.plotly_chart(fig_mood, use_container_width=True)
    else:
        st.warning("Analysis data not available.")


# ========================= TAB 4: TIME SERIES ===============================

with tab_timeseries:
    st.header("Time Series Analysis")

    if timeseries_df is not None:
        col_f1, col_f2 = st.columns(2)

        metric_types = sorted(timeseries_df["metric_type"].unique())
        with col_f1:
            selected_metric = st.selectbox("Metric Type", metric_types, index=0)

        topic_ids = sorted(timeseries_df["topic_id"].unique())
        with col_f2:
            selected_topics = st.multiselect(
                "Topic IDs (leave empty for aggregate -1)",
                topic_ids,
                default=[-1] if -1 in topic_ids else topic_ids[:1],
            )

        if not selected_topics:
            selected_topics = [-1] if -1 in topic_ids else topic_ids[:1]

        mask = (
            (timeseries_df["metric_type"] == selected_metric) &
            (timeseries_df["topic_id"].isin(selected_topics))
        )
        ts_filtered = timeseries_df[mask].copy()

        if len(ts_filtered) > 0:
            ts_filtered["date"] = pd.to_datetime(ts_filtered["date"])
            ts_filtered = ts_filtered.sort_values("date")

            fig_ts = go.Figure()
            for tid in selected_topics:
                tid_data = ts_filtered[ts_filtered["topic_id"] == tid]
                fig_ts.add_trace(go.Scatter(
                    x=tid_data["date"],
                    y=tid_data["value"],
                    mode="lines",
                    name=f"Topic {tid} — Value",
                    opacity=0.6,
                ))
                if tid_data["trend"].notna().any():
                    fig_ts.add_trace(go.Scatter(
                        x=tid_data["date"],
                        y=tid_data["trend"],
                        mode="lines",
                        name=f"Topic {tid} — Trend",
                        line=dict(dash="dash", width=2),
                    ))

                burst_data = tid_data[tid_data["burst_score"].notna() & (tid_data["burst_score"] > 0)]
                if len(burst_data) > 0:
                    fig_ts.add_trace(go.Scatter(
                        x=burst_data["date"],
                        y=burst_data["value"],
                        mode="markers",
                        name=f"Topic {tid} — Bursts",
                        marker=dict(size=10, symbol="star", color="red"),
                    ))

            fig_ts.update_layout(
                title=f"Time Series: {selected_metric}",
                xaxis_title="Date",
                yaxis_title="Value",
                height=500,
            )
            st.plotly_chart(fig_ts, use_container_width=True)

            if ts_filtered["ma_short"].notna().any():
                st.subheader("Moving Average Crossover")
                fig_ma = go.Figure()
                for tid in selected_topics:
                    tid_data = ts_filtered[ts_filtered["topic_id"] == tid]
                    fig_ma.add_trace(go.Scatter(
                        x=tid_data["date"], y=tid_data["ma_short"],
                        name=f"Topic {tid} — MA Short (3d)",
                        line=dict(width=1),
                    ))
                    fig_ma.add_trace(go.Scatter(
                        x=tid_data["date"], y=tid_data["ma_long"],
                        name=f"Topic {tid} — MA Long (14d)",
                        line=dict(width=1, dash="dash"),
                    ))
                fig_ma.update_layout(height=400, title="Short vs Long Moving Average")
                st.plotly_chart(fig_ma, use_container_width=True)

            if ts_filtered["prophet_forecast"].notna().any():
                st.subheader("Prophet Forecast")
                for tid in selected_topics:
                    tid_data = ts_filtered[ts_filtered["topic_id"] == tid]
                    forecast_data = tid_data[tid_data["prophet_forecast"].notna()]
                    if len(forecast_data) > 0:
                        fig_prophet = go.Figure()
                        fig_prophet.add_trace(go.Scatter(
                            x=tid_data["date"], y=tid_data["value"],
                            name="Actual", line=dict(color="#3498db"),
                        ))
                        fig_prophet.add_trace(go.Scatter(
                            x=forecast_data["date"], y=forecast_data["prophet_forecast"],
                            name="Forecast", line=dict(color="#e74c3c", dash="dash"),
                        ))
                        if forecast_data["prophet_lower"].notna().any():
                            fig_prophet.add_trace(go.Scatter(
                                x=forecast_data["date"], y=forecast_data["prophet_upper"],
                                mode="lines", line=dict(width=0), showlegend=False,
                            ))
                            fig_prophet.add_trace(go.Scatter(
                                x=forecast_data["date"], y=forecast_data["prophet_lower"],
                                mode="lines", line=dict(width=0), showlegend=False,
                                fill="tonexty", fillcolor="rgba(231,76,60,0.15)",
                            ))
                        fig_prophet.update_layout(
                            title=f"Prophet Forecast — Topic {tid}",
                            height=400,
                        )
                        st.plotly_chart(fig_prophet, use_container_width=True)
        else:
            st.info("No data for the selected filters.")

        st.subheader("Time Series Statistics")
        ts_stats = {
            "Total Series": timeseries_df["series_id"].nunique(),
            "Date Range": f"{timeseries_df['date'].min()} -> {timeseries_df['date'].max()}",
            "Data Points": format_number(len(timeseries_df)),
            "Burst Events": int((timeseries_df["burst_score"].notna() & (timeseries_df["burst_score"] > 0)).sum()),
            "Changepoints": int(timeseries_df["is_changepoint"].sum()),
        }
        for k, v in ts_stats.items():
            st.text(f"  {k}: {v}")
    else:
        st.warning("Time series data not available.")


# ========================= TAB 5: WORD CLOUD ================================

with tab_wordcloud:
    st.header("Word Cloud Analysis")

    if raw_df is not None:
        wc_col1, wc_col2, wc_col3 = st.columns(3)

        with wc_col1:
            wc_lang_options = ["All", "Korean (ko)", "English (en)"]
            wc_lang = st.selectbox("Language Filter", wc_lang_options, key="wc_lang")

        with wc_col2:
            wc_group_options = ["All"] + [
                f"{k} — {v}" for k, v in sorted(GROUP_NAMES.items())
            ]
            wc_group = st.selectbox("Group Filter", wc_group_options, key="wc_group")

        with wc_col3:
            wc_max_words = st.slider("Max Words", 50, 300, 150, step=25, key="wc_max")

        wc_filtered = raw_df.copy()

        if wc_lang == "Korean (ko)":
            wc_filtered = wc_filtered[wc_filtered["language"] == "ko"]
        elif wc_lang == "English (en)":
            wc_filtered = wc_filtered[wc_filtered["language"] == "en"]

        if wc_group != "All":
            group_letter = wc_group.split(" — ")[0]
            group_sources = {
                sid for sid, g in SOURCE_GROUPS.items() if g == group_letter
            }
            wc_filtered = wc_filtered[wc_filtered["source_id"].isin(group_sources)]

        st.caption(f"Analyzing {len(wc_filtered):,} articles")

        if len(wc_filtered) == 0:
            st.warning("No articles match the selected filters.")
        else:
            texts = wc_filtered["body"].fillna("").tolist()
            langs = wc_filtered["language"].fillna("en").tolist()

            with st.spinner("Extracting words (Korean NLP + English tokenization)..."):
                word_freq = extract_word_frequencies(texts, langs)

            if not word_freq:
                st.warning("No words extracted. Try different filters.")
            else:
                st.success(f"Extracted {len(word_freq):,} unique words")

                st.subheader("Word Cloud")

                has_korean = any(
                    "\uac00" <= ch <= "\ud7a3"
                    for w in list(word_freq.keys())[:100]
                    for ch in w
                )
                font_path = None
                if has_korean:
                    for fp in [
                        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
                        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
                        "/Library/Fonts/NanumGothic.ttf",
                    ]:
                        if Path(fp).exists():
                            font_path = fp
                            break

                wc = WordCloud(
                    width=1200,
                    height=600,
                    max_words=wc_max_words,
                    background_color="white",
                    colormap="viridis",
                    font_path=font_path,
                    prefer_horizontal=0.7,
                    min_font_size=10,
                    max_font_size=120,
                    relative_scaling=0.5,
                )
                wc.generate_from_frequencies(word_freq)

                fig_wc, ax_wc = plt.subplots(figsize=(14, 7))
                ax_wc.imshow(wc, interpolation="bilinear")
                ax_wc.axis("off")
                st.pyplot(fig_wc)
                plt.close(fig_wc)

                st.subheader("Top 30 Words by Frequency")
                top_words = sorted(
                    word_freq.items(), key=lambda x: x[1], reverse=True
                )[:30]
                top_df = pd.DataFrame(top_words, columns=["word", "count"])

                fig_top = px.bar(
                    top_df,
                    x="count",
                    y="word",
                    orientation="h",
                    title="Top 30 Most Frequent Words",
                    labels={"word": "Word", "count": "Frequency"},
                    text_auto=True,
                    height=700,
                    color="count",
                    color_continuous_scale="viridis",
                )
                fig_top.update_layout(
                    yaxis=dict(autorange="reversed"),
                    showlegend=False,
                )
                st.plotly_chart(fig_top, use_container_width=True)

                col_ws1, col_ws2, col_ws3 = st.columns(3)
                col_ws1.metric("Unique Words", f"{len(word_freq):,}")
                col_ws2.metric("Total Word Count", f"{sum(word_freq.values()):,}")
                col_ws3.metric(
                    "Most Frequent",
                    f"{top_words[0][0]} ({top_words[0][1]:,})" if top_words else "N/A",
                )
    else:
        st.warning("Raw article data not available.")


# ========================= TAB 6: ARTICLE EXPLORER =========================

with tab_explorer:
    st.header("Article Explorer")

    if merged_df is not None:
        col_e1, col_e2, col_e3 = st.columns(3)

        with col_e1:
            sources = ["All"] + sorted(merged_df["source"].dropna().unique().tolist())
            selected_source = st.selectbox("Source", sources)

        with col_e2:
            languages = ["All"] + sorted(merged_df["language"].dropna().unique().tolist())
            selected_lang = st.selectbox("Language", languages)

        with col_e3:
            search_query = st.text_input("Search in title", "")

        filtered = merged_df.copy()
        if selected_source != "All":
            filtered = filtered[filtered["source"] == selected_source]
        if selected_lang != "All":
            filtered = filtered[filtered["language"] == selected_lang]
        if search_query:
            filtered = filtered[
                filtered["title"].str.contains(search_query, case=False, na=False)
            ]

        st.caption(f"Showing {len(filtered)} of {len(merged_df)} articles")

        sort_col = st.selectbox(
            "Sort by",
            ["published_at", "importance_score", "sentiment_score", "topic_probability"],
            index=0,
        )
        sort_asc = st.checkbox("Ascending", value=False)

        if sort_col in filtered.columns:
            filtered = filtered.sort_values(sort_col, ascending=sort_asc, na_position="last")

        display_cols = [
            "title", "source", "language", "published_at",
            "sentiment_label", "sentiment_score",
            "topic_id", "topic_label",
            "steeps_category", "importance_score",
        ]
        display_cols = [c for c in display_cols if c in filtered.columns]

        st.dataframe(
            filtered[display_cols].head(100),
            use_container_width=True,
            hide_index=True,
            height=500,
        )

        st.subheader("Article Detail")
        if len(filtered) > 0:
            article_titles = filtered["title"].head(50).tolist()
            selected_title = st.selectbox("Select an article", article_titles)
            row = filtered[filtered["title"] == selected_title].iloc[0]

            col_d1, col_d2 = st.columns([2, 1])
            with col_d1:
                st.markdown(f"**{row['title']}**")
                st.caption(f"Source: {row.get('source', 'N/A')} | "
                           f"Language: {row.get('language', 'N/A')} | "
                           f"Published: {row.get('published_at', 'N/A')}")
                body = row.get("body", "")
                if isinstance(body, str) and body:
                    st.text_area("Body", body[:3000], height=300, disabled=True)

            with col_d2:
                st.markdown("**Analysis**")
                for field in ["sentiment_label", "sentiment_score", "steeps_category",
                              "importance_score", "topic_id", "topic_label", "topic_probability"]:
                    if field in row.index and pd.notna(row[field]):
                        label = field.replace("_", " ").title()
                        st.text(f"{label}: {row[field]}")

                emotion_cols = [c for c in row.index if c.startswith("emotion_")]
                if emotion_cols:
                    st.markdown("**Emotions**")
                    emo_data = {c.replace("emotion_", "").title(): row[c]
                                for c in emotion_cols if pd.notna(row[c])}
                    if emo_data:
                        fig_emo = px.bar(
                            x=list(emo_data.keys()),
                            y=list(emo_data.values()),
                            labels={"x": "", "y": "Score"},
                            height=250,
                        )
                        fig_emo.update_layout(margin=dict(t=10, b=10))
                        st.plotly_chart(fig_emo, use_container_width=True)

    elif articles_df is not None:
        st.dataframe(articles_df.head(100), use_container_width=True, hide_index=True)
    else:
        st.warning("Article data not available.")


# ---------------------------------------------------------------------------
# Sidebar — Cross Analysis summary + meta
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Cross Analysis")

    if cross_df is not None and len(cross_df) > 0:
        analysis_types = cross_df["analysis_type"].value_counts()
        st.markdown("**Technique Results**")
        for atype, cnt in analysis_types.items():
            st.text(f"  {atype}: {cnt:,}")

        st.markdown("---")
        st.markdown("**Top Entity Pairs (by strength)**")
        top_cross = (
            cross_df[cross_df["strength"].notna()]
            .nlargest(10, "strength")[["source_entity", "target_entity", "relationship", "strength"]]
        )
        if len(top_cross) > 0:
            st.dataframe(top_cross, use_container_width=True, hide_index=True)
    else:
        st.info("No cross-analysis data.")

    if networks_df is not None and len(networks_df) > 0:
        st.markdown("---")
        st.markdown("**Network Stats**")
        st.text(f"  Edges: {len(networks_df):,}")
        st.text(f"  Unique entities: {pd.concat([networks_df['entity_a'], networks_df['entity_b']]).nunique():,}")
        st.text(f"  Communities: {networks_df['community_id'].nunique()}")

    st.markdown("---")
    st.caption("GlobalNews Crawling & Analysis Pipeline")
    st.caption(f"Available dates: {len(all_dates)} | Current: {period} view")
