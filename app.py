import streamlit as st
import pandas as pd
import numpy as np
import calendar
from datetime import datetime

# ---------- 설정 ----------
st.set_page_config(layout="wide")
st.title("🔥 AI 폭염위험지수 리포트 대시보드")

# ---------- 실제 데이터 로드 및 위험도 계산 ----------
@st.cache_data
def load_data():
    df = pd.read_excel("최종리포트데이터.xlsx", parse_dates=["date"])
    df["예측 환자수"] = pd.to_numeric(df["예측 환자수"], errors="coerce")
    df["예측 위험도"] = pd.cut(df["예측 환자수"],
                           bins=[-1, 0, 2, 5, 10, float('inf')],
                           labels=["🟢 매우 낮음", "🟡 낮음", "🟠 보통", "🔴 높음", "🔥 매우 높음"])
    return df

data = load_data()

# ---------- 컬러 매핑 ----------
def get_color(risk):
    return {
        "🟢 매우 낮음": "#d4f4fa",
        "🟡 낮음": "#fef3c7",
        "🟠 보통": "#fdba74",
        "🔴 높음": "#f87171",
        "🔥 매우 높음": "#e11d48"
    }.get(risk, "#e5e7eb")

# ---------- 날짜 박스 그리드 ----------
st.markdown("### 📆 2025년 7월 위험도 히트맵")
selected_date = None
cols = st.columns(7)
for week in calendar.Calendar().monthdayscalendar(2025, 7):
    cols = st.columns(7)
    for i, day in enumerate(week):
        if day == 0:
            cols[i].empty()
        else:
            date_str = f"2025-07-{day:02d}"
            row = data[data.date == date_str]
            if not row.empty:
                risk = row.iloc[0]["예측 위험도"]
                color = get_color(risk)
                with cols[i]:
                    if st.button(f" ", key=date_str, use_container_width=True):
                        selected_date = date_str
                    st.markdown(f"<div style='margin-top:-50px;background:{color};border-radius:6px;padding:20px 0;text-align:center;color:black;font-weight:bold;font-size:20px'>{day}</div>", unsafe_allow_html=True)
            else:
                cols[i].markdown(f"<div style='background:#e5e7eb;border-radius:6px;padding:20px 0;text-align:center;color:gray;font-weight:bold;font-size:20px'>{day}</div>", unsafe_allow_html=True)

# ---------- 리포트 ----------
if selected_date:
    report = data[data.date == selected_date].iloc[0]
    st.markdown("---")
    st.subheader(f"📄 {selected_date} 리포트")
    st.markdown(f"""
    - **기상 정보**: 최고기온 {report['최고기온(°C)']:.1f}℃ / 평균기온 {report['평균기온(°C)']:.1f}℃ / 습도 {report['습도(%)']:.1f}%
    - **AI 예측 위험지수**: {report['예측 위험도']}
    - **2025년 실제 환자수**: {int(report['2025 실제 환자수'])}명
    - **2024년 환자수**: {int(report['2024 실제 환자수'])}명
    """)
