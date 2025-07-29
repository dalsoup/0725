import streamlit as st
import pandas as pd
import numpy as np
import calendar
from datetime import datetime

# ---------- 앱 설정 ----------
st.set_page_config(page_title="Heatwave Risk Dashboard", page_icon="🔥", layout="wide")
st.markdown("""
    <h1 style='font-size: 2.5rem; font-weight: 700; margin-bottom: 10px;'>🔥 2025년 Heatwave Risk Calendar</h1>
    <p style='color: gray; font-size: 1.1rem;'>예측 위험도에 따라 날짜를 선택하고 리포트를 확인하세요.</p>
""", unsafe_allow_html=True)

# ---------- 데이터 로드 및 위험도 계산 ----------
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
st.markdown("""
<style>
    .date-box {
        border-radius: 10px;
        padding: 24px 0;
        font-weight: bold;
        font-size: 22px;
        text-align: center;
        margin-bottom: 6px;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

selected_date = None
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
                    if st.button(" ", key=date_str, use_container_width=True):
                        selected_date = date_str
                    st.markdown(
                        f"<div class='date-box' style='background:{color};'>{day}</div>",
                        unsafe_allow_html=True)
            else:
                cols[i].markdown(f"<div class='date-box' style='background:#e5e7eb;color:gray'>{day}</div>", unsafe_allow_html=True)

# ---------- 리포트 ----------
if selected_date:
    report = data[data.date == selected_date].iloc[0]
    st.markdown("---")
    st.markdown(f"""
        <h2 style='margin-top: 10px;'>📅 {selected_date} 리포트</h2>
        <ul style='font-size: 1.1rem;'>
            <li><strong>기상 정보:</strong> 최고기온 {report['최고기온(°C)']:.1f}℃ / 평균기온 {report['평균기온(°C)']:.1f}℃ / 습도 {report['습도(%)']:.1f}%</li>
            <li><strong>AI 예측 위험지수:</strong> {report['예측 위험도']}</li>
            <li><strong>2025년 실제 환자수:</strong> {int(report['2025 실제 환자수'])}명</li>
            <li><strong>2024년 환자수:</strong> {int(report['2024 실제 환자수'])}명</li>
        </ul>
    """, unsafe_allow_html=True)
