
import streamlit as st
import pandas as pd
import calendar

# 데이터 불러오기
df = pd.read_excel("최종리포트데이터.xlsx")
df["date"] = pd.to_datetime(df["date"])
df["day"] = df["date"].dt.day
df["month_day"] = df["date"].dt.strftime("%m-%d")

# 위험도 색상 매핑
def get_color(risk):
    return {
        "🟢 매우 낮음": "#d4f4fa",
        "🟡 낮음": "#fef3c7",
        "🟠 보통": "#fdba74",
        "🔴 높음": "#f87171",
        "🔥 매우 높음": "#e11d48"
    }.get(risk, "#e5e7eb")

# Streamlit 설정
st.set_page_config(layout="wide")
st.title("🔥 AI 폭염위험지수 리포트 대시보드")

st.markdown("### 📆 2025년 7월 위험도 히트맵")

selected_date = None
calendar.setfirstweekday(calendar.SUNDAY)

# 달력 그리기
for week in calendar.Calendar().monthdayscalendar(2025, 7):
    cols = st.columns(7)
    for i, day in enumerate(week):
        if day == 0:
            cols[i].empty()
        else:
            date_str = f"2025-07-{day:02d}"
            row = df[df["date"] == date_str]
            if not row.empty:
                risk = row.iloc[0]["예측 위험도"]
                color = get_color(risk)
                if cols[i].button(f"{day}\n{risk}", key=date_str):
                    selected_date = date_str
                cols[i].markdown(f"<div style='background:{color};border-radius:6px;padding:8px;text-align:center;color:black;font-weight:bold'>{day}<br>{risk}</div>", unsafe_allow_html=True)

# 리포트 표시
if selected_date:
    st.markdown("---")
    st.subheader(f"📄 {selected_date} 리포트")
    row = df[df["date"] == selected_date].iloc[0]

    st.markdown(f"""
    - **기상 정보**: 최고기온 {row['최고기온(°C)']}℃ / 평균기온 {row['평균기온(°C)']}℃ / 습도 {row['습도(%)']:.1f}%
    - **AI 예측 위험지수**: {row['예측 위험도']}
    - **2025년 실제 환자수**: {int(row['2025 실제 환자수'])}명
    - **2024년 환자수**: {int(row['2024 실제 환자수'])}명
    """)
