import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

# ---------- 앱 설정 ----------
st.set_page_config(page_title="Heatwave Risk Dashboard", page_icon="🔥", layout="wide")
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Pretendard&display=swap');
        html, body, [class*="css"]  {
            font-family: 'Pretendard', sans-serif;
        }
    </style>
    <h1 style='font-size: 2.5rem; font-weight: 700; margin-bottom: 10px;'>🔥 2025년 Heatwave Risk Dashboard</h1>
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

# ---------- 날짜 선택 드롭다운 ----------
dates = data["date"].dt.strftime("%Y-%m-%d").tolist()
selected_date = st.selectbox("날짜를 선택하세요", dates)

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

    # ---------- SHAP 설명 ----------
    st.markdown("### 🤖 AI 판단 근거 (SHAP 분석)")
    try:
        model = joblib.load("trained_model.pkl")
        shap_data = pd.read_excel("모델 입력용 데이터.xlsx")
        X = shap_data.drop(columns=["date"])
        explainer = shap.Explainer(model)
        shap_values = explainer(X)
        row_index = shap_data[shap_data["date"] == selected_date].index[0]

        # 시각화
        st.set_option('deprecation.showPyplotGlobalUse', False)
        fig, ax = plt.subplots()
        shap.plots.bar(shap_values[row_index], max_display=10, show=False)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"SHAP 분석을 표시하는 데 문제가 발생했습니다: {e}")
