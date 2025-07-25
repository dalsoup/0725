import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import requests
from sklearn.ensemble import RandomForestRegressor

# ==================== 기본 설정 ====================
st.set_page_config(page_title="Heat Illness Predictor", layout="wide")
st.title("🌡️ Heat Illness Risk Predictor")

# ==================== 유틸 함수 ====================

def get_weather_from_kma(region_name):
    """
    기상청 API 대체용 예시 함수 (실제 API 연결 시 수정 필요)
    """
    sample_data = {
        "서울특별시": [35.0, 33.1, 30.5, 27.8, 65.2],
        "부산광역시": [33.2, 32.0, 29.7, 26.1, 70.5]
    }
    return sample_data.get(region_name, [np.nan] * 5)

def classify_risk(predicted_count):
    if predicted_count == 0:
        return "🟢 매우 낮음"
    elif predicted_count <= 2:
        return "🟡 낮음"
    elif predicted_count <= 5:
        return "🟠 보통"
    elif predicted_count <= 10:
        return "🔴 높음"
    else:
        return "🔥 매우 높음"

def generate_report(pred, last_year):
    diff = pred - last_year
    if diff > 0:
        comment = f"작년 같은 날보다 환자가 약 {diff:.1f}명 많을 것으로 예상됩니다."
    elif diff < 0:
        comment = f"작년 같은 날보다 환자가 약 {-diff:.1f}명 적을 것으로 예상됩니다."
    else:
        comment = "작년과 같은 수준의 환자 수가 예상됩니다."

    if pred >= 11:
        rec = "💡 위기 경보 수준입니다. 외부 활동 자제 및 보상 조건 확인이 필요합니다."
    elif pred >= 6:
        rec = "⚠️ 위험 경고 수준입니다. 외출을 자제하세요."
    elif pred >= 3:
        rec = "🔅 주의 필요. 푸시 알림을 권장합니다."
    else:
        rec = "✅ 위험 수준은 낮지만 더위를 피하세요."
    return f"{comment}\n\n{rec}"

# ==================== 모델 로드 ====================
model = joblib.load("trained_model.pkl")

# ==================== UI: 입력 폼 ====================
st.subheader("📅 Step 1. 날짜 및 지역 선택")
today = datetime.date.today()
selected_date = st.date_input("예측 날짜를 선택하세요", min_value=today, max_value=today + datetime.timedelta(days=7), value=today)
region = st.selectbox("광역자치단체 선택", ["서울특별시", "부산광역시", "대구광역시", "인천광역시", "광주광역시", "대전광역시", "울산광역시", "세종특별자치시", "경기도", "강원특별자치도", "충청북도", "충청남도", "전라북도", "전라남도", "경상북도", "경상남도", "제주특별자치도"])

st.markdown("---")
st.subheader("🌤️ Step 2. 기상 조건 입력")

if st.button("📡 기상청 API로 불러오기"):
    temp_values = get_weather_from_kma(region)
    st.session_state["weather"] = temp_values
else:
    temp_values = st.session_state.get("weather", [np.nan]*5)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    max_feel = st.number_input("최고체감온도(°C)", value=temp_values[0] if not np.isnan(temp_values[0]) else 0.0)
with col2:
    max_temp = st.number_input("최고기온(°C)", value=temp_values[1] if not np.isnan(temp_values[1]) else 0.0)
with col3:
    mean_temp = st.number_input("평균기온(°C)", value=temp_values[2] if not np.isnan(temp_values[2]) else 0.0)
with col4:
    min_temp = st.number_input("최저기온(°C)", value=temp_values[3] if not np.isnan(temp_values[3]) else 0.0)
with col5:
    humidity = st.number_input("평균상대습도(%)", value=temp_values[4] if not np.isnan(temp_values[4]) else 0.0)

# ==================== 예측 ====================
st.markdown("---")
st.subheader("🔮 Step 3. 예측")

if st.button("예측하기"):
    input_data = pd.DataFrame([{
        "최고체감온도(°C)": max_feel,
        "최고기온(°C)": max_temp,
        "평균기온(°C)": mean_temp,
        "최저기온(°C)": min_temp,
        "평균상대습도(%)": humidity,
        "연도": selected_date.year,
        "월": selected_date.month,
    }])

    pred = model.predict(input_data)[0]
    pred = max(0, round(pred, 1))

    # 예시 전년도 값
    last_year_estimate = pred - np.random.randint(-3, 3)
    last_year_estimate = max(0, last_year_estimate)

    st.success(f"✅ 예측 환자 수: {pred}명")
    st.info(f"위험 등급: {classify_risk(pred)}")
    st.markdown("---")
    st.subheader("📝 예측 리포트")
    st.write(generate_report(pred, last_year_estimate))
