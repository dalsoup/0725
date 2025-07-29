
import streamlit as st
import pandas as pd
import joblib
import datetime

# 모델 및 데이터 로딩
model = joblib.load("trained_model.pkl")
weather_df = pd.read_excel("모델 입력용 데이터.xlsx")
weather_df["date"] = pd.to_datetime(weather_df["date"])

# 페이지 설정
st.set_page_config(page_title="온열질환 예측 대시보드", layout="centered")
st.title("🔥 온열질환 예측 대시보드")
st.write("2025년 7월 · 청운효자동 기준")

# 날짜 선택 UI
selected_date = st.date_input("날짜 선택", value=datetime.date(2025, 7, 1),
                              min_value=datetime.date(2025, 7, 1),
                              max_value=datetime.date(2025, 7, 28))

# 해당 날짜의 데이터 추출
row = weather_df[weather_df["date"] == pd.to_datetime(selected_date)]

if row.empty:
    st.error("선택한 날짜의 기상 데이터가 없습니다.")
else:
    input_data = row[['최고체감온도(°C)', '최고기온(°C)', '평균기온(°C)', '최저기온(°C)', '평균상대습도(%)']]
    pred = model.predict(input_data)[0]

    def get_risk_level(val):
        if val == 0: return "🟢 매우 낮음"
        elif val <= 2: return "🟡 낮음"
        elif val <= 5: return "🟠 보통"
        elif val <= 10: return "🔴 높음"
        else: return "🔥 매우 높음"

    risk = get_risk_level(pred)

    st.subheader("예측 결과")
    st.metric("예측 환자 수", f"{pred:.2f}명")
    st.metric("위험 등급", risk)

    if "🔥" in risk:
        st.warning("🚨 매우 높음: 외출 자제 및 냉방기기 사용 권고")
    elif "🔴" in risk:
        st.info("🔴 높음: 노약자 야외활동 주의")
    elif "🟠" in risk:
        st.info("🟠 보통: 충분한 수분 섭취 필요")
    elif "🟡" in risk:
        st.success("🟡 낮음: 무리한 야외활동 자제")
    else:
        st.success("🟢 매우 낮음: 위험 없음")
