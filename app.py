import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, timedelta

# 모델 로드
model = joblib.load("trained_model.pkl")

# 위험 등급 함수
def get_risk_level(patients):
    if patients == 0:
        return "🟢 매우 낮음"
    elif 1 <= patients <= 2:
        return "🟡 낮음"
    elif 3 <= patients <= 5:
        return "🟠 보통"
    elif 6 <= patients <= 10:
        return "🔴 높음"
    else:
        return "🔥 매우 높음"

# Streamlit UI 설정
st.set_page_config(page_title="온열질환 예측", layout="centered")
st.title("🌡️ 온열질환자 수 예측 대시보드")
st.markdown("**오늘부터 최대 7일 후까지 예측이 가능합니다.**")

# 날짜 입력
min_date = datetime.now().date()
max_date = min_date + timedelta(days=7)
selected_date = st.date_input("📅 날짜 선택", min_value=min_date, max_value=max_date)

# 지역 선택
region = st.selectbox("🌍 광역자치단체 선택", [
    "서울특별시", "부산광역시", "대구광역시", "인천광역시", "광주광역시", "대전광역시", "울산광역시",
    "세종특별자치시", "경기도", "강원도", "충청북도", "충청남도", "전라북도",
    "전라남도", "경상북도", "경상남도", "제주특별자치도"
])

# 기상 조건 수동 입력
st.markdown("### ☀️ 기상 조건 입력")
col1, col2 = st.columns(2)
with col1:
    max_feel_temp = st.number_input("최고체감온도(°C)", value=33.0)
    max_temp = st.number_input("최고기온(°C)", value=31.0)
    avg_temp = st.number_input("평균기온(°C)", value=29.0)
with col2:
    min_temp = st.number_input("최저기온(°C)", value=26.0)
    avg_humidity = st.number_input("평균상대습도(%)", value=75.0)

# 예측 버튼
if st.button("예측하기"):
    # 모델 입력용 데이터프레임 (학습 시 사용한 칼럼만 사용)
    input_data = pd.DataFrame([{
        "최고체감온도(°C)": max_feel_temp,
        "최고기온(°C)": max_temp,
        "평균기온(°C)": avg_temp,
        "최저기온(°C)": min_temp,
        "평균상대습도(%)": avg_humidity
    }])

    # 예측
    pred = model.predict(input_data)[0]
    risk = get_risk_level(pred)

    # 결과 표시
    st.subheader("📊 예측 결과")
    st.write(f"예측 날짜: **{selected_date.strftime('%Y-%m-%d')}**")
    st.write(f"선택 지역: **{region}**")
    st.write(f"예측 환자 수: **{int(pred)} 명**")
    st.write(f"위험 등급: **{risk}**")

    # 대응 안내
    st.markdown("---")
    st.subheader("📄 대응 분석 리포트")
    if pred == 0:
        st.info("위험이 매우 낮습니다. 별도의 대응이 필요하지 않습니다.")
    elif pred <= 2:
        st.info("낮은 수준의 위험입니다. 실외 활동 전 수분 섭취를 권장합니다.")
    elif pred <= 5:
        st.warning("주의가 필요합니다. 온열질환 예방 수칙을 안내해주세요.")
    elif pred <= 10:
        st.error("위험 경보 수준입니다. 노약자 외출 자제 권고가 필요합니다.")
    else:
        st.error("위기 경보입니다. 자동 보상 트리거 발동 가능성이 있습니다.")
