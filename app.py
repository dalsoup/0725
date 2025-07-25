
import streamlit as st
import pandas as pd
import joblib
import datetime

# 모델 불러오기
model = joblib.load("trained_model.pkl")

# 리스크 판단 함수
def get_risk_level(pred):
    if pred == 0:
        return "🟢 매우 낮음"
    elif pred <= 2:
        return "🟡 낮음"
    elif pred <= 5:
        return "🟠 보통"
    elif pred <= 10:
        return "🔴 높음"
    else:
        return "🔥 매우 높음"

# UI 시작
st.title("🔥 온열질환 예측 대시보드")
st.write("폭염으로 인한 온열질환자를 기상 조건 기반으로 예측합니다.")

st.markdown("## 📅 Step 1. 날짜 선택")
today = datetime.date.today()
date_selected = st.date_input("날짜를 선택하세요", min_value=today, max_value=today + datetime.timedelta(days=7))

st.markdown("## 🗺️ Step 2. 지역 선택")
region = st.selectbox("광역자치단체를 선택하세요", [
    "서울특별시", "부산광역시", "대구광역시", "인천광역시", "광주광역시",
    "대전광역시", "울산광역시", "세종특별자치시", "경기도", "강원도",
    "충청북도", "충청남도", "전라북도", "전라남도", "경상북도", "경상남도", "제주특별자치도"
])

st.markdown("## 🌡️ Step 3. 기상 조건 입력")
col1, col2 = st.columns(2)
with col1:
    max_feel = st.number_input("최고체감온도(°C)", min_value=0.0, max_value=60.0, value=33.0)
    max_temp = st.number_input("최고기온(°C)", min_value=0.0, max_value=60.0, value=32.0)
    avg_temp = st.number_input("평균기온(°C)", min_value=0.0, max_value=50.0, value=28.5)
with col2:
    min_temp = st.number_input("최저기온(°C)", min_value=0.0, max_value=40.0, value=25.0)
    humidity = st.number_input("평균상대습도(%)", min_value=0.0, max_value=100.0, value=70.0)

if st.button("📊 예측하기"):
    input_data = pd.DataFrame([{
        "광역자치단체": region,
        "최고체감온도(°C)": max_feel,
        "최고기온(°C)": max_temp,
        "평균기온(°C)": avg_temp,
        "최저기온(°C)": min_temp,
        "평균상대습도(%)": humidity,
    }])

    input_for_model = input_data.drop(columns=["광역자치단체"])
    pred = model.predict(input_for_model)[0]
    risk = get_risk_level(pred)

    st.markdown("## ✅ 예측 결과")
    st.write(f"예측된 환자 수: **{pred}명**")
    st.write(f"위험 등급: **{risk}**")

    # 추가 분석 리포트
    st.markdown("### 📌 분석 리포트")
    st.markdown(f"""
**선택한 날짜:** {date_selected.strftime('%Y-%m-%d')}  
**지역:** {region}  
**대응 권장 사항:**  
- 🟢 매우 낮음: 무대응  
- 🟡 낮음: 모자, 선크림 착용  
- 🟠 보통: 충분한 수분 섭취, 푸시 알림 권장  
- 🔴 높음: 외출 자제 권고  
- 🔥 매우 높음: 자동 보상 트리거, 적극적 대응
""")
