import streamlit as st
import pandas as pd
import joblib

# 모델 및 feature 목록 불러오기
model = joblib.load("trained_model.pkl")
features = joblib.load("feature_names.pkl")

# 엑셀 파일 불러오기
@st.cache_data
def load_excel_data():
    temp_df = pd.read_excel("서울_1시간 기온.xlsx")
    reh_df = pd.read_excel("서울_습도.xlsx")
    tmx_df = pd.read_excel("서울_일최고기온.xlsx")
    tmn_df = pd.read_excel("서울_일최저기온.xlsx")
    wind_df = pd.read_excel("서울_풍속.xlsx")
    return temp_df, reh_df, tmx_df, tmn_df, wind_df

temp_df, reh_df, tmx_df, tmn_df, wind_df = load_excel_data()

# UI 구성
st.title("🔥 폭염 위험도 예측 대시보드")
st.caption("2025년 7월 24일 ~ 28일 기간 중 날짜와 시간 선택 시 실시간 기상정보 기반으로 AI가 폭염 위험도를 예측합니다.")

col1, col2, col3 = st.columns(3)
with col1:
    region = st.selectbox("지역", ["서울특별시"], index=0)
with col2:
    date_selected = st.selectbox("날짜 선택", ["2025-07-24", "2025-07-25", "2025-07-26", "2025-07-27", "2025-07-28"])
with col3:
    time_selected = st.selectbox("시간 선택", [f"{h:02}:00" for h in range(24)])

if st.button("🔍 폭염 위험도 조회"):

    # 날짜/시간 → 숫자 변환
    selected_day = int(date_selected[-2:])
    selected_hour = int(time_selected.split(":")[0]) * 100

    try:
        t_avg = temp_df[(temp_df["day"] == selected_day) & (temp_df["hour"] == selected_hour)]["value"].values[0]
        humidity = reh_df[(reh_df["day"] == selected_day) & (reh_df["hour"] == selected_hour)]["value"].values[0]
        wind = wind_df[(wind_df["day"] == selected_day) & (wind_df["hour"] == selected_hour)]["value"].values[0]
        t_max = tmx_df[tmx_df["day"] == selected_day]["value"].values[-1]
        t_min = tmn_df[tmn_df["day"] == selected_day]["value"].values[-1]

        # 실시간 기상정보 출력
        st.markdown("### ☁️ 실시간 기상 정보")
        st.markdown(f"- 평균기온: **{t_avg}℃**")
        st.markdown(f"- 일 최고기온: **{t_max}℃**")
        st.markdown(f"- 일 최저기온: **{t_min}℃**")
        st.markdown(f"- 습도: **{humidity}%**")
        st.markdown(f"- 풍속: **{wind} m/s**")

        # 예측
        input_df = pd.DataFrame([{
            "최고체감온도(°C)": t_max + 1.5,
            "최고기온(°C)": t_max,
            "평균기온(°C)": t_avg,
            "최저기온(°C)": t_min,
            "평균상대습도(%)": humidity
        }])[features]

        pred = model.predict(input_df)[0]

        def get_risk_level(x):
            if x == 0: return "🟢 매우 낮음"
            elif x <= 2: return "🟡 낮음"
            elif x <= 5: return "🟠 보통"
            elif x <= 10: return "🔴 높음"
            else: return "🔥 매우 높음"

        risk = get_risk_level(pred)
        baseline = 5.3  # 예시 비교값
        diff = pred - baseline

        st.markdown("### 🔥 예측된 폭염 위험도")
        st.markdown(f"**위험 등급:** {risk}")
        st.markdown(f"**예상 온열질환자 수:** {pred:.2f}명")
        st.markdown(f"**비교:** 하루 전보다 {'+' if diff >= 0 else ''}{diff:.2f}명")

    except Exception as e:
        st.error("❌ 선택하신 날짜 및 시간에 해당하는 데이터를 찾을 수 없습니다.")
        st.error(f"오류 내용: {e}")
