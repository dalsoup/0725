import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# 모델 및 피처 불러오기
model = joblib.load("trained_model.pkl")
features = joblib.load("feature_names.pkl")

# 정제된 CSV 데이터 로딩
@st.cache_data
def load_cleaned_data():
    def load(path):
        df = pd.read_excel(path)
        df.columns = ["day", "hour", "forecast", "value"]
        df = df[pd.to_numeric(df["day"], errors="coerce").notnull()].copy()
        df["day"] = df["day"].astype(int)
        start_date = datetime(2025, 7, 1)
        df["date"] = df["day"].apply(lambda x: start_date + pd.Timedelta(days=x - 1))
        df["hour"] = df["forecast"].astype(int) // 100
        return df[["date", "hour", "value"]]

    temp = load("청운효자동_1시간기온_20250701_20250728.xlsx")
    reh = load("청운효자동_습도_20250701_20250728.xlsx")
    tmx = load("청운효자동_일최고기온_20250701_20250728.xlsx")
    tmn = load("청운효자동_일최저기온_20250701_20250728.xlsx")
    wind = load("청운효자동_풍속_20250701_20250728.xlsx")
    return temp, reh, tmx, tmn, wind

# 앱 시작
st.title("폭염 위험도 예측 대시보드")
st.caption("기상청 단기예보 데이터를 기반으로 온열질환 위험도를 예측합니다.")

temp_df, reh_df, tmx_df, tmn_df, wind_df = load_cleaned_data()

# 날짜 및 시간 선택
date_options = sorted(temp_df["date"].unique())
date_selected = st.selectbox("📅 날짜 선택", date_options)

hour_options = sorted(temp_df[temp_df["date"] == date_selected]["hour"].unique())
hour_selected = st.selectbox("⏰ 시간 선택", hour_options)

# 예측 버튼
if st.button("🔍 폭염 위험도 예측"):
    try:
        t_avg = temp_df[(temp_df["date"] == date_selected) & (temp_df["hour"] == hour_selected)]["value"].values[0]
        humidity = reh_df[(reh_df["date"] == date_selected) & (reh_df["hour"] == hour_selected)]["value"].values[0]
        wind = wind_df[(wind_df["date"] == date_selected) & (wind_df["hour"] == hour_selected)]["value"].values[0]
        t_max = tmx_df[tmx_df["date"] == date_selected]["value"].values[-1]
        t_min = tmn_df[tmn_df["date"] == date_selected]["value"].values[-1]

        st.markdown("### ☁️ 실시간 기상 정보")
        st.markdown(f"- 평균기온: **{t_avg}℃**")
        st.markdown(f"- 일 최고기온: **{t_max}℃**")
        st.markdown(f"- 일 최저기온: **{t_min}℃**")
        st.markdown(f"- 습도: **{humidity}%**")
        st.markdown(f"- 풍속: **{wind} m/s**")

        # 입력값 구성 및 예측
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

        st.markdown("### 🔥 예측된 폭염 위험도")
        st.markdown(f"**위험 등급:** {risk}")
        st.markdown(f"**예상 온열질환자 수:** {pred:.2f}명")

    except Exception as e:
        st.error("❌ 해당 시각의 기상 정보가 존재하지 않습니다.")
        st.error(str(e))
