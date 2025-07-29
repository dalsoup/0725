
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, timedelta

# 모델 및 피처 불러오기
model = joblib.load("trained_model.pkl")
features = joblib.load("feature_names.pkl")

# 데이터 로드 함수
@st.cache_data
def load_cleaned_data():
    def load(path):
        df = pd.read_excel(path, header=None)
        df.columns = ["day", "hour", "forecast", "value"]
        df = df[pd.to_numeric(df["day"], errors="coerce").notnull()].copy()
        df["day"] = df["day"].astype(int)
        df["hour_clean"] = df["hour"].astype(int) // 100
        start_date = datetime(2025, 7, 1)
        df["date"] = df["day"].apply(lambda x: start_date + timedelta(days=x - 1))
        return df[["date", "hour_clean", "value"]]

    temp = load("청운효자동_1시간기온_20250701_20250728.xlsx")
    reh = load("청운효자동_습도_20250701_20250728.xlsx")
    tmx = load("청운효자동_일최고기온_20250701_20250728.xlsx")
    tmn = load("청운효자동_일최저기온_20250701_20250728.xlsx")
    wind = load("청운효자동_풍속_20250701_20250728.xlsx")
    return temp, reh, tmx, tmn, wind

# 데이터 로드
temp_df, reh_df, tmx_df, tmn_df, wind_df = load_cleaned_data()

# ---------------- UI 시작 ----------------
st.set_page_config(layout="wide")
st.title("🔥 폭염 위험도 예측 대시보드")

# 지역 선택
region = st.selectbox("지역 선택", ["서울특별시 (청운효자동)"])

# 날짜 선택
date_selected = st.date_input("날짜 선택", min_value=datetime(2025,7,1), max_value=datetime(2025,7,27))

# 시간 선택 (기상청 단기예보 시간대)
valid_hours = [2, 5, 8, 11, 14, 17, 20, 23]
hour_selected = st.selectbox("시간 선택", valid_hours)

# 선택한 시간대에 데이터가 있는지 확인
has_data = not temp_df[(temp_df["date"] == date_selected) & (temp_df["hour_clean"] == hour_selected)].empty

if not has_data:
    st.warning("선택한 시각에 대한 기온 데이터가 없습니다. 다른 날짜나 시간을 선택해 주세요.")
else:
    if st.button("🔍 폭염 위험도 예측"):
        try:
            t_avg = temp_df[(temp_df["date"] == date_selected) & (temp_df["hour_clean"] == hour_selected)]["value"].values[0]
            humidity = reh_df[(reh_df["date"] == date_selected) & (reh_df["hour_clean"] == hour_selected)]["value"].values[0]
            wind = wind_df[(wind_df["date"] == date_selected) & (wind_df["hour_clean"] == hour_selected)]["value"].values[0]
            t_max = tmx_df[tmx_df["date"] == date_selected]["value"].values[-1]
            t_min = tmn_df[tmn_df["date"] == date_selected]["value"].values[-1]

            # 작년 비교
            prev_year_date = date_selected.replace(year=2024)
            try:
                prev_t_avg = temp_df[(temp_df["date"] == prev_year_date) & (temp_df["hour_clean"] == hour_selected)]["value"].values[0]
                prev_pred = model.predict(pd.DataFrame([{
                    "최고체감온도(°C)": t_max + 1.5,
                    "최고기온(°C)": t_max,
                    "평균기온(°C)": prev_t_avg,
                    "최저기온(°C)": t_min,
                    "평균상대습도(%)": humidity
                }])[features])[0]
            except:
                prev_pred = None

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

            # 결과 카드
            st.markdown("### 📊 예측 결과 카드")
            col1, col2, col3 = st.columns(3)
            col1.metric("🔥 폭염 위험 등급", risk)
            col2.metric("🤒 예상 온열질환자 수", f"{pred:.2f}명", 
                        delta=f"{pred - prev_pred:.2f}명" if prev_pred is not None else "N/A")
            col3.metric("🌡 평균기온", f"{t_avg:.1f}℃", 
                        delta=f"{t_avg - prev_t_avg:.1f}℃" if prev_pred is not None else "N/A")

            with st.expander("🔎 기상 정보 상세 보기"):
                st.markdown(f"- 🌡 일 최고기온: **{t_max}℃**")
                st.markdown(f"- 🧊 일 최저기온: **{t_min}℃**")
                st.markdown(f"- 💧 습도: **{humidity}%**")
                st.markdown(f"- 🍃 풍속: **{wind} m/s**")

        except Exception as e:
            st.error("❌ 예측 중 오류가 발생했습니다.")
            st.error(str(e))
