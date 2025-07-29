
import streamlit as st
import pandas as pd
import joblib

# 🔹 모델 및 피처 로드
model = joblib.load("trained_model.pkl")
features = joblib.load("feature_names.pkl")

# 🔹 CSV 불러오기
@st.cache_data
def load_data():
    def clean(df):
        df.columns = ["day", "hour", "forecast", "value"]
        return df
    temp_df = clean(pd.read_csv("청운효자동_1시간기온_20250701_20250728.csv"))
    reh_df = clean(pd.read_csv("청운효자동_습도_20250701_20250728.csv"))
    tmx_df = clean(pd.read_csv("청운효자동_일최고기온_20250701_20250728.csv"))
    tmn_df = clean(pd.read_csv("청운효자동_일최저기온_20250701_20250728.csv"))
    wind_df = clean(pd.read_csv("청운효자동_풍속_20250701_20250728.csv"))
    return temp_df, reh_df, tmx_df, tmn_df, wind_df

# 🔹 전년도 온열질환자 수 로드
@st.cache_data
def load_baseline_data():
    df = pd.read_excel("ML_7_8월_2021_2025_dataset.xlsx")
    df["일시"] = pd.to_datetime(df["일시"])
    return df[df["광역자치단체"] == "서울"]  # 또는 청운효자동 해당 구역으로 수정 가능

temp_df, reh_df, tmx_df, tmn_df, wind_df = load_data()
baseline_df = load_baseline_data()

# 🔹 UI
st.title("🔥 폭염 위험도 예측 대시보드")
st.caption("2025년 7월 1일 ~ 28일 사이 청운효자동의 기상데이터를 기반으로 AI가 폭염 위험도를 예측하고, 전년도 환자수와 비교합니다.")

col1, col2 = st.columns(2)
with col1:
    date_selected = st.selectbox("날짜 선택", sorted(temp_df["day"].unique()))
with col2:
    hours = sorted(temp_df[temp_df["day"] == date_selected]["hour"].unique())
    hour_options = [f"{int(h)//100:02}:00" for h in hours]
    time_selected = st.selectbox("시간 선택", hour_options)
    selected_hour = int(time_selected.split(":")[0]) * 100 if time_selected else None

if st.button("🔍 폭염 위험도 조회") and selected_hour is not None:
    try:
        t_avg = temp_df[(temp_df["day"] == date_selected) & (temp_df["hour"] == selected_hour)]["value"].values[0]
        humidity = reh_df[(reh_df["day"] == date_selected) & (reh_df["hour"] == selected_hour)]["value"].values[0]
        wind = wind_df[(wind_df["day"] == date_selected) & (wind_df["hour"] == selected_hour)]["value"].values[0]
        t_max = tmx_df[tmx_df["day"] == date_selected]["value"].values[-1]
        t_min = tmn_df[tmn_df["day"] == date_selected]["value"].values[-1]

        st.markdown("### ☁️ 실시간 기상 정보")
        st.markdown(f"- 평균기온: **{t_avg}℃**")
        st.markdown(f"- 일 최고기온: **{t_max}℃**")
        st.markdown(f"- 일 최저기온: **{t_min}℃**")
        st.markdown(f"- 습도: **{humidity}%**")
        st.markdown(f"- 풍속: **{wind} m/s**")

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

        # 🔸 전년도 환자 수 비교
        month = 7  # 고정
        day_num = int(str(date_selected)[-2:])
        prev = baseline_df[
            (baseline_df["연도"] == 2024) &
            (baseline_df["월"] == month) &
            (baseline_df["일시"].dt.day == day_num)
        ]["환자수"].values
        baseline = prev[0] if len(prev) > 0 else 0
        diff = pred - baseline

        st.markdown("### 🔥 예측된 폭염 위험도")
        st.markdown(f"**위험 등급:** {risk}")
        st.markdown(f"**예상 온열질환자 수 (2025년):** {pred:.2f}명")
        st.markdown(f"**전년도 같은 날(2024년) 실제 환자 수:** {baseline:.2f}명")
        st.markdown(f"**전년도 대비 변화:** {'+' if diff >= 0 else ''}{diff:.2f}명")

    except Exception as e:
        st.error("❌ 선택하신 날짜 및 시간에 해당하는 데이터를 찾을 수 없습니다.")
        st.error(f"오류 내용: {e}")
