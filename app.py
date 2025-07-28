import streamlit as st
import pandas as pd
import datetime
import joblib
import requests
import math
from urllib.parse import unquote
import matplotlib.pyplot as plt

# ----------- STYLE (Dark Mode) -----------
st.set_page_config(layout="wide")
st.markdown("""
<style>
html, body, .stApp {
    background-color: #0e1117 !important;
    color: #ffffff !important;
}
div[data-testid="column"] > div {
    background-color: #1e1e1e;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 16px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.3);
}
.stButton > button {
    background-color: #2563eb;
    color: white;
    font-weight: 600;
    padding: 0.6rem 1.2rem;
    border-radius: 8px;
    border: none;
}
.stButton > button:hover {
    background-color: #1d4ed8;
}
.stNumberInput input,
.stSelectbox div,
div.st-cj {
    background-color: #2c2f36 !important;
    color: #ffffff !important;
    border: 1px solid #444c56 !important;
    border-radius: 6px;
    padding: 0.4rem 0.6rem;
}
.stMetricLabel, .stMetricValue {
    color: #ffffff !important;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ----------- MODEL LOAD -----------
model = joblib.load("trained_model.pkl")

# ----------- API KEY -----------
KMA_API_KEY = unquote(st.secrets["KMA"]["API_KEY"])

# ----------- FUNCTIONS -----------
def get_risk_level(pred):
    if pred == 0: return "🟢 매우 낮음"
    elif pred <= 2: return "🟡 낮음"
    elif pred <= 5: return "🟠 보통"
    elif pred <= 10: return "🔴 높음"
    else: return "🔥 매우 높음"

def convert_latlon_to_xy(lat, lon):
    RE, GRID = 6371.00877, 5.0
    SLAT1, SLAT2, OLON, OLAT = 30.0, 60.0, 126.0, 38.0
    XO, YO = 43, 136
    DEGRAD = math.pi / 180.0
    re = RE / GRID
    slat1, slat2 = SLAT1 * DEGRAD, SLAT2 * DEGRAD
    olon, olat = OLON * DEGRAD, OLAT * DEGRAD
    sn = math.log(math.cos(slat1) / math.cos(slat2)) / math.log(math.tan(math.pi * 0.25 + slat2 * 0.5) / math.tan(math.pi * 0.25 + slat1 * 0.5))
    sf = math.tan(math.pi * 0.25 + slat1 * 0.5)**sn * math.cos(slat1) / sn
    ro = re * sf / (math.tan(math.pi * 0.25 + olat * 0.5)**sn)
    ra = re * sf / (math.tan(math.pi * 0.25 + lat * DEGRAD * 0.5)**sn)
    theta = lon * DEGRAD - olon
    if theta > math.pi: theta -= 2.0 * math.pi
    if theta < -math.pi: theta += 2.0 * math.pi
    theta *= sn
    x = ra * math.sin(theta) + XO + 0.5
    y = ro - ra * math.cos(theta) + YO + 0.5
    return int(x), int(y)

def get_weather_from_api(region_name):
    latlon = region_to_latlon.get(region_name, (37.5665, 126.9780))
    nx, ny = convert_latlon_to_xy(*latlon)
    now = datetime.datetime.now()
    base_time = max([h for h in [2, 5, 8, 11, 14, 17, 20, 23] if now.hour >= h], default=23)
    base_date = now.strftime("%Y%m%d") if now.hour >= base_time else (now - datetime.timedelta(days=1)).strftime("%Y%m%d")
    params = {
        "serviceKey": KMA_API_KEY, "numOfRows": "300", "pageNo": "1", "dataType": "JSON",
        "base_date": base_date, "base_time": f"{base_time:02d}00", "nx": nx, "ny": ny
    }
    try:
        r = requests.get("http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst", params=params, timeout=10, verify=False)
        items = r.json().get("response", {}).get("body", {}).get("items", {}).get("item", [])
        df = pd.DataFrame(items)
        df = df[df["category"].isin(["T3H", "TMX", "TMN", "REH", "WSD"])]
        df["fcstHour"] = df["fcstTime"].astype(int) // 100
        df["hour_diff"] = abs(df["fcstHour"] - now.hour)
        closest = df.loc[df.groupby("category")["hour_diff"].idxmin()].set_index("category")
        return {c: float(closest.loc[c]["fcstValue"]) if c in closest.index else None for c in ["TMX", "TMN", "REH", "T3H", "WSD"]}
    except: return {}

region_to_latlon = {
    "서울특별시": (37.5665, 126.9780), "부산광역시": (35.1796, 129.0756), "대구광역시": (35.8722, 128.6025),
    "인천광역시": (37.4563, 126.7052), "광주광역시": (35.1595, 126.8526), "대전광역시": (36.3504, 127.3845),
    "울산광역시": (35.5384, 129.3114), "세종특별자치시": (36.4800, 127.2890), "경기도": (37.4138, 127.5183),
    "강원도": (37.8228, 128.1555), "충청북도": (36.6358, 127.4917), "충청남도": (36.5184, 126.8000),
    "전라북도": (35.7167, 127.1442), "전라남도": (34.8161, 126.4630), "경상북도": (36.5760, 128.5056),
    "경상남도": (35.4606, 128.2132), "제주특별자치도": (33.4996, 126.5312)
}

# ----------- UI -----------
st.markdown("### 👋 Hello, User")
col1, col2, col3 = st.columns([3, 2, 1])
with col1: st.caption("폭염에 따른 온열질환 발생 예측 플랫폼")
with col2: date_selected = st.date_input("Select period", value=(datetime.date.today(), datetime.date.today()))
with col3: st.button("📤 새 리포트")

st.markdown("#### ☁️ 오늘의 기상정보")
region = st.selectbox("지역 선택", list(region_to_latlon.keys()))
data = get_weather_from_api(region)

col1, col2, col3 = st.columns(3)
with col1:
    max_temp = st.number_input("최고기온(°C)", value=data.get("TMX", 32.0), step=0.1)
    max_feel = st.number_input("최고체감온도(°C)", value=max_temp + 1.5, step=0.1)
with col2:
    min_temp = st.number_input("최저기온(°C)", value=data.get("TMN", 25.0), step=0.1)
    humidity = st.number_input("평균상대습도(%)", value=data.get("REH", 70.0), step=1.0)
with col3:
    avg_temp = st.number_input("평균기온(°C)", value=data.get("T3H", 28.5), step=0.1)

input_df = pd.DataFrame([{ 
    "광역자치단체": region,
    "최고체감온도(°C)": max_feel,
    "최고기온(°C)": max_temp,
    "평균기온(°C)": avg_temp,
    "최저기온(°C)": min_temp,
    "평균상대습도(%)": humidity
}])
pred = model.predict(input_df.drop(columns=["광역자치단체"]))[0]
risk = get_risk_level(pred)

# ----------- PREDICTION LOGGING -----------
input_df["날짜"] = datetime.date.today().strftime("%Y-%m-%d")
input_df["예측환자수"] = pred
log_path = "prediction_log.csv"
try:
    if not pd.io.common.file_exists(log_path):
        input_df.to_csv(log_path, index=False)
    else:
        log_df = pd.read_csv(log_path)
        combined = pd.concat([log_df, input_df], ignore_index=True)
        combined.drop_duplicates(subset=["날짜", "광역자치단체"], keep="last", inplace=True)
        combined.to_csv(log_path, index=False)
except Exception as e:
    st.warning(f"[예측값 저장 실패] {e}")

# ----------- SUMMARY CARDS -----------
st.markdown("#### 📊 요약")
sum1, sum2, sum3, sum4 = st.columns(4)
sum1.metric("예측 환자 수", f"{pred:.2f}명")
sum2.metric("위험 등급", risk)
sum3.metric("최고기온", f"{max_temp:.1f}°C")
sum4.metric("습도", f"{humidity:.1f}%")

# ----------- VISUALIZATION -----------
st.markdown("#### 📈 예측 기록 그래프")
try:
    df_log = pd.read_csv("prediction_log.csv")
    df_log["날짜"] = pd.to_datetime(df_log["날짜"])
    region_filter = st.selectbox("지역별 기록 보기", sorted(df_log["광역자치단체"].unique()))
    filtered = df_log[df_log["광역자치단체"] == region_filter].sort_values("날짜")
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(filtered["날짜"], filtered["예측환자수"], marker="o", color="#3182f6", label="2025 예측값")
    last_year = filtered.copy(); last_year["날짜"] = last_year["날짜"] - pd.DateOffset(years=1)
    ax.plot(last_year["날짜"], last_year["예측환자수"], linestyle="--", color="#9ca3af", label="2024 동일일 예측")
    for _, row in filtered.iterrows():
        if row["예측환자수"] > 10:
            ax.axvspan(row["날짜"] - pd.Timedelta(days=0.5), row["날짜"] + pd.Timedelta(days=0.5), color="#fee2e2", alpha=0.4)
    for _, row in filtered.iterrows():
        emoji = get_risk_level(row["예측환자수"]).split()[0]
        ax.text(row["날짜"], row["예측환자수"] + 0.5, emoji, fontsize=9, ha="center")
    ax.set_title(f"{region_filter} 예측 환자수 추이", fontsize=14)
    ax.set_xlabel("날짜"); ax.set_ylabel("예측환자수")
    ax.legend(); ax.grid(True, linestyle="--", alpha=0.3)
    st.pyplot(fig)
    st.download_button("📥 예측 기록 CSV 다운로드", df_log.to_csv(index=False).encode("utf-8-sig"), file_name="prediction_log.csv", mime="text/csv")
except:
    st.info("📁 예측 기록이 아직 충분하지 않거나 그래프를 그릴 수 없습니다.")
