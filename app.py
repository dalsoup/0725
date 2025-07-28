import streamlit as st
import pandas as pd
import datetime
import plotly.express as px
import joblib
import requests
import math
from urllib.parse import unquote

# ----------- STYLE (Toss-like) -----------
st.markdown("""
<style>
html, body, .stApp {
    background-color: #ffffff !important;
    font-family: 'Pretendard', 'Noto Sans KR', sans-serif;
}
div[data-testid="column"] > div {
    background-color: #f9fafb;
    border-radius: 12px;
    padding: 24px 24px 16px 24px;
    margin-bottom: 16px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}
.stButton > button {
    background-color: #3182f6;
    color: white;
    font-weight: 600;
    padding: 0.6rem 1.2rem;
    border-radius: 8px;
    border: none;
}
.stButton > button:hover {
    background-color: #2563eb;
}
.stNumberInput input {
    background-color: #ffffff;
    border-radius: 6px;
    padding: 0.4rem 0.6rem;
}
.stMetricLabel {
    font-weight: 600;
    color: #6b7280;
}
</style>
""", unsafe_allow_html=True)

# ----------- MODEL LOAD -----------
model = joblib.load("trained_model.pkl")

# ----------- WEATHER FUNCTIONS -----------
KMA_API_KEY = unquote(st.secrets["KMA"]["API_KEY"])

def convert_latlon_to_xy(lat, lon):
    RE = 6371.00877
    GRID = 5.0
    SLAT1 = 30.0
    SLAT2 = 60.0
    OLON = 126.0
    OLAT = 38.0
    XO = 43
    YO = 136
    DEGRAD = math.pi / 180.0
    re = RE / GRID
    slat1 = SLAT1 * DEGRAD
    slat2 = SLAT2 * DEGRAD
    olon = OLON * DEGRAD
    olat = OLAT * DEGRAD
    sn = math.tan(math.pi * 0.25 + slat2 * 0.5) / math.tan(math.pi * 0.25 + slat1 * 0.5)
    sn = math.log(math.cos(slat1) / math.cos(slat2)) / math.log(sn)
    sf = math.tan(math.pi * 0.25 + slat1 * 0.5)
    sf = math.pow(sf, sn) * math.cos(slat1) / sn
    ro = math.tan(math.pi * 0.25 + olat * 0.5)
    ro = re * sf / math.pow(ro, sn)
    ra = math.tan(math.pi * 0.25 + lat * DEGRAD * 0.5)
    ra = re * sf / math.pow(ra, sn)
    theta = lon * DEGRAD - olon
    if theta > math.pi:
        theta -= 2.0 * math.pi
    if theta < -math.pi:
        theta += 2.0 * math.pi
    theta *= sn
    x = ra * math.sin(theta) + XO + 0.5
    y = ro - ra * math.cos(theta) + YO + 0.5
    return int(x), int(y)

region_to_latlon = {
    "서울특별시": (37.5665, 126.9780),
    "부산광역시": (35.1796, 129.0756),
    "대구광역시": (35.8722, 128.6025),
    "인천광역시": (37.4563, 126.7052),
    "광주광역시": (35.1595, 126.8526),
    "대전광역시": (36.3504, 127.3845),
    "울산광역시": (35.5384, 129.3114),
    "세종특별자치시": (36.4800, 127.2890),
    "경기도": (37.4138, 127.5183),
    "강원도": (37.8228, 128.1555),
    "충청북도": (36.6358, 127.4917),
    "충청남도": (36.5184, 126.8000),
    "전라북도": (35.7167, 127.1442),
    "전라남도": (34.8161, 126.4630),
    "경상북도": (36.5760, 128.5056),
    "경상남도": (35.4606, 128.2132),
    "제주특별자치도": (33.4996, 126.5312)
}

def get_base_time(now):
    valid_times = [2, 5, 8, 11, 14, 17, 20, 23]
    hour = now.hour
    for t in reversed(valid_times):
        if hour >= t:
            return f"{t:02d}00", now.strftime("%Y%m%d")
    return "2300", (now - datetime.timedelta(days=1)).strftime("%Y%m%d")

def get_weather_from_api(region_name):
    lat, lon = region_to_latlon.get(region_name, (37.5665, 126.9780))
    nx, ny = convert_latlon_to_xy(lat, lon)
    now = datetime.datetime.now()
    base_time, base_date = get_base_time(now)
    url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
    params = {
        "serviceKey": KMA_API_KEY,
        "numOfRows": "300",
        "pageNo": "1",
        "dataType": "JSON",
        "base_date": base_date,
        "base_time": base_time,
        "nx": nx,
        "ny": ny
    }
    try:
        response = requests.get(url, params=params, timeout=10, verify=False)
        data = response.json()
        items = data.get("response", {}).get("body", {}).get("items", {}).get("item", [])
    except:
        return {}
    df = pd.DataFrame(items)
    df = df[df["category"].isin(["T3H", "TMX", "TMN", "REH", "WSD"])]
    df["fcstHour"] = df["fcstTime"].astype(int) // 100
    now_hour = now.hour
    df["hour_diff"] = abs(df["fcstHour"] - now_hour)
    closest = df.loc[df.groupby("category")["hour_diff"].idxmin()]
    closest = closest.set_index("category")
    def get(c):
        return float(closest.loc[c]["fcstValue"]) if c in closest.index else None
    return {
        "max_temp": get("TMX"),
        "min_temp": get("TMN"),
        "humidity": get("REH"),
        "avg_temp": get("T3H"),
        "wind": get("WSD")
    }

# ----------- HEADER -----------
st.markdown("### 👋 Hello, User")
col1, col2, col3 = st.columns([3, 2, 1])
with col1:
    st.caption("폭염에 따른 온열질환 발생 예측 플랫폼")
with col2:
    date_selected = st.date_input("Select period", value=(datetime.date.today(), datetime.date.today()))
with col3:
    st.button("📤 새 리포트")

# ----------- INPUTS -----------
st.markdown("#### ☁️ 오늘의 기상정보")
region = st.selectbox("지역 선택", list(region_to_latlon.keys()))
data = get_weather_from_api(region)
col1, col2, col3 = st.columns(3)
with col1:
    max_temp = st.number_input("최고기온(°C)", value=data.get("max_temp", 32.0), step=0.1)
    max_feel = st.number_input("최고체감온도(°C)", value=max_temp + 1.5, step=0.1)
with col2:
    min_temp = st.number_input("최저기온(°C)", value=data.get("min_temp", 25.0), step=0.1)
    humidity = st.number_input("평균상대습도(%)", value=data.get("humidity", 70.0), step=1.0)
with col3:
    avg_temp = st.number_input("평균기온(°C)", value=data.get("avg_temp", 28.5), step=0.1)

# ----------- PREDICT -----------
input_df = pd.DataFrame([{ 
    "날짜": datetime.date.today().strftime("%Y-%m-%d"),
    "광역자치단체": region,
    "최고체감온도(°C)": max_feel,
    "최고기온(°C)": max_temp,
    "평균기온(°C)": avg_temp,
    "최저기온(°C)": min_temp,
    "평균상대습도(%)": humidity
}])
pred = model.predict(input_df.drop(columns=["날짜", "광역자치단체"]))[0]
input_df["예측환자수"] = pred

# CSV 파일에 저장
try:
    log_path = "prediction_log.csv"
    if not pd.io.common.file_exists(log_path):
        input_df.to_csv(log_path, index=False)
    else:
        past_df = pd.read_csv(log_path)
        combined = pd.concat([past_df, input_df], ignore_index=True)
        combined.drop_duplicates(subset=["날짜", "광역자치단체"], keep="last", inplace=True)
        combined.to_csv(log_path, index=False)
except Exception as e:
    st.warning(f"[예측값 저장 실패] {e}")

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

risk = get_risk_level(pred)

# ----------- SUMMARY CARDS -----------
st.markdown("#### 📊 요약")
sum1, sum2, sum3, sum4 = st.columns(4)
sum1.metric("예측 환자 수", f"{pred:.2f}명")
sum2.metric("위험 등급", risk)
sum3.metric("최고기온", f"{max_temp:.1f}°C")
sum4.metric("습도", f"{humidity:.1f}%")

# ----------- COMPARISON HISTORY -----------
st.markdown("#### 📁 예측 기록 보기")
try:
    df_log = pd.read_csv("prediction_log.csv")
    df_today = df_log[df_log["날짜"] == datetime.date.today().strftime("%Y-%m-%d")]
    st.dataframe(df_today.sort_values("예측환자수", ascending=False), use_container_width=True)
except:
    st.info("예측 기록이 아직 없습니다.")