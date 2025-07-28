import streamlit as st
import pandas as pd
import datetime
import joblib
import requests
import math
from urllib.parse import unquote
import matplotlib.pyplot as plt

# ----------- STYLE -----------
st.set_page_config(layout="centered")
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
    height: 45px !important;
    margin-top: 32px;
}
.stButton > button:hover {
    background-color: #1d4ed8;
}
.stNumberInput input,
.stSelectbox > div > div,
.stSelectbox select,
div.st-cj,
.css-1cpxqw2.edgvbvh3, .stDateInput input {
    background-color: #2c2f36 !important;
    color: #ffffff !important;
    border: 1px solid #444c56 !important;
    border-radius: 6px;
    padding: 0.4rem 0.6rem;
    font-size: 14px !important;
    height: 45px !important;
}
.stMetricLabel, .stMetricValue {
    color: #ffffff !important;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ----------- LOAD MODEL & FEATURES -----------
model = joblib.load("trained_model.pkl")
features = joblib.load("feature_names.pkl")

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

def get_latest_base_time():
    now = datetime.datetime.now()
    hour = now.hour
    if hour < 2: return "2300"
    elif hour < 5: return "0200"
    elif hour < 8: return "0500"
    elif hour < 11: return "0800"
    elif hour < 14: return "1100"
    elif hour < 17: return "1400"
    elif hour < 20: return "1700"
    elif hour < 23: return "2000"
    else: return "2300"

def get_weather_from_api(region_name, target_date):
    latlon = region_to_latlon.get(region_name, (37.5665, 126.9780))
    nx, ny = convert_latlon_to_xy(*latlon)

    base_date = (target_date - datetime.timedelta(days=1)).strftime("%Y%m%d")
    base_time = get_latest_base_time()

    params = {
        "serviceKey": KMA_API_KEY,
        "numOfRows": "300", "pageNo": "1", "dataType": "JSON",
        "base_date": base_date, "base_time": base_time,
        "nx": nx, "ny": ny
    }

    try:
        r = requests.get("http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst", params=params, timeout=10, verify=False)
        items = r.json().get("response", {}).get("body", {}).get("items", {}).get("item", [])
        df = pd.DataFrame(items)
        df = df[df["category"].isin(["TMX", "TMN", "REH", "WSD"])]
        df = df[df["fcstDate"] == target_date.strftime("%Y%m%d")]
        summary = {}
        for cat in ["TMX", "TMN", "REH", "WSD"]:
            values = df[df["category"] == cat]["fcstValue"].astype(float)
            if not values.empty:
                summary[cat] = values.mean() if cat == "REH" else values.iloc[0]
        return summary
    except Exception as e:
        return {}

def calculate_avg_temp(tmx, tmn):
    if tmx is not None and tmn is not None:
        return round((tmx + tmn) / 2, 1)
    return None

# ----------- UI -----------

region_to_latlon = {
    "서울특별시": (37.5665, 126.9780), "부산광역시": (35.1796, 129.0756), "대구광역시": (35.8722, 128.6025),
    "인천광역시": (37.4563, 126.7052), "광주광역시": (35.1595, 126.8526), "대전광역시": (36.3504, 127.3845),
    "울산광역시": (35.5384, 129.3114), "세종특별자치시": (36.4800, 127.2890), "경기도": (37.4138, 127.5183),
    "강원도": (37.8228, 128.1555), "충청북도": (36.6358, 127.4917), "충청남도": (36.5184, 126.8000),
    "전라북도": (35.7167, 127.1442), "전라남도": (34.8161, 126.4630), "경상북도": (36.5760, 128.5056),
    "경상남도": (35.4606, 128.2132), "제주특별자치도": (33.4996, 126.5312)
}

st.markdown("### ☀️ 온열질환 예측 대시보드")
st.caption("폭염에 따른 온열질환자 수를 예측합니다.")

c1, c2, c3 = st.columns([2, 2, 1])
with c1:
    region = st.selectbox("지역 선택", list(region_to_latlon.keys()))
with c2:
    today = datetime.date.today()
    date_selected = st.date_input("날짜 선택", value=today, min_value=today, max_value=today + datetime.timedelta(days=6))
with c3:
    predict_clicked = st.button("예측하기")

if predict_clicked:
    weather = get_weather_from_api(region, date_selected)
    if not weather or weather.get("TMX") is None or weather.get("TMN") is None:
        st.error("예보 데이터를 불러오지 못했습니다. 잠시 후 다시 시도해주세요.")
        st.stop()

    avg_temp = calculate_avg_temp(weather["TMX"], weather["TMN"])

    st.markdown("#### ☁️ 기상 정보 요약")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("최고기온", f"{weather['TMX']:.1f}℃")
    col2.metric("최저기온", f"{weather['TMN']:.1f}℃")
    col3.metric("평균기온", f"{avg_temp:.1f}℃")
    col4.metric("습도", f"{weather['REH']:.1f}%")

    input_data = {
        "최고체감온도(°C)": weather["TMX"] + 1.5,
        "최고기온(°C)": weather["TMX"],
        "평균기온(°C)": avg_temp,
        "최저기온(°C)": weather["TMN"],
        "평균상대습도(%)": weather["REH"]
    }
    input_df = pd.DataFrame([input_data])[features]

    pred = model.predict(input_df)[0]
    risk = get_risk_level(pred)

    st.markdown("#### 💡 온열질환자 예측")
    col1, col2 = st.columns(2)
    col1.metric("예측 환자 수", f"{pred:.2f}명")
    col2.metric("위험 등급", risk)

    baseline = 6.8
    diff = pred - baseline
    st.caption(f"전년도 대비 {'+' if diff >= 0 else ''}{diff:.1f}명")

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
