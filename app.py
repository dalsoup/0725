import streamlit as st
import pandas as pd
import datetime
import joblib
import requests
import math
from urllib.parse import unquote
import matplotlib.pyplot as plt

# ----------- STYLE (Dark Mode) -----------
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
    max-width: 100%;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    line-height: 1.2rem;
    height: 45px !important;
}
.stMetricLabel, .stMetricValue {
    color: #ffffff !important;
    font-weight: 600;
}
.css-13sd7wv.edgvbvh3 p {
    font-size: 13px;
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

def get_weather_from_api(region_name, target_date):
    latlon = region_to_latlon.get(region_name, (37.5665, 126.9780))
    nx, ny = convert_latlon_to_xy(*latlon)

    # ✅ base_date를 target_date의 하루 전으로 고정
    base_date = (target_date - datetime.timedelta(days=1)).strftime("%Y%m%d")
    base_time = "2300"

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
        df = df[df["category"].isin(["T3H", "TMX", "TMN", "REH", "WSD"])]
        target_str = target_date.strftime("%Y%m%d")
        df = df[df["fcstDate"] == target_str]
        summary = {}
        for cat in ["TMX", "TMN", "REH", "T3H", "WSD"]:
            values = df[df["category"] == cat]["fcstValue"].astype(float)
            if not values.empty:
                summary[cat] = values.mean() if cat in ["REH", "T3H"] else values.iloc[0]
        return summary
    except:
        return {}

def calculate_avg_temp(tmx, tmn):
    if tmx is not None and tmn is not None:
        return round((tmx + tmn) / 2, 1)
    return None

region_to_latlon = {
    "서울특별시": (37.5665, 126.9780), "부산광역시": (35.1796, 129.0756), "대구광역시": (35.8722, 128.6025),
    "인천광역시": (37.4563, 126.7052), "광주광역시": (35.1595, 126.8526), "대전광역시": (36.3504, 127.3845),
    "울산광역시": (35.5384, 129.3114), "세종특별자치시": (36.4800, 127.2890), "경기도": (37.4138, 127.5183),
    "강원도": (37.8228, 128.1555), "충청북도": (36.6358, 127.4917), "충청남도": (36.5184, 126.8000),
    "전라북도": (35.7167, 127.1442), "전라남도": (34.8161, 126.4630), "경상북도": (36.5760, 128.5056),
    "경상남도": (35.4606, 128.2132), "제주특별자치도": (33.4996, 126.5312)
}

# ----------- HEADER UI -----------
st.markdown("### 👋 Hello, User")
st.caption("폭염에 따른 온열질환 발생 예측 플랫폼")

c1, c2, c3 = st.columns([2, 2, 1])
with c1:
    region = st.selectbox("지역 선택", list(region_to_latlon.keys()), label_visibility="visible", key="region_select")
with c2:
    today = datetime.date.today()
    date_selected = st.date_input("날짜 선택", value=today, min_value=today, max_value=today + datetime.timedelta(days=5))
with c3:
    predict_clicked = st.button("예측하기")

if predict_clicked and region and date_selected:
    weather = get_weather_from_api(region, date_selected)
    avg_temp = calculate_avg_temp(weather.get("TMX"), weather.get("TMN"))
    st.markdown("#### ☁️ 오늘의 기상정보")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("최고기온", f"{weather.get('TMX', 0):.1f}℃")
    col2.metric("최저기온", f"{weather.get('TMN', 0):.1f}℃")
    col3.metric("평균기온", f"{avg_temp:.1f}℃" if avg_temp is not None else "-℃")
    col4.metric("습도", f"{weather.get('REH', 0):.1f}%")

    input_df = pd.DataFrame([{ 
        "광역자치단체": region,
        "최고체감온도(°C)": weather.get("TMX", 0) + 1.5,
        "최고기온(°C)": weather.get("TMX", 0),
        "평균기온(°C)": avg_temp or 0,
        "최저기온(°C)": weather.get("TMN", 0),
        "평균상대습도(%)": weather.get("REH", 0)
    }])
    pred = model.predict(input_df.drop(columns=["광역자치단체"]))[0]
    risk = get_risk_level(pred)

    st.markdown("#### 💡 온열질환자 예측")
    c1, c2 = st.columns(2)
    c1.metric("예측 온열질환자 수", f"{pred:.2f}명")
    c2.metric("위험 등급", risk)

    diff = pred - 6.8
    if diff >= 0:
        st.caption(f"전년도 대비 +{diff:.1f}명")
    else:
        st.caption(f"전년도 대비 {diff:.1f}명")

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
