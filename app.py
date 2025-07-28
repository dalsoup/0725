import streamlit as st
import pandas as pd
import datetime
import joblib
import requests
import math
from urllib.parse import unquote

# -------------------- STYLE --------------------
st.set_page_config(layout="centered")
st.markdown("""
<link href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css" rel="stylesheet" />
<style>
html, body, .stApp {
    background-color: #0e1117 !important;
    color: #ffffff !important;
    font-family: 'Pretendard', sans-serif !important;
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
    padding: 0.4rem 1.2rem;
    border-radius: 8px;
    border: none;
    height: 45px !important;
    font-size: 14px;
    line-height: 1.2rem;
    margin-top: 0 !important;
}
.stButton > button:hover {
    background-color: #1d4ed8;
}
.stDateInput input,
.stSelectbox > div > div,
.stSelectbox select {
    background-color: #2c2f36 !important;
    color: #ffffff !important;
    border: 1px solid #444c56 !important;
    border-radius: 6px;
    padding: 0.4rem 0.6rem;
    font-size: 14px;
    height: 45px !important;
    line-height: 1.2rem;
}
</style>
""", unsafe_allow_html=True)

# -------------------- MODEL & KEY --------------------
model = joblib.load("trained_model.pkl")
KMA_API_KEY = unquote(st.secrets["KMA"]["API_KEY"])

# -------------------- REGION --------------------
region_to_latlon = {
    "서울특별시": (37.5665, 126.9780), "부산광역시": (35.1796, 129.0756),
    "대구광역시": (35.8722, 128.6025), "인천광역시": (37.4563, 126.7052),
    "광주광역시": (35.1595, 126.8526), "대전광역시": (36.3504, 127.3845),
    "울산광역시": (35.5384, 129.3114), "세종특별자치시": (36.4800, 127.2890),
    "경기도": (37.4138, 127.5183), "강원도": (37.8228, 128.1555),
    "충청북도": (36.6358, 127.4917), "충청남도": (36.5184, 126.8000),
    "전라북도": (35.7167, 127.1442), "전라남도": (34.8161, 126.4630),
    "경상북도": (36.5760, 128.5056), "경상남도": (35.4606, 128.2132),
    "제주특별자치도": (33.4996, 126.5312)
}

# -------------------- UTILS --------------------
def get_base_time(now):
    valid = [23, 20, 17, 14, 11, 8, 5, 2]
    for v in valid:
        if now.hour >= v:
            return f"{v:02d}00"
    return "2300"

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
    theta = (theta + 2 * math.pi) % (2 * math.pi)
    theta *= sn
    x = ra * math.sin(theta) + XO + 0.5
    y = ro - ra * math.cos(theta) + YO + 0.5
    return int(x), int(y)

def get_weather(region_name, target_date):
    latlon = region_to_latlon.get(region_name)
    nx, ny = convert_latlon_to_xy(*latlon)
    now = datetime.datetime.now()
    base_date = now.strftime("%Y%m%d") if target_date == now.date() else (target_date - datetime.timedelta(days=1)).strftime("%Y%m%d")
    base_time = get_base_time(now) if target_date == now.date() else "2300"

    params = {
        "serviceKey": KMA_API_KEY,
        "numOfRows": "500", "pageNo": "1", "dataType": "JSON",
        "base_date": base_date, "base_time": base_time,
        "nx": nx, "ny": ny
    }

    r = requests.get("http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst", params=params)
    items = r.json().get("response", {}).get("body", {}).get("items", {}).get("item", [])
    df = pd.DataFrame(items)
    df = df[df["fcstDate"] == target_date.strftime("%Y%m%d")]
    df = df[df["category"].isin(["T3H", "TMX", "TMN", "REH", "WSD"])]
    summary = {}
    for cat in ["TMX", "TMN", "REH", "T3H", "WSD"]:
        values = df[df["category"] == cat]["fcstValue"].astype(float)
        if not values.empty:
            summary[cat] = values.mean() if cat in ["REH", "T3H"] else values.iloc[0]
    return summary

def calculate_avg_temp(tmx, tmn):
    try:
        return round((float(tmx) + float(tmn)) / 2, 1)
    except:
        return None

def calculate_heat_index(temp_c, humidity):
    try:
        T = float(temp_c)
        R = float(humidity)
        return round(
            -8.784695 + 1.61139411*T + 2.338549*R
            - 0.14611605*T*R - 0.012308094*T**2
            - 0.016424828*R**2 + 0.002211732*T**2*R
            + 0.00072546*T*R**2 - 0.000003582*T**2*R**2,
            1
        )
    except:
        return None

def get_risk_level(pred):
    if pred == 0: return "🟢 매우 낮음"
    elif pred <= 2: return "🟡 낮음"
    elif pred <= 5: return "🟠 보통"
    elif pred <= 10: return "🔴 높음"
    else: return "🔥 매우 높음"

# -------------------- UI --------------------
st.markdown("### 광역자치단체별 온열질환 위험도 예측 AI 플랫폼")

today = datetime.date.today()
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    region = st.selectbox("지역 선택", list(region_to_latlon.keys()))
with col2:
    date = st.date_input("날짜를 선택하세요", min_value=today, max_value=today + datetime.timedelta(days=3))
with col3:
    st.markdown(" ")
    predict = st.button("예측하기")

# 🔮 예측하기 버튼 눌렀을 때
if predict:
    with st.spinner("📡 기상청 데이터를 불러오는 중..."):
        weather = get_weather(region, date)

    if not weather:
        st.error("기상 데이터가 부족하여 예측할 수 없습니다.")
    else:
        tmax = weather.get("TMX")
        tmin = weather.get("TMN")
        avg_temp = calculate_avg_temp(tmax, tmin)
        t3h = weather.get("T3H", avg_temp)
        humidity = weather.get("REH", 60)
        wind = weather.get("WSD", 1)

        # 🔐 안전하게 변환
        try:
            t3h = float(t3h)
        except:
            t3h = avg_temp
        try:
            humidity = float(humidity)
        except:
            humidity = 60

        heat_index = calculate_heat_index(t3h, humidity)

        # 🔮 모델 입력값 정비
        input_df = pd.DataFrame([{
            "최고체감온도(°C)": heat_index,
            "최고기온(°C)": tmax,
            "평균기온(°C)": t3h,
            "최저기온(°C)": tmin,
            "평균상대습도(%)": humidity
        }])

        try:
            pred = model.predict(input_df.values)[0]
            risk = get_risk_level(pred)

            st.markdown("### ☁️ 오늘의 기상정보")
            st.metric("최고기온", f"{tmax}℃" if tmax is not None else "-℃")
            st.metric("최저기온", f"{tmin}℃" if tmin is not None else "-℃")
            st.metric("평균기온", f"{avg_temp}℃" if avg_temp is not None else "-℃")
            st.metric("습도", f"{humidity}%")
            st.metric("체감온도", f"{heat_index}℃" if heat_index is not None else "-℃")

            st.markdown("### 💡 온열질환자 예측")
            st.metric("예측 온열질환자 수", f"{pred:.2f}명")
            st.metric("위험 등급", risk)

        except Exception as e:
            st.error(f"예측 중 오류가 발생했습니다: {e}")
            st.write("입력값:", input_df)