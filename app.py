import streamlit as st
import pandas as pd
import joblib
import datetime
import requests
import math
from urllib.parse import unquote

# 모델 불러오기
model = joblib.load("trained_model.pkl")

# secrets에서 기상청 API 키 불러오기
KMA_API_KEY = unquote(st.secrets["KMA"]["API_KEY"])

# 위경도 → 기상청 격자 좌표 변환 함수
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

def calculate_feels_like(temp, wind_speed):
    return round(13.12 + 0.6215*temp - 11.37*(wind_speed**0.16) + 0.3965*temp*(wind_speed**0.16), 1)

def get_weather_from_ultra_now(region_name):
    lat, lon = region_to_latlon.get(region_name, (37.5665, 126.9780))
    nx, ny = convert_latlon_to_xy(lat, lon)
    now = datetime.datetime.now()
    base_time = (now - datetime.timedelta(hours=1)).strftime("%H00")
    base_date = now.strftime("%Y%m%d")

    url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst"
    params = {
        "serviceKey": KMA_API_KEY,
        "numOfRows": "100",
        "pageNo": "1",
        "dataType": "JSON",
        "base_date": base_date,
        "base_time": base_time,
        "nx": nx,
        "ny": ny
    }

    try:
        response = requests.get(url, params=params, timeout=5, verify=False)
        data = response.json()
        items = data.get("response", {}).get("body", {}).get("items", {}).get("item", [])
        df = pd.DataFrame(items)
        df = df.set_index("category")
        temp = float(df.loc["T1H"]["obsrValue"]) if "T1H" in df.index else None
        wind = float(df.loc["WSD"]["obsrValue"]) if "WSD" in df.index else None
        hum = float(df.loc["REH"]["obsrValue"]) if "REH" in df.index else None
        feel = calculate_feels_like(temp, wind) if temp and wind else None
        return {"avg_temp": temp, "humidity": hum, "wind": wind, "max_feel": feel}
    except:
        return {}

def get_weather_from_ultra_fcst(region_name):
    lat, lon = region_to_latlon.get(region_name, (37.5665, 126.9780))
    nx, ny = convert_latlon_to_xy(lat, lon)
    now = datetime.datetime.now()
    base_time = (now - datetime.timedelta(hours=1)).strftime("%H00")
    base_date = now.strftime("%Y%m%d")

    url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtFcst"
    params = {
        "serviceKey": KMA_API_KEY,
        "numOfRows": "100",
        "pageNo": "1",
        "dataType": "JSON",
        "base_date": base_date,
        "base_time": base_time,
        "nx": nx,
        "ny": ny
    }

    try:
        response = requests.get(url, params=params, timeout=5, verify=False)
        data = response.json()
        items = data.get("response", {}).get("body", {}).get("items", {}).get("item", [])
        df = pd.DataFrame(items)
        now_hour = now.hour
        df["fcstHour"] = df["fcstTime"].astype(int) // 100
        df["hour_diff"] = abs(df["fcstHour"] - now_hour)
        df = df[df["category"].isin(["T1H", "REH", "WSD"])]
        closest = df.loc[df.groupby("category")["hour_diff"].idxmin()].set_index("category")
        temp = float(closest.loc["T1H"]["fcstValue"]) if "T1H" in closest.index else None
        wind = float(closest.loc["WSD"]["fcstValue"]) if "WSD" in closest.index else None
        hum = float(closest.loc["REH"]["fcstValue"]) if "REH" in closest.index else None
        feel = calculate_feels_like(temp, wind) if temp and wind else None
        return {"avg_temp": temp, "humidity": hum, "wind": wind, "max_feel": feel}
    except:
        return {}

def get_weather_combined(region_name):
    data = get_weather_from_api(region_name)
    if data: return data

    data = get_weather_from_ultra_now(region_name)
    if data: return data

    data = get_weather_from_ultra_fcst(region_name)
    return data

# ------------------ UI 시작 -------------------

st.subheader("온열질환 예측 대시보드")

col1, col2 = st.columns(2)
with col1:
    date_selected = st.date_input("날짜", datetime.date.today(),
                                   min_value=datetime.date.today(),
                                   max_value=datetime.date.today() + datetime.timedelta(days=5))
with col2:
    region = st.selectbox("광역자치단체", list(region_to_latlon.keys()))

if "show_inputs" not in st.session_state:
    st.session_state.show_inputs = False

if date_selected and region:
    if st.button("☁️ 기상정보 확인하기"):
        st.session_state.show_inputs = True

if st.session_state.show_inputs:
    with st.container():
        st.markdown("**기상 정보**")

        use_api = st.checkbox("기상청 단기예보 API 사용", key="api_checkbox")

        if use_api:
            weather_data = get_weather_from_api(region) or {}
        else:
            weather_data = {}

        st.write("필요시 직접 수정 후 예측 버튼을 누르세요.")
        col1, col2, col3 = st.columns(3)
        with col1:
            max_temp = st.number_input("최고기온(°C)", value=weather_data.get("max_temp", 32.0), key="max_temp")
            max_feel = st.number_input("최고체감온도(°C)", value=weather_data.get("max_feel", 33.0), key="max_feel")
        with col2:
            min_temp = st.number_input("최저기온(°C)", value=weather_data.get("min_temp", 25.0), key="min_temp")
            humidity = st.number_input("평균상대습도(%)", value=weather_data.get("humidity", 70.0), key="humidity")
        with col3:
            avg_temp = st.number_input("평균기온(°C)", value=weather_data.get("avg_temp", 28.5), key="avg_temp")

        if use_api or any([weather_data.get(k) is not None for k in ["max_temp", "min_temp", "avg_temp"]]):
            if st.button("📊 온열질환 예측하기"):
                input_df = pd.DataFrame([{ 
                    "광역자치단체": region,
                    "최고체감온도(°C)": max_feel,
                    "최고기온(°C)": max_temp,
                    "평균기온(°C)": avg_temp,
                    "최저기온(°C)": min_temp,
                    "평균상대습도(%)": humidity
                }])
                pred = model.predict(input_df.drop(columns=["광역자치단체"]))[0]

                def get_risk_level(pred):
                    if pred == 0: return "🟢 매우 낮음"
                    elif pred <= 2: return "🟡 낮음"
                    elif pred <= 5: return "🟠 보통"
                    elif pred <= 10: return "🔴 높음"
                    else: return "🔥 매우 높음"

                risk = get_risk_level(pred)

                st.markdown("### ✅ **예측 결과**", unsafe_allow_html=True)
                st.success(f"**예측 환자 수: {pred:.2f}명**", icon="✅")
                st.markdown(f"### 위험 등급: **{risk}**")
