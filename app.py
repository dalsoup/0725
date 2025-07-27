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

# 시도명 → 위도/경도 매핑
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

# base_time 계산 함수 (최신 예보 수집 가능 시점 고려)
def get_valid_base_time(now):
    valid_times = [23, 20, 17, 14, 11, 8, 5, 2]
    for t in valid_times:
        base_dt = now.replace(hour=t, minute=0, second=0, microsecond=0)
        if now >= base_dt + datetime.timedelta(minutes=40):
            return f"{t:02d}00", now.strftime("%Y%m%d")
    return "2300", (now - datetime.timedelta(days=1)).strftime("%Y%m%d")

# Streamlit UI 시작
st.title("🔥 온열질환 예측 대시보드")

# 날짜 및 지역 선택 (한 줄)
col1, col2 = st.columns(2)
with col1:
    date_selected = st.date_input("예측 날짜", datetime.date.today())
with col2:
    region = st.selectbox("광역자치단체", list(region_to_latlon.keys()))

# 기상청 API 호출 여부
use_api = st.checkbox("🌤️ 기상청 단기예보 API 자동 사용")
weather_data = {}

# 기상 데이터 불러오기 함수

def get_weather_from_api(region_name):
    lat, lon = region_to_latlon.get(region_name, (37.5665, 126.9780))
    nx, ny = convert_latlon_to_xy(lat, lon)
    now = datetime.datetime.now()

    tried = 0
    while tried < 3:
        base_time, base_date = get_valid_base_time(now - datetime.timedelta(hours=tried*3))
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
        response = requests.get(url, params=params, timeout=10, verify=False)
        items = response.json().get("response", {}).get("body", {}).get("items", {}).get("item", [])
        df = pd.DataFrame(items)
        if "T3H" in df.get("category", []):
            break
        tried += 1
    else:
        st.error("T3H 항목 누락 - 기상청 API 응답 불완전")
        return None

    df["fcstHour"] = df["fcstTime"].astype(int) // 100
    now_hour = now.hour
    df["hour_diff"] = abs(df["fcstHour"] - now_hour)
    latest = df[df["category"].isin(["TMX", "TMN", "REH", "WSD", "T3H"])]
    closest = latest.loc[latest.groupby("category")["hour_diff"].idxmin()].set_index("category")

    temp = float(closest.loc["T3H"]["fcstValue"])
    wind = float(closest.loc["WSD"]["fcstValue"])
    max_temp = float(closest.loc["TMX"]["fcstValue"])
    min_temp = float(closest.loc["TMN"]["fcstValue"])
    hum = float(closest.loc["REH"]["fcstValue"])
    feel = calculate_feels_like(temp, wind)

    fcst_time_row = closest.reset_index().iloc[0]
    fcst_date = fcst_time_row.get("fcstDate", base_date)
    fcst_time = fcst_time_row.get("fcstTime", base_time)
    formatted_time = f"{fcst_date[:4]}-{fcst_date[4:6]}-{fcst_date[6:]} {fcst_time[:2]}:00"
    st.caption(f"예보 시각 기준: {formatted_time} (가장 근접한 시각의 데이터)")

    st.table(pd.DataFrame({
        "항목": ["예보기온(T3H)", "풍속(WSD)", "습도(REH)", "최고기온(TMX)", "최저기온(TMN)", "체감온도"],
        "값": [temp, wind, hum, max_temp, min_temp, feel]
    }))

    return {
        "max_temp": max_temp,
        "humidity": hum,
        "min_temp": min_temp,
        "avg_temp": temp,
        "max_feel": feel
    }

if use_api:
    weather_data = get_weather_from_api(region) or {}

# 입력 UI 구성
col1, col2 = st.columns(2)
with col1:
    max_feel = weather_data.get("max_feel") or st.number_input("최고체감온도(°C)", 0.0, 60.0, 33.0)
    max_temp = weather_data.get("max_temp") or st.number_input("최고기온(°C)", 0.0, 60.0, 32.0)
    avg_temp = weather_data.get("avg_temp") or st.number_input("평균기온(°C)", 0.0, 50.0, 28.5)
with col2:
    min_temp = weather_data.get("min_temp") or st.number_input("최저기온(°C)", 0.0, 40.0, 25.0)
    humidity = weather_data.get("humidity") or st.number_input("평균상대습도(%)", 0.0, 100.0, 70.0)

# 예측 실행 버튼
if st.button("📊 예측하기"):
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
    st.markdown("## ✅ 예측 결과")
    st.write(f"예측 환자 수: **{pred}명**")
    st.write(f"위험 등급: **{risk}**")
