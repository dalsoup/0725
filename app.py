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
    RE, GRID, SLAT1, SLAT2, OLON, OLAT = 6371.00877, 5.0, 30.0, 60.0, 126.0, 38.0
    XO, YO = 43, 136
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

# 지역명 → 위경도
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

# 체감온도 계산
def calculate_feels_like(temp, wind_speed):
    return round(13.12 + 0.6215*temp - 11.37*(wind_speed**0.16) + 0.3965*temp*(wind_speed**0.16), 1)

# 기상 데이터 통합
def get_weather_combined(region_name):
    lat, lon = region_to_latlon[region_name]
    nx, ny = convert_latlon_to_xy(lat, lon)
    now = datetime.datetime.now()

    def get_ultra_base_time():
        hour, minute = now.hour, now.minute
        if minute < 45:
            hour -= 1
        return f"{hour:02}30"

    def get_vilage_base_time():
        candidates = [2, 5, 8, 11, 14, 17, 20, 23]
        base_hour = max([h for h in candidates if h <= now.hour], default=23)
        return f"{base_hour:02}00"

    base_date = now.strftime("%Y%m%d")
    ultra_base_time = get_ultra_base_time()
    vilage_base_time = get_vilage_base_time()

    try:
        # 초단기예보
        ultra_url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtFcst"
        ultra_params = {
            "serviceKey": KMA_API_KEY,
            "numOfRows": "100",
            "pageNo": "1",
            "dataType": "JSON",
            "base_date": base_date,
            "base_time": ultra_base_time,
            "nx": nx,
            "ny": ny
        }
        ultra_res = requests.get(ultra_url, params=ultra_params, timeout=10, verify=False)
        ultra_items = ultra_res.json()['response']['body']['items']['item']
        df_ultra = pd.DataFrame(ultra_items)
        df_ultra = df_ultra[df_ultra['category'].isin(['T1H', 'REH', 'WSD'])].set_index("category")

        temp = float(df_ultra.loc["T1H"]["fcstValue"]) if "T1H" in df_ultra.index else None
        wind = float(df_ultra.loc["WSD"]["fcstValue"]) if "WSD" in df_ultra.index else None
        hum = float(df_ultra.loc["REH"]["fcstValue"]) if "REH" in df_ultra.index else None
        feel = calculate_feels_like(temp, wind) if temp is not None and wind is not None else None

        # 단기예보
        vilage_url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
        vilage_params = {
            "serviceKey": KMA_API_KEY,
            "numOfRows": "1000",
            "pageNo": "1",
            "dataType": "JSON",
            "base_date": base_date,
            "base_time": vilage_base_time,
            "nx": nx,
            "ny": ny
        }
        vilage_res = requests.get(vilage_url, params=vilage_params, timeout=10, verify=False)
        vilage_items = vilage_res.json()['response']['body']['items']['item']
        df_vilage = pd.DataFrame(vilage_items)
        today_str = (now + datetime.timedelta(hours=1)).strftime("%Y%m%d")
        df_vilage = df_vilage[(df_vilage["fcstDate"] == today_str) & (df_vilage["category"].isin(["TMX", "TMN"]))]
        df_vilage = df_vilage.set_index("category")

        max_temp = float(df_vilage.loc["TMX"]["fcstValue"]) if "TMX" in df_vilage.index else None
        min_temp = float(df_vilage.loc["TMN"]["fcstValue"]) if "TMN" in df_vilage.index else None

        return {
            "avg_temp": temp,
            "humidity": hum,
            "wind": wind,
            "max_feel": feel,
            "max_temp": max_temp,
            "min_temp": min_temp
        }

    except Exception as e:
        print("기상청 API 오류:", e)
        return {}

# UI 시작
st.title("🔥 온열질환 예측 대시보드")

# 날짜 및 지역 선택
col1, col2 = st.columns(2)
with col1:
    date_selected = st.date_input("예측 날짜 선택", datetime.date.today())
with col2:
    region = st.selectbox("광역자치단체 선택", list(region_to_latlon.keys()))

# 자동 불러오기
use_auto = st.checkbox("기상청 API 자동 불러오기")
weather_data = get_weather_combined(region) if use_auto else {}
if not isinstance(weather_data, dict):
    weather_data = {}

st.caption("필요시 직접 수정 후 예측 버튼을 누르세요.")

# 기상 변수 입력
col1, col2, col3 = st.columns(3)
with col1:
    max_temp = st.number_input("최고기온(°C)", value=weather_data.get("max_temp", 32.0))
    min_temp = st.number_input("최저기온(°C)", value=weather_data.get("min_temp", 25.0))
with col2:
    avg_temp = st.number_input("평균기온(°C)", value=weather_data.get("avg_temp", 28.5))
    humidity = st.number_input("평균상대습도(%)", value=weather_data.get("humidity", 70.0))
with col3:
    max_feel = st.number_input("최고체감온도(°C)", value=weather_data.get("max_feel", 33.0))

# 예측 버튼
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

    st.markdown("## ✅ 예측 결과")
    st.success(f"예측 환자 수: {pred:.2f}명")
    st.markdown(f"### 위험 등급: **{risk}**")
