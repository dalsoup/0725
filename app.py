import streamlit as st
import pandas as pd
import joblib
import datetime
import requests
import math
from urllib.parse import unquote

model = joblib.load("trained_model.pkl")
KMA_API_KEY = unquote(st.secrets["KMA"]["API_KEY"])

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

def convert_latlon_to_xy(lat, lon):
    RE, GRID, SLAT1, SLAT2, OLON, OLAT = 6371.00877, 5.0, 30.0, 60.0, 126.0, 38.0
    XO, YO = 43, 136
    DEGRAD = math.pi / 180.0
    re = RE / GRID
    slat1, slat2 = SLAT1 * DEGRAD, SLAT2 * DEGRAD
    olon, olat = OLON * DEGRAD, OLAT * DEGRAD
    sn = math.log(math.cos(slat1)/math.cos(slat2)) / math.log(math.tan(math.pi*0.25+slat2*0.5)/math.tan(math.pi*0.25+slat1*0.5))
    sf = math.tan(math.pi*0.25 + slat1*0.5)
    sf = math.pow(sf, sn) * math.cos(slat1) / sn
    ro = math.tan(math.pi*0.25 + olat*0.5)
    ro = re * sf / math.pow(ro, sn)
    ra = math.tan(math.pi*0.25 + lat*DEGRAD*0.5)
    ra = re * sf / math.pow(ra, sn)
    theta = lon * DEGRAD - olon
    theta = theta - 2.0 * math.pi if theta > math.pi else theta
    theta = theta + 2.0 * math.pi if theta < -math.pi else theta
    theta *= sn
    x = ra * math.sin(theta) + XO + 0.5
    y = ro - ra * math.cos(theta) + YO + 0.5
    return int(x), int(y)

def calculate_feels_like(temp, wind):
    return round(13.12 + 0.6215*temp - 11.37*(wind**0.16) + 0.3965*temp*(wind**0.16), 1)

def get_today_weather_data(region_name):
    lat, lon = region_to_latlon[region_name]
    nx, ny = convert_latlon_to_xy(lat, lon)
    now = datetime.datetime.now()
    base_date = now.strftime("%Y%m%d")
    base_time = now.strftime("%H%M")

    try:
        obs_url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst"
        obs_params = {
            "serviceKey": KMA_API_KEY, "numOfRows": "100", "pageNo": "1",
            "dataType": "JSON", "base_date": base_date, "base_time": base_time,
            "nx": nx, "ny": ny
        }
        res_obs = requests.get(obs_url, params=obs_params, timeout=10, verify=False)
        df_obs = pd.DataFrame(res_obs.json()['response']['body']['items']['item']).set_index("category")

        temp = float(df_obs.loc["T1H"]["obsrValue"])
        wind = float(df_obs.loc["WSD"]["obsrValue"])
        hum = float(df_obs.loc["REH"]["obsrValue"])
        feel = calculate_feels_like(temp, wind)

        fcst_url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
        fcst_params = {
            "serviceKey": KMA_API_KEY, "numOfRows": "1000", "pageNo": "1",
            "dataType": "JSON", "base_date": base_date, "base_time": "0500",
            "nx": nx, "ny": ny
        }
        res_fcst = requests.get(fcst_url, params=fcst_params, timeout=10, verify=False)
        df_fcst = pd.DataFrame(res_fcst.json()['response']['body']['items']['item'])
        df_fcst = df_fcst[df_fcst["fcstDate"] == base_date].set_index("category")
        max_temp = float(df_fcst.loc["TMX"]["fcstValue"])
        min_temp = float(df_fcst.loc["TMN"]["fcstValue"])

        return {"avg_temp": temp, "humidity": hum, "wind": wind, "max_feel": feel,
                "max_temp": max_temp, "min_temp": min_temp}

    except Exception as e:
        print("🔴 오늘 예측 데이터 오류:", e)
        return {}

def get_future_weather_data(region_name, target_date):
    lat, lon = region_to_latlon[region_name]
    nx, ny = convert_latlon_to_xy(lat, lon)
    base_date = datetime.datetime.now().strftime("%Y%m%d")
    base_time_ultra = (datetime.datetime.now() - datetime.timedelta(hours=1)).strftime("%H") + "30"
    target_date_str = target_date.strftime("%Y%m%d")

    try:
        fcst1_url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtFcst"
        fcst1_params = {
            "serviceKey": KMA_API_KEY, "numOfRows": "100", "pageNo": "1",
            "dataType": "JSON", "base_date": base_date, "base_time": base_time_ultra,
            "nx": nx, "ny": ny
        }
        res1 = requests.get(fcst1_url, params=fcst1_params, timeout=10, verify=False)
        df1 = pd.DataFrame(res1.json()['response']['body']['items']['item']).set_index("category")

        temp = float(df1.loc["T1H"]["fcstValue"])
        wind = float(df1.loc["WSD"]["fcstValue"])
        hum = float(df1.loc["REH"]["fcstValue"])
        feel = calculate_feels_like(temp, wind)

        fcst2_url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
        fcst2_params = {
            "serviceKey": KMA_API_KEY, "numOfRows": "1000", "pageNo": "1",
            "dataType": "JSON", "base_date": base_date, "base_time": "0500",
            "nx": nx, "ny": ny
        }
        res2 = requests.get(fcst2_url, params=fcst2_params, timeout=10, verify=False)
        df2 = pd.DataFrame(res2.json()['response']['body']['items']['item'])
        df2 = df2[df2["fcstDate"] == target_date_str].set_index("category")
        max_temp = float(df2.loc["TMX"]["fcstValue"])
        min_temp = float(df2.loc["TMN"]["fcstValue"])

        return {"avg_temp": temp, "humidity": hum, "wind": wind, "max_feel": feel,
                "max_temp": max_temp, "min_temp": min_temp}

    except Exception as e:
        print("🔴 내일 예측 데이터 오류:", e)
        return {}

st.title("🔥 온열질환 예측 대시보드")
date_selected = st.date_input("날짜 선택", datetime.date.today())
region = st.selectbox("광역자치단체 선택", list(region_to_latlon.keys()))
use_auto = st.checkbox("기상 정보 자동 불러오기")

weather_data = {}
today = datetime.date.today()
if use_auto:
    if date_selected == today:
        st.caption("📡 실황 + 단기예보 데이터로 입력값 구성 중...")
        weather_data = get_today_weather_data(region)
    else:
        st.caption("📡 초단기예보 + 단기예보 데이터로 입력값 구성 중...")
        weather_data = get_future_weather_data(region, date_selected)
    st.write("✅ 불러온 weather_data:", weather_data)

if st.button("📊 온열질환 예측하기"):
    input_df = pd.DataFrame([{ 
        "광역자치단체": region,
        "최고체감온도(°C)": weather_data.get("max_feel", 33.0),
        "최고기온(°C)": weather_data.get("max_temp", 32.0),
        "평균기온(°C)": weather_data.get("avg_temp", 28.5),
        "최저기온(°C)": weather_data.get("min_temp", 25.0),
        "평균상대습도(%)": weather_data.get("humidity", 70.0)
    }])

    st.write("🔍 예측에 사용된 입력값:", input_df)

    pred = model.predict(input_df.drop(columns=["광역자치단체"]))[0]

    def get_risk_level(pred):
        if pred == 0: return "🟢 매우 낮음"
        elif pred <= 2: return "🟡 낮음"
        elif pred <= 5: return "🟠 보통"
        elif pred <= 10: return "🔴 높음"
        else: return "🔥 매우 높음"

    st.success(f"예측 환자 수: {pred:.2f}명")
    st.markdown(f"### 위험 등급: **{get_risk_level(pred)}**")