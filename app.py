import streamlit as st
import pandas as pd
import joblib
import datetime
import requests
import math
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

st.markdown("""
    <style>
        html, body, [class*="css"]  {
            font-family: 'Pretendard', sans-serif;
            background-color: #f8f9fb;
        }
        .main {
            padding: 2rem;
        }
        .card {
            padding: 1.5rem;
            background: white;
            border-radius: 1rem;
            box-shadow: 0 1px 6px rgba(0,0,0,0.05);
            margin-bottom: 1.5rem;
        }
        .risk-box {
            font-size: 1.1rem;
            font-weight: bold;
            padding: 0.5rem 1rem;
            border-radius: 1rem;
            display: inline-block;
        }
        .verylow { background-color: #d4edda; color: #155724; }
        .low { background-color: #fff3cd; color: #856404; }
        .medium { background-color: #ffeeba; color: #856404; }
        .high { background-color: #f8d7da; color: #721c24; }
        .veryhigh { background-color: #f5c6cb; color: #721c24; }
    </style>
""", unsafe_allow_html=True)

model = joblib.load("trained_model.pkl")
KMA_API_KEY = st.secrets["KMA"]["API_KEY"]

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

def get_base_time(now):
    for t in reversed([2, 5, 8, 11, 14, 17, 20, 23]):
        if now.hour >= t:
            return f"{t:02d}00", now.strftime("%Y%m%d")
    return "2300", (now - datetime.timedelta(days=1)).strftime("%Y%m%d")

def convert_latlon_to_xy(lat, lon):
    RE = 6371.00877; GRID = 5.0; SLAT1 = 30.0; SLAT2 = 60.0; OLON = 126.0; OLAT = 38.0; XO = 43; YO = 136
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
    if theta > math.pi: theta -= 2.0 * math.pi
    if theta < -math.pi: theta += 2.0 * math.pi
    theta *= sn
    x = ra * math.sin(theta) + XO + 0.5
    y = ro - ra * math.cos(theta) + YO + 0.5
    return int(x), int(y)

def get_weather_from_api(region_name):
    lat, lon = region_to_latlon[region_name]
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
        st.write("✅ 상태코드:", response.status_code)
        st.json(response.json())  # 디버깅용 응답 확인

        data = response.json()
        items = data["response"]["body"]["items"]["item"]
    except Exception as e:
        st.error(f"❌ API 요청 실패: {e}")
        return None

    df = pd.DataFrame(items)
    df["fcstHour"] = df["fcstTime"].astype(int) // 100
    now_hour = now.hour
    df["hour_diff"] = abs(df["fcstHour"] - now_hour)
    df = df[df["category"].isin(["TMX", "TMN", "REH", "WSD", "T3H"])]
    df = df.loc[df.groupby("category")["hour_diff"].idxmin()].set_index("category")
    temp = float(df.loc["T3H"]["fcstValue"]) if "T3H" in df.index else None
    wind = float(df.loc["WSD"]["fcstValue"]) if "WSD" in df.index else 1.5
    max_temp = float(df.loc["TMX"]["fcstValue"]) if "TMX" in df.index else None
    min_temp = float(df.loc["TMN"]["fcstValue"]) if "TMN" in df.index else None
    hum = float(df.loc["REH"]["fcstValue"]) if "REH" in df.index else None
    if temp is None and max_temp and min_temp:
        temp = round((max_temp + min_temp) / 2, 1)
    feel = 13.12 + 0.6215 * temp - 11.37 * (wind ** 0.16) + 0.3965 * temp * (wind ** 0.16)
    return {"max_temp": max_temp, "min_temp": min_temp, "humidity": hum, "wind": wind, "avg_temp": temp, "max_feel": round(feel, 1)}
