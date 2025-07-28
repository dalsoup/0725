import streamlit as st
import pandas as pd
import joblib
import requests
import datetime
import math
import folium
from streamlit_folium import st_folium
from urllib.parse import unquote

# -----------------------
# Toss 스타일 CSS
# -----------------------
st.markdown("""
<style>
body {
    font-family: 'Pretendard', sans-serif;
}
.big-title {
    font-size: 32px;
    font-weight: bold;
    margin-bottom: 5px;
}
.subtitle {
    font-size: 18px;
    color: #666;
    margin-bottom: 30px;
}
.risk-box {
    font-size: 18px;
    padding: 12px 20px;
    border-radius: 12px;
    margin: 10px 0;
    font-weight: 600;
}
.risk-🟢 { background-color: #e0f2f1; color: #00695c; }
.risk-🟡 { background-color: #fff9c4; color: #f57f17; }
.risk-🟠 { background-color: #ffe0b2; color: #ef6c00; }
.risk-🔴 { background-color: #ef9a9a; color: #c62828; }
.risk-🔥 { background-color: #ff8a80; color: #b71c1c; }
</style>
""", unsafe_allow_html=True)

# -----------------------
# UI 시작
# -----------------------
st.markdown('<div class="big-title">🔥 온열질환 실시간 예측</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">기상청 API + AI 기반으로 전국 위험도를 지도에 시각화합니다.</div>', unsafe_allow_html=True)

# -----------------------
# 기상청 설정 및 변환
# -----------------------
KMA_API_KEY = unquote("tY8VLwuj4xgpv5%2FRvL9hClEoJJMLlK5qtBj%2FpaNY%2FAfX2nFuXUg3utCLLK%2FBdE6Dh3td1JQrqGUv3Ml4Xw7GwA%3D%3D")

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
    RE, GRID = 6371.00877, 5.0
    SLAT1, SLAT2, OLON, OLAT = 30.0, 60.0, 126.0, 38.0
    XO, YO = 43, 136
    DEGRAD = math.pi / 180.0
    re = RE / GRID
    slat1, slat2 = SLAT1 * DEGRAD, SLAT2 * DEGRAD
    olon, olat = OLON * DEGRAD, OLAT * DEGRAD
    sn = math.log(math.cos(slat1)/math.cos(slat2)) / math.log(math.tan(math.pi/4+slat2/2)/math.tan(math.pi/4+slat1/2))
    sf = math.pow(math.tan(math.pi/4 + slat1/2), sn) * math.cos(slat1) / sn
    ro = re * sf / math.pow(math.tan(math.pi/4 + olat/2), sn)
    ra = re * sf / math.pow(math.tan(math.pi/4 + lat*DEGRAD/2), sn)
    theta = lon*DEGRAD - olon
    theta = (theta + math.pi) % (2 * math.pi) - math.pi
    x = ra * math.sin(sn*theta) + XO + 0.5
    y = ro - ra * math.cos(sn*theta) + YO + 0.5
    return int(x), int(y)

# 위험도 판정
def get_risk_level(count):
    if count == 0:
        return "🟢"
    elif count <= 2:
        return "🟡"
    elif count <= 5:
        return "🟠"
    elif count <= 10:
        return "🔴"
    else:
        return "🔥"

# 실시간 기상정보 가져오기
@st.cache_data(show_spinner=False)
def get_weather(region):
    lat, lon = region_to_latlon[region]
    nx, ny = convert_latlon_to_xy(lat, lon)
    today = datetime.datetime.now().strftime("%Y%m%d")
    base_time = "0500"
    url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
    params = {
        "serviceKey": KMA_API_KEY, "numOfRows": "1000", "pageNo": "1",
        "dataType": "JSON", "base_date": today, "base_time": base_time,
        "nx": nx, "ny": ny
    }
    res = requests.get(url, params=params)
    items = res.json()['response']['body']['items']['item']
    df = pd.DataFrame(items)
    df = df[df["fcstDate"] == today].set_index("category")

    def feels_like(t, w): return round(13.12 + 0.6215*t - 11.37*(w**0.16) + 0.3965*t*(w**0.16), 1)

    avg_temp = float(df.loc["TMP"]["fcstValue"])
    humidity = float(df.loc["REH"]["fcstValue"])
    wind = float(df.loc["WSD"]["fcstValue"])
    max_temp = float(df.loc["TMX"]["fcstValue"])
    min_temp = float(df.loc["TMN"]["fcstValue"])
    max_feel = feels_like(avg_temp, wind)

    return {
        "avg_temp": avg_temp, "humidity": humidity, "wind": wind,
        "max_temp": max_temp, "min_temp": min_temp, "max_feel": max_feel
    }

# 전국 수집 + 예측
def get_all_predictions():
    model = joblib.load("trained_model.pkl")
    results = []
    for region in region_to_latlon:
        try:
            data = get_weather(region)
            X = pd.DataFrame([[
                data["max_feel"], data["max_temp"],
                data["avg_temp"], data["min_temp"],
                data["humidity"]
            ]], columns=["max_feel", "max_temp", "avg_temp", "min_temp", "humidity"])
            y = model.predict(X)[0]
            results.append({"지역": region, "예측환자수": round(y), "위험도": get_risk_level(y)})
        except:
            continue
    return pd.DataFrame(results)

# 지도 출력
def draw_map(df):
    m = folium.Map(location=[36.5, 127.8], zoom_start=7)
    for _, row in df.iterrows():
        region = row["지역"]
        lat, lon = region_to_latlon[region]
        color = {
            "🟢": "#B2DFDB", "🟡": "#FFF59D", "🟠": "#FFB74D",
            "🔴": "#EF5350", "🔥": "#C62828"
        }[row["위험도"]]
        folium.CircleMarker(
            location=[lat, lon], radius=20, color=color,
            fill=True, fill_opacity=0.8,
            popup=f"{region} : {row['위험도']} ({row['예측환자수']}명)"
        ).add_to(m)
    return m

# -----------------------
# 예측 실행 및 지도 표시
# -----------------------
with st.spinner("실시간 기상 데이터를 불러오고 있습니다..."):
    df_pred = get_all_predictions()
    st.markdown("### 🌡️ 오늘 전국 위험도 예측")
    for _, row in df_pred.iterrows():
        st.markdown(f"<div class='risk-box risk-{row['위험도']}'>{row['지역']} — {row['위험도']} ({row['예측환자수']}명)</div>", unsafe_allow_html=True)

    st.markdown("### 🗺️ 지도 보기")
    st_folium(draw_map(df_pred), width=700, height=500)
