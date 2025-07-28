import streamlit as st
import pandas as pd
import joblib
import datetime
from urllib.parse import unquote
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

def get_base_time(now):
    for t in reversed([2, 5, 8, 11, 14, 17, 20, 23]):
        if now.hour >= t:
            return f"{t:02d}00", now.strftime("%Y%m%d")
    return "2300", (now - datetime.timedelta(days=1)).strftime("%Y%m%d")

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
        data = requests.get(url, params=params, timeout=10, verify=False).json()
        items = data["response"]["body"]["items"]["item"]
    except Exception: return None
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

def get_risk_level(pred):
    if pred == 0: return "<span class='risk-box verylow'>🟢 매우 낮음</span>", "✔️ 평소처럼 생활하되 충분한 수분 섭취를 유지하세요."
    elif pred <= 2: return "<span class='risk-box low'>🟡 낮음</span>", "🔅 직사광선 피하고 정오 전후 실내 생활을 권장합니다."
    elif pred <= 5: return "<span class='risk-box medium'>🟠 보통</span>", "🚨 수시로 물을 마시고 노약자 외출은 자제하세요."
    elif pred <= 10: return "<span class='risk-box high'>🔴 높음</span>", "❗ 실외 활동 최소화 및 냉방기 점검 필요."
    else: return "<span class='risk-box veryhigh'>🔥 매우 높음</span>", "⚠️ 외출 금지, 커뮤니티와 연락 유지 및 냉방시설 우선 이용!"

st.markdown("## 🧠 지역별 온열질환 예측")

selected = st.selectbox("지역 선택", list(region_to_latlon.keys()))
date_selected = st.date_input("예측 날짜 선택", value=datetime.date.today(), min_value=datetime.date.today(), max_value=datetime.date.today() + datetime.timedelta(days=5))

if st.button("📊 예측 실행"):
    weather = get_weather_from_api(selected)
    if weather:
        input_df = pd.DataFrame([{
            "최고체감온도(°C)": weather["max_feel"],
            "최고기온(°C)": weather["max_temp"],
            "평균기온(°C)": weather["avg_temp"],
            "최저기온(°C)": weather["min_temp"],
            "평균상대습도(%)": weather["humidity"]
        }])
        pred = model.predict(input_df)[0]
        risk, guide = get_risk_level(pred)

        st.markdown(f"""
        <div class='card'>
            <h4 style='margin-top:0;'>📊 {selected} ({date_selected.strftime('%Y-%m-%d')})</h4>
            <div style='font-size:2.5rem;font-weight:600;'>{pred:.1f}명</div>
            <div style='margin-top:0.5rem;'>{risk}</div>
            <p style='margin-top:1rem;font-size:0.95rem;'>{guide}</p>
        </div>
        """, unsafe_allow_html=True)

        # 예측 히스토리 차트 (예시 데이터)
        st.markdown("### 📈 최근 예측 히스토리 (예시)")
        dates = [(date_selected - datetime.timedelta(days=i)).strftime("%m/%d") for i in range(4, -1, -1)]
        preds = [3.1, 4.5, 6.0, 7.2, pred]  # 실제 구현 시 session_state 또는 DB로 관리

        fig, ax = plt.subplots()
        ax.plot(dates, preds, marker='o', color='#0061f2', linewidth=2)
        ax.set_xlabel("날짜")
        ax.set_ylabel("예측 환자 수")
        ax.set_title("최근 5일 예측 추이")
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.error("기상청 API에서 데이터를 불러올 수 없습니다.")