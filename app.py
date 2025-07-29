import streamlit as st
import pandas as pd
import joblib
import datetime
import requests
from urllib.parse import unquote
import math

# ---------------- 앱 설정 ----------------
st.set_page_config(page_title="Heatwave Dashboard", layout="wide")
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Pretendard&display=swap');
        html, body, [class*="css"]  {
            font-family: 'Pretendard', sans-serif;
        }
    </style>
    <h1 style='font-size: 2.5rem;'>🔥 2025년 Heatwave Risk Dashboard</h1>
""", unsafe_allow_html=True)

# ---------------- MODEL & API 설정 ----------------
model = joblib.load("trained_model.pkl")
KMA_API_KEY = unquote(st.secrets["KMA"]["API_KEY"])

region_to_latlon = {
    "서울특별시": (37.5665, 126.9780), "부산광역시": (35.1796, 129.0756), "대구광역시": (35.8722, 128.6025),
    "인천광역시": (37.4563, 126.7052), "광주광역시": (35.1595, 126.8526), "대전광역시": (36.3504, 127.3845),
    "울산광역시": (35.5384, 129.3114), "세종특별자치시": (36.4800, 127.2890)
}

def get_risk_level(val):
    if val == 0: return "🟢 매우 낮음"
    elif val <= 2: return "🟡 낮음"
    elif val <= 5: return "🔶 보통"
    elif val <= 10: return "🔴 높음"
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
    theta *= sn
    x = ra * math.sin(theta) + XO + 0.5
    y = ro - ra * math.cos(theta) + YO + 0.5
    return int(x), int(y)

def get_base_time(target_date):
    today = datetime.date.today()
    now = datetime.datetime.now()

    if target_date > today:
        return (target_date - datetime.timedelta(days=1)).strftime("%Y%m%d"), "2300"

    hour = now.hour
    if hour < 2: base = "2300"; day = today - datetime.timedelta(days=1)
    elif hour < 5: base = "0200"; day = today
    elif hour < 8: base = "0500"; day = today
    elif hour < 11: base = "0800"; day = today
    elif hour < 14: base = "1100"; day = today
    elif hour < 17: base = "1400"; day = today
    elif hour < 20: base = "1700"; day = today
    elif hour < 23: base = "2000"; day = today
    else: base = "2300"; day = today
    return day.strftime("%Y%m%d"), base

def get_weather(region_name, target_date):
    latlon = region_to_latlon.get(region_name, (37.5665, 126.9780))
    nx, ny = convert_latlon_to_xy(*latlon)
    base_date, base_time = get_base_time(target_date)

    params = {
        "serviceKey": KMA_API_KEY,
        "numOfRows": "300", "pageNo": "1", "dataType": "JSON",
        "base_date": base_date, "base_time": base_time,
        "nx": nx, "ny": ny
    }

    try:
        r = requests.get("http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst", params=params, timeout=10)
        items = r.json().get("response", {}).get("body", {}).get("items", {}).get("item", [])
        df = pd.DataFrame(items)
        df = df[df["category"].isin(["TMX", "TMN", "REH"])]
        df = df[df["fcstDate"] == target_date.strftime("%Y%m%d")]
        return {
            "TMX": df[df["category"] == "TMX"]["fcstValue"].astype(float).iloc[0],
            "TMN": df[df["category"] == "TMN"]["fcstValue"].astype(float).iloc[0],
            "REH": df[df["category"] == "REH"]["fcstValue"].astype(float).mean()
        }
    except:
        return {}

def calculate_avg_temp(tmx, tmn):
    return round((tmx + tmn) / 2, 1)

@st.cache_data
def load_report():
    df = pd.read_excel("최종리포트데이터.xlsx", parse_dates=["date"])
    try:
        df["예측 환자수"] = pd.to_numeric(df["예측 환자수"], errors="coerce")
    except KeyError:
        st.error("❌ '예측 환자수' 컬럼이 없습니다.")
    return df

# ---------------- UI ----------------
st.markdown("### 날짜 및 지역 선택")
col1, col2 = st.columns(2)
with col1:
    region = st.selectbox("📍 지역 선택", list(region_to_latlon.keys()))
with col2:
    date_selected = st.date_input("📅 날짜 선택", value=datetime.date.today())

report_df = load_report()
today = datetime.date.today()

if date_selected < today:
    row = report_df[report_df["date"] == pd.Timestamp(date_selected)].iloc[0]
    risk = row["예측 위험도"]
    st.markdown(f"#### ✅ <b>{date_selected}</b> 리포트 (출처: 저장된 리포트)", unsafe_allow_html=True)
    st.markdown(f"""
    - 최고기온: {row['최고기온(°C)']}℃  
    - 평균기온: {row['평균기온(°C)']}℃  
    - 습도: {row['습도(%)']}%  
    - AI 예측 환자수: {row['예측 환자수']}명  
    - 위험도: {risk}  
    - 실제 환자수(2025): {row['2025 실제 환자수']}명  
    - 비교값(2024): {row['2024 실제 환자수']}명  
    """)
else:
    weather = get_weather(region, date_selected)
    if not weather:
        st.error("⚠️ 기상 정보를 불러오지 못했습니다.")
    else:
        avg_temp = calculate_avg_temp(weather["TMX"], weather["TMN"])
        X = pd.DataFrame([{
            "최고체감온도(°C)": weather["TMX"] + 1.5,
            "최고기온(°C)": weather["TMX"],
            "평균기온(°C)": avg_temp,
            "최저기온(°C)": weather["TMN"],
            "평균상대습도(%)": weather["REH"]
        }])
        X = X[model.feature_names_in_]
        pred = model.predict(X)[0]
        risk = get_risk_level(pred)
        st.markdown(f"#### ⚡ <b>{date_selected}</b> 예측 결과 (출처: 실시간 예측)</b>", unsafe_allow_html=True)
        st.markdown(f"""
        - 최고기온: {weather["TMX"]}℃  
        - 평균기온: {avg_temp}℃  
        - 습도: {weather["REH"]:.1f}%  
        - AI 예측 환자수: {pred:.2f}명  
        - 위험도: {risk}  
        """)
