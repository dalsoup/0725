import streamlit as st
import pandas as pd
import datetime
import joblib
import requests
import math
from urllib.parse import unquote
import certifi  # ✅ SSL 인증서 번들

st.set_page_config(layout="centered")
model = joblib.load("trained_model.pkl")
feature_names = joblib.load("feature_names.pkl")
KMA_API_KEY = unquote(st.secrets["KMA"]["API_KEY"])

def get_best_available_base_datetime(target_date):
    now = datetime.datetime.now()
    today = now.date()
    times = ["2300", "2000", "1700", "1400", "1100", "0800", "0500", "0200"]
    current_time = int(f"{now.hour:02d}00")
    for t in times:
        if int(t) <= current_time:
            base_time = t
            break
    else:
        base_time = "2300"
        target_date -= datetime.timedelta(days=1)
    base_date = today.strftime("%Y%m%d") if target_date > today else target_date.strftime("%Y%m%d")
    return base_date, base_time

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
    sn = math.log(math.cos(slat1)/math.cos(slat2)) / math.log(math.tan(math.pi/4+slat2/2)/math.tan(math.pi/4+slat1/2))
    sf = math.tan(math.pi/4+slat1/2)**sn * math.cos(slat1)/sn
    ro = re * sf / (math.tan(math.pi/4+olat/2)**sn)
    ra = re * sf / (math.tan(math.pi/4+lat*DEGRAD/2)**sn)
    theta = lon * DEGRAD - olon
    if theta > math.pi: theta -= 2*math.pi
    if theta < -math.pi: theta += 2*math.pi
    theta *= sn
    x = ra * math.sin(theta) + XO + 0.5
    y = ro - ra * math.cos(theta) + YO + 0.5
    return int(x), int(y)

def get_weather(region_name, target_date):
    latlon = region_to_latlon.get(region_name, (37.5665, 126.9780))
    nx, ny = convert_latlon_to_xy(*latlon)
    base_date, base_time = get_best_available_base_datetime(target_date)

    params = {
        "serviceKey": KMA_API_KEY,
        "numOfRows": "300", "pageNo": "1", "dataType": "JSON",
        "base_date": base_date, "base_time": base_time,
        "nx": nx, "ny": ny
    }

    try:
        r = requests.get(
            "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst",
            params=params,
            timeout=20,
            verify=certifi.where()  # ✅ 안전한 SSL 연결
        )
        items = r.json().get("response", {}).get("body", {}).get("items", {}).get("item", [])

        if not items:
            st.error("❌ 예보 항목이 비어 있습니다. 아직 발표되지 않았거나 잘못된 요청일 수 있습니다.")
            return {}

        df = pd.DataFrame(items)

        if "fcstDate" not in df.columns:
            st.error("❌ 응답 데이터에 'fcstDate' 컬럼이 없습니다.")
            return {}

        df["fcstDate"] = df["fcstDate"].astype(str)
        target_str = target_date.strftime("%Y%m%d")
        df = df[df["fcstDate"] == target_str]
        df = df[df["category"].isin(["T3H", "TMX", "TMN", "REH"])]

        summary = {}
        for cat in ["TMX", "TMN", "REH", "T3H"]:
            vals = df[df["category"] == cat]["fcstValue"].astype(float)
            if not vals.empty:
                summary[cat] = vals.mean() if cat in ["REH", "T3H"] else vals.iloc[0]

        return summary

    except Exception as e:
        st.error(f"⚠️ API 호출 실패: {e}")
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

# ==================== UI ====================
st.title("🔥 온열질환 예측 대시보드")
region = st.selectbox("지역 선택", list(region_to_latlon.keys()))
today = datetime.date.today()
date_selected = st.date_input("예측 날짜", value=today, min_value=today, max_value=today + datetime.timedelta(days=5))

if st.button("예측하기"):
    weather = get_weather(region, date_selected)
    if not weather:
        st.error("기상 데이터를 불러올 수 없습니다.")
        st.stop()

    tmx, tmn = weather.get("TMX"), weather.get("TMN")
    avg_temp = calculate_avg_temp(tmx, tmn)

    st.markdown("#### ☁️ 오늘의 기상정보")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("최고기온", f"{tmx:.1f}℃" if tmx else "-")
    col2.metric("최저기온", f"{tmn:.1f}℃" if tmn else "-")
    col3.metric("평균기온", f"{avg_temp:.1f}℃" if avg_temp is not None else "-")
    col4.metric("습도", f"{weather.get('REH', 0):.1f}%" if weather.get("REH") is not None else "-")

    input_df = pd.DataFrame([{
        "최고체감온도(°C)": tmx + 1.5 if tmx else 0,
        "최고기온(°C)": tmx or 0,
        "평균기온(°C)": avg_temp or 0,
        "최저기온(°C)": tmn or 0,
        "평균상대습도(%)": weather.get("REH", 0)
    }])

    missing = [col for col in feature_names if col not in input_df.columns]
    if missing:
        st.error(f"입력 누락 피처: {missing}")
        st.stop()

    X_input = input_df[feature_names].copy()
    try:
        X_input.columns = model.get_booster().feature_names
    except:
        st.error("모델의 feature 이름 설정 실패")
        st.stop()

    pred = model.predict(X_input)[0]
    risk = get_risk_level(pred)

    st.markdown("#### 💡 온열질환자 예측")
    c1, c2 = st.columns(2)
    c1.metric("예측 환자 수", f"{pred:.2f}명")
    c2.metric("위험 등급", risk)
    st.caption(f"전년도 평균(6.8명) 대비 {'+' if pred - 6.8 >= 0 else ''}{pred - 6.8:.1f}명")
