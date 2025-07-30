import streamlit as st
import pandas as pd
import datetime
import joblib
import requests
import math
import os
import base64
from urllib.parse import unquote

# ----------------------- 📦 설정 -----------------------
st.set_page_config(layout="centered")
model = joblib.load("trained_model.pkl")
feature_names = joblib.load("feature_names.pkl")

KMA_API_KEY = unquote(st.secrets["KMA"]["API_KEY"])
ASOS_API_KEY = unquote(st.secrets["ASOS"]["API_KEY"])
GITHUB_USERNAME = st.secrets["GITHUB"]["USERNAME"]
GITHUB_REPO = st.secrets["GITHUB"]["REPO"]
GITHUB_BRANCH = st.secrets["GITHUB"]["BRANCH"]
GITHUB_TOKEN = st.secrets["GITHUB"]["TOKEN"]
GITHUB_FILENAME = "ML_asos_dataset.csv"

region_to_stn_id = {
    "서울특별시": 108, "부산광역시": 159, "대구광역시": 143, "인천광역시": 112,
    "광주광역시": 156, "대전광역시": 133, "울산광역시": 152, "세종특별자치시": 131,
    "경기도": 119, "강원도": 101, "충청북도": 131, "충청남도": 133,
    "전라북도": 146, "전라남도": 165, "경상북도": 137, "경상남도": 155, "제주특별자치도": 184
}

region_to_latlon = {
    "서울특별시": (37.5665, 126.9780), "부산광역시": (35.1796, 129.0756), "대구광역시": (35.8722, 128.6025),
    "인천광역시": (37.4563, 126.7052), "광주광역시": (35.1595, 126.8526), "대전광역시": (36.3504, 127.3845),
    "울산광역시": (35.5384, 129.3114), "세종특별자치시": (36.4800, 127.2890), "경기도": (37.4138, 127.5183),
    "강원도": (37.8228, 128.1555), "충청북도": (36.6358, 127.4917), "충청남도": (36.5184, 126.8000),
    "전라북도": (35.7167, 127.1442), "전라남도": (34.8161, 126.4630), "경상북도": (36.5760, 128.5056),
    "경상남도": (35.4606, 128.2132), "제주특별자치도": (33.4996, 126.5312)
}

# ----------------------- 🔁 공통 함수 -----------------------
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

def get_fixed_base_datetime(target_date):
    today = datetime.date.today()
    now = datetime.datetime.now()
    if target_date == today:
        hour = now.hour
        if hour >= 23: bt = "2300"
        elif hour >= 20: bt = "2000"
        elif hour >= 17: bt = "1700"
        elif hour >= 14: bt = "1400"
        elif hour >= 11: bt = "1100"
        elif hour >= 8: bt = "0800"
        elif hour >= 5: bt = "0500"
        else: bt = "0200"
        return today.strftime("%Y%m%d"), bt
    else:
        return today.strftime("%Y%m%d"), "0500"

def get_weather(region_name, target_date):
    latlon = region_to_latlon.get(region_name, (37.5665, 126.9780))
    nx, ny = convert_latlon_to_xy(*latlon)
    base_date, base_time = get_fixed_base_datetime(target_date)
    params = {
        "serviceKey": KMA_API_KEY,
        "numOfRows": "1000",
        "pageNo": "1",
        "dataType": "JSON",
        "base_date": base_date,
        "base_time": base_time,
        "nx": nx,
        "ny": ny
    }
    try:
        r = requests.get("http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst", params=params, timeout=10, verify=False)
        items = r.json().get("response", {}).get("body", {}).get("items", {}).get("item", [])
        df = pd.DataFrame(items)
        df["fcstDate"] = df["fcstDate"].astype(str)
        target_str = target_date.strftime("%Y%m%d")
        if target_str not in df["fcstDate"].values:
            return {}, base_date, base_time
        df = df[df["fcstDate"] == target_str]
        df = df[df["category"].isin(["T3H", "TMX", "TMN", "REH"])]
        summary = {}
        for cat in ["TMX", "TMN", "REH", "T3H"]:
            vals = df[df["category"] == cat]["fcstValue"].astype(float)
            if not vals.empty:
                summary[cat] = vals.mean() if cat in ["REH", "T3H"] else vals.iloc[0]
        return summary, base_date, base_time
    except:
        return {}, base_date, base_time

def get_asos_weather(region, ymd):
    stn_id = region_to_stn_id[region]
    url = f"http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList?serviceKey={ASOS_API_KEY}&pageNo=1&numOfRows=10&dataType=JSON&dataCd=ASOS&dateCd=DAY&startDt={ymd}&endDt={ymd}&stnIds={stn_id}"
    r = requests.get(url, timeout=10, verify=False)
    j = r.json()
    item = j.get("response", {}).get("body", {}).get("items", {}).get("item", [])[0]
    return {
        "TMX": float(item["maxTa"]),
        "TMN": float(item["minTa"]),
        "REH": float(item["avgRhm"])
    }

def get_risk_level(pred):
    if pred == 0: return "🟢 매우 낮음"
    elif pred <= 2: return "🟡 낮음"
    elif pred <= 5: return "🟠 보통"
    elif pred <= 10: return "🔴 높음"
    else: return "🔥 매우 높음"

def predict_from_weather(tmx, tmn, reh):
    avg_temp = round((tmx + tmn) / 2, 1)
    input_df = pd.DataFrame([{ 
        "최고체감온도(°C)": tmx + 1.5,
        "최고기온(°C)": tmx,
        "평균기온(°C)": avg_temp,
        "최저기온(°C)": tmn,
        "평균상대습도(%)": reh
    }])
    X = input_df[feature_names].copy()
    X.columns = model.get_booster().feature_names
    pred = model.predict(X)[0]
    return pred, avg_temp, input_df

# ----------------------- 🧭 UI 시작 -----------------------
st.title("🔥 온열질환 예측 및 학습데이터 기록기")
tab1, tab2 = st.tabs(["📊 예측하기", "📥 학습데이터 기록"])
# ====================================================================
# 🔮 예측 탭
# ====================================================================
with tab1:
    st.header("📊 온열질환자 예측")
    region = st.selectbox("지역 선택", list(region_to_stn_id.keys()), key="region_pred")
    today = datetime.date.today()
    date_selected = st.date_input("날짜 선택", value=today, min_value=datetime.date(2021, 7, 1), max_value=today + datetime.timedelta(days=5))

    if st.button("🔍 예측하기"):
        if date_selected >= today:
            weather, base_date, base_time = get_weather(region, date_selected)
        else:
            ymd = date_selected.strftime("%Y%m%d")
            weather = get_asos_weather(region, ymd)

        if not weather:
            st.error("❌ 기상 정보 없음")
            st.stop()

        tmx, tmn, reh = weather.get("TMX", 0), weather.get("TMN", 0), weather.get("REH", 0)
        pred, avg_temp, input_df = predict_from_weather(tmx, tmn, reh)
        risk = get_risk_level(pred)

        st.markdown("#### ☁️ 기상정보")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("최고기온", f"{tmx:.1f}℃")
        col2.metric("최저기온", f"{tmn:.1f}℃")
        col3.metric("평균기온", f"{avg_temp:.1f}℃")
        col4.metric("습도", f"{reh:.1f}%")

        with st.expander("🧪 입력값 확인"):
            st.dataframe(input_df)

        st.markdown("#### 💡 예측 결과")
        c1, c2 = st.columns(2)
        c1.metric("예측 환자 수", f"{pred:.2f}명")
        c2.metric("위험 등급", risk)
        st.caption(f"전년도 평균(6.8명) 대비 {'+' if pred - 6.8 >= 0 else ''}{pred - 6.8:.1f}명")

# ====================================================================
# 📥 학습 데이터 기록 탭
# ====================================================================
with tab2:
    st.header("📥 질병청 엑셀 업로드")
    with st.form(key="upload_form"):
        uploaded_file = st.file_uploader("엑셀 파일 (시트명은 지역명)", type=["xlsx"])
        region = st.selectbox("지역 선택 (시트명과 동일)", list(region_to_stn_id.keys()), key="region_excel")
        date_selected = st.date_input("기록할 날짜", value=today, key="record_date")
        submit_button = st.form_submit_button("📅 저장하기")

    if uploaded_file and submit_button:
        try:
            df_raw = pd.read_excel(uploaded_file, sheet_name=region, header=None)
            ymd = date_selected.strftime("%Y-%m-%d")

            if "합계" in df_raw.iloc[0].astype(str).tolist():
                # ✅ 서울시형 구조 (가로)
                df_raw.columns = df_raw.iloc[1]
                df = df_raw[2:].reset_index(drop=True)
                df.rename(columns={df.columns[0]: "일자"}, inplace=True)
                df["일자"] = pd.to_datetime(df["일자"], errors="coerce").dt.strftime("%Y-%m-%d")
                df_day = df[df["일자"] == ymd]
                if df_day.empty:
                    st.warning("📭 선택한 날짜에 해당하는 환자 수 정보가 없습니다.")
                    st.stop()

                # 일자를 제외한 나머지 열 전체 환자수 합산
                환자수 = pd.to_numeric(df_day.drop(columns=["일자"]).values.flatten(), errors="coerce").sum()

            else:
                # ✅ 일반 시군구 구조 (세로)
                df_raw.columns = df_raw.iloc[2]
                df = df_raw[3:].reset_index(drop=True)
                df.columns = df.columns.map(lambda x: str(x).strip().replace("\n", "").replace(" ", ""))
                일자_col = next(col for col in df.columns if "일자" in col)
                환자수_col = next((col for col in df.columns if "합계" in str(df[col].iloc[0])), None)
                if 환자수_col is None:
                    st.error("❌ '합계' 값이 있는 열을 찾을 수 없습니다.")
                    st.stop()
                df[일자_col] = pd.to_datetime(df[일자_col], errors='coerce').dt.strftime("%Y-%m-%d")
                df = df[[일자_col, 환자수_col]]
                df.columns = ["일자", "환자수"]
                df["환자수"] = pd.to_numeric(df["환자수"], errors="coerce")
                df = df[df["일자"] == ymd]
                if df.empty:
                    st.warning("📭 선택한 날짜에 해당하는 환자 수 정보가 없습니다.")
                    st.stop()
                환자수 = int(df["환자수"].iloc[0])

            # ✅ 기상 정보 결합
            weather = get_asos_weather(region, date_selected.strftime("%Y%m%d"))
            tmx = weather.get("TMX", 0)
            tmn = weather.get("TMN", 0)
            reh = weather.get("REH", 0)
            avg_temp = round((tmx + tmn) / 2, 1)

            input_row = {
                "일자": ymd,
                "지역": region,
                "최고체감온도(°C)": tmx + 1.5,
                "최고기온(°C)": tmx,
                "평균기온(°C)": avg_temp,
                "최저기온(°C)": tmn,
                "평균상대습도(%)": reh,
                "환자수": 환자수
            }

            # ✅ CSV 파일 저장 및 GitHub 푸시
            csv_path = GITHUB_FILENAME
            if os.path.exists(csv_path):
                existing = pd.read_csv(csv_path)
                existing = existing[~((existing["일자"] == ymd) & (existing["지역"] == region))]
                df_all = pd.concat([existing, pd.DataFrame([input_row])], ignore_index=True)
            else:
                df_all = pd.DataFrame([input_row])
            df_all.to_csv(csv_path, index=False, encoding="utf-8-sig")

            # GitHub 업로드
            with open(csv_path, "rb") as f:
                content = f.read()
            b64_content = base64.b64encode(content).decode("utf-8")
            api_url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{GITHUB_REPO}/contents/{GITHUB_FILENAME}"
            r = requests.get(api_url, headers={"Authorization": f"Bearer {GITHUB_TOKEN}"})
            sha = r.json().get("sha") if r.status_code == 200 else None
            payload = {
                "message": f"Update {GITHUB_FILENAME} with new data for {ymd} {region}",
                "content": b64_content,
                "branch": GITHUB_BRANCH
            }
            if sha:
                payload["sha"] = sha
            headers = {
                "Authorization": f"Bearer {GITHUB_TOKEN}",
                "Accept": "application/vnd.github+json"
            }
            r = requests.put(api_url, headers=headers, json=payload)
            if r.status_code in [200, 201]:
                st.success("✅ GitHub 저장 완료")
                st.info(f"🔗 [파일 바로 확인하기](https://github.com/{GITHUB_USERNAME}/{GITHUB_REPO}/blob/{GITHUB_BRANCH}/{GITHUB_FILENAME})")
            else:
                st.warning(f"⚠️ GitHub 저장 실패: {r.status_code} {r.text[:200]}")

        except Exception as e:
            st.error(f"❌ 처리 중 오류 발생: {e}")


