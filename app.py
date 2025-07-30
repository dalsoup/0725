import streamlit as st
import pandas as pd
import datetime
import joblib
import requests
import math
from urllib.parse import unquote
import os
import base64

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

st.set_page_config(layout="centered")
model = joblib.load("trained_model.pkl")
feature_names = joblib.load("feature_names.pkl")
KMA_API_KEY = unquote(st.secrets["KMA"]["API_KEY"])
ASOS_API_KEY = unquote(st.secrets["ASOS"]["API_KEY"])

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

st.title("🔥 온열질환 예측 및 학습데이터 기록기")
region = st.selectbox("지역 선택", list(region_to_stn_id.keys()))
today = datetime.date.today()
min_day = datetime.date(2021, 7, 1)
max_day = today + datetime.timedelta(days=5)
date_selected = st.date_input("날짜 선택", value=today, min_value=min_day, max_value=max_day)

use_asos = date_selected < today

# --- 예측 or 기록 분기 ---
if st.button("조회하기"):
    if not use_asos:
        st.info("📡 오늘 이후 → 단기예보 API 기반 예측")

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
                    st.error(f"❌ 예보 데이터에 {target_str} 날짜가 포함되어 있지 않습니다.")
                    return {}, base_date, base_time

                df = df[df["fcstDate"] == target_str]
                df = df[df["category"].isin(["T3H", "TMX", "TMN", "REH"])]

                summary = {}
                for cat in ["TMX", "TMN", "REH", "T3H"]:
                    vals = df[df["category"] == cat]["fcstValue"].astype(float)
                    if not vals.empty:
                        summary[cat] = vals.mean() if cat in ["REH", "T3H"] else vals.iloc[0]

                return summary, base_date, base_time

            except Exception as e:
                st.error(f"⚠️ API 호출 실패: {e}")
                return {}, base_date, base_time

        def calculate_avg_temp(tmx, tmn):
            if tmx is not None and tmn is not None:
                return round((tmx + tmn) / 2, 1)
            return None

        weather, base_date, base_time = get_weather(region, date_selected)
        if not weather:
            st.stop()

        st.caption(f"📡 사용된 예보 기준 시각 → base_date: `{base_date}`, base_time: `{base_time}`")

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

        st.subheader("🧪 모델 입력값 확인")
        st.dataframe(input_df)

        X_input = input_df[feature_names].copy()
        X_input.columns = model.get_booster().feature_names

        pred = model.predict(X_input)[0]
        def get_risk_level(pred):
            if pred == 0: return "🟢 매우 낮음"
            elif pred <= 2: return "🟡 낮음"
            elif pred <= 5: return "🟠 보통"
            elif pred <= 10: return "🔴 높음"
            else: return "🔥 매우 높음"
        risk = get_risk_level(pred)

        st.markdown("#### 💡 온열질환자 예측")
        c1, c2 = st.columns(2)
        c1.metric("예측 환자 수", f"{pred:.2f}명")
        c2.metric("위험 등급", risk)
        st.caption(f"전년도 평균(6.8명) 대비 {'+' if pred - 6.8 >= 0 else ''}{pred - 6.8:.1f}명")

    else:
        st.info("🕰 과거 날짜 → ASOS + 엑셀 기반 학습데이터 기록")

        # 1️⃣ ASOS 기반 예측 먼저 수행 (예측 모드와 동일)
        st.markdown("#### ☁️ 오늘의 기상정보 (ASOS 기준)")
        stn_id = region_to_stn_id[region]
        ymd = date_selected.strftime("%Y%m%d")
        url = f"http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList?serviceKey={ASOS_API_KEY}&pageNo=1&numOfRows=10&dataType=JSON&dataCd=ASOS&dateCd=DAY&startDt={ymd}&endDt={ymd}&stnIds={stn_id}"
        r = requests.get(url, timeout=10, verify=False)
        if "application/json" not in r.headers.get("Content-Type", ""):
            st.error("❌ JSON 형식이 아닌 응답입니다. 아래 내용을 확인하세요.")
            st.text(r.text[:500])
            st.stop()
        j = r.json()
        item = j.get("response", {}).get("body", {}).get("items", {}).get("item", [])[0]

        tmx = float(item["maxTa"])
        tmn = float(item["minTa"])
        reh = float(item["avgRhm"])
        avg = round((tmx + tmn) / 2, 1)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("최고기온", f"{tmx:.1f}℃")
        col2.metric("최저기온", f"{tmn:.1f}℃")
        col3.metric("평균기온", f"{avg:.1f}℃")
        col4.metric("습도", f"{reh:.1f}%")

        input_df = pd.DataFrame([{
            "최고체감온도(°C)": round(tmx + 1.5, 1),
            "최고기온(°C)": tmx,
            "평균기온(°C)": avg,
            "최저기온(°C)": tmn,
            "평균상대습도(%)": reh
        }])

        st.subheader("🧪 모델 입력값 확인")
        st.dataframe(input_df)

        X_input = input_df[feature_names].copy()
        X_input.columns = model.get_booster().feature_names

        pred = model.predict(X_input)[0]
        def get_risk_level(pred):
            if pred == 0: return "🟢 매우 낮음"
            elif pred <= 2: return "🟡 낮음"
            elif pred <= 5: return "🟠 보통"
            elif pred <= 10: return "🔴 높음"
            else: return "🔥 매우 높음"
        risk = get_risk_level(pred)

        st.markdown("#### 💡 온열질환자 예측")
        c1, c2 = st.columns(2)
        c1.metric("예측 환자 수", f"{pred:.2f}명")
        c2.metric("위험 등급", risk)

        # 2️⃣ 엑셀 업로드로 실제 환자수 추가 기록
        with st.form(key=f"upload_form_{ymd}_{region}"):
            uploaded_file = st.file_uploader("질병청 온열질환 엑셀 업로드 (시트명 = 지역명)", type=["xlsx"])
            submit_upload = st.form_submit_button("📥 업로드 및 학습 데이터 저장")
        if uploaded_file and submit_upload:
            try:
                sheet_df = pd.read_excel(uploaded_file, sheet_name=region, engine="openpyxl")
                patient_col = [col for col in sheet_df.columns if "환자수" in col or "환자 수" in col]
                date_col = [col for col in sheet_df.columns if "일자" in col or "날짜" in col or "기준일" in col]

                if not patient_col or not date_col:
                    st.error("❌ 시트에 '환자수'와 '일자' 관련 컬럼이 필요합니다.")
                    st.stop()

                df = sheet_df[[date_col[0], patient_col[0]]].copy()
                df.columns = ["일자", "환자수"]
                df["일자"] = pd.to_datetime(df["일자"]).dt.date
                filtered = df[df["일자"] == date_selected]

                if filtered.empty:
                    st.warning("⚠️ 해당 날짜에 환자수가 없습니다.")
                    st.stop()

                환자수 = int(filtered.iloc[0]["환자수"])

                input_row = {
                    "일자": ymd,
                    "지역": region,
                    "최고체감온도(°C)": round(tmx + 1.5, 1),
                    "최고기온(°C)": tmx,
                    "평균기온(°C)": avg,
                    "최저기온(°C)": tmn,
                    "평균상대습도(%)": reh,
                    "환자수": 환자수
                }

                st.success(f"✅ {ymd} {region} → 환자수 {환자수}명 기록 완료")
                st.dataframe(pd.DataFrame([input_row]))

                csv_path = "ML_asos_dataset.csv"
                if os.path.exists(csv_path):
                    existing = pd.read_csv(csv_path)
                    existing = existing[~((existing["일자"] == ymd) & (existing["지역"] == region))]
                    df = pd.concat([existing, pd.DataFrame([input_row])], ignore_index=True)
                else:
                    df = pd.DataFrame([input_row])

                df.to_csv(GITHUB_FILENAME, index=False, encoding="utf-8-sig")

                try:
                    # GitHub에 업로드
                    with open(GITHUB_FILENAME, "rb") as f:
                        content = f.read()
                    b64_content = base64.b64encode(content).decode("utf-8")
                    api_url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{GITHUB_REPO}/contents/{GITHUB_FILENAME}"

                    # 기존 파일 SHA 가져오기 (있으면 업데이트)
                    r = requests.get(api_url, headers={"Authorization": f"Bearer {GITHUB_TOKEN}"})
                    if r.status_code == 200:
                        sha = r.json()["sha"]
                    else:
                        sha = None

                    commit_msg = f"Update {GITHUB_FILENAME} with new data for {ymd} {region}"
                    payload = {
                        "message": commit_msg,
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
                    else:
                        st.warning(f"⚠️ GitHub 저장 실패: {r.status_code} {r.text[:200]}")

                except Exception as e:
                    st.error(f"❌ GitHub 업로드 중 오류: {e}")

            except Exception as e:
                st.error(f"❌ 처리 중 오류 발생: {e}")

