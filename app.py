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
    X = input_df[feature_names] 
    pred = model.predict(X)[0]
    return pred, avg_temp, input_df

def get_last_year_patient_count(current_date, region, static_file="ML_7_8월_2021_2025_dataset.xlsx"):
    try:
        # 현재 날짜의 전년도 날짜 구하기
        last_year_date = current_date - datetime.timedelta(days=365)

        # 엑셀 파일 읽기
        df_all = pd.read_excel(static_file, engine="openpyxl")

        # 날짜 컬럼 처리
        df_all["일시"] = pd.to_datetime("1899-12-30") + pd.to_timedelta(df_all["일시"], unit="D")
        df_all["일자"] = df_all["일시"].dt.strftime("%Y-%m-%d")

        # 전년도 동일 일자의 데이터 필터링
        cond = (df_all["일자"] == last_year_date.strftime("%Y-%m-%d")) & (df_all["광역자치단체"] == region)
        row = df_all[cond]

        if not row.empty:
            return int(row["환자수"].values[0])
        else:
            return None
    except Exception as e:
        return None

# ----------------------- 🧭 UI 시작 -----------------------
st.title("HeatAI")
tab1, tab2 = st.tabs(["📊 폭염트리거 예측하기", "📥 AI 학습 데이터 추가"])
# ====================================================================
# 🔮 예측 탭
# ====================================================================
with tab1:
    st.header("📊 폭염트리거 예측하기")
    
    # 📅 날짜 제한: 오늘 ~ 오늘 + 4일
    today = datetime.date.today()
    min_pred_date = today
    max_pred_date = today + datetime.timedelta(days=4)

    region = st.selectbox("지역 선택", list(region_to_stn_id.keys()), key="region_pred")
    date_selected = st.date_input(
        "날짜 선택",
        value=today,
        min_value=min_pred_date,
        max_value=max_pred_date
    )

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

        # 전년도 환자 수 비교
        last_year_count = get_last_year_patient_count(date_selected, region)
        if last_year_count is not None:
            delta_from_last_year = pred - last_year_count
            st.markdown(f"📅 **전년도({(date_selected - datetime.timedelta(days=365)).strftime('%Y-%m-%d')}) 동일 날짜 환자수**: **{last_year_count}명**")
            st.markdown(f"📈 **전년 대비 증감**: {'+' if delta_from_last_year >= 0 else ''}{delta_from_last_year:.1f}명")
        else:
            st.markdown("📭 전년도 동일 날짜의 환자 수 데이터를 찾을 수 없습니다.")

# ====================================================================
# 📥 AI 학습 데이터 추가
# ====================================================================
with tab2:
    with st.expander("ℹ️ tab2 사용법"):
        st.markdown("""
        **✅ 사용 방법**  
        1. 날짜와 자치구를 선택하세요.  
        2. 아래 링크에서 질병청의 온열질환자 엑셀 파일을 다운로드해 업로드하세요.  
           👉 [온열질환 응급실감시체계 다운로드](https://www.kdca.go.kr/board/board.es?mid=a20205030102&bid=0004&&cg_code=C01)  
        3. **저장하기** 버튼을 누르면, 해당 데이터는 tab3의 자치구별 피해점수 산정을 위한 입력값으로 자동 반영됩니다.
        """)

    st.header("📥 자치구별 실제 폭염 데이터 저장하기")

# ✅ 1. 날짜, 광역시도, 자치구 선택
today = datetime.date.today()
min_record_date = datetime.date(2021, 5, 1)
max_record_date = today - datetime.timedelta(days=1)

date_selected = st.date_input("📅 기록할 날짜", value=max_record_date, min_value=min_record_date, max_value=max_record_date)
region = st.selectbox("🌐 광역시도 선택", ["서울특별시"], key="region_excel")
gu = st.selectbox("🏘️ 자치구 선택", [
    '종로구', '중구', '용산구', '성동구', '광진구', '동대문구', '중랑구', '성북구', '강북구', '도봉구',
    '노원구', '은평구', '서대문구', '마포구', '양천구', '강서구', '구로구', '금천구', '영등포구',
    '동작구', '관악구', '서초구', '강남구', '송파구', '강동구'
])

# ✅ 2. 질병청 엑셀 파일 업로드
uploaded_file = st.file_uploader("📎 질병청 환자수 파일 업로드 (.xlsx, 시트명: 서울특별시)", type=["xlsx"])
    if uploaded_file:
    try:
        df_raw = pd.read_excel(uploaded_file, sheet_name="서울특별시", header=None)
        districts = df_raw.iloc[0, 1::2].tolist()
        dates = df_raw.iloc[3:, 0].tolist()
        df_values = df_raw.iloc[3:, 1::2]
        df_values.columns = districts
        df_values.insert(0, "일자", dates)
        df_long = df_values.melt(id_vars=["일자"], var_name="자치구", value_name="환자수")
        df_long["일자"] = pd.to_datetime(df_long["일자"], errors="coerce").dt.strftime("%Y-%m-%d")
        df_long["환자수"] = pd.to_numeric(df_long["환자수"], errors="coerce").fillna(0).astype(int)
        df_long["지역"] = "서울특별시"

        # ✅ 3. 선택된 날짜+자치구의 환자수 확인
        ymd = date_selected.strftime("%Y-%m-%d")
        selected = df_long[(df_long["일자"] == ymd) & (df_long["자치구"] == gu)]
        if selected.empty:
            st.warning(f"❌ {ymd} {gu} 환자수 데이터가 없습니다.")
            st.stop()
        환자수 = int(selected["환자수"].values[0])

        # ✅ 4. 기상청 ASOS API로 실제 기온 데이터 가져오기
        from app import get_asos_weather  # 기존 함수 재활용
        weather = get_asos_weather(region, ymd.replace("-", ""))
        tmx = weather.get("TMX", 0)
        tmn = weather.get("TMN", 0)
        reh = weather.get("REH", 0)
        avg_temp = round((tmx + tmn) / 2, 1)

        # ✅ 5. 통합 표 표시
        st.markdown("### ✅ 저장될 학습 데이터")
            preview_df = pd.DataFrame([{ 
                "일자": ymd,
                "지역": region,
                "자치구": gu,
                "최고체감온도(°C)": tmx + 1.5,
                "최고기온(°C)": tmx,
                "평균기온(°C)": avg_temp,
                "최저기온(°C)": tmn,
                "평균상대습도(%)": reh,
                "환자수": 환자수
            }])

        # ✅ 6. GitHub 저장 버튼
        if st.button("💾 GitHub에 저장하기"):
            csv_path = "ML_asos_dataset.csv"
            if os.path.exists(csv_path):
                try:
                    existing = pd.read_csv(csv_path, encoding="utf-8-sig")
                except UnicodeDecodeError:
                    existing = pd.read_csv(csv_path, encoding="cp949")
                existing = existing[~((existing["일자"] == ymd) & (existing["자치구"] == gu))]
                df_all = pd.concat([existing, preview_df], ignore_index=True)
            else:
                df_all = preview_df

            df_all.to_csv(csv_path, index=False, encoding="utf-8-sig")

            # ✅ GitHub API 업로드
            from urllib.parse import unquote
            GITHUB_USERNAME = st.secrets["GITHUB"]["USERNAME"]
            GITHUB_REPO = st.secrets["GITHUB"]["REPO"]
            GITHUB_BRANCH = st.secrets["GITHUB"]["BRANCH"]
            GITHUB_TOKEN = st.secrets["GITHUB"]["TOKEN"]
            GITHUB_FILENAME = "ML_asos_dataset.csv"

            with open(csv_path, "rb") as f:
                content = f.read()
            b64_content = base64.b64encode(content).decode("utf-8")

            api_url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{GITHUB_REPO}/contents/{GITHUB_FILENAME}"
            r = requests.get(api_url, headers={"Authorization": f"Bearer {GITHUB_TOKEN}"})
            sha = r.json().get("sha") if r.status_code == 200 else None

            payload = {
                "message": f"Update {GITHUB_FILENAME} with new data for {ymd} {region} {gu}",
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

