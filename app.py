import streamlit as st
import pandas as pd
import datetime
import requests
import os
import base64
from urllib.parse import unquote

from utils import (
    get_weather, get_asos_weather, get_risk_level,
    calculate_avg_temp, region_to_stn_id
)
from model_utils import predict_from_weather

# ----------------------- 📦 설정 -----------------------
st.set_page_config(layout="centered")

KMA_API_KEY = unquote(st.secrets["KMA"]["API_KEY"])
ASOS_API_KEY = unquote(st.secrets["ASOS"]["API_KEY"])
GITHUB_USERNAME = st.secrets["GITHUB"]["USERNAME"]
GITHUB_REPO = st.secrets["GITHUB"]["REPO"]
GITHUB_BRANCH = st.secrets["GITHUB"]["BRANCH"]
GITHUB_TOKEN = st.secrets["GITHUB"]["TOKEN"]
GITHUB_FILENAME = "ML_asos_dataset.csv"

# ----------------------- 🧭 UI 시작 -----------------------
st.title("HeatAI")
tab1, tab2 = st.tabs(["📊 폭염트리거 예측하기", "📥 AI 학습 데이터 추가"])

# ====================================================================
# 🔮 예측 탭
# ====================================================================
with tab1:
    st.header("📊 폭염트리거 예측하기")

    today = datetime.date.today()
    min_pred_date = today
    max_pred_date = today + datetime.timedelta(days=4)

    region = st.selectbox("지역 선택", list(region_to_stn_id.keys()), key="region_tab1")
    date_selected = st.date_input("날짜 선택", value=today, min_value=min_pred_date, max_value=max_pred_date, key="date_tab1")

    if st.button("🔍 예측하기", key="predict_tab1"):
        if date_selected >= today:
            weather, base_date, base_time = get_weather(region, date_selected, KMA_API_KEY)
        else:
            ymd = date_selected.strftime("%Y%m%d")
            weather = get_asos_weather(region, ymd, ASOS_API_KEY)

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

        def get_last_year_patient_count(current_date, region, static_file="ML_7_8월_2021_2025_dataset.xlsx"):
            try:
                last_year_date = current_date - datetime.timedelta(days=365)
                df_all = pd.read_excel(static_file, engine="openpyxl")
                df_all["일시"] = pd.to_datetime("1899-12-30") + pd.to_timedelta(df_all["일시"], unit="D")
                df_all["일자"] = df_all["일시"].dt.strftime("%Y-%m-%d")
                cond = (df_all["일자"] == last_year_date.strftime("%Y-%m-%d")) & (df_all["광역자치단체"] == region)
                row = df_all[cond]
                if not row.empty:
                    return int(row["환자수"].values[0])
                else:
                    return None
            except:
                return None

        last_year_count = get_last_year_patient_count(date_selected, region)
        if last_year_count is not None:
            delta = pred - last_year_count
            st.markdown(f"📅 **전년도({(date_selected - datetime.timedelta(days=365)).strftime('%Y-%m-%d')}) 동일 날짜 환자수**: **{last_year_count}명**")
            st.markdown(f"📈 **전년 대비 증감**: {'+' if delta >= 0 else ''}{delta:.1f}명")
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

    today = datetime.date.today()
    min_record_date = datetime.date(2021, 5, 1)
    max_record_date = today - datetime.timedelta(days=1)

    date_selected = st.date_input("📅 기록할 날짜", value=max_record_date, min_value=min_record_date, max_value=max_record_date, key="date_tab2")
    region = st.selectbox("🌐 광역시도 선택", ["서울특별시"], key="region_tab2")
    gu = st.selectbox("🏘️ 자치구 선택", [
        '종로구', '중구', '용산구', '성동구', '광진구', '동대문구', '중랑구', '성북구', '강북구', '도봉구',
        '노원구', '은평구', '서대문구', '마포구', '양천구', '강서구', '구로구', '금천구', '영등포구',
        '동작구', '관악구', '서초구', '강남구', '송파구', '강동구'
    ], key="gu_tab2")

    uploaded_file = st.file_uploader("📎 질병청 환자수 파일 업로드 (.xlsx, 시트명: 서울특별시)", type=["xlsx"], key="upload_tab2")

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

            ymd = date_selected.strftime("%Y-%m-%d")
            selected = df_long[(df_long["일자"] == ymd) & (df_long["자치구"] == gu)]
            if selected.empty:
                st.warning(f"❌ {ymd} {gu} 환자수 데이터가 없습니다.")
                st.stop()
            환자수 = int(selected["환자수"].values[0])

            weather = get_asos_weather(region, ymd.replace("-", ""), ASOS_API_KEY)
            tmx = weather.get("TMX", 0)
            tmn = weather.get("TMN", 0)
            reh = weather.get("REH", 0)
            avg_temp = calculate_avg_temp(tmx, tmn)

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
            st.dataframe(preview_df)

            if st.button("💾 GitHub에 저장하기", key="save_tab2"):
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
