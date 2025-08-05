import streamlit as st
import pandas as pd
import datetime
import requests
import os
import base64
import io
from urllib.parse import unquote
import subprocess
import sys


from utils import (
    get_weather, get_asos_weather, get_risk_level,
    calculate_avg_temp, region_to_stn_id, compute_heat_index
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
tab1, tab2, tab3 = st.tabs(["📊 폭염 예측 및 위험도 분석", "📥 실제 피해 기록 및 데이터 입력", "📍 자치구별 피해점수 및 보상 분석"])

with tab1:
    # ✅ 사용법 안내
    with st.expander("📊 tab1에서 입력된 정보는 이렇게 활용됩니다"):
        st.markdown("""
        1. 기상청의 단기예보 API를 통해 자동으로 수집된 기상 정보를 바탕으로,  
           AI가 선택한 지역의 예측 온열질환자 수를 산출합니다.

        2. 예측 모델은 4개년 7,8월의 실제 기상 조건과 온열질환자 수 데이터를 학습했습니다.  
           현재 입력된 기상 조건이 과거 어떤 날과 유사한지를 바탕으로  
           AI가 발생 가능성이 높은 환자 수를 추정합니다.

        3. 예측된 환자 수는 위험도 등급(🟢~🔥)으로 변환되어 시민에게 전달되며,  
           tab3의 자치구별 피해점수 계산에 활용되는 입력값(P_pred)으로도 사용됩니다.

        📍 기상청 예보는 지점 단위(광역시도) 기준으로 제공되므로,  
        현재는 자치구 단위가 아닌 광역시도 단위로만 예측이 가능합니다.

        📅 예측 가능한 날짜는 2025년 7월 1일부터 8월 31일까지입니다.
        """)

# ✅ 작년 동일 날짜 환자 수 조회 함수
def get_last_year_patient_count(current_date, region, static_file="ML_7_8월_2021_2025_dataset.xlsx"):
    try:
        last_year_date = (current_date - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
        df_all = pd.read_excel(static_file, engine="openpyxl")

        # 날짜 형식 처리
        if "일시" in df_all.columns and pd.api.types.is_numeric_dtype(df_all["\uc77c\uc2dc"]):
            df_all["\uc77c\uc2dc"] = pd.to_datetime("1899-12-30") + pd.to_timedelta(df_all["\uc77c\uc2dc"], unit="D")
            df_all["\uc77c\uc790"] = df_all["\uc77c\uc2dc"].dt.strftime("%Y-%m-%d")
        elif "일자" not in df_all.columns and "일시" in df_all.columns:
            df_all["\uc77c\uc790"] = pd.to_datetime(df_all["\uc77c\uc2dc"]).dt.strftime("%Y-%m-%d")

        cond = (df_all["\uc77c\uc790"] == last_year_date) & (df_all["\uad11\uc5ed\uc790\uce58\ub2e8\uccb4"] == region)
        row = df_all[cond]
        return int(row["\ud658\uc790\uc218"].values[0]) if not row.empty else None

    except Exception as e:
        st.warning(f"⚠️ 작년 환자수 불러오기 오류: {e}")
        return None

# ✅ 날짜 선택 범위 설정
min_pred_date = datetime.date(2025, 7, 1)
max_pred_date = datetime.date(2025, 8, 31)

# ✅ 지역 및 날짜 선택 UI
region = st.selectbox("지역 선택", list(region_to_stn_id.keys()), key="region_tab1")
date_selected = st.date_input("날짜 선택", value=min_pred_date, min_value=min_pred_date, max_value=max_pred_date, key="date_tab1")

# ✅ 예측 버튼 클릭 시 실행
if st.button("🔍 예측하기", key="predict_tab1"):
    today = datetime.date.today()

    if date_selected >= today:
        weather, base_date, base_time = get_weather(region, date_selected, KMA_API_KEY)
    else:
        ymd = date_selected.strftime("%Y%m%d")
        weather = get_asos_weather(region, ymd, ASOS_API_KEY)

    if not weather:
        st.error("❌ 기상 정보 없음")
        st.stop()

    tmx = weather.get("TMX", 0)
    tmn = weather.get("TMN", 0)
    reh = weather.get("REH", 0)
    heat_index = compute_heat_index(tmx, reh)

    pred, avg_temp, input_df = predict_from_weather(tmx, tmn, reh)
    risk = get_risk_level(pred)

    with st.expander("🤪 입력값 확인"):
        st.dataframe(input_df)

    st.markdown("#### 💡 예측 결과")
    c1, c2 = st.columns(2)
    c1.metric("예측 환자 수", f"{pred:.2f}명")
    c2.metric("위험 등급", risk)

    last_year_count = get_last_year_patient_count(date_selected, region)
    if last_year_count is not None:
        delta = pred - last_year_count
        st.markdown(f"""
        📅 **전년도({(date_selected - datetime.timedelta(days=365)).strftime('%Y-%m-%d')}) 동일 날짜 환자수**: **{last_year_count}명**  
        📈 **전년 대비 증가**: {'+' if delta >= 0 else ''}{delta:.1f}명
        """)
    else:
        st.info("ℹ️ 전년도 동일 날짜의 환자 수 데이터를 찾을 수 없습니다.")

    SAVE_FILE = "ML_asos_total_prediction.csv"
    today_str = date_selected.strftime("%Y-%m-%d")

    try:
        df_total = pd.read_csv(SAVE_FILE, encoding="utf-8-sig")
    except FileNotFoundError:
        df_total = pd.DataFrame(columns=["\uc77c\uc790", "\uc11c\uc6b8\uc2dc\uc608\uce21\ud658\uc790\uc218"])

    new_row = pd.DataFrame([{ "\uc77c\uc790": today_str, "\uc11c\uc6b8\uc2dc\uc608\uce21\ud658\uc790\uc218": round(pred, 2) }])
    df_total = pd.concat([df_total, new_row], ignore_index=True)
    df_total.to_csv(SAVE_FILE, index=False, encoding="utf-8-sig")
    st.success(f"✅ 예측값이 '{SAVE_FILE}'에 저장되었습니다.")

    with open(SAVE_FILE, "rb") as f:
        content = f.read()
    b64_content = base64.b64encode(content).decode("utf-8")

    api_url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{GITHUB_REPO}/contents/{SAVE_FILE}"
    r = requests.get(api_url, headers={"Authorization": f"Bearer {GITHUB_TOKEN}"})
    sha = r.json().get("sha") if r.status_code == 200 else None

    payload = {
        "message": f"[tab1] {date_selected} 예측값 업데이트",
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
        st.success("✅ GitHub에 예측값 저장 완료")
        st.info(f"🔗 [GitHub에서 확인하기](https://github.com/{GITHUB_USERNAME}/{GITHUB_REPO}/blob/{GITHUB_BRANCH}/{SAVE_FILE})")
    else:
        st.warning(f"⚠️ GitHub 저장 실패: {r.status_code} / {r.text[:200]}")

with tab2:
    # ✅ 사용법 안내
    with st.expander("📊 tab2에서 입력된 정보는 이렇게 활용됩니다"):
        st.markdown("""
        1. 서울특별시 각 자치구의 **실제 폭염 피해 여부(1 또는 0)**를 수집하여,  
           **tab3의 피해점수 계산**에 활용됩니다.  
           (※ 온열질환자 수가 **1명 이상 발생한 경우 → 피해 발생(1)**로 간주합니다.)

        2. 실제 기상조건과 온열질환자 수를 함께 저장하여,  
           **tab1의 머신러닝 예측 모델 학습 데이터로 자동 반영**됩니다.

        📅 **정확한 피해 판단을 위해 오늘 날짜는 제외되며, 어제까지의 정보만 저장할 수 있습니다.**  
        이는 질병관리청의 온열질환자 통계가 **하루 단위로 집계되어 익일에 공개**되기 때문입니다.

        📂 아래 링크에서 질병청의 온열질환자 엑셀 파일을 다운로드해 업로드해주세요.  
        👉 [온열질환 응급실감시체계 다운로드](https://www.kdca.go.kr/board/board.es?mid=a20205030102&bid=0004&&cg_code=C01)

        ✅ 모든 입력이 완료되면 **저장하기** 버튼을 눌러주세요.  
        입력된 데이터는 자동으로 내부 DB에 기록되어 다른 탭에서 즉시 활용됩니다.
        """)

    region = st.selectbox("🌐 광역시도 선택", ["서울특별시"], key="region_tab2")

    all_gus = [
        '종로구', '중구', '용산구', '성동구', '광진구', '동대문구', '중랑구', '성북구', '강북구', '도봉구',
        '노원구', '은평구', '서대문구', '마포구', '양천구', '강서구', '구로구', '금천구', '영등포구',
        '동작구', '관악구', '서초구', '강남구', '송파구', '강동구'
    ]
    gus = st.multiselect("🏘️ 자치구 선택 (선택하지 않으면 전체)", all_gus, key="gu_tab2_multi")
    if not gus:
        gus = all_gus

    min_record_date = datetime.date(2025, 7, 1)
    max_record_date = datetime.date.today() - datetime.timedelta(days=1)

    date_selected = st.date_input(
        "📅 저장할 날짜 선택", 
        value=max_record_date, 
        min_value=min_record_date, 
        max_value=max_record_date,
        key="date_tab2"
    )

    uploaded_file = st.file_uploader("📎 질병청 환자수 파일 업로드 (.xlsx, 시트명: 서울특별시)", type=["xlsx"], key="upload_tab2")

    if uploaded_file and date_selected:
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

            preview_list = []
            ymd = date_selected.strftime("%Y-%m-%d")

            weather = get_asos_weather(region, ymd.replace("-", ""), ASOS_API_KEY)
            tmx = weather.get("TMX", 0)
            tmn = weather.get("TMN", 0)
            reh = weather.get("REH", 0)
            avg_temp = calculate_avg_temp(tmx, tmn)
            heat_index = compute_heat_index(tmx, reh)

            for gu in gus:
                selected = df_long[(df_long["일자"] == ymd) & (df_long["자치구"] == gu)]
                if selected.empty:
                    st.warning(f"❌ {ymd} {gu} 환자수 데이터가 없습니다.")
                    continue

                환자수 = int(selected["환자수"].values[0])
                preview_list.append({
                    "일자": ymd,
                    "지역": region,
                    "자치구": gu,
                    "최고체감온도(°C)": heat_index,
                    "최고기온(°C)": tmx,
                    "평균기온(°C)": avg_temp,
                    "최저기온(°C)": tmn,
                    "평균상대습도(%)": reh,
                    "환자수": 환자수
                })

            if not preview_list:
                st.warning("❌ 선택한 날짜와 자치구 조합에 저장할 데이터가 없습니다.")
                st.stop()

            preview_df = pd.DataFrame(preview_list)
            st.markdown("#### ✅ 저장될 학습 데이터 미리보기")
            st.dataframe(preview_df)

            if st.button("💾 GitHub에 저장하고 모델 재학습하기", key="save_and_train_tab2"):
                csv_path = "ML_asos_dataset.csv"

                if os.path.exists(csv_path):
                    try:
                        existing = pd.read_csv(csv_path, encoding="utf-8-sig")
                    except UnicodeDecodeError:
                        existing = pd.read_csv(csv_path, encoding="cp949")
                else:
                    existing = pd.DataFrame()

                merge_keys = ["일자", "자치구"]
                if not existing.empty:
                    existing = existing[~existing.set_index(merge_keys).index.isin(preview_df.set_index(merge_keys).index)]
                df_all = pd.concat([existing, preview_df], ignore_index=True)
                df_all.to_csv(csv_path, index=False, encoding="utf-8-sig")
                st.success("✅ 학습 데이터 저장 완료 (로컬)")

                try:
                    with open(csv_path, "rb") as f:
                        content = f.read()
                    b64_content = base64.b64encode(content).decode("utf-8")

                    api_url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{GITHUB_REPO}/contents/{GITHUB_FILENAME}"
                    r = requests.get(api_url, headers={"Authorization": f"Bearer {GITHUB_TOKEN}"})
                    sha = r.json().get("sha") if r.status_code == 200 else None

                    payload = {
                        "message": f"Update {GITHUB_FILENAME} with new data for {region} ({len(preview_list)} entries)",
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
                        st.info(f"🔗 [GitHub에서 보기](https://github.com/{GITHUB_USERNAME}/{GITHUB_REPO}/blob/{GITHUB_BRANCH}/{GITHUB_FILENAME})")
                    else:
                        st.warning(f"⚠️ GitHub 저장 실패: {r.status_code} {r.text[:200]}")

                except Exception as e:
                    st.error(f"❌ 처리 중 오류 발생: {e}")
                    st.stop()

                st.info("📈 머신러닝 모델 재학습 중입니다...")
                try:
                    result = subprocess.run([sys.executable, "train_model.py"], capture_output=True, text=True, check=True)
                    st.success("✅ 모델 재학습 완료")
                    st.text_area("📄 학습 로그", result.stdout, height=300)
                except subprocess.CalledProcessError as e:
                    st.error("❌ 모델 학습 실패")
                    st.text_area("🚨 오류 로그", e.stderr or str(e), height=300)

        except Exception as e:
            st.error(f"❌ 처리 중 오류 발생: {e}")

with tab3:
    # ✅ 사용법 안내
    with st.expander("📊 tab3에서 산출된 정보는 이렇게 활용됩니다"):
        st.markdown("""
        1. tab1에서 예측한 **광역시도별 온열질환자 수**를 자치구별 **사회적 취약성(S)**에    
           따라 분배하고, 정규화하여 자치구별 **예측 환자수(P_pred)**로 변환합니다.
        2. tab2에서 수집된 **실제 온열질환자 수(P_real)**,  
           그리고 자치구별 **사회적(S)** 및 **환경적(E)** 취약성 지표와 함께  
           **종합적인 피해점수(0~100점)**를 계산합니다.
        3. 피해점수가 높을수록 해당 자치구가 더 큰 폭염 피해 위험에 노출된 것으로 간주되며,  
           위험등급(🟢~🔥)과 **보상금 산정 기준**으로 활용됩니다.

        📌 피해점수 계산은 다음 요소를 기반으로 이루어집니다:
        - 🧓 S: 고령자 비율, 야외 근로자 비율, 열취약 인구 비율  
        - 🌍 E: 열섬지수, 녹지율, 냉방보급률 (표준화된 환경지표)  
        - 📈 P_pred: AI 예측 환자 수 (자치구별 분배 및 정규화)  
        - 📉 P_real: 실제 환자 수 (1명 이상이면 1.0, 없으면 0.0)

        피해점수 = 100 × (0.2 × S + 0.2 × E + 0.5 × P_pred + 0.1 × P_real)
        """)

    # ✅ 함수 정의
    def format_debug_log(row, date_str):
        return f"""[피해점수 계산 로그 - {row['자치구']} / {date_str}]
[S 계산] - 고령자비율 = {row['고령자비율']:.4f}, 야외근로자비율 = {row['야외근로자비율']:.4f}, 열취약인구비율 = {row['열쾌적취약인구비율']:.4f} → S = {row['S']:.4f}
[E 계산] - 열섬지수 = {row['열섬지수_std']:.4f}, 녹지율 = {row['녹지율_std']:.4f}, 냉방보급률 = {row['냉방보급률_std']:.4f} → E = {row['E']:.4f}
[P 계산] - 예측환자수 = {row['P_pred_raw']:.2f}명 → 정규화(P_pred) = {row['P_pred']:.4f}
[R 계산] - 실제환자수 = {row['환자수']}, 변환(P_real) = {1.0 if row['환자수'] >= 1 else 0.0}
🧮 피해점수 = {row['피해점수']:.2f} / 위험등급: {row['위험등급']} / 보상금: {row['보상금']}원
"""

    def calculate_social_index(row):
        return (
            0.5 * row["고령자비율"] +
            0.3 * row["야외근로자비율"] +
            0.2 * row["열쾌적취약인구비율"]
        )

    def standardize_column(df, col):
        min_val = df[col].min()
        max_val = df[col].max()
        range_val = max_val - min_val if max_val != min_val else 1
        return (df[col] - min_val) / range_val

    def calculate_environment_index(row):
        return (
            0.5 * row["열섬지수_std"] +
            0.3 * (1 - row["녹지율_std"]) +
            0.2 * (1 - row["냉방보급률_std"])
        )

    def distribute_pred_by_s(merged_df, total_pred):
        """
        S 비율에 따라 자치구별 예측환자수 분배 및 √ 정규화
        (25명은 서울시 하루 최대 예측 환자수 기준으로 정규화에 사용됨)
        """
        s_sum = merged_df["S"].sum()
        merged_df["P_pred_raw"] = total_pred * (merged_df["S"] / s_sum)
        merged_df["P_pred"] = (merged_df["P_pred_raw"] / 25) ** 0.5
        return merged_df

    def calculate_damage_score_v2(s, e, p_pred, p_real):
        return 100 * (0.2 * s + 0.2 * e + 0.5 * p_pred + 0.1 * p_real)

    def score_to_grade(score):
        if score < 20: return "🟢 매우 낮음"
        elif score < 30: return "🟡 낮음"
        elif score < 40: return "🟠 보통"
        elif score < 50: return "🔴 높음"
        else: return "🔥 매우 높음"

    def calc_payout(score):
        if score < 30: return 0
        elif score < 40: return 5000
        elif score < 50: return 10000
        else: return 20000

    def load_csv_with_fallback(path):
        for enc in ["utf-8-sig", "cp949", "euc-kr"]:
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise UnicodeDecodeError(f"❌ 인코딩 실패: {path}")

    def load_csv_from_github(filename):
        try:
            github_url =   f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/{GITHUB_BRANCH}/{filename}"
            r = requests.get(github_url)
            r.raise_for_status()
            return pd.read_csv(io.StringIO(r.text), encoding="utf-8-sig")
        except Exception as e:
            st.error(f"❌ GitHub에서 {filename} 불러오기 실패: {e}")
            return pd.DataFrame()


    # ✅ 메인 실행
    try:
        col1, col2 = st.columns(2)
        with col1:
            selected_date = st.date_input("📅 분석 날짜 선택", datetime.date.today())
            ymd = selected_date.strftime("%Y-%m-%d")
        with col2:
            selected_gu = None

        ml_data = load_csv_from_github("ML_asos_dataset.csv")
        if ml_data.empty:
            st.warning("❗️기록된 학습 데이터가 없습니다. tab2에서 데이터를 먼저 저장해주세요.")
            st.stop()
        ml_data = ml_data[ml_data["일자"] == ymd]
        static_data = load_csv_with_fallback("seoul_static_data.csv")
        merged_all = pd.merge(static_data, ml_data, on="자치구", how="left")

        if merged_all.empty:
            st.warning("❗️선택한 날짜의 데이터가 없습니다.")
            st.stop()

        df_total = load_csv_from_github("ML_asos_total_prediction.csv")
        pred_row = df_total[df_total["일자"] == ymd]

        if pred_row.empty:
            st.warning(f"⚠️ {ymd} 예측값이 존재하지 않습니다. tab1에서 먼저 예측을 수행하세요.")
            st.stop()

        seoul_pred = float(pred_row["서울시예측환자수"].values[0])

        merged_all["S"] = merged_all.apply(calculate_social_index, axis=1)
        for col in ["열섬지수", "녹지율", "냉방보급률"]:
            merged_all[f"{col}_std"] = standardize_column(merged_all, col)
        merged_all["E"] = merged_all.apply(calculate_environment_index, axis=1)

        merged_all = distribute_pred_by_s(merged_all, seoul_pred)

        merged_all["피해점수"] = merged_all.apply(
            lambda row: calculate_damage_score_v2(
                row["S"], row["E"], row["P_pred"], 1.0 if row["환자수"] >= 1 else 0.0
            ),
            axis=1
        )
        merged_all["위험등급"] = merged_all["피해점수"].apply(score_to_grade)
        merged_all["보상금"] = merged_all["피해점수"].apply(calc_payout)

        col1, col2 = st.columns(2)
        with col1:
            selected_gu = st.selectbox("🏘️ 자치구 선택", sorted(merged_all["자치구"].unique()))
        with col2:
            subs_count = st.number_input(f"{selected_gu} 가입자 수", min_value=0, step=1, key="subs_tab3")

        merged = merged_all[merged_all["자치구"] == selected_gu].copy()
        if merged.empty:
            st.warning("선택한 자치구에 해당하는 데이터가 없습니다.")
            st.stop()

        merged["가입자수"] = subs_count
        merged["예상총보상금"] = merged["보상금"] * subs_count
        st.success(f"💰 예상 보상금액: {int(merged['예상총보상금'].sum()):,}원")

        show_cols = ["자치구", "피해점수", "위험등급", "보상금", "가입자수", "예상총보상금"]
        st.dataframe(merged[show_cols], use_container_width=True)

        st.markdown("#### 📊 피해점수 분포")
        st.bar_chart(data=merged_all.set_index("자치구")["피해점수"])

        # ✅ 단일 자치구 디버깅 로그
        row = merged.iloc[0]
        single_log = format_debug_log(row, ymd)

        with st.expander(f"🔎 {selected_gu} 디버깅 로그"):
            st.code(single_log, language="text")
            st.download_button(
                label="📄 현재 자치구 디버깅 로그 다운로드",
                data=single_log.encode("utf-8-sig"),
                file_name=f"피해점수_디버깅_{ymd}_{selected_gu}.txt",
                mime="text/plain"
            )

        # ✅ 전체 자치구 디버깅 로그
        all_debug_logs = "\n".join([
            format_debug_log(row, ymd) for _, row in merged_all.iterrows()
        ])
        st.download_button(
            label="📥 전체 자치구 디버깅 로그 다운로드",
            data=all_debug_logs.encode("utf-8-sig"),
            file_name=f"전체_피해점수_디버깅_{ymd}.txt",
            mime="text/plain"
        )

    except Exception as e:
        st.error(f"❌ 처리 중 오류가 발생했습니다: {e}")

