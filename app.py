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
tab1, tab2, tab3 = st.tabs(["📊 폭염 피해 AI 예측 탭", "📥 실제 폭염 피해 기록 탭", "📍 자치구별 피해점수 및 보상금 계산탭"])

# ====================================================================
# 🔮 예측 탭
# ====================================================================
with tab1:
    st.header("📊 폭염 피해 AI 예측")

    # ✅ 날짜 선택 범위 설정: 2021-05-01 ~ 오늘+30일
    today = datetime.date.today()
    min_pred_date = datetime.date(2021, 5, 1)
    max_pred_date = today + datetime.timedelta(days=30)

    # ✅ 지역 및 날짜 선택
    region = st.selectbox("지역 선택", list(region_to_stn_id.keys()), key="region_tab1")
    date_selected = st.date_input(
        "날짜 선택",
        value=today,
        min_value=min_pred_date,
        max_value=max_pred_date,
        key="date_tab1"
    )

    if st.button("🔍 예측하기", key="predict_tab1"):
        # ✅ 기상 데이터 불러오기
        if date_selected >= today:
            weather, base_date, base_time = get_weather(region, date_selected, KMA_API_KEY)
        else:
            ymd = date_selected.strftime("%Y%m%d")
            weather = get_asos_weather(region, ymd, ASOS_API_KEY)

        # ✅ 예측 가능한 기상 정보가 없을 경우 중단
        if not weather:
            st.error("❌ 기상 정보 없음")
            st.stop()

        # ✅ 기온/습도 추출
        tmx = weather.get("TMX", 0)
        tmn = weather.get("TMN", 0)
        reh = weather.get("REH", 0)

        # ✅ 예측 수행
        pred, avg_temp, input_df = predict_from_weather(tmx, tmn, reh)
        risk = get_risk_level(pred)

        # ✅ 기상 정보 출력
        st.markdown("#### ☁️ 기상정보")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("최고기온", f"{tmx:.1f}℃")
        col2.metric("최저기온", f"{tmn:.1f}℃")
        col3.metric("평균기온", f"{avg_temp:.1f}℃")
        col4.metric("습도", f"{reh:.1f}%")

        with st.expander("🧪 입력값 확인"):
            st.dataframe(input_df)

        # ✅ 예측 결과 출력
        st.markdown("#### 💡 예측 결과")
        c1, c2 = st.columns(2)
        c1.metric("예측 환자 수", f"{pred:.2f}명")
        c2.metric("위험 등급", risk)

        # ✅ 전년도 환자수 비교
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

        # ✅ 예측값 CSV로 저장 (tab3에서 활용)
        SAVE_FILE = "ML_asos_total_prediction.csv"
        try:
            df_total = pd.read_csv(SAVE_FILE, encoding="utf-8-sig")
        except FileNotFoundError:
            df_total = pd.DataFrame(columns=["일자", "서울시예측환자수"])

        new_row = pd.DataFrame([{
            "일자": date_selected.strftime("%Y-%m-%d"),
            "서울시예측환자수": round(pred, 2)
        }])

        df_total = df_total[df_total["일자"] != new_row.iloc[0]["일자"]]
        df_total = pd.concat([df_total, new_row], ignore_index=True)
        df_total.to_csv(SAVE_FILE, index=False, encoding="utf-8-sig")

        st.success(f"✅ 예측값이 '{SAVE_FILE}'에 저장되었습니다.")


# ====================================================================
# 📥 실제 폭염 피해 기록 탭
# ====================================================================
with tab2:
    with st.expander("ℹ️ tab2 사용법"):
        st.markdown("""
        **✅ 사용 방법**  
        1. 날짜(복수 가능)와 자치구(전체 또는 일부)를 선택하세요.  
        2. 아래 링크에서 질병청의 온열질환자 엑셀 파일을 다운로드해 업로드하세요.  
           👉 [온열질환 응급실감시체계 다운로드](https://www.kdca.go.kr/board/board.es?mid=a20205030102&bid=0004&&cg_code=C01)  
        3. **저장하기** 버튼을 누르면, 해당 데이터는 tab3의 자치구별 피해점수 산정을 위한 입력값으로 자동 반영됩니다.
        """)

    st.header("📥 실제 폭염 피해 기록하기")

    region = st.selectbox("🌐 광역시도 선택", ["서울특별시"], key="region_tab2")

    all_gus = [
        '종로구', '중구', '용산구', '성동구', '광진구', '동대문구', '중랑구', '성북구', '강북구', '도봉구',
        '노원구', '은평구', '서대문구', '마포구', '양천구', '강서구', '구로구', '금천구', '영등포구',
        '동작구', '관악구', '서초구', '강남구', '송파구', '강동구'
    ]

    gus = st.multiselect("🏘️ 자치구 선택 (선택하지 않으면 전체)", all_gus, key="gu_tab2_multi")
    if not gus:
        gus = all_gus

    # ✅ 날짜 선택: pandas → date 변환
    min_record_date = datetime.date(2021, 5, 1)
    max_record_date = datetime.date.today() - datetime.timedelta(days=1)
    date_range = [d.date() for d in pd.date_range(min_record_date, max_record_date, freq='D')]

    dates_selected = st.multiselect("📅 기록할 날짜 (복수 선택 가능)", date_range, default=[max_record_date])

    uploaded_file = st.file_uploader("📎 질병청 환자수 파일 업로드 (.xlsx, 시트명: 서울특별시)", type=["xlsx"], key="upload_tab2")

    if uploaded_file and dates_selected:
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

            for date_selected in dates_selected:
                ymd = date_selected.strftime("%Y-%m-%d")

                weather = get_asos_weather(region, ymd.replace("-", ""), ASOS_API_KEY)
                tmx = weather.get("TMX", 0)
                tmn = weather.get("TMN", 0)
                reh = weather.get("REH", 0)
                avg_temp = calculate_avg_temp(tmx, tmn)

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
                        "최고체감온도(°C)": tmx + 1.5,
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
            st.markdown("### ✅ 저장될 학습 데이터 미리보기")
            st.dataframe(preview_df)

            if st.button("💾 GitHub에 저장하기", key="save_tab2_multi"):
                csv_path = "ML_asos_dataset_by_gu.csv"
                if os.path.exists(csv_path):
                    try:
                        existing = pd.read_csv(csv_path, encoding="utf-8-sig")
                    except UnicodeDecodeError:
                        existing = pd.read_csv(csv_path, encoding="cp949")
                    for row in preview_list:
                        existing = existing[~((existing["일자"] == row["일자"]) & (existing["자치구"] == row["자치구"]))]
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

with tab3:
    st.header("📍 자치구별 피해점수 및 보상금 계산하기")

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

    def calculate_damage_score(s, e, pred_count, real_count):
        P_pred = min(pred_count / 25, 1.0)
        P_real = 1.0 if real_count >= 1 else 0.0
        return 100 * (0.4 * s + 0.3 * e + 0.2 * P_pred + 0.1 * P_real)

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

    try:
        selected_date = st.date_input("📅 분석 날짜 선택", datetime.date.today())
        ymd = selected_date.strftime("%Y-%m-%d")

        def load_csv_with_fallback(path):
            for enc in ["utf-8-sig", "cp949", "euc-kr"]:
                try:
                    return pd.read_csv(path, encoding=enc)
                except UnicodeDecodeError:
                    continue
            raise UnicodeDecodeError(f"❌ 인코딩 실패: {path}")

        ml_data = load_csv_with_fallback("ML_asos_dataset.csv")
        static_data = load_csv_with_fallback("seoul_static_data.csv")

        merged_all = pd.merge(static_data, ml_data, on="자치구", how="left")
        merged_all = merged_all[merged_all["일자"] == ymd].copy()

        if merged_all.empty:
            st.warning("❗️선택한 날짜의 데이터가 없습니다.")
            st.stop()

        df_total = load_csv_with_fallback("ML_asos_total_prediction.csv")
        pred_row = df_total[df_total["일자"] == ymd]

        if pred_row.empty:
            st.warning(f"⚠️ {ymd} 예측값이 존재하지 않습니다. tab1에서 먼저 예측을 수행하세요.")
            st.stop()

        seoul_pred = float(pred_row["서울시예측환자수"].values[0])
        merged_all["예측환자수"] = seoul_pred

        merged_all["S"] = merged_all.apply(calculate_social_index, axis=1)
        for col in ["열섬지수", "녹지율", "냉방보급률"]:
            merged_all[f"{col}_std"] = standardize_column(merged_all, col)
        merged_all["E"] = merged_all.apply(calculate_environment_index, axis=1)

        merged_all["피해점수"] = merged_all.apply(
            lambda row: calculate_damage_score(
                row["S"], row["E"], row["예측환자수"], row["환자수"]
            ),
            axis=1
        )
        merged_all["위험등급"] = merged_all["피해점수"].apply(score_to_grade)
        merged_all["보상금"] = merged_all["피해점수"].apply(calc_payout)

        selected_gu = st.selectbox("🏘️ 자치구 선택", sorted(merged_all["자치구"].unique()))
        merged = merged_all[merged_all["자치구"] == selected_gu].copy()

        subs_count = st.number_input(f"{selected_gu} 가입자 수", min_value=0, step=1, key="subs_tab3")
        merged["가입자수"] = subs_count
        merged["예상총보상금"] = merged["보상금"] * subs_count
        st.success(f"💰 예상 보상금액: {int(merged['예상총보상금'].values[0]):,}원")

        show_cols = ["자치구", "피해점수", "위험등급", "보상금", "가입자수", "예상총보상금"]
        st.dataframe(merged[show_cols], use_container_width=True)

        st.markdown("### 📊 피해점수 분포")
        st.bar_chart(data=merged_all.set_index("자치구")["피해점수"])

        # 로그 및 다운로드
        row = merged.iloc[0]
        s_val = row["S"]
        e_val = row["E"]
        pred_count = row["예측환자수"]
        real_count = row["환자수"]
        score = row["피해점수"]
        P_pred = min(pred_count / 25, 1.0)
        P_real = 1.0 if real_count >= 1 else 0.0

        single_log = f"""
[피해점수 계산 로그 - {selected_gu} / {ymd}]
--------------------------------------------------
[S 계산]
- 고령자비율              = {row['고령자비율']:.4f}
- 야외근로자비율          = {row['야외근로자비율']:.4f}
- 열쾌적취약인구비율      = {row['열쾌적취약인구비율']:.4f}
=> S = {s_val:.4f}

[E 계산]
- 열섬지수 (표준화)       = {row['열섬지수_std']:.4f}
- 녹지율 (표준화)         = {row['녹지율_std']:.4f}
- 냉방보급률 (표준화)     = {row['냉방보급률_std']:.4f}
=> E = {e_val:.4f}

[환자 수 입력값]
- 예측환자수              = {pred_count:.1f}명 → 정규화 = {P_pred:.3f}
- 실제환자수              = {real_count:.1f}명 → 정규화 = {P_real:.1f}

[최종 피해점수 계산]
피해점수 = 100 × (0.4×{s_val:.4f} + 0.3×{e_val:.4f} + 0.2×{P_pred:.4f} + 0.1×{P_real:.1f})  
         = {score:.2f}
--------------------------------------------------
"""
        with st.expander(f"🔎 {selected_gu} 디버깅 로그"):
            st.code(single_log, language="text")
            st.download_button(
                label="📄 현재 자치구 디버깅 로그 다운로드",
                data=single_log.encode("utf-8-sig"),
                file_name=f"피해점수_디버깅_{ymd}_{selected_gu}.txt",
                mime="text/plain"
            )

        all_debug_logs = ""
        for _, row in merged_all.iterrows():
            s_val = row["S"]
            e_val = row["E"]
            pred_count = row["예측환자수"]
            real_count = row["환자수"]
            score = row["피해점수"]
            P_pred = min(pred_count / 25, 1.0)
            P_real = 1.0 if real_count >= 1 else 0.0

            log = f"""
[피해점수 계산 로그 - {row['자치구']} / {ymd}]
--------------------------------------------------
[S 계산]
- 고령자비율              = {row['고령자비율']:.4f}
- 야외근로자비율          = {row['야외근로자비율']:.4f}
- 열쾌적취약인구비율      = {row['열쾌적취약인구비율']:.4f}
=> S = {s_val:.4f}

[E 계산]
- 열섬지수 (표준화)       = {row['열섬지수_std']:.4f}
- 녹지율 (표준화)         = {row['녹지율_std']:.4f}
- 냉방보급률 (표준화)     = {row['냉방보급률_std']:.4f}
=> E = {e_val:.4f}

[환자 수 입력값]
- 예측환자수              = {pred_count:.1f}명 → 정규화 = {P_pred:.3f}
- 실제환자수              = {real_count:.1f}명 → 정규화 = {P_real:.1f}

[최종 피해점수 계산]
피해점수 = 100 × (0.4×{s_val:.4f} + 0.3×{e_val:.4f} + 0.2×{P_pred:.4f} + 0.1×{P_real:.1f})  
         = {score:.2f}
--------------------------------------------------
"""
            all_debug_logs += log + "\n"

        st.download_button(
            label="📥 전체 자치구 디버깅 로그 다운로드",
            data=all_debug_logs.encode("utf-8-sig"),
            file_name=f"전체_피해점수_디버깅_{ymd}.txt",
            mime="text/plain"
        )

        with st.expander("📘 피해점수 구성 요소 안내"):
            st.markdown("""
> 🧓 **S (사회적 취약성 지수)**  
고령자, 야외 근로자, 열에 취약한 인구의 비율을 반영한 지수입니다.

> 🌍 **E (환경적 취약성 지수)**  
열섬 현상, 녹지율, 냉방 보급률 등을 표준화하여 반영한 지수입니다.

> 📈 **예측환자수**  
서울시 전체 예측 환자 수를 그대로 사용하며, 모든 자치구에 동일하게 적용됩니다. (25명 기준으로 정규화, 최대 1.0)

> 📉 **실제환자수**  
1명 이상 발생 시 1.0, 그렇지 않으면 0.0으로 계산됩니다.

> 🧮 **피해점수 계산식**  
피해점수 = 100 × (0.4 × S + 0.3 × E + 0.2 × P_pred + 0.1 × P_real)
""")

    except Exception as e:
        st.error(f"❌ 분석 실패: {e}")
