import streamlit as st
import pandas as pd
import datetime
import datetime as dt
import requests
import os
import base64
import io
import json
import time
import math
from urllib.parse import unquote, quote
import urllib.parse
import subprocess
import sys


from utils import (
    get_weather, get_asos_weather, get_risk_level,
    calculate_avg_temp, region_to_stn_id,convert_latlon_to_xy, get_fixed_base_datetime
)
from model_utils import predict_from_weather

# ----------------------- 설정 -----------------------
st.set_page_config(layout="centered")

KMA_API_KEY = unquote(st.secrets["KMA"]["API_KEY"])
ASOS_API_KEY = unquote(st.secrets["ASOS"]["API_KEY"])
GITHUB_USERNAME = st.secrets["GITHUB"]["USERNAME"]
GITHUB_REPO = st.secrets["GITHUB"]["REPO"]
GITHUB_BRANCH = st.secrets["GITHUB"]["BRANCH"]
GITHUB_TOKEN = st.secrets["GITHUB"]["TOKEN"]
GITHUB_FILENAME = "ML_asos_dataset.csv"

# ----------------------- UI 시작 -----------------------
st.title("Weather Pay")
tab1, tab2, tab3 = st.tabs(["학습 데이터 입력", "환자 수 지표 산출", "피해점수 계산 및 보상"])

with tab1:
    with st.expander("이 탭에서는 무엇을 하나요?"):
        st.markdown("""
        이 탭은 AI 모델의 학습을 위한 사용자 입력 학습 데이터를 추가하는 기능을 수행합니다. 
        사용자는 질병관리청 온열질환자 통계를 기반으로, 
        서울특별시 자치구별 실제 환자 수와 해당 일자의 기상 정보를 기록할 수 있습니다.

        이 데이터는 기존의 과거 기초 학습 데이터셋(2021~2024년 7·8월)과 함께 사용되어 
        XGBoost 기반 모델을 최신 상태로 재학습하는 데 활용됩니다.
        이 과정은 TAB2의 예측 정확도를 향상시키며, TAB3의 피해점수 신뢰도를 높여줍니다.

        아래 링크에서 질병청의 온열질환자 엑셀 파일을 다운로드해 업로드해주세요.  
        [온열질환 응급실감시체계 다운로드](https://www.kdca.go.kr/board/board.es?mid=a20205030102&bid=0004&&cg_code=C01)

        ---

        **수행 내용 요약**:

        1. 질병청 엑셀 파일 업로드 → 자치구별 환자 수 자동 추출
        2. 선택 날짜의 기상정보 자동 수집 (ASOS API)
        3. 자치구별 환자수 + 기상정보 → 학습용 데이터프레임 생성
        4. `ML_asos_dataset.csv`에 저장 후 자동 재학습 수행
""")

    region = st.selectbox("광역시도 선택", ["서울특별시"], key="region_tab1")

    all_gus = [
        '종로구', '중구', '용산구', '성동구', '광진구', '동대문구', '중랑구', '성북구', '강북구', '도봉구',
        '노원구', '은평구', '서대문구', '마포구', '양천구', '강서구', '구로구', '금천구', '영등포구',
        '동작구', '관악구', '서초구', '강남구', '송파구', '강동구'
    ]
    gus = st.multiselect("자치구 선택 (선택하지 않으면 전체)", all_gus, key="gu_tab1_multi")
    if not gus:
        gus = all_gus

    min_record_date = datetime.date(2025, 7, 1)
    max_record_date = datetime.date.today() - datetime.timedelta(days=1)

    date_selected = st.date_input(
        "저장할 날짜 선택", 
        value=max_record_date, 
        min_value=min_record_date, 
        max_value=max_record_date,
        key="date_tab1"
    )

    uploaded_file = st.file_uploader("질병청 환자수 파일 업로드 (.xlsx, 시트명: 서울특별시)", type=["xlsx"], key="upload_tab1")

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
            pred, avg_temp, heat_index, input_df = predict_from_weather(tmx, tmn, reh)


            for gu in gus:
                selected = df_long[(df_long["일자"] == ymd) & (df_long["자치구"] == gu)]
                if selected.empty:
                    st.warning(f"{ymd} {gu} 환자수 데이터가 없습니다.")
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
                st.warning("선택한 날짜와 자치구 조합에 저장할 데이터가 없습니다.")
                st.stop()

            preview_df = pd.DataFrame(preview_list)
            st.markdown("#### 저장될 학습 데이터 미리보기")
            st.dataframe(preview_df)

            if st.button("GitHub에 저장하고 모델 재학습하기", key="save_and_train_tab1"):
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
                st.success("학습 데이터 저장 완료 (로컬)")

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
                        st.success("GitHub 저장 완료")
                        st.info(f"[GitHub에서 보기](https://github.com/{GITHUB_USERNAME}/{GITHUB_REPO}/blob/{GITHUB_BRANCH}/{GITHUB_FILENAME})")
                    else:
                        st.warning(f"GitHub 저장 실패: {r.status_code} {r.text[:200]}")

                except Exception as e:
                    st.error(f"처리 중 오류 발생: {e}")
                    st.stop()

                st.info("머신러닝 모델 재학습 중입니다...")
                try:
                    result = subprocess.run([sys.executable, "train_model.py"], capture_output=True, text=True, check=True)
                    st.success("모델 재학습 완료")
                    st.text_area("학습 로그", result.stdout, height=300)
                except subprocess.CalledProcessError as e:
                    st.error("모델 학습 실패")
                    st.text_area("오류 로그", e.stderr or str(e), height=300)

        except Exception as e:
            st.error(f"처리 중 오류 발생: {e}")

with tab2:
    # ---------- 보조 함수들 ----------
    def _ultra_now_safe(nx: int, ny: int, api_key: str) -> dict:
        """초단기실황 REH/T1H: 00/30 교차 + 전시간 폴백, resultCode 체크"""
        def candidates():
            now_kst = dt.datetime.now(dt.timezone.utc).astimezone(KST)
            ref = now_kst - dt.timedelta(minutes=40)
            hh = ref.strftime("%H")
            mm = "30" if ref.minute >= 30 else "00"
            d0 = ref.strftime("%Y%m%d")
            c = [
                  (d0, f"{hh}{mm}"),
                  (d0, f"{hh}{'00' if mm=='30' else '30'})",
            ]
            prev = ref - dt.timedelta(hours=1)
            d1, h1 = prev.strftime("%Y%m%d"), prev.strftime("%H")
            c += [(d1, f"{h1}30"), (d1, f"{h1}00")]
            # 중복 제거
            seen, out = set(), []
            for bd, bt in c:
                k = bd + bt
                if k not in seen:
                    seen.add(k); out.append((bd, bt))
            return out

        def call(bd, bt):
            url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst"
            params = {
                "serviceKey": unquote(api_key),
                "dataType": "JSON",
                "numOfRows": "100",
                "pageNo": "1",
                "base_date": bd,
                "base_time": bt,
                "nx": nx, "ny": ny,
            }
            r = requests.get(url, params=params, timeout=8)
            r.raise_for_status()
            resp = r.json().get("response", {})
            if resp.get("header", {}).get("resultCode") != "00":
                return None
            items = resp.get("body", {}).get("items", {}).get("item", [])
            reh = t1h = None
            for it in items:
                cat = it.get("category")
                try:
                    val = float(it.get("obsrValue"))
                except (TypeError, ValueError):
                    continue
                if cat == "REH": reh = val
                elif cat == "T1H": t1h = val
            if reh is None and t1h is None:
                return None
            bdate = items[0].get("baseDate", bd) if items else bd
            btime = items[0].get("baseTime", bt) if items else bt
            return {"REH": reh, "T1H": t1h, "base_date": bdate, "base_time": btime}

        for bd, bt in candidates():
            try:
                out = call(bd, bt)
                if out:
                    return out
            except Exception:
                continue
        return {"REH": None, "T1H": None, "base_date": None, "base_time": None}

    def _today_tmx_tmn_safe(nx: int, ny: int, api_key: str, base_date: str, base_time: str) -> dict:
        """단기예보에서 오늘 TMX/TMN만 추출 (KST 기준)"""
        url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
        params = {
            "serviceKey": unquote(api_key),
            "numOfRows": "1000",
            "pageNo": "1",
            "dataType": "JSON",
            "base_date": base_date,
            "base_time": base_time,
            "nx": nx, "ny": ny,
        }
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            resp = r.json().get("response", {})
            if resp.get("header", {}).get("resultCode") != "00":
                return {"TMX": None, "TMN": None}
            items = resp.get("body", {}).get("items", {}).get("item", [])
            today_kst = dt.datetime.now(dt.timezone.utc).astimezone(KST).strftime("%Y%m%d")
            tmx = tmn = None
            for it in items:
                if it.get("fcstDate") != today_kst:
                    continue
                cat = it.get("category")
                try:
                    v = float(it.get("fcstValue"))
                except (TypeError, ValueError):
                    continue
                if cat == "TMX": tmx = v
                elif cat == "TMN": tmn = v
            return {"TMX": tmx, "TMN": tmn}
        except Exception:
            return {"TMX": None, "TMN": None}

    def _reverse_geocode_to_gu(lat: float, lon: float) -> dict:
        """좌표 → {'city': '서울특별시', 'gu': '동작구'} (정규화 포함)"""
        try:
            url = "https://nominatim.openstreetmap.org/reverse"
            params = {"format": "jsonv2", "lat": lat, "lon": lon, "addressdetails": 1, "zoom": 14}
            r = requests.get(url, params=params, headers={"User-Agent": "weatherpay/1.0"}, timeout=8)
            addr = r.json().get("address", {})
            city_raw = (addr.get("city") or addr.get("state") or addr.get("region") or "").strip()
            gu_raw = (addr.get("city_district") or addr.get("district") or addr.get("county") or "").strip()
            city = {
                "Seoul": "서울특별시",
                "Seoul Metropolitan City": "서울특별시",
                "서울특별자치시": "서울특별시",
                "서울특별시": "서울특별시",
            }.get(city_raw, city_raw)
            gu = gu_raw
            if gu.endswith("-gu"): gu = gu.replace("-gu", "구")
            if gu and not gu.endswith("구"): gu = gu + "구"
            return {"city": city, "gu": gu}
        except Exception:
            return {}

    def _haversine_km(a, b):
        import math
        lat1, lon1 = math.radians(a[0]), math.radians(a[1])
        lat2, lon2 = math.radians(b[0]), math.radians(b[1])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        h = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
        return 6371 * 2 * math.asin(math.sqrt(h))

    def _nearest_gu(lat_f: float, lon_f: float, gu_centers: dict) -> str:
        best_gu, best_d = None, 1e9
        for gu, center in gu_centers.items():
            d = _haversine_km((lat_f, lon_f), center)
            if d < best_d:
                best_d, best_gu = d, gu
        return best_gu

    # ---------- 상단 헤더 (컬럼 배치) ----------
    colH1, colH2 = st.columns([1, 1])
    with colH1:
        st.subheader("실시간 위험 점수")
        st.caption("내 위치(자치구) 기준으로 현재 기상조건을 반영해 온열질환 위험을 추정합니다.")
    with colH2:
        now_kst = dt.datetime.now(dt.timezone.utc).astimezone(KST).strftime("%Y-%m-%d %H:%M")
        st.markdown(f"<div style='text-align:right;color:#6b7280;'>기준시각(실황): {now_kst} KST</div>", unsafe_allow_html=True)

    # 위치 버튼 (브라우저 권한 → 쿼리스트링 lat/lon)
    st.components.v1.html(
        """
        <script>
        function setLoc(){
          if(!navigator.geolocation){ alert('위치 접근이 지원되지 않습니다.'); return; }
          navigator.geolocation.getCurrentPosition(function(pos){
            const lat = pos.coords.latitude.toFixed(6);
            const lon = pos.coords.longitude.toFixed(6);
            const url = new URL(window.location.href);
            url.searchParams.set('lat', lat);
            url.searchParams.set('lon', lon);
            window.location.href = url.toString();
          }, function(err){ alert('위치 접근이 거부되었거나 실패했습니다.'); });
        }
        </script>
        <button onclick="setLoc()" style="padding:8px 12px;border-radius:10px;border:1px solid #ddd;cursor:pointer;">내 위치로 찾기</button>
        """,
        height=50,
    )

    # ---------- 날짜 분기 ----------
    today = dt.date.today()
    if date_selected > today:
        weather_fcst, base_date, base_time = get_weather(region, date_selected, KMA_API_KEY)
        weather = weather_fcst
    elif date_selected < today:
        ymd = date_selected.strftime("%Y%m%d")
        weather = get_asos_weather(region, ymd, ASOS_API_KEY)
    else:
        weather_fcst, base_date, base_time = get_weather(region, date_selected, KMA_API_KEY)
        lat0, lon0 = region_to_latlon[region]
        nx0, ny0 = convert_latlon_to_xy(lat0, lon0)
        ultra0 = _ultra_now_safe(nx0, ny0, KMA_API_KEY)
        weather = {
            "TMX": weather_fcst.get("TMX"),
            "TMN": weather_fcst.get("TMN"),
            "REH": ultra0.get("REH") if ultra0.get("REH") is not None else weather_fcst.get("REH"),
        }

    # ---------- 자치구 선택 (자동/수동 + 최근접 보정) ----------
    params = st.query_params
    q_lat = params.get("lat", None)
    q_lon = params.get("lon", None)

    seoul_gus = [
        '종로구','중구','용산구','성동구','광진구','동대문구','중랑구','성북구','강북구','도봉구',
        '노원구','은평구','서대문구','마포구','양천구','강서구','구로구','금천구','영등포구',
        '동작구','관악구','서초구','강남구','송파구','강동구'
    ]
    gu_centers = {
        '종로구': (37.5731,126.9793), '중구': (37.5636,126.9976), '용산구': (37.5323,126.9907), '성동구': (37.5634,127.0368),
        '광진구': (37.5384,127.0823), '동대문구': (37.5744,127.0396), '중랑구': (37.6063,127.0927), '성북구': (37.5894,127.0167),
        '강북구': (37.6396,127.0259), '도봉구': (37.6688,127.0471), '노원구': (37.6542,127.0568), '은평구': (37.6176,126.9227),
        '서대문구': (37.5792,126.9368), '마포구': (37.5663,126.9018), '양천구': (37.5169,126.8665), '강서구': (37.5509,126.8495),
        '구로구': (37.4954,126.8879), '금천구': (37.4568,126.8956), '영등포구': (37.5264,126.8963), '동작구': (37.5126,126.9393),
        '관악구': (37.4784,126.9516), '서초구': (37.4836,127.0327), '강남구': (37.5172,127.0473), '송파구': (37.5145,127.1059),
        '강동구': (37.5301,127.1238)
    }

    detected_gu = None
    lat_f = lon_f = None
    if q_lat and q_lon:
        try:
            lat_f, lon_f = float(q_lat), float(q_lon)
            rg = _reverse_geocode_to_gu(lat_f, lon_f)
            if rg.get("city") == "서울특별시" and rg.get("gu") in seoul_gus:
                detected_gu = rg["gu"]
            else:
                detected_gu = _nearest_gu(lat_f, lon_f, gu_centers)
        except Exception:
            pass

    colA, colB = st.columns([1, 1])
    with colA:
        if detected_gu and detected_gu in seoul_gus:
            selected_gu = detected_gu
            st.success(f"내 위치 인식: {selected_gu}")
        else:
            selected_gu = st.selectbox("자치구 선택", seoul_gus, index=seoul_gus.index('송파구'))
    with colB:
        st.empty()  # (상단 기준시각은 이미 헤더에 표시)

    # ---------- 격자 좌표 산출 ----------
    if lat_f is None or lon_f is None:
        lat_f, lon_f = gu_centers[selected_gu]
    nx, ny = convert_latlon_to_xy(lat_f, lon_f)

    # ---------- 오늘일 때: 초단기실황 + TMX/TMN 재확인 & 기준시각 업데이트 ----------
    if date_selected == today:
        ultra = _ultra_now_safe(nx, ny, KMA_API_KEY)
        bd, bt = get_fixed_base_datetime(today)
        tmx_tmn = _today_tmx_tmn_safe(nx, ny, KMA_API_KEY, bd, bt)
        tmx = tmx_tmn.get("TMX") or ultra.get("T1H")
        tmn = tmx_tmn.get("TMN") or ultra.get("T1H")
        reh = ultra.get("REH") or weather.get("REH")

        # 응답 기준시각이 있으면 상단 표기 갱신
        if ultra.get("base_date") and ultra.get("base_time"):
            bdisp = f"{ultra['base_date']} {ultra['base_time'][:2]}:{ultra['base_time'][2:]}"
            colH2.markdown(
                f"<div style='text-align:right;color:#6b7280;'>기준시각(실황): {bdisp} KST</div>",
                unsafe_allow_html=True
            )
    else:
        tmx, tmn, reh = weather.get("TMX"), weather.get("TMN"), weather.get("REH")

    # ---------- 모델 입력 검증 ----------
    if not all(v is not None for v in [tmx, tmn, reh]):
        st.error("실시간 기상 입력을 충분히 확보하지 못했습니다. 잠시 후 다시 시도해주세요.")
        st.stop()

    # ---------- 예측 ----------
    pred, avg_temp, heat_index, input_df = predict_from_weather(tmx, tmn, reh)
    risk = get_risk_level(pred)

    # ---------- 출력 ----------
    st.markdown("#### 입력값(실시간)")
    st.dataframe(input_df, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("자치구", selected_gu)
    c2.metric("예측 환자 수(도시기준)", f"{pred:.2f}명")
    c3.metric("위험 등급", risk)

    st.info("무더위 휴식시간 준수, 수분·그늘·휴식 확보, 냉방 취약계층 보호를 권고합니다. 더 자세한 보상·피해점수는 Tab3에서 확인하세요.")

with tab3:
    with st.expander("이 탭에서는 무엇을 하나요?"):
        st.markdown("""
        이 탭은 HeatAI의 핵심 기능으로, 
        **예측 환자 수(P_pred)**, **실제 환자 수(P_real)**, 
        그리고 자치구별 고정된 지표(S, E)와 폭염 지속성(H)을 활용해
        **자치구별 피해점수를 정량적으로 계산**하고, 
        위험 등급과 **예상 보상금액**까지 시뮬레이션합니다.

        ---
        **수행 내용 요약**:

        1. 사회적 지표(S): 고령자 비율, 야외근로자 비율, 열취약 인구 비율
        2. 환경적 지표(E): 열섬지수, 녹지율, 냉방보급률 (표준화)
        3. P_pred: tab2에서 예측된 서울시 전체 온열 환자 수 
        4. P_real: tab1에서 입력된 자치구별 실제 온열 환자 수 
        5. H: 최근 7일간 최고체감온도 기반 폭염 지속 가중치 계산

        **사전 피해점수** = 100 * (0.25 * S + 0.25 * E + 0.5 * P_pred) * H  
        **사후 피해점수** = 100 * (0.2 * S + 0.2 * E + 0.5 * P_pred + 0.1 * P_real) * H
        """)

    # 함수 정의
    def calculate_heatwave_multiplier(temps):
        count_33 = count_35 = max_33 = max_35 = 0
        for t in temps:
            if t >= 33:
                count_33 += 1
                max_33 = max(max_33, count_33)
            else:
                count_33 = 0
            if t >= 35:
                count_35 += 1
                max_35 = max(max_35, count_35)
            else:
                count_35 = 0
        if max_35 >= 2:
            return 1.3
        elif max_33 >= 2:
            return 1.15
        else:
            return 1.0

    def calculate_damage_score_prescore(s, e, p_pred):
        return 100 * (0.25 * s + 0.25 * e + 0.5 * p_pred)

    def calculate_damage_score_final(s, e, p_pred, p_real):
        return 100 * (0.2 * s + 0.2 * e + 0.5 * p_pred + 0.1 * p_real)

    def score_to_grade(score):
        if score < 30: return "낮음"
        elif score < 40: return "보통"
        elif score < 50: return "높음"
        else: return "매우 높음"

    def calc_payout(score):
        if score < 30: return 0
        elif score < 40: return 5000
        elif score < 50: return 10000
        else: return 20000

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
        s_sum = merged_df["S"].sum()
        merged_df["P_pred_raw"] = total_pred * (merged_df["S"] / s_sum)
        merged_df["P_pred"] = (merged_df["P_pred_raw"] / 25) ** 0.5
        return merged_df

    def load_csv_with_fallback(path):
        for enc in ["utf-8-sig", "cp949", "euc-kr"]:
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise UnicodeDecodeError(f"인코딩 실패: {path}")

    def load_csv_from_github(filename):
        try:
            github_url =   f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/{GITHUB_BRANCH}/{filename}"
            r = requests.get(github_url)
            r.raise_for_status()
            return pd.read_csv(io.StringIO(r.text), encoding="utf-8-sig")
        except Exception as e:
            st.error(f"GitHub에서 {filename} 불러오기 실패: {e}")
            return pd.DataFrame()


    # 메인 실행
    try:
        # 날짜 선택 (단일 칼럼)
        today = datetime.date.today()
        min_date = today - datetime.timedelta(days=6)
        selected_date = st.date_input("분석 기준일 선택 (최근 7일)", today, min_value=min_date, max_value=today)
        ymd = selected_date.strftime("%Y-%m-%d")

        ml_data = load_csv_from_github("ML_asos_dataset.csv")
        if ml_data.empty:
            st.warning("기록된 학습 데이터가 없습니다. tab2에서 데이터를 먼저 저장해주세요.")
            st.stop()
        ml_data = ml_data[ml_data["일자"] == ymd]
        static_data = load_csv_with_fallback("seoul_static_data.csv")
        merged_all = pd.merge(static_data, ml_data, on="자치구", how="left")

        if merged_all.empty:
            st.warning("선택한 날짜의 데이터가 없습니다.")
            st.stop()

        df_total = load_csv_from_github("ML_asos_total_prediction.csv")
        pred_row = df_total[df_total["일자"] == ymd]

        if pred_row.empty:
            st.warning(f"{ymd} 예측값이 존재하지 않습니다. tab1에서 먼저 예측을 수행하세요.")
            st.stop()

        seoul_pred = float(pred_row["서울시예측환자수"].values[0])

        merged_all["S"] = merged_all.apply(calculate_social_index, axis=1)
        for col in ["열섬지수", "녹지율", "냉방보급률"]:
            merged_all[f"{col}_std"] = standardize_column(merged_all, col)
        merged_all["E"] = merged_all.apply(calculate_environment_index, axis=1)

        merged_all = distribute_pred_by_s(merged_all, seoul_pred)

        merged_all["피해점수_사전"] = merged_all.apply(
            lambda row: calculate_damage_score_prescore(
                row["S"], row["E"], row["P_pred"]
            ),
            axis=1
        )

        merged_all["피해점수"] = merged_all.apply(
            lambda row: calculate_damage_score_final(
                row["S"], row["E"], row["P_pred"], 1.0 if row["환자수"] >= 1 else 0.0
            ),
            axis=1
        )

        # 폭염 지속성 가중치 계산 및 반영
        heatwave_temps = merged_all.sort_values("일자").groupby("자치구")["최고체감온도(°C)"].apply(list)
        merged_all["H"] = merged_all["자치구"].map(lambda gu: calculate_heatwave_multiplier(heatwave_temps.get(gu, [])))
        merged_all["피해점수_사전"] *= merged_all["H"]
        merged_all["피해점수"] *= merged_all["H"]

        merged_all["위험등급"] = merged_all["피해점수"].apply(score_to_grade)
        merged_all["보상금"] = merged_all["피해점수"].apply(calc_payout)

        col1, col2 = st.columns(2)
        with col1:
            selected_gu = st.selectbox("자치구 선택", sorted(merged_all["자치구"].unique()))
        with col2:
            subs_count = st.number_input(f"{selected_gu} 가입자 수", min_value=0, step=1, key="subs_tab3")

        merged = merged_all[merged_all["자치구"] == selected_gu].copy()
        if merged.empty:
            st.warning("선택한 자치구에 해당하는 데이터가 없습니다.")
            st.stop()

        merged["가입자수"] = subs_count
        merged["예상총보상금"] = merged["보상금"] * subs_count
        st.success(f"예상 보상금액: {int(merged['예상총보상금'].sum()):,}원")

        show_cols = ["자치구", "피해점수_사전", "피해점수", "H", "위험등급", "보상금", "가입자수", "예상총보상금"]
        st.markdown("#### 자치구별 피해점수 비교")
        st.dataframe(
            merged[show_cols],
            use_container_width=True
        )

        st.markdown("#### 피해점수 분포 (사후 기준)")
        st.bar_chart(data=merged_all.set_index("자치구")["피해점수"])

    except Exception as e:
        st.error(f"피해점수 그래프 생성 중 오류 발생: {e}")
