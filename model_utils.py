import pandas as pd
import joblib
import math

# ✅ 모델 및 피처 로드
model = joblib.load("trained_model.pkl")
feature_names = joblib.load("feature_names.pkl")

# ✅ 1. Stull의 추정식 기반 습구온도(Tw) 계산 함수
def compute_tw_stull(ta, rh):
    """
    ta: 기온 (°C)
    rh: 상대습도 (%)
    return: Tw (습구온도)
    """
    try:
        tw = (
            ta * math.atan(0.151977 * math.sqrt(rh + 8.313659))
            + math.atan(ta + rh)
            - math.atan(rh - 1.67633)
            + 0.00391838 * math.pow(rh, 1.5) * math.atan(0.023101 * rh)
            - 4.686035
        )
        return round(tw, 3)
    except:
        return None

# ✅ 2. 기상청 체감온도 계산 (2022 개정식)
def compute_heat_index_kma2022(ta, rh):
    """
    ta: 기온 (°C)
    rh: 상대습도 (%)
    return: 체감온도 (°C)
    """
    tw = compute_tw_stull(ta, rh)
    if tw is None:
        return None

    heat_index = (
        -0.2442
        + 0.55399 * tw
        + 0.45535 * ta
        - 0.0022 * (tw ** 2)
        + 0.00278 * tw * ta
        + 3.0
    )
    return round(heat_index, 1)

# ✅ 3. 예측 함수 (기상 정보 → 예측 환자 수)
def predict_from_weather(tmx, tmn, reh):
    """
    tmx: 최고기온 (°C)
    tmn: 최저기온 (°C)
    reh: 평균상대습도 (%)
    return: (예측 환자 수, 평균기온, 체감온도, 입력데이터프레임)
    """
    avg_temp = round((tmx + tmn) / 2, 1)
    heat_index = compute_heat_index_kma2022(tmx, reh)

    if heat_index is None:
        raise ValueError("❌ 체감온도 계산 실패")

    input_df = pd.DataFrame([{
        "최고체감온도(°C)": heat_index,
        "최고기온(°C)": tmx,
        "평균기온(°C)": avg_temp,
        "최저기온(°C)": tmn,
        "평균상대습도(%)": reh
    }])

    X = input_df[feature_names]
    pred = model.predict(X)[0]
    return pred, avg_temp, heat_index, input_df
