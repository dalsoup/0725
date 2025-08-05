import requests
import pandas as pd
import datetime
import math

# ----------------------- 파라미터 정렬 -----------------------
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

# ----------------------- 체감온도 산식 (기상청 2022 개정) -----------------------
def compute_tw_stull(ta, rh):
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

def compute_heat_index_kma2022(ta, rh):
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

# ----------------------- 날씨 API 함수들 -----------------------
def get_weather(region_name, target_date, KMA_API_KEY):
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

def get_asos_weather(region, ymd, ASOS_API_KEY):
    stn_id = region_to_stn_id[region]
    url = f"http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList"
    params = {
        "serviceKey": ASOS_API_KEY,
        "pageNo": 1,
        "numOfRows": 10,
        "dataType": "JSON",
        "dataCd": "ASOS",
        "dateCd": "DAY",
        "startDt": ymd,
        "endDt": ymd,
        "stnIds": stn_id
    }
    try:
        r = requests.get(url, params=params, timeout=10, verify=False)
        item = r.json().get("response", {}).get("body", {}).get("items", {}).get("item", [])[0]
        return {
            "TMX": float(item["maxTa"]),
            "TMN": float(item["minTa"]),
            "REH": float(item["avgRhm"])
        }
    except:
        return {}

def get_risk_level(pred):
    if pred == 0: return "🟢 매우 낮음"
    elif pred <= 2: return "🟡 낮음"
    elif pred <= 5: return "🟠 보통"
    elif pred <= 10: return "🔴 높음"
    else: return "🔥 매우 높음"

def calculate_avg_temp(tmx, tmn):
    if tmx is not None and tmn is not None:
        return round((tmx + tmn) / 2, 1)
    return None

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
