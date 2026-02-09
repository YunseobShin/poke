# app.py
import os
import random
from datetime import datetime, timedelta

import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# OpenAI (official SDK)
# pip install openai
from openai import OpenAI


# =========================
# Boot
# =========================
load_dotenv()

st.set_page_config(page_title="AI 습관 트래커 (포켓몬)", page_icon="🎮", layout="wide")


# =========================
# Helpers
# =========================
HABITS = [
    ("🌅", "기상 미션"),
    ("💧", "물 마시기"),
    ("📚", "공부/독서"),
    ("🏃", "운동하기"),
    ("😴", "수면"),
]

CITIES = [
    "Seoul",
    "Busan",
    "Incheon",
    "Daegu",
    "Daejeon",
    "Gwangju",
    "Ulsan",
    "Suwon",
    "Changwon",
    "Jeju",
]

COACH_STYLES = {
    "스파르타 코치": "엄격하고 단호한 톤. 변명 컷. 행동 중심. 짧고 강하게.",
    "따뜻한 멘토": "공감과 격려 중심. 작은 성취를 칭찬. 현실적인 조언.",
    "게임 마스터": "RPG 내레이션. 퀘스트/경험치/보상 느낌. 몰입감 있게.",
}

STAT_LABELS_KO = {
    "hp": "HP",
    "attack": "공격",
    "defense": "방어",
    "special-attack": "특수공격",
    "special-defense": "특수방어",
    "speed": "스피드",
}


def safe_get(url: str, timeout: int = 10):
    try:
        return requests.get(url, timeout=timeout)
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=60 * 10)
def get_weather(city: str, api_key: str):
    """
    OpenWeatherMap 현재 날씨
    - 한국어
    - 섭씨
    실패 시 None
    """
    if not api_key:
        return None
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric",
        "lang": "kr",
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        j = r.json()
        return {
            "city": city,
            "temp_c": j.get("main", {}).get("temp"),
            "feels_like_c": j.get("main", {}).get("feels_like"),
            "humidity": j.get("main", {}).get("humidity"),
            "desc": (j.get("weather", [{}])[0] or {}).get("description"),
            "wind_mps": j.get("wind", {}).get("speed"),
            "icon": (j.get("weather", [{}])[0] or {}).get("icon"),
        }
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=60 * 60)
def get_pokemon():
    """
    PokeAPI: 1세대(1~151) 랜덤 포켓몬
    - 공식 아트워크 URL
    - 이름, 도감 번호, 타입, 스탯
    실패 시 None
    """
    pid = random.randint(1, 151)
    url = f"https://pokeapi.co/api/v2/pokemon/{pid}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        j = r.json()

        name = j.get("name")
        dex = j.get("id")
        types = [t["type"]["name"] for t in j.get("types", []) if "type" in t]

        stats = {}
        for s in j.get("stats", []):
            key = s.get("stat", {}).get("name")
            val = s.get("base_stat")
            if key and isinstance(val, int):
                stats[key] = val

        artwork = (
            j.get("sprites", {})
            .get("other", {})
            .get("official-artwork", {})
            .get("front_default")
        )

        return {
            "name": name,
            "dex": dex,
            "types": types,
            "stats": stats,
            "artwork": artwork,
        }
    except Exception:
        return None


def _coach_system_prompt(style_label: str) -> str:
    base = f"""
너는 'AI 습관 코치'다. 사용자의 오늘 습관 체크인 + 기분 + 날씨 + 포켓몬 정보를 보고,
행동을 유도하는 1일 리포트를 작성한다.

코치 스타일: {style_label}
스타일 가이드: {COACH_STYLES.get(style_label, "")}

출력 규칙:
- 반드시 아래 섹션을 순서대로 출력한다.
- 한국어로 작성한다.
- 군더더기 없이, 하지만 읽기 즐겁게.

출력 형식(그대로 유지):
1) 컨디션 등급: (S/A/B/C/D 중 하나) - 한 줄 코멘트
2) 습관 분석:
- 잘한 점 2개
- 아쉬운 점 1개
- 내일 1% 개선 액션 1개
3) 날씨 코멘트: (날씨/기온/체감/습도 중 2개 이상을 엮어서 현실적인 조언)
4) 내일 미션(체크박스 기반): 3개 (각각 1줄, 구체적으로)
5) 오늘의 파트너 포켓몬:
- 포켓몬: 이름(#도감번호)
- 타입:
- 스탯 하이라이트: (스탯 2개를 골라 숫자와 함께)
- 응원 멘트: (스탯을 은유로 연결해서, 한 문단)

등급 기준 힌트(너가 판단):
- S: 5개 습관 중 4~5개 + 기분 8~10
- A: 3~4개 + 기분 7~10
- B: 2~3개 + 기분 5~8
- C: 1~2개 또는 기분 3~5
- D: 0~1개 + 기분 1~3
""".strip()
    return base


def generate_report(
    openai_api_key: str,
    coach_style: str,
    habits_checked: list[str],
    mood: int,
    weather: dict | None,
    pokemon: dict | None,
):
    """
    OpenAI Responses API
    모델: gpt-5-mini
    실패 시 (None, error_message)
    """
    if not openai_api_key:
        return None, "OpenAI API Key가 필요합니다."

    w = weather or {}
    p = pokemon or {}

    weather_text = (
        f"- 도시: {w.get('city')}\n"
        f"- 날씨: {w.get('desc')}\n"
        f"- 기온(섭씨): {w.get('temp_c')}°C / 체감: {w.get('feels_like_c')}°C\n"
        f"- 습도: {w.get('humidity')}% / 바람: {w.get('wind_mps')} m/s\n"
        if weather
        else "- (날씨 정보 없음)\n"
    )

    pokemon_text = (
        f"- 이름: {p.get('name')} / 도감번호: {p.get('dex')}\n"
        f"- 타입: {', '.join(p.get('types', []) or [])}\n"
        f"- 스탯: {p.get('stats')}\n"
        if pokemon
        else "- (포켓몬 정보 없음)\n"
    )

    user_payload = f"""
[오늘 체크인]
- 완료한 습관: {', '.join(habits_checked) if habits_checked else '없음'}
- 기분(1~10): {mood}

[날씨]
{weather_text}

[포켓몬]
{pokemon_text}

주의:
- 포켓몬/날씨 정보가 없으면, 없는 상태에서도 설득력 있게 리포트를 작성해라.
""".strip()

    try:
        client = OpenAI(api_key=openai_api_key)
        resp = client.responses.create(
            model="gpt-5-mini",
            instructions=_coach_system_prompt(coach_style),
            input=user_payload,
            # 안전하게 텍스트 포맷 명시 (Responses API 레퍼런스 기준)
            text={"format": {"type": "text"}},
        )
        return (resp.output_text or "").strip(), None
    except Exception as e:
        return None, f"OpenAI 호출 실패: {e}"


def build_demo_week(today_rate: int, today_checked: int, today_mood: int):
    """
    데모용 6일 + 오늘 1일 = 7일 데이터
    """
    base = datetime.now().date()
    dates = [base - timedelta(days=d) for d in range(6, 0, -1)] + [base]

    # 샘플(6일)
    sample = [
        {"date": dates[0], "achv_rate": 40, "checked": 2, "mood": 5},
        {"date": dates[1], "achv_rate": 60, "checked": 3, "mood": 6},
        {"date": dates[2], "achv_rate": 80, "checked": 4, "mood": 7},
        {"date": dates[3], "achv_rate": 20, "checked": 1, "mood": 4},
        {"date": dates[4], "achv_rate": 60, "checked": 3, "mood": 6},
        {"date": dates[5], "achv_rate": 40, "checked": 2, "mood": 5},
    ]
    # 오늘
    sample.append({"date": dates[6], "achv_rate": today_rate, "checked": today_checked, "mood": today_mood})
    df = pd.DataFrame(sample)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return df


def type_ko(t: str) -> str:
    # 최소한의 감성 번역(필요하면 확장)
    mapping = {
        "grass": "풀",
        "fire": "불꽃",
        "water": "물",
        "bug": "벌레",
        "normal": "노말",
        "poison": "독",
        "electric": "전기",
        "ground": "땅",
        "fairy": "페어리",
        "fighting": "격투",
        "psychic": "에스퍼",
        "rock": "바위",
        "ghost": "고스트",
        "ice": "얼음",
        "dragon": "드래곤",
        "flying": "비행",
        "steel": "강철",
        "dark": "악",
    }
    return mapping.get(t, t)


# =========================
# Sidebar: API Keys
# =========================
with st.sidebar:
    st.header("🔑 API Keys")

    env_openai = os.getenv("OPENAI_API_KEY", "")
    env_weather = os.getenv("OPENWEATHER_API_KEY", "") or os.getenv("OPENWEATHERMAP_API_KEY", "")

    openai_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.get("openai_key", env_openai),
        placeholder="sk-...",
    )
    weather_key = st.text_input(
        "OpenWeatherMap API Key",
        type="password",
        value=st.session_state.get("weather_key", env_weather),
        placeholder="OWM key",
    )

    st.session_state["openai_key"] = openai_key
    st.session_state["weather_key"] = weather_key

    st.caption("-.env에서도 자동 로드됩니다 (OPENAI_API_KEY / OPENWEATHER_API_KEY).")


# =========================
# Main UI
# =========================
st.title("🎮 AI 습관 트래커 (포켓몬)")
st.write("오늘의 습관을 체크하고 - 날씨/포켓몬/AI 코치 리포트로 하루를 정리해보자.")


st.subheader("✅ 습관 체크인")

c1, c2 = st.columns(2)
habit_state = {}

# 2열 배치: 5개를 번갈아 배치
for i, (emoji, label) in enumerate(HABITS):
    target_col = c1 if i % 2 == 0 else c2
    with target_col:
        habit_state[label] = st.checkbox(f"{emoji} {label}", value=False)

mood = st.slider("🙂 기분(1~10)", min_value=1, max_value=10, value=6)

sel1, sel2 = st.columns([1, 1])
with sel1:
    city = st.selectbox("🏙️ 도시 선택", CITIES, index=0)
with sel2:
    coach_style = st.radio("🧭 코치 스타일", list(COACH_STYLES.keys()), horizontal=True)

checked_habits = [k for k, v in habit_state.items() if v]
checked_count = len(checked_habits)
achv_rate = int(round((checked_count / len(HABITS)) * 100, 0))

st.divider()

# =========================
# Metrics + Weekly Chart
# =========================
m1, m2, m3 = st.columns(3)
m1.metric("달성률", f"{achv_rate}%")
m2.metric("달성 습관", f"{checked_count}/{len(HABITS)}")
m3.metric("기분", f"{mood}/10")

df_week = build_demo_week(achv_rate, checked_count, mood)

st.subheader("📊 최근 7일 달성률")
st.bar_chart(df_week[["achv_rate"]], height=220)

st.divider()

# =========================
# Fetch Weather & Pokemon (on-demand but cheap)
# =========================
weather = get_weather(city, st.session_state.get("weather_key", ""))
pokemon = get_pokemon()

# =========================
# Generate Report
# =========================
st.subheader("🧠 AI 코치 리포트")

btn = st.button("컨디션 리포트 생성", type="primary", use_container_width=True)

if btn:
    with st.spinner("리포트 생성 중..."):
        report, err = generate_report(
            openai_api_key=st.session_state.get("openai_key", ""),
            coach_style=coach_style,
            habits_checked=checked_habits,
            mood=mood,
            weather=weather,
            pokemon=pokemon,
        )

    if err:
        st.error(err)
    else:
        # 2열: 날씨 카드 + 포켓몬 카드
        left, right = st.columns(2)

        with left:
            st.markdown("### ☁️ 오늘의 날씨")
            if weather:
                icon = weather.get("icon")
                icon_url = f"https://openweathermap.org/img/wn/{icon}@2x.png" if icon else None
                if icon_url:
                    st.image(icon_url, width=72)
                st.markdown(
                    f"""
- **도시**: {weather.get("city")}
- **날씨**: {weather.get("desc")}
- **기온**: {weather.get("temp_c")}°C (체감 {weather.get("feels_like_c")}°C)
- **습도**: {weather.get("humidity")}%
- **바람**: {weather.get("wind_mps")} m/s
""".strip()
                )
            else:
                st.info("날씨 정보를 가져오지 못했어요 - OpenWeatherMap API Key/도시를 확인해줘.")

        with right:
            st.markdown("### 🧩 오늘의 포켓몬")
            if pokemon:
                name = pokemon.get("name") or "unknown"
                dex = pokemon.get("dex") or "?"
                types = pokemon.get("types") or []
                types_ko = [type_ko(t) for t in types]

                st.markdown(f"**{name} (#{dex})**  -  타입: `{', '.join(types_ko) if types_ko else 'N/A'}`")

                if pokemon.get("artwork"):
                    st.image(pokemon["artwork"], use_container_width=True)

                # 스탯 바 차트 (빨간색)
                stats = pokemon.get("stats") or {}
                stat_items = []
                for k, v in stats.items():
                    if k in STAT_LABELS_KO:
                        stat_items.append({"stat": STAT_LABELS_KO[k], "value": v})

                if stat_items:
                    import altair as alt

                    df_stats = pd.DataFrame(stat_items)
                    chart = (
                        alt.Chart(df_stats)
                        .mark_bar(color="red")
                        .encode(
                            x=alt.X("value:Q", title="스탯"),
                            y=alt.Y("stat:N", sort="-x", title=""),
                            tooltip=["stat", "value"],
                        )
                        .properties(height=220)
                    )
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.caption("스탯 데이터가 비어있어요.")
            else:
                st.info("포켓몬 정보를 가져오지 못했어요 - 네트워크 상태를 확인해줘.")

        st.divider()

        st.markdown("### 📝 AI 리포트")
        st.write(report)

        # 공유용 텍스트
        share = f"""[AI 습관 트래커 - 오늘의 컨디션]
- 달성률: {achv_rate}% ({checked_count}/{len(HABITS)})
- 완료: {', '.join(checked_habits) if checked_habits else '없음'}
- 기분: {mood}/10
- 도시: {city}

{report}
""".strip()

        st.markdown("### 📌 공유용 텍스트")
        st.code(share, language="text")

# =========================
# Footer: API Guide
# =========================
with st.expander("📎 API 안내 (키 발급/사용)"):
    st.markdown(
        """
**OpenAI API**
- 환경변수: `OPENAI_API_KEY`
- 사이드바에 입력하면 앱에서 바로 사용

**OpenWeatherMap API**
- 환경변수: `OPENWEATHER_API_KEY` (또는 `OPENWEATHERMAP_API_KEY`)
- 현재 날씨 API를 사용 - 도시(영문) 기준으로 조회

**PokeAPI**
- 키 필요 없음
- 1세대(1~151) 랜덤 포켓몬 데이터 조회
"""
    )
