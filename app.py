import os
import re
import io
import json
import time
import base64
import html
import uuid
import random
import copy
import hashlib
import functools
import math
import zipfile
import threading
import pandas as pd
from typing import List, Dict

import streamlit as st
import streamlit.components.v1 as components
from google import genai
from google.genai import types

try:
    from supabase import create_client
except ImportError:
    create_client = None

# -----------------------------
# 1. 기본 설정, 비밀번호 잠금 및 API 보안 로드
# -----------------------------
st.set_page_config(page_title="4pm 올인원 교재 시스템", layout="wide")

def get_secret(key: str, default: str = "") -> str:
    try:
        if key in st.secrets:
            return str(st.secrets[key]).strip()
    except Exception:
        pass
    return os.environ.get(key, default).strip()

APP_PASSWORD = get_secret("APP_PASSWORD", "")
if APP_PASSWORD:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if not st.session_state.authenticated:
        st.markdown("## 🔒 4pm 올인원 교재 시스템 (전용 웹 스튜디오)")
        pwd_input = st.text_input("접속 비밀번호를 입력하세요", type="password")
        if st.button("로그인", type="primary"):
            if pwd_input == APP_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("❌ 비밀번호가 일치하지 않습니다.")
        st.stop()

LOCAL_FALLBACK_KEY = ""  # 로컬 PC 단독 실행 시에만 입력
API_KEY = get_secret("GEMINI_API_KEY", LOCAL_FALLBACK_KEY)

# -----------------------------
# 1-B. AI 제공사 (Gemini · OpenAI GPT · Anthropic Claude)
#   - Secrets에 키가 있는 제공사만 사이드바에 나오고, 그 키로 쓸 수 있는 모델 목록을 자동으로 불러옴
#   - 모델 지정은 '제공사:모델' 문자열 (예: openai:gpt-5, anthropic:claude-opus-5-5). 예전 설정처럼 앞부분이 없으면 Gemini
#   - 이미지 생성은 Gemini 이미지 모델만 지원 (Claude는 이미지 생성 기능이 없음)
# -----------------------------
AI_PROVIDERS = {"gemini": "Google Gemini", "openai": "OpenAI GPT", "anthropic": "Anthropic Claude"}
AI_KEYS = {
    "gemini": API_KEY if API_KEY != "여기에_선생님의_API_키를_넣어주세요" else "",
    "openai": get_secret("OPENAI_API_KEY", ""),
    "anthropic": get_secret("ANTHROPIC_API_KEY", ""),
}
AI_DEFAULT_MODELS = {
    "gemini": ["gemini-2.5-flash", "gemini-3.5-flash", "gemini-3.1-pro", "gemini-3.1-flash-lite"],
    "openai": ["gpt-6.1-sol", "gpt-6.1-terra", "gpt-5", "gpt-5-mini"],
    "anthropic": ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"],
}
# Claude 안전 필터가 요청을 거절하면 같은 요청을 다른 모델로 다시 처리 (서버 측 대체 모델)
CLAUDE_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1"}
# 쉬운 작업(변형문제 하·중 등)에 쓰는 같은 제공사의 가벼운 모델 (값이 대략 1/2~1/5)
AI_LIGHT_MODELS = {"gemini": "gemini-3.5-flash-lite", "openai": "gpt-6.1-terra", "anthropic": "claude-sonnet-5-5"}

# 변형문제 모델 고르기 참고표 — 2026-11-03 같은 지문([2024년 10월 고1] 37번)으로 삽입 최상(제일고)·어법 조합형 최상(운양고)·빈칸 상(제일고)을
# 모델마다 만들게 한 뒤, Claude Opus 5.5와 GPT-6.1 Sol이 누가 만든지 모르게 채점 (정답·유일성·지문 충실·스타일·난이도 1~5)
AI_MODEL_GUIDE = [
    # 모델, 문제당 원, 정답 일치, 지문 충실, 스타일(5점), 난이도(5점), 한 줄 평
    ("openai:gpt-6.1-sol", 42, "100%", "100%", 3.5, 2.0, "⭐ 추천 · 스타일 점수 1위, 정답·지문 모두 정확, 값도 중간"),
    ("openai:gpt-6.1-terra", 20, "100%", "100%", 3.2, 2.0, "💰 가성비 · Sol보다 약간 단순하지만 정확, 하·중에 알맞음"),
    ("anthropic:claude-sonnet-5-5", 75, "100%", "100%", 3.3, 2.0, "정확하고 지문 충실 · 값이 GPT-6.1 Sol의 약 2배"),
    ("anthropic:claude-opus-5-5", 110, "83%", "67%", 3.3, 2.0, "가장 비쌈 · 이번 시험에서는 원문 문장을 바꿔 해석한 경우가 있음"),
    ("gemini:gemini-3.8-flash", 17, "100%", "17%", 3.3, 2.0, "싸지만 지문을 통째로 다시 씀(원문 표현 0~59%) → 운양고(원문 거의 그대로)와 안 맞음"),
    ("gemini:gemini-3.5-flash", 35, "100%", "67%", 3.0, 1.7, "3.8 Flash보다 비싸고 지문 재작성이 심함 · 바꾸는 것을 권장"),
    ("gemini:gemini-3.1-pro-preview", 70, "83%", "33%", 3.2, 2.0, "어법 조합형 정답이 틀림 · 지문 변형 심함"),
    ("anthropic:claude-haiku-5-5", 5, "100%", "100%", 2.8, 1.3, "매우 쌈 · 빈칸 선택지 비문 1건, 하 난이도 전용"),
    ("openai:gpt-6-luna", 3, "100%", "67%", 2.5, 1.7, "매우 쌈 · 밑줄 기호 누락 등 형식 오류, 비추천"),
]

def render_model_guide():
    """사이드바: 변형문제용 모델 고르기 참고표"""
    with st.expander("📊 모델 고르기 참고표 (변형문제)", expanded=False):
        st.dataframe(pd.DataFrame([{"모델": ai_model_label(m), "문제당": f"약 {won}원", "정답": acc, "지문 충실": fid, "스타일": sty, "난이도": dif, "평": note}
                                   for m, won, acc, fid, sty, dif, note in AI_MODEL_GUIDE]), hide_index=True, use_container_width=True)
        st.caption("2026-11-03 같은 지문으로 3문제(삽입 최상·어법 조합형 최상·빈칸 상)씩 만들어 Claude Opus와 GPT-6.1 Sol이 채점했어요. "
                   "지문 충실 = 원문 정보를 빼거나 뜻을 바꾸지 않은 비율. 표본이 작아 참고용이에요. "
                   "난이도 점수는 모든 모델이 낮았어요(5점 중 2점 안팎) — 모델보다 출제 방식을 보강해야 하는 부분이에요.")

def light_model_for(model_spec: str) -> str:
    provider, model = split_model_spec(model_spec)
    light = AI_LIGHT_MODELS.get(provider, "")
    return f"{provider}:{light}" if light and model != light else model_spec

if not any(AI_KEYS.values()):
    st.error("AI API 키를 찾지 못했습니다. Streamlit Cloud의 Secrets에 GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY 중 하나 이상을 등록해주세요.")
    st.stop()

@st.cache_resource
def get_genai_client(api_key: str):
    return genai.Client(api_key=api_key)

client = get_genai_client(API_KEY) if AI_KEYS["gemini"] else None

@functools.lru_cache(maxsize=4)
def _openai_client(api_key: str):
    from openai import OpenAI  # 쓸 때만 불러옴
    return OpenAI(api_key=api_key)

@functools.lru_cache(maxsize=4)
def _gemini_batch_client(api_key: str):
    """반값 모드용 Gemini 연결 (한 번 만든 것을 계속 씀 — 그때그때 만들면 요청 중에 닫혀 버림)"""
    return genai.Client(api_key=api_key)

@functools.lru_cache(maxsize=4)
def _anthropic_client(api_key: str):
    import anthropic  # 쓸 때만 불러옴
    return anthropic.Anthropic(api_key=api_key)

class AiFile:
    """AI에 함께 보내는 파일 (PDF·이미지). 제공사마다 알맞은 형식으로 바꿔 보냄"""
    def __init__(self, data: bytes, mime: str, name: str = "file"):
        self.data, self.mime, self.name = data, mime, name

def ai_model_label(spec: str) -> str:
    """'anthropic:claude-opus-5-5' → 'Anthropic Claude · claude-opus-5-5'"""
    p, m = split_model_spec(spec)
    return f"{AI_PROVIDERS.get(p, p)} · {m}"

def split_model_spec(spec: str) -> (str, str):
    p, _, m = str(spec or "").partition(":")
    return (p, m) if m and p in AI_PROVIDERS else ("gemini", str(spec or ""))

def _schema_closed(schema):
    """Claude 구조화 출력용: 모든 object에 additionalProperties: false"""
    if isinstance(schema, dict):
        out = {k: _schema_closed(v) for k, v in schema.items()}
        if out.get("type") == "object":
            out.setdefault("additionalProperties", False)
        return out
    if isinstance(schema, list):
        return [_schema_closed(v) for v in schema]
    return schema

GEMINI_INLINE_MAX_BYTES = 15_000_000  # 이보다 큰 PDF는 Gemini 파일 업로드로 보냄

def _gemini_part(gc, f: AiFile):
    if len(f.data) <= GEMINI_INLINE_MAX_BYTES:
        return types.Part.from_bytes(data=f.data, mime_type=f.mime)
    up = gc.files.upload(file=io.BytesIO(f.data), config=types.UploadFileConfig(mime_type=f.mime, display_name=f.name[:100]))
    for _ in range(60):
        if str(getattr(up, "state", "")).upper().endswith("ACTIVE"):
            break
        time.sleep(3)
        up = gc.files.get(name=up.name)
    return up

# ---- AI 사용량·비용 ----
#   호출마다 토큰(입력·캐시로 읽은 입력·캐시에 새로 쓴 입력·출력)을 모아 달마다 모델별로 더해 DB(ai_usage 문서)에 저장
#   단가(1M 토큰당 달러)는 2026-11 기준 공개 가격. 금액은 화면에 보일 때 토큰 수 × 지금 단가로 다시 계산 (단가를 고치면 지난 기록도 맞게 바뀜)
#   정확한 청구액은 각 회사 콘솔(결제 화면)이 기준 (부가세 별도)
AI_PRICES = [  # (모델 이름에 들어 있는 글자, 입력, 캐시 읽기, 출력) — 위에서부터 먼저 맞는 것을 씀
    # 이미지 모델: 출력 단가는 '이미지 출력' 단가 (1K 이미지 1장 = 1120토큰 → 3.1 flash 약 $0.067 · 3 pro 약 $0.134, 2026-11 Google 공개 가격)
    ("gemini-3.1-flash-image", 0.50, 0.05, 60.0), ("gemini-3-pro-image", 2.0, 0.20, 120.0),
    ("claude-fable-5-1", 10, 0.25, 50), ("claude-fable", 10, 1.0, 50), ("claude-mythos", 10, 1.0, 50),
    ("claude-opus-5-5", 4, 0.20, 20), ("claude-opus", 5, 0.50, 25), ("claude-sonnet", 2, 0.20, 10), ("claude-haiku", 0.10, 0.01, 0.50),
    ("gpt-6.1-sol", 2.0, 0.10, 10.0), ("gpt-6.1-terra", 1.0, 0.05, 5.0), ("gpt-6-astra", 10.0, 1.0, 50.0), ("gpt-6-sol", 2.0, 0.20, 10.0),
    ("gpt-6-luna", 0.10, 0.01, 0.50), ("gpt-5.6-sol", 4.0, 0.40, 20.0), ("gpt-5.6-terra", 2.0, 0.20, 12.0), ("gpt-5.6-luna", 0.20, 0.02, 1.20),
    ("gpt-5.5", 5.0, 0.50, 30.0), ("gpt-5.4-mini", 0.75, 0.075, 4.50), ("gpt-5.4-nano", 0.20, 0.02, 1.25), ("gpt-5.4", 2.50, 0.25, 15.0),
    ("gpt-5-nano", 0.05, 0.005, 0.40), ("gpt-5-mini", 0.25, 0.025, 2.0), ("gpt-5", 1.25, 0.125, 10.0),
    ("gpt-4.1-mini", 0.40, 0.10, 1.60), ("gpt-4o-mini", 0.15, 0.075, 0.60), ("gpt-4", 2.0, 0.50, 8.0), ("o4-mini", 1.10, 0.275, 4.40), ("o3", 2.0, 0.50, 8.0),
    ("gemini-3.8-flash", 0.75, 0.075, 3.75),  # 2026-12-31까지 할인가 (2027년부터 1.50 / 0.15 / 7.50)
    ("gemini-3.7-flash", 1.50, 0.15, 7.50), ("gemini-3.6-flash", 1.50, 0.15, 7.50),
    ("gemini-3.5-flash-lite", 0.30, 0.03, 2.50), ("gemini-3.5-flash", 1.50, 0.15, 9.0), ("gemini-3.1-flash-lite", 0.25, 0.025, 1.50),
    ("gemini-3.1-pro", 2.0, 0.20, 12.0), ("gemini-3-flash", 0.50, 0.05, 3.0), ("gemini-2.5-pro", 1.25, 0.125, 10.0),
    ("gemini-2.5-flash-lite", 0.10, 0.01, 0.40), ("gemini-2.5-flash", 0.30, 0.03, 2.50),
    ("flash-lite", 0.30, 0.03, 2.50), ("flash", 1.50, 0.15, 9.0), ("pro", 2.0, 0.20, 12.0),
]
AI_PRICE_DEFAULT = (2.0, 0.20, 10.0)
AI_USAGE_FILE = "ai_usage.json"  # DB 문서 이름 ai_usage (로컬 실행이면 같은 이름 파일)
LLM_USAGE: List[Dict] = []  # 이번 실행에서 최근 호출 기록 (시험용)

def ai_price(model: str):
    m = str(model).lower()
    return next((t[1:] for t in AI_PRICES if t[0] in m), AI_PRICE_DEFAULT)

BATCH_TAG = " (batch)"  # 반값 모드로 쓴 사용량은 모델 이름 뒤에 이 표시를 붙여 따로 기록 (금액 절반으로 계산)

def ai_cost_usd(model: str, inp: int, cached: int, write: int, out: int) -> float:
    """inp는 캐시 읽기·쓰기를 포함한 전체 입력. Claude 캐시 쓰기는 입력 단가의 1.25배 · 반값 모드(Batch API)는 모두 절반"""
    p_in, p_cache, p_out = ai_price(model)
    plain = max(0, inp - cached - write)
    usd = (plain * p_in + cached * p_cache + write * p_in * 1.25 + out * p_out) / 1_000_000
    return usd * 0.5 if BATCH_TAG in str(model) else usd

@st.cache_resource
def _usage_buffer() -> Dict:
    """아직 DB에 쓰지 않은 사용량 (재실행·백그라운드 작업이 함께 쓰는 하나뿐인 통)"""
    return {"lock": threading.Lock(), "io": threading.Lock(), "pending": {}, "last": time.time()}

def flush_ai_usage():
    """모아 둔 사용량을 DB의 달별·모델별 합계에 더함 (실패하면 다음에 다시)"""
    buf = _usage_buffer()
    with buf["io"]:
        with buf["lock"]:
            pending, buf["pending"], buf["last"] = buf["pending"], {}, time.time()
        if not pending:
            return
        try:
            sb = globals().get("supabase") if globals().get("IS_CLOUD_DB") else None
            path = os.path.join(globals().get("BASE_DIR", "."), AI_USAGE_FILE)
            doc = _store_get(sb, path, {}) or {}
            months = doc.setdefault("months", {})
            for month, models in pending.items():
                for spec, d in models.items():
                    tot = months.setdefault(month, {}).setdefault(spec, {})
                    for k, v in d.items():
                        tot[k] = round(tot.get(k, 0) + v, 6)
            doc["updated_at"] = time.strftime("%Y-%m-%d %H:%M")
            _store_put(sb, path, doc)
        except Exception:
            with buf["lock"]:
                for month, models in pending.items():
                    for spec, d in models.items():
                        tot = buf["pending"].setdefault(month, {}).setdefault(spec, {})
                        for k, v in d.items():
                            tot[k] = tot.get(k, 0) + v

def _log_usage(provider: str, model: str, inp: int, cached: int, out: int, write: int = 0):
    inp, cached, out, write = int(inp or 0), int(cached or 0), int(out or 0), int(write or 0)
    usd = ai_cost_usd(model, inp, cached, write, out)
    LLM_USAGE.append({"provider": provider, "model": model, "in": inp, "cached": cached, "write": write, "out": out, "usd": usd, "t": time.time()})
    del LLM_USAGE[:-500]
    try:
        buf = _usage_buffer()
        with buf["lock"]:
            d = buf["pending"].setdefault(time.strftime("%Y-%m"), {}).setdefault(f"{provider}:{model}", {})
            for k, v in (("calls", 1), ("in", inp), ("cached", cached), ("write", write), ("out", out), ("usd", usd)):
                d[k] = d.get(k, 0) + v
            due = time.time() - buf["last"] > 30
        if due:
            flush_ai_usage()
    except Exception:
        pass

def load_ai_usage() -> Dict:
    """DB 합계 + 아직 쓰지 않은 사용량"""
    flush_ai_usage()
    return _store_get(supabase if IS_CLOUD_DB else None, os.path.join(BASE_DIR, AI_USAGE_FILE), {}) or {}

@st.cache_data(ttl=30, show_spinner=False)
def _ai_usage_cached() -> Dict:
    return load_ai_usage()

def render_ai_spend():
    """사이드바: 이번 달·전체 AI 사용 금액 (추정)"""
    months = (_ai_usage_cached().get("months") or {})
    this = months.get(time.strftime("%Y-%m"), {})
    cost = lambda spec, d: ai_cost_usd(split_model_spec(spec)[1], d.get("in", 0), d.get("cached", 0), d.get("write", 0), d.get("out", 0))
    usd_month = sum(cost(k, d) for k, d in this.items())
    usd_all = sum(cost(k, d) for m in months.values() for k, d in m.items())
    with st.expander(f"💵 AI 사용 금액 · 이번 달 ${usd_month:,.2f}", expanded=False):
        if not this:
            st.caption("이번 달에는 아직 기록이 없어요. (이 기능을 넣은 뒤부터 기록됩니다)")
        for spec, d in sorted(this.items(), key=lambda x: -cost(*x)):
            hit = d.get("cached", 0) / max(1, d.get("in", 0))
            st.markdown(f"<b>{html.escape(ai_model_label(spec))}</b><br>${cost(spec, d):,.2f} · {int(d.get('calls', 0))}번 · "
                        f"입력 {d.get('in', 0) / 1000:,.0f}K(캐시 {hit:.0%}) · 출력 {d.get('out', 0) / 1000:,.0f}K", unsafe_allow_html=True)
        st.caption(f"기록한 뒤 전체 ${usd_all:,.2f} (약 {usd_all * 1400:,.0f}원) · 공개 단가로 계산한 추정치예요. 정확한 청구액은 각 회사 결제 화면에서 확인하세요.")

@st.cache_resource
def _gemini_cache_store() -> Dict:
    return {"lock": threading.Lock(), "caches": {}}

GEMINI_CACHE_TTL = 900  # Gemini 명시 캐시 유지 시간(초). 같은 앞부분을 이 시간 안에 다시 쓰면 입력의 약 1/10 값

def _gemini_cached_prefix(gc, model: str, text: str) -> str:
    """앞부분 글을 Gemini 캐시에 올려 두고 이름을 돌려줌 (자동 캐시는 잘 안 맞아서 직접 만듦)"""
    store = _gemini_cache_store()
    k = hashlib.md5(f"{model}\n{text}".encode("utf-8")).hexdigest()
    with store["lock"]:
        hit = store["caches"].get(k)
        if hit and hit["exp"] > time.time() + 60:
            return hit["name"]
        c = gc.caches.create(model=model, config=types.CreateCachedContentConfig(
            contents=[types.Content(role="user", parts=[types.Part(text=text)])], ttl=f"{GEMINI_CACHE_TTL}s"))
        store["caches"][k] = {"name": c.name, "exp": time.time() + GEMINI_CACHE_TTL}
        return c.name

def llm_json(model_spec: str, contents, schema: Dict, temperature: float = 0.3, gemini_client=None, cache_first: bool = False) -> Dict:
    """JSON 스키마에 맞춘 답을 받아 dict로 돌려줌. contents: 글(str)과 AiFile의 목록 (또는 글 하나)
    cache_first=True: 첫 글 조각이 여러 호출에서 똑같은 앞부분이라는 뜻
      Claude: 캐시 표시를 붙여 다음 호출부터 약 1/20 값 · GPT: 첫 조각을 instructions로 따로 보내 자동 캐시(약 1/20) · Gemini: 명시 캐시를 만들어 약 1/10"""
    provider, model = split_model_spec(model_spec)
    items = contents if isinstance(contents, list) else [contents]
    key = AI_KEYS.get(provider, "")
    if not key:
        raise RuntimeError(f"{AI_PROVIDERS.get(provider, provider)} API 키가 Secrets에 없습니다.")

    if provider == "openai":
        # GPT-6.1은 바뀌는 글과 한 메시지에 섞으면 캐시가 전혀 안 됨 (2026-11-04 시험: 0토큰) → 같은 앞부분은 instructions로 따로
        instructions = None
        if cache_first and len(items) > 1 and isinstance(items[0], str):
            instructions, items = items[0], items[1:]
        parts = []
        for it in items:
            if isinstance(it, AiFile):
                b64 = base64.b64encode(it.data).decode()
                if it.mime.startswith("image/"):
                    parts.append({"type": "input_image", "image_url": f"data:{it.mime};base64,{b64}"})
                else:
                    parts.append({"type": "input_file", "filename": it.name, "file_data": f"data:{it.mime};base64,{b64}"})
            else:
                parts.append({"type": "input_text", "text": str(it)})
        res = _openai_client(key).responses.create(
            model=model, input=[{"role": "user", "content": parts}], **({"instructions": instructions} if instructions else {}),
            text={"format": {"type": "json_schema", "name": "result", "schema": schema, "strict": False}})
        u = getattr(res, "usage", None)
        if u:
            _log_usage(provider, model, u.input_tokens, getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0), u.output_tokens)
        return json.loads(res.output_text.strip())

    if provider == "anthropic":
        blocks = []
        for it in items:
            if isinstance(it, AiFile):
                b64 = base64.b64encode(it.data).decode()
                kind = "image" if it.mime.startswith("image/") else "document"
                blocks.append({"type": kind, "source": {"type": "base64", "media_type": it.mime, "data": b64}})
            else:
                blocks.append({"type": "text", "text": str(it)})
        if cache_first and blocks and blocks[0]["type"] == "text":
            blocks[0]["cache_control"] = {"type": "ephemeral"}
        ac = _anthropic_client(key)
        req = dict(model=model, max_tokens=64000, messages=[{"role": "user", "content": blocks}],
                   output_config={"format": {"type": "json_schema", "schema": _schema_closed(schema)}})
        if model in CLAUDE_FALLBACK_MODELS:
            stream_ctx = ac.beta.messages.stream(betas=["server-side-fallback-2026-06-01"], fallbacks=[{"model": "claude-opus-4-8"}], **req)
        else:
            stream_ctx = ac.messages.stream(**req)
        with stream_ctx as stream:
            msg = stream.get_final_message()
        if msg.stop_reason == "refusal":
            raise RuntimeError("Claude가 요청을 거절했습니다 (refusal).")
        if msg.stop_reason == "max_tokens":
            raise RuntimeError("Claude 답이 너무 길어 중간에 끊겼습니다 (max_tokens).")
        text = next(b.text for b in msg.content if b.type == "text")
        u = msg.usage
        _log_usage(provider, model, (u.input_tokens or 0) + (u.cache_read_input_tokens or 0) + (u.cache_creation_input_tokens or 0),
                   u.cache_read_input_tokens, u.output_tokens, u.cache_creation_input_tokens)
        return json.loads(text.strip())

    gc = gemini_client or client
    cfg = dict(temperature=temperature, response_mime_type="application/json", response_json_schema=schema)
    if cache_first and len(items) > 1 and isinstance(items[0], str):
        try:
            cfg["cached_content"] = _gemini_cached_prefix(gc, model, items[0])
            items = items[1:]
        except Exception:
            pass  # 캐시를 못 만들면(너무 짧음·지원 안 함 등) 그냥 전부 보냄
    parts = [(_gemini_part(gc, it) if isinstance(it, AiFile) else it) for it in items]
    res = gc.models.generate_content(model=model, contents=parts if len(parts) > 1 else parts[0], config=types.GenerateContentConfig(**cfg))
    u = getattr(res, "usage_metadata", None)
    if u:
        _log_usage(provider, model, u.prompt_token_count, u.cached_content_token_count, (u.candidates_token_count or 0) + (u.thoughts_token_count or 0))
    return json.loads(res.text.strip())

# ---- 반값 모드: 각 회사 Batch API (급하지 않은 요청을 모아 보내면 모든 토큰 50% 할인, 결과는 보통 몇 분~몇 시간 · 길면 24시간) ----
#   요청은 모두 [같은 앞부분, 바뀌는 글] 두 조각의 글. 캐시 방식은 llm_json과 같게 (GPT instructions · Claude cache_control)
def llm_batch_submit(model_spec: str, reqs: List, schema: Dict, temperature: float = 0.3) -> str:
    """reqs: [(custom_id, (앞부분, 바뀌는 글)), ...] → 회사 쪽 batch id"""
    provider, model = split_model_spec(model_spec)
    key = AI_KEYS.get(provider, "")
    if not key:
        raise RuntimeError(f"{AI_PROVIDERS.get(provider, provider)} API 키가 Secrets에 없습니다.")
    if provider == "openai":
        lines = [json.dumps({"custom_id": cid, "method": "POST", "url": "/v1/responses", "body": {
            "model": model, "instructions": first, "input": [{"role": "user", "content": [{"type": "input_text", "text": rest}]}],
            "text": {"format": {"type": "json_schema", "name": "result", "schema": schema, "strict": False}}}}, ensure_ascii=False)
            for cid, (first, rest) in reqs]
        oc = _openai_client(key)
        f = oc.files.create(file=("batch.jsonl", "\n".join(lines).encode("utf-8")), purpose="batch")
        return oc.batches.create(input_file_id=f.id, endpoint="/v1/responses", completion_window="24h").id
    if provider == "anthropic":
        sch = _schema_closed(schema)
        return _anthropic_client(key).messages.batches.create(requests=[{"custom_id": cid, "params": {
            "model": model, "max_tokens": 64000,
            "messages": [{"role": "user", "content": [{"type": "text", "text": first, "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": rest}]}],
            "output_config": {"format": {"type": "json_schema", "schema": sch}}}} for cid, (first, rest) in reqs]).id
    # Gemini 반값 모드는 response_json_schema를 무시함 (2026-11-04 시험: 형식이 다른 목록으로 옴) → response_schema로 보냄
    src = [{"contents": [{"role": "user", "parts": [{"text": first}, {"text": rest}]}], "metadata": {"key": cid},
            "config": {"temperature": temperature, "response_mime_type": "application/json", "response_schema": schema}}
           for cid, (first, rest) in reqs]
    return _gemini_batch_client(key).batches.create(model=model, src=src, config={"display_name": "4pm-batch"}).name

def _oai_body_text(body: Dict) -> str:
    for o in body.get("output") or []:
        if o.get("type") == "message":
            for c in o.get("content") or []:
                if c.get("type") == "output_text":
                    return c.get("text", "")
    return ""

def _json_or_none(text):
    try:
        return json.loads(str(text or "").strip())
    except Exception:
        return None

def llm_batch_collect(model_spec: str, batch_id: str, ids: List[str], log_usage: bool = True) -> (bool, Dict, str):
    """→ (끝났는지, {custom_id: 답 dict 또는 None}, 상태 설명). 끝나지 않았으면 회사에 상태만 한 번 물어봄 (가벼움)"""
    provider, model = split_model_spec(model_spec)
    key = AI_KEYS.get(provider, "")
    tag, out = model + BATCH_TAG, {}
    if provider == "openai":
        oc = _openai_client(key)
        b = oc.batches.retrieve(batch_id)
        if b.status in ("validating", "in_progress", "finalizing", "cancelling"):
            return False, {}, b.status
        if b.output_file_id:
            for line in oc.files.content(b.output_file_id).text.splitlines():
                r = _json_or_none(line)
                if not r:
                    continue
                body = (r.get("response") or {}).get("body") or {}
                u = body.get("usage") or {}
                if u and log_usage:
                    _log_usage(provider, tag, u.get("input_tokens"), (u.get("input_tokens_details") or {}).get("cached_tokens"), u.get("output_tokens"))
                out[r.get("custom_id")] = _json_or_none(_oai_body_text(body))
        return True, out, b.status
    if provider == "anthropic":
        ac = _anthropic_client(key)
        b = ac.messages.batches.retrieve(batch_id)
        if b.processing_status != "ended":
            return False, {}, b.processing_status
        for r in ac.messages.batches.results(batch_id):
            if r.result.type != "succeeded":
                out[r.custom_id] = None
                continue
            msg = r.result.message
            u = msg.usage
            if log_usage:
                _log_usage(provider, tag, (u.input_tokens or 0) + (u.cache_read_input_tokens or 0) + (u.cache_creation_input_tokens or 0),
                           u.cache_read_input_tokens, u.output_tokens, u.cache_creation_input_tokens)
            out[r.custom_id] = _json_or_none(next((x.text for x in msg.content if x.type == "text"), "")) if msg.stop_reason != "max_tokens" else None
        return True, out, "ended"
    j = _gemini_batch_client(key).batches.get(name=batch_id)
    state = getattr(j.state, "name", str(j.state))
    if state not in ("JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED", "JOB_STATE_PARTIALLY_SUCCEEDED"):
        return False, {}, state
    for i, r in enumerate((j.dest.inlined_responses if j.dest else None) or []):
        cid = (r.metadata or {}).get("key") or (ids[i] if i < len(ids) else "")
        res = r.response
        u = getattr(res, "usage_metadata", None) if res else None
        if u and log_usage:
            _log_usage(provider, tag, u.prompt_token_count, u.cached_content_token_count, (u.candidates_token_count or 0) + (u.thoughts_token_count or 0))
        try:
            out[cid] = _json_or_none(res.text) if res else None
        except Exception:
            out[cid] = None
    return True, out, state

@st.cache_data(ttl=3600, show_spinner=False)
def list_ai_models(provider: str) -> List[str]:
    """그 제공사 키로 쓸 수 있는 글 생성 모델 목록 (불러오지 못하면 기본 목록)"""
    key = AI_KEYS.get(provider, "")
    skip = ("image", "tts", "audio", "embedding", "live", "realtime", "transcribe", "search", "moderation", "robotics", "computer", "instruct", "codex", "dall-e", "whisper", "veo", "imagen", "aqa", "banana", "customtools")
    try:
        if provider == "gemini":
            names = [m.name.split("/")[-1] for m in get_genai_client(key).models.list()
                     if "generateContent" in (getattr(m, "supported_actions", None) or []) and m.name.split("/")[-1].startswith("gemini")]
        elif provider == "openai":
            names = [m.id for m in _openai_client(key).models.list() if m.id.startswith(("gpt-", "o1", "o3", "o4"))]
        else:
            names = [m.id for m in _anthropic_client(key).models.list()]
        names = sorted({n for n in names if not any(s in n.lower() for s in skip)}, key=natural_sort_key, reverse=True)
        return names or AI_DEFAULT_MODELS[provider]
    except Exception:
        return AI_DEFAULT_MODELS[provider]


# Supabase 클라우드 DB 연결 초기화 (URL 뒤에 경로가 붙어도 순수 도메인만 자동 추출)
raw_supabase_url = get_secret("SUPABASE_URL", "https://evwqyfmnqiklxcbwaqee.supabase.co")
m_url = re.match(r"(https?://[^/\s]+)", raw_supabase_url)
SUPABASE_URL = m_url.group(1) if m_url else raw_supabase_url
SUPABASE_KEY = get_secret("SUPABASE_KEY", "")

@st.cache_resource
def get_supabase_client(url: str, key: str):
    if url and key and create_client is not None:
        try:
            return create_client(url, key)
        except Exception as e:
            st.warning(f"⚠️ Supabase 연결 실패 (로컬 모드로 전환됩니다): {e}")
    return None

supabase = get_supabase_client(SUPABASE_URL, SUPABASE_KEY)
IS_CLOUD_DB = supabase is not None

# -----------------------------
# 2. 공통 유틸리티 및 하이브리드(로컬+클라우드) 데이터 관리
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VICTORY_FILE = os.path.join(BASE_DIR, "victory_library.json")
ANATOMY_FILE = os.path.join(BASE_DIR, "anatomy_library.json")
CORE_FILE = os.path.join(BASE_DIR, "core_library.json")
VOCA_FILE = os.path.join(BASE_DIR, "voca_library.json")
PAST_VOCA_FILE = os.path.join(BASE_DIR, "past_voca_library.json")
MOCK_FILE = os.path.join(BASE_DIR, "mock_exam_library.json")
DNA_FILE = os.path.join(BASE_DIR, "exam_dna_rules.json")
IMAGE_DIR = os.path.join(BASE_DIR, "image_assets")
os.makedirs(IMAGE_DIR, exist_ok=True)

def filepath_to_doc_key(filepath: str) -> str:
    return os.path.splitext(os.path.basename(filepath))[0]

# app_storage 문서 쓰기는 모두 여기를 거침
#   - updated_at에 저장 시각을 찍음 (예전에는 처음 만든 시각에서 바뀌지 않았음) → 다른 곳이 먼저 저장했는지 가볍게 확인
#   - 저장한 내용을 돌려받지 않음 (예전에는 저장할 때마다 문서 전체를 다시 내려받아 DB 전송량이 두 배였음)
try:
    from postgrest import ReturnMethod as _PgReturnMethod
    _UPSERT_MIN = {"returning": _PgReturnMethod.minimal}
except Exception:
    _UPSERT_MIN = {}

def _doc_upsert(sb, doc_key: str, content) -> str:
    """문서 저장 → 찍은 저장 시각을 돌려줌"""
    from datetime import datetime, timezone
    ver = datetime.now(timezone.utc).isoformat()
    sb.table("app_storage").upsert({"doc_key": doc_key, "content": content, "updated_at": ver},
                                   on_conflict="doc_key", **_UPSERT_MIN).execute()
    return ver

DEFAULT_EXAM_DNA = {
    "updated_at": "2026-10 (운양고 1·2·3학년 7개 시험지 & 제일고 2학년 1개 시험지 기출 DNA 탑재)",
    "rules_text": """[운양고 & 제일고 실제 내신 출제 DNA 핵심 규칙]
1. 상위 개념어(Meta-Vocabulary) 치환 원칙:
   - 지문의 구체적 서술을 주제/제목/요지/요약문에서는 반드시 학술적 상위 개념어로 압축할 것.
   - 실제 기출 예시: 'help a child in distress' -> 'altruistic behavior(이타적 행동)', 'start off hating' -> 'aversion(혐오)', 'associated with a reward' -> 'paired with positive reinforcement(긍정적 강화)', 'depriving them of personal choices' -> 'autonomy(자율성)', 'keep its location a secret' -> 'withhold its whereabouts(소재지를 감추다)', 'defensive, rigid state' -> 'emotionally guarded and unyielding posture'.
2. 요약문 빈칸 (A), (B) 출제 공식:
   - 지문 원문 단어를 그대로 빈칸으로 뚫지 말고, 원문의 핵심 인과관계/대조관계를 압축하는 학술 동사·명사·형용사·부사로 패러프레이징하여 (A), (B) 타겟으로 설정할 것.
   - 실제 기출 예시: trivialize -> downplay, defend -> safeguard, break down -> deconstruct / divide, capture energy -> harness, copy verbatim -> blindly / mindlessly copy, chronic -> persistent / continual, confuse -> malfunction, shaped by context -> conditioned / integrate.
3. 문맥상 어휘 추론(밑줄 변형) 및 함정 반의어 패턴:
   - 흐름을 정반대로 뒤집는 방향성 반의어 쌍을 적극 제시할 것 (예: facilitation <-> inhibition, impede <-> guarantee, lift burdens <-> impose burdens, permit <-> prevent, corridor <-> barrier, integration <-> fragmentation).
   - 철자나 형태가 유사해 착시를 일으키는 함정 어휘(Confusable words)를 주의 깊게 다룰 것 (예: priceless[매우 귀중한] vs valueless[가치 없는], replace[대체하다] vs misplace[잘못 두다], elicit[이끌어내다] vs eliminate[제거하다], feasible vs unfeasible).
4. 영문 선지 및 <보기> 진위형(T/F) 문장 스타일:
   - 요지/주장/주제/제목 및 <보기> 일치·불일치 문항은 모두 수준 높은 영문으로 출제되며, 원문의 인과관계를 역전시키거나 'only/solely/completely/independent of' 같은 극단적 한정어를 넣어 오답(False) 진술문을 구성함.""",
    "history": [
        {
            "title": "[기본 탑재] 운양고(고1·2·3) 7부 & 제일고(고2) 1부 기출 통합 분석",
            "summary": "상위 개념어 패러프레이징, 요약문 (A)/(B) 학술 어휘 치환, 방향성 역전 반의어 및 형태 유사 함정 어휘(priceless vs valueless 등), <보기> 영문 일치/불일치 변형 패턴 반영 완료"
        }
    ]
}

def load_exam_dna() -> Dict:
    doc_key = filepath_to_doc_key(DNA_FILE)
    if IS_CLOUD_DB:
        try:
            res = supabase.table("app_storage").select("content").eq("doc_key", doc_key).execute()
        except Exception as e:
            # DB 읽기가 잠깐 실패한 것 → 기본값을 화면에만 쓰고 DB에는 절대 저장하지 않음 (예전엔 기본값으로 덮어써서 규칙·영쌤 포인트가 사라질 수 있었음)
            st.warning(f"⚠️ 출제 DNA를 DB에서 불러오지 못해 기본값으로 보여 줘요. 저장할 때는 DB 최신 내용을 다시 읽어서 그 위에 반영해요. ({str(e)[:80]})")
            return copy.deepcopy(DEFAULT_EXAM_DNA)
        if res.data and res.data[0].get("content"):
            return res.data[0]["content"]
        # 여기부터는 DB에 출제 DNA 문서가 정말 없을 때만 (처음 설치 등)
    if os.path.exists(DNA_FILE):
        try:
            with open(DNA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if IS_CLOUD_DB:
                save_exam_dna(data)
            return data
        except Exception:
            pass
    save_exam_dna(copy.deepcopy(DEFAULT_EXAM_DNA))
    return copy.deepcopy(DEFAULT_EXAM_DNA)

def exam_dna_update(fn) -> bool:
    """출제 DNA를 DB에서 새로 읽어 fn으로 바꿀 부분만 고친 뒤 저장.
    창마다 처음 열 때 읽은 사본을 통째로 저장하면 다른 창에서 그사이 바꾼 규칙·영쌤 포인트·학습 기록이 되돌아가서, 저장할 때마다 최신본 위에 반영함"""
    latest = None
    if IS_CLOUD_DB:
        try:
            res = supabase.table("app_storage").select("content").eq("doc_key", filepath_to_doc_key(DNA_FILE)).execute()
            latest = res.data[0].get("content") if res.data else None
        except Exception as e:
            st.error(f"❌ 출제 DNA 최신본을 읽지 못해서 저장하지 않았어요. 잠시 뒤 다시 눌러 주세요. ({str(e)[:80]})")
            return False
    if not isinstance(latest, dict) or not latest:
        latest = copy.deepcopy(st.session_state.get("exam_dna") or DEFAULT_EXAM_DNA)
    fn(latest)
    save_exam_dna(latest)
    st.session_state.exam_dna = latest
    return True

def save_exam_dna(dna_data: Dict):
    doc_key = filepath_to_doc_key(DNA_FILE)
    if IS_CLOUD_DB:
        try:
            _doc_upsert(supabase, doc_key, dna_data)
        except Exception as e:
            st.warning(f"⚠️ 클라우드 출제 DNA 저장 실패: {e}")
    try:
        with open(DNA_FILE, "w", encoding="utf-8") as f:
            json.dump(dna_data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# 사이드바 로고·학원명 등 설정을 클라우드에 저장해 로그인할 때마다 다시 올리지 않도록 함
SETTINGS_FILE = os.path.join(BASE_DIR, "app_settings.json")
DEFAULT_LOGO_URL = "https://www.genspark.ai/api/files/s/Bvmm1owU"

def load_app_settings() -> Dict:
    doc_key = filepath_to_doc_key(SETTINGS_FILE)
    if IS_CLOUD_DB:
        try:
            res = supabase.table("app_storage").select("content").eq("doc_key", doc_key).execute()
            if res.data and isinstance(res.data[0].get("content"), dict):
                return res.data[0]["content"]
        except Exception:
            pass
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_app_settings(settings: Dict):
    doc_key = filepath_to_doc_key(SETTINGS_FILE)
    if IS_CLOUD_DB:
        try:
            _doc_upsert(supabase, doc_key, settings)
        except Exception as e:
            st.warning(f"⚠️ 설정 클라우드 저장 실패: {e}")
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def logo_upload_to_data_url(uploaded, max_side: int = 400) -> str:
    """로고는 교재 머리말에 작게 쓰이므로 400px PNG로 줄여서 저장 (투명 배경 유지)"""
    from PIL import Image
    try:
        im = Image.open(io.BytesIO(uploaded.getvalue()))
        im.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
    except Exception:
        return logo_to_url(uploaded)

def clean_label(text) -> str:
    if not text:
        return ""
    text = str(text)
    text = re.sub(r'<[^<]+?>', '', text)
    if "color:#" in text or "font-size" in text:
        text = text.split(">")[-1].split("}")[-1].strip()
    return text.strip()

def natural_sort_key(item):
    lbl = item.get("label", "") if isinstance(item, dict) else str(item)
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'([0-9]+)', lbl)]

def extract_group_prefix(label: str) -> str:
    lbl = clean_label(label)
    if lbl.startswith("[") and "]" in lbl:
        in_bracket = lbl[1:lbl.find("]")].strip()
        parts = in_bracket.split()
        return parts[0] if parts else in_bracket
    prefix = re.sub(r'[\s_\-#]*(?:q|no\.?|문항|번)?\s*\d+\s*(?:번|문항)?$', '', lbl, flags=re.IGNORECASE).strip()
    return prefix if prefix else "기타"

def extract_question_number(label: str):
    nums = re.findall(r'\d+', clean_label(label))
    return int(nums[-1]) if nums else None

# -----------------------------
# 2-A. 기출 어휘 시험지 단위 자동 통합기 (기존 분할 데이터 + 신규 업로드 모두 시험지별로 통합)
# -----------------------------
def parse_past_exam_label(label: str) -> (str, str):
    """'[운양고 26년 1학기 중간 고1] 18번' -> ('[운양고 26년 1학기 중간 고1]', '18번') 분리"""
    lbl = clean_label(label)
    m = re.match(r'^(\[[^\]]+\])\s*(.*)$', lbl)
    if m:
        exam_base = m.group(1).strip()
        rem = re.sub(r'\s*기출\s*어휘.*$', '', m.group(2).strip()).strip()
        return exam_base, rem
    m2 = re.match(r'^(.*?)\s*((?:논술형|서술형|문항|Q|q)?\s*\d+\s*(?:번|문항)?(?:\s*~\s*\d+\s*번?)?)$', lbl)
    if m2 and m2.group(1).strip():
        return f"[{m2.group(1).strip()}]", m2.group(2).strip()
    return (lbl if lbl.startswith("[") else f"[{lbl}]"), ""

def extract_q_order_from_category(cat_str: str) -> (int, int):
    """단어의 출처 배지에서 문항 번호를 추출하여 1번 -> 2번 -> ... -> 논술형 순으로 정렬"""
    s = str(cat_str)
    m_sub = re.search(r'(?:논술형|서술형)\s*(\d+)', s)
    if m_sub:
        return (2, int(m_sub.group(1)))
    m_num = re.search(r'(\d+)\s*(?:번|문항)', s)
    if m_num:
        return (1, int(m_num.group(1)))
    return (3, 999)

def consolidate_past_voca_records(records: List[Dict]) -> (List[Dict], bool):
    """문항별로 쪼개진 기출 어휘 레코드들을 시험지 단위(예: [운양고 26년 1학기 중간 고1])로 자동 병합"""
    if not records:
        return [], False

    grouped = {}
    changed = False

    for p in records:
        orig_lbl = clean_label(p.get("label", ""))
        exam_base, q_str = parse_past_exam_label(orig_lbl)
        clean_exam_name = exam_base.strip("[]").strip()

        if orig_lbl != exam_base or exam_base in grouped:
            changed = True

        if exam_base not in grouped:
            grouped[exam_base] = {
                "id": p.get("id") or str(uuid.uuid4()),
                "label": exam_base,
                "data": {"voca_list": []},
                "body_text": f"{exam_base} 기출 통합 어휘",
                "academy_name": p.get("academy_name", "포피엠 입시 영어"),
                "school_name": p.get("school_name", clean_exam_name.split()[0] if clean_exam_name else "")
            }

        target_list = grouped[exam_base]["data"]["voca_list"]
        existing_keys = {
            (str(x.get("word", "")).strip().lower(), str(x.get("category", "")).strip())
            for x in target_list
        }

        for w in p.get("data", {}).get("voca_list", []):
            w_copy = dict(w)
            cat = str(w_copy.get("category", "")).strip()
            # 만약 단어 category에 문항 번호가 빠져 있다면 라벨에서 추출한 문항 번호를 자동 기입
            if q_str and (q_str not in cat):
                src_tag = cat if cat else "기출어휘"
                cat = f"{clean_exam_name} {q_str} [{src_tag}]"
                w_copy["category"] = cat
                changed = True

            dedup_key = (str(w_copy.get("word", "")).strip().lower(), cat)
            if dedup_key not in existing_keys:
                target_list.append(w_copy)
                existing_keys.add(dedup_key)

    consolidated_list = list(grouped.values())
    for item in consolidated_list:
        v_list = item["data"]["voca_list"]
        sorted_v = sorted(v_list, key=lambda x: extract_q_order_from_category(x.get("category", "")))
        if sorted_v != v_list:
            item["data"]["voca_list"] = sorted_v
            changed = True

    consolidated_list.sort(key=natural_sort_key)
    if len(consolidated_list) != len(records):
        changed = True

    return consolidated_list, changed

def save_base64_to_image_file(data_url: str, custom_filename: str = "") -> str:
    if not data_url or "," not in data_url:
        return ""
    try:
        _, b64_str = data_url.split(",", 1)
        img_bytes = base64.b64decode(b64_str)
        if custom_filename:
            filename = custom_filename if custom_filename.endswith(".png") else f"{custom_filename}.png"
        else:
            img_hash = hashlib.md5(img_bytes).hexdigest()[:14]
            filename = f"{img_hash}.png"
        img_path = os.path.join(IMAGE_DIR, filename)
        if not os.path.exists(img_path):
            with open(img_path, "wb") as img_f:
                img_f.write(img_bytes)
        if IS_CLOUD_DB:
            try:
                supabase.storage.from_("image_assets").upload(
                    path=filename,
                    file=img_bytes,
                    file_options={"content-type": "image/png", "upsert": "true"}
                )
            except Exception:
                try:
                    supabase.table("app_images").upsert({
                        "img_id": filename,
                        "b64_data": f"data:image/png;base64,{b64_str}"
                    }, on_conflict="img_id").execute()
                except Exception:
                    pass
        return img_path
    except Exception:
        return ""

def normalize_passage_records(data: List[Dict]) -> (List[Dict], bool):
    needs_save = False
    for p in data:
        if not p.get("id"):
            p["id"] = str(uuid.uuid4())
            needs_save = True
        cleaned = clean_label(p.get("label", ""))
        if p.get("label") != cleaned:
            p["label"] = cleaned
            needs_save = True
        if p.get("image_path"):
            base_fn = os.path.basename(str(p["image_path"]).replace("\\", "/"))
            norm_path = os.path.join(IMAGE_DIR, base_fn)
            if p["image_path"] != norm_path:
                # 이 컴퓨터의 폴더 경로로만 바꿈 (파일 이름은 그대로라 다시 저장할 필요 없음
                #  — 예전에는 PC 앱·배포 앱을 번갈아 열 때마다 열자마자 교재 전체를 다시 저장했음)
                p["image_path"] = norm_path
        if "image_data_url" in p and len(p.get("image_data_url", "")) > 1000:
            file_id = p.get("id", str(uuid.uuid4()))
            img_path = save_base64_to_image_file(p["image_data_url"], custom_filename=f"{file_id}.png")
            if img_path:
                p["image_path"] = img_path
                p["image_data_url"] = ""
                needs_save = True
    data.sort(key=natural_sort_key)
    return data, needs_save

# ---- 교재 문서 저장 안전장치 ----
#   교재·원문·기출 어휘는 종류마다 문서 하나에 목록 전체가 들어 있어서, 저장할 때 목록 전체를 다시 씀.
#   그래서 다른 창(PC·폰)이나 서버의 생성 작업이 그 사이 저장한 지문이 있으면, 오래된 목록으로 덮어써 사라질 수 있었음.
#   → 저장할 때마다 updated_at에 시각을 찍고, 저장 직전에 그 시각만 읽어(가벼움) 내가 불러온 뒤 바뀌었으면
#     최신 목록을 받아 '내가 고친·추가한·지운 지문'만 그 위에 반영해서 저장함.
def _doc_versions() -> Dict:
    """모든 문서의 마지막 저장 시각 {doc_key: updated_at} (내용은 안 읽어서 가벼움)"""
    if not IS_CLOUD_DB:
        return {}
    res = supabase.table("app_storage").select("doc_key,updated_at").execute()
    return {r["doc_key"]: r.get("updated_at") for r in (res.data or [])}

def _doc_version(doc_key: str):
    res = supabase.table("app_storage").select("updated_at").eq("doc_key", doc_key).execute()
    return res.data[0].get("updated_at") if res.data else None

def _ver_key(v) -> tuple:
    """DB가 돌려주는 시각 글자는 소수점 뒤 0을 빼기도 해서, 비교할 때는 6자리로 맞춤"""
    m = re.match(r"(\d{4}-\d\d-\d\d)[T ](\d\d:\d\d:\d\d)(?:\.(\d+))?", str(v or ""))
    return (m.group(1), m.group(2), (m.group(3) or "").ljust(6, "0")[:6]) if m else (str(v or ""),)

def _item_fp(p: Dict) -> str:
    """지문 하나의 내용 지문(指紋) — 이미지 경로는 컴퓨터마다 폴더가 달라서 파일 이름만 비교"""
    q = p
    if isinstance(p, dict) and p.get("image_path"):
        q = {**p, "image_path": os.path.basename(str(p["image_path"]).replace("\\", "/"))}
    return hashlib.md5(json.dumps(q, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()

def _lib_meta() -> Dict:
    """이 창이 마지막으로 DB와 맞춘 시점의 {doc_key: {"ver": 시각, "fp": {지문 id: 내용 지문}}}"""
    if "_lib_meta" not in st.session_state:
        st.session_state["_lib_meta"] = {}
    return st.session_state["_lib_meta"]

def _remember_lib(doc_key: str, ver, items: List[Dict]):
    _lib_meta()[doc_key] = {"ver": ver, "fp": {p.get("id"): _item_fp(p) for p in items if isinstance(p, dict) and p.get("id")}}

@st.cache_resource(show_spinner=False)
def _lib_mem_cache() -> Dict:
    """서버 하나가 같이 쓰는 교재 원본 {doc_key: (저장 시각, JSON 글자)}.
    새로 연 창·새로고침은 DB의 저장 시각이 같으면 내용을 다시 내려받지 않고 여기서 읽음 (앱 여는 시간·DB 전송량 절약)"""
    return {}

def _merge_library(base_fp: Dict, local: List[Dict], remote: List[Dict]) -> (List[Dict], int):
    """remote(DB 최신) 위에 이 창에서 바꾼 것만 반영. 돌려주는 수: 다른 곳에서 추가·수정돼 함께 살린 지문 수"""
    local_by = {p.get("id"): p for p in local if isinstance(p, dict)}
    out, seen, n_theirs = [], set(), 0
    local_new_labels = {p.get("label") for p in local if isinstance(p, dict) and p.get("id") not in base_fp}
    for r in remote:
        if not isinstance(r, dict):
            continue
        rid = r.get("id")
        seen.add(rid)
        if rid in base_fp:
            if rid not in local_by:
                continue                      # 이 창에서 지움
            lp = local_by[rid]
            if _item_fp(lp) != base_fp[rid]:
                out.append(lp)                # 이 창에서 고침 → 이 창 것
            else:
                out.append(r)                 # 이 창에선 그대로 → DB 최신 것
                n_theirs += _item_fp(r) != base_fp[rid]
        elif rid in local_by:
            out.append(local_by[rid])
        elif r.get("label") in local_new_labels:
            continue                          # 같은 지문을 이 창에서도 새로 만듦 → 이 창 것 하나만
        else:
            out.append(r)                     # 다른 곳에서 새로 추가
            n_theirs += 1
    for p in local:
        if not isinstance(p, dict) or p.get("id") in seen:
            continue
        if p.get("id") in base_fp and _item_fp(p) == base_fp[p.get("id")]:
            continue                          # 다른 곳에서 지웠고 이 창에선 안 고침 → 지운 대로
        out.append(p)                         # 이 창에서 새로 추가(또는 다른 곳에서 지웠지만 이 창에서 고친 것)
    return out, n_theirs

def load_from_file(filepath: str, keep: List[Dict] = None) -> List[Dict]:
    """keep: 다시 불러오기에서 지금 화면에 있는 목록. DB 읽기가 실패하면 이 목록을 그대로 둠 (빈 목록으로 바꾸지 않음)"""
    doc_key = filepath_to_doc_key(filepath)
    data = None
    ver = None

    if IS_CLOUD_DB:
        try:
            cache = _lib_mem_cache()
            hit = cache.get(doc_key)
            ver = _doc_version(doc_key) if hit else None
            if hit and ver and _ver_key(hit[0]) == _ver_key(ver):
                data = json.loads(hit[1])
            else:
                res = supabase.table("app_storage").select("content,updated_at").eq("doc_key", doc_key).execute()
                if res.data and isinstance(res.data[0].get("content"), list):
                    data = res.data[0]["content"]
                    ver = res.data[0].get("updated_at")
                    cache[doc_key] = (ver, json.dumps(data, ensure_ascii=False))
        except Exception as e:
            data, ver = None, None  # ver를 비워 두면 다음 저장 때 반드시 DB 최신과 합침
            st.warning(f"⚠️ 클라우드 데이터 로드 중 오류 ({doc_key}): {e}")
            if keep is not None:
                return keep  # 지금 목록과 그때 맞춘 기록을 그대로 둠 → 저장하면 DB 최신과 합쳐짐

    if data is None:
        data = []
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                st.error(f"⚠️ 파일 로드 중 에러 발생 ({os.path.basename(filepath)}): {e}")
                data = []
        if not isinstance(data, list):
            data = []

    data, needs_save = normalize_passage_records(data)
    if filepath == PAST_VOCA_FILE:
        data, c_changed = consolidate_past_voca_records(data)
        if c_changed:
            needs_save = True

    if IS_CLOUD_DB:
        _remember_lib(doc_key, ver, data)
    if needs_save:
        save_to_file(filepath, data)
    return data

def save_to_file(filepath: str, passages: List[Dict], replace: bool = False):
    """replace=True: 데이터 이식의 '전체 바꾸기'처럼 목록을 그대로 덮어쓸 때 (다른 곳 변경과 합치지 않음)"""
    doc_key = filepath_to_doc_key(filepath)
    for p in passages:
        if isinstance(p, dict) and not p.get("id"):
            p["id"] = str(uuid.uuid4())
    passages.sort(key=natural_sort_key)
    if IS_CLOUD_DB:
        try:
            # 이 창이 DB와 맞춘 기록이 없으면(불러오기 실패 등) 기준을 비워 두고 DB 최신과 합침 → DB 지문은 하나도 안 지워짐
            meta = _lib_meta().get(doc_key) or {}
            if not replace:
                now_ver = _doc_version(doc_key)
                if now_ver and _ver_key(now_ver) != _ver_key(meta.get("ver")):
                    hit = _lib_mem_cache().get(doc_key)
                    if hit and hit[0] and _ver_key(hit[0]) == _ver_key(now_ver):
                        remote = json.loads(hit[1])  # 같은 서버의 생성 작업이 방금 저장한 것 → 다시 받지 않음
                    else:
                        res = supabase.table("app_storage").select("content").eq("doc_key", doc_key).execute()
                        remote = res.data[0].get("content") if res.data else None
                    if isinstance(remote, list):
                        merged, n_theirs = _merge_library(meta.get("fp") or {}, passages, remote)
                        passages[:] = merged
                        passages.sort(key=natural_sort_key)
                        if n_theirs:
                            st.toast(f"🔀 다른 창·생성 작업에서 그사이 저장된 {n_theirs}개도 함께 살려 저장했어요")
            ver = _doc_upsert(supabase, doc_key, passages)
            _remember_lib(doc_key, ver, passages)
            _lib_mem_cache()[doc_key] = (ver, json.dumps(passages, ensure_ascii=False))
        except Exception as e:
            st.warning(f"⚠️ 클라우드 자동 저장 실패 ({doc_key}): {e}")
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(passages, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# 통합 세션 스테이트 초기화
_lib_load_note = st.empty()
if "saved_v" not in st.session_state:
    _lib_load_note.info("📚 교재·원문 데이터를 불러오는 중이에요... 창을 처음 열 때만 잠깐 걸려요.")
if "saved_v" not in st.session_state:
    st.session_state.saved_v = load_from_file(VICTORY_FILE)
if "v_selected" not in st.session_state:
    st.session_state.v_selected = []

if "saved_a" not in st.session_state:
    st.session_state.saved_a = load_from_file(ANATOMY_FILE)
if "a_selected" not in st.session_state:
    st.session_state.a_selected = []

if "saved_c" not in st.session_state:
    st.session_state.saved_c = load_from_file(CORE_FILE)
if "c_selected" not in st.session_state:
    st.session_state.c_selected = []

if "saved_w" not in st.session_state:
    st.session_state.saved_w = load_from_file(VOCA_FILE)
if "w_selected" not in st.session_state:
    st.session_state.w_selected = []

if "saved_past_voca" not in st.session_state:
    st.session_state.saved_past_voca = load_from_file(PAST_VOCA_FILE)
else:
    # 이미 세션에 로드된 기존 문항별 쪼개진 데이터도 즉시 시험지 단위로 자동 병합
    st.session_state.saved_past_voca, _migrated = consolidate_past_voca_records(st.session_state.saved_past_voca)
    if _migrated:
        save_to_file(PAST_VOCA_FILE, st.session_state.saved_past_voca)

if "past_v_selected" not in st.session_state:
    st.session_state.past_v_selected = []

# 한판승·Self-Note·Core·Voca가 같이 쓰는 선택 목록 (지문 이름 순서 = 출력 순서). 새로고침·재로그인 시 초기화
if "shared_sel_labels" not in st.session_state:
    st.session_state.shared_sel_labels = []
SHARED_SELECTION_TYPES = ("victory", "anatomy", "core", "voca")

if "saved_mock" not in st.session_state:
    st.session_state.saved_mock = load_from_file(MOCK_FILE)

if "app_settings" not in st.session_state:
    st.session_state.app_settings = load_app_settings()

if "exam_dna" not in st.session_state:
    st.session_state.exam_dna = load_exam_dna()
_lib_load_note.empty()

@st.cache_data(show_spinner=False, max_entries=30)
def cached_read_image_base64(img_filename: str) -> str:
    local_p = os.path.join(IMAGE_DIR, img_filename)
    if os.path.exists(local_p):
        mime = "image/jpeg" if img_filename.lower().endswith((".jpg", ".jpeg")) else "image/png"
        with open(local_p, "rb") as f:
            return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"
    if IS_CLOUD_DB:
        return f"{SUPABASE_URL}/storage/v1/object/public/image_assets/{img_filename}"
    return ""

def get_image_base64(p: Dict) -> str:
    path = p.get("image_path", "")
    if path:
        img_filename = os.path.basename(str(path).replace("\\", "/"))
        res = cached_read_image_base64(img_filename)
        if res:
            return res
    return p.get("image_data_url", "")

def logo_to_url(uploaded) -> str:
    if uploaded is None:
        return "https://www.genspark.ai/api/files/s/Bvmm1owU"
    mime = uploaded.type if hasattr(uploaded, 'type') and uploaded.type else 'image/png'
    return f"data:{mime};base64,{base64.b64encode(uploaded.getvalue()).decode()}"

def esc(t) -> str:
    return html.escape(str(t)) if t else ""

def nl2br(t) -> str:
    return esc(t).replace("\n", "<br>")

def md_safe(t) -> str:
    """st.markdown에서 '41~42번'의 ~가 취소선으로 바뀌지 않도록 이스케이프"""
    return str(t).replace("~", "\\~")


# -----------------------------
# 2-B. 단행본 표지(1P) + 공란(2P) + 저작권 뒤표지(마지막P) 빌더
# -----------------------------
BOOK_COVER_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800;900&family=Noto+Sans+KR:wght@300;400;500;700;900&display=swap');
.book-special-page {
    width: 210mm; height: 297mm; box-sizing: border-box; margin: 10mm auto;
    background: #ffffff; box-shadow: 0 0 15px rgba(0,0,0,0.10);
    page-break-after: always; break-after: page; position: relative; overflow: hidden;
    font-family: 'Inter', 'Noto Sans KR', sans-serif;
}
.book-cover-inner {
    width: 100%; height: 100%; box-sizing: border-box; padding: 26mm 24mm 22mm 24mm;
    display: flex; flex-direction: column; justify-content: space-between; position: relative;
}
.cover-accent-bar {
    position: absolute; top: 0; left: 0; width: 100%; height: 10mm;
    background: linear-gradient(90deg, #0f172a 0%, #0047AB 65%, #38bdf8 100%);
}
.cover-side-line {
    position: absolute; top: 26mm; bottom: 22mm; left: 16mm; width: 1.5px; background: #e2e8f0;
}
.cover-top-meta {
    display: flex; justify-content: space-between; align-items: center;
    border-bottom: 1.5px solid #0f172a; padding-bottom: 14px;
}
.cover-series-tag {
    font-size: 11px; font-weight: 800; letter-spacing: 3.5px; color: #0047AB; text-transform: uppercase;
}
.cover-logo-circle {
    width: 46px; height: 46px; border-radius: 50%; border: 2px solid #0047AB;
    background-size: cover; background-position: center;
}
.cover-hero { margin: auto 0; padding: 15mm 0; }
.cover-edition-pill {
    display: inline-block; background: #0f172a; color: #38bdf8; font-size: 10.5px;
    font-weight: 800; letter-spacing: 2.5px; padding: 5px 12px; border-radius: 4px;
    margin-bottom: 18px; text-transform: uppercase;
}
.cover-main-title {
    font-size: 54px; font-weight: 900; color: #0f172a; letter-spacing: -1.5px;
    line-height: 1.08; margin: 0 0 14px 0;
}
.cover-main-title span { color: #0047AB; }
.cover-subtitle-en { font-size: 15px; font-weight: 600; color: #475569; letter-spacing: 0.5px; margin-bottom: 8px; }
.cover-subtitle-ko { font-size: 13.5px; font-weight: 500; color: #64748b; }
.cover-divider { width: 65px; height: 5px; background: #0047AB; margin: 28px 0; }
.cover-author-credit { font-size: 14px; font-weight: 700; color: #1e293b; letter-spacing: 1.2px; text-transform: uppercase; }
.cover-author-credit span { color: #0047AB; font-weight: 900; }
.cover-bottom-area {
    display: flex; justify-content: space-between; align-items: flex-end;
    border-top: 1px solid #cbd5e1; padding-top: 18px;
}
.cover-Creator-block { font-size: 11.5px; color: #475569; line-height: 1.6; }
.cover-student-box {
    width: 210px; border: 1.5px solid #0f172a; border-radius: 6px;
    padding: 10px 14px; font-size: 11.5px; color: #0f172a; background: #f8fafc;
}
.cover-student-row {
    display: flex; justify-content: space-between; border-bottom: 1px dashed #cbd5e1;
    padding: 5px 0; font-weight: 700;
}
.cover-student-row:last-child { border-bottom: none; }
.back-cover-inner {
    width: 100%; height: 100%; box-sizing: border-box; padding: 25mm 24mm 22mm 24mm;
    display: flex; flex-direction: column; justify-content: flex-end; align-items: flex-end;
}
.copyright-box-right { text-align: right; max-width: 125mm; border-right: 4px solid #0047AB; padding-right: 14px; }
.copyright-title-en { font-size: 12px; font-weight: 900; color: #0f172a; letter-spacing: 1.5px; margin-bottom: 6px; }
.copyright-ko-legal { font-size: 10.5px; font-weight: 700; color: #334155; line-height: 1.65; word-break: keep-all; margin-bottom: 6px; }
.copyright-en-sub { font-size: 9px; color: #94a3b8; letter-spacing: 0.4px; line-height: 1.4; }
/* 표지·공란·뒤표지는 본문 페이지 여백과 상관없이 여백 0인 A4 한 장 전체로 인쇄
   (본문 @page 여백 안에 210mm 표지를 넣으면 넘쳐서 크롬이 문서 전체를 축소함) */
@page bookcover { size: A4; margin: 0; }
@media print {
    .book-special-page {
        page: bookcover;
        width: 210mm !important; height: 296mm !important; margin: 0 !important;
        box-shadow: none !important; border: none !important;
        page-break-after: always !important; break-after: page !important;
    }
    .book-last-page { page-break-after: auto !important; break-after: auto !important; }
}
"""

def booklet_stats(book_type: str, passages: List[Dict]) -> Dict:
    """표지·소개 쪽에 넣을 숫자를 고른 지문 데이터에서 직접 셈 (AI 호출 없음)"""
    datas = [p.get("data") or {} for p in passages]
    st_ = {"passages": len(passages)}
    if book_type == "victory":
        rows = [r for d in datas for r in d.get("analysis", []) or []]
        st_["sentences"] = len(rows)
        st_["points"] = sum(len([g for g in r.get("grammar_points", []) or [] if g.get("title") or g.get("explain")]) or (1 if r.get("grammar") else 0) for r in rows)
    elif book_type == "anatomy":
        st_["quizzes"] = sum(len(d.get("quizzes", []) or []) for d in datas)
        st_["grammar"] = sum(len(d.get("grammar_points", []) or []) for d in datas)
    elif book_type == "core":
        st_["sentences"] = sum(len(d.get(k, []) or []) for d in datas for k in ("topics", "titles", "main_ideas", "summaries"))
        st_["tf"] = sum(len(d.get("paraphrased_tf", []) or []) for d in datas)
        try:
            exams = list((load_gichul_map().get("exams") or {}).values())
        except Exception:
            exams = []
        years = sorted({int(m.group(1)) for ex in exams for m in [re.search(r"(\d{2})\s*년", str(ex.get("exam_name", "")))] if m})
        st_["exams"] = len(exams)
        st_["years"] = (2000 + years[0], 2000 + years[-1]) if years else None
    else:  # voca, past_voca
        recs = consolidate_past_voca_records(passages)[0] if book_type == "past_voca" else passages
        items = [it for p in recs for it in (p.get("data") or {}).get("voca_list", []) or []]
        is_core = lambda it: any(k in str(it.get("category", "지문핵심")).upper() for k in ["CORE", "패러", "요약", "주제", "선지"])
        st_["exams"] = len(recs)
        st_["words"] = len(items)
        st_["core_words"] = sum(1 for it in items if is_core(it))
        st_["key_words"] = st_["words"] - st_["core_words"]
        st_["related"] = sum(len([x for x in (it.get("synonyms", []) or []) + (it.get("antonyms", []) or []) if x.get("word")]) for it in items)
    return st_


# 소개 쪽 'ANATOMY 시리즈, 언제 무엇을 보나요?' 표 (모든 교재 공통)
SERIES_ROADMAP = [("One-Shot", "수업 전·후", "구문 분석·문법을 혼자 정리하는 참고서 (예습용으로도 좋아요)"),
                  ("Self-Note", "수업 시간", "시험에 나올 내용을 직접 필기하고, 그 필기로 복습"),
                  ("Core", "수업 후·시험 직전", "수업 후 복습, 시험 직전 주제·제목·요지·요약문 핵심 정리"),
                  ("Voca", "매일", "시험 범위 지문에서 출제될 수 있는 관련 단어를 모두 담은 단어장"),
                  ("Vision", "공부를 마친 뒤", "우리 학교 스타일 변형문제를 충분히 풀며 실전 연습")]


def build_booklet_front_and_back(book_type: str, h_img: str, f_img: str, academy: str, school: str, passages: List[Dict]) -> (str, str):
    """한판승·Self-Note·Core·Voca·기출 어휘 단행본의 앞쪽 4쪽(앞표지·사용법·이렇게 만들었습니다·이런 점이 다릅니다)과
    뒤쪽(빈 쪽 넣기용 공란·저작권·뒤표지). 모양은 ANATOMY VISION과 같은 QB_BOOK_CSS를 씀"""
    acad = esc(academy or "포피엠 입시 영어")
    sch = esc(school or "")
    s = booklet_stats(book_type, passages)
    np_ = s["passages"]

    def stat_line(parts):  # 0인 항목은 빼고 ' · '로 이음 (예전 데이터는 일부 항목이 없을 수 있음)
        return " · ".join(t for t, v in parts if v)

    def howto(items):
        lis = "".join(f"<li><b>{h}</b><br>{b}</li>" for h, b in items)
        return f"<div class='vpage'><div class='kicker'>HOW TO USE</div><h2>이 교재 사용법</h2><ol class='howto'>{lis}</ol></div>"

    def made(kick, h2, lead, intro, steps):
        st_html = "".join(f"<div class='step'><h3><span>STEP {i}</span>{t}</h3><ul>{''.join(f'<li>{x}</li>' for x in xs)}</ul></div>"
                          for i, (t, xs) in enumerate(steps, 1))
        return f"<div class='vpage'><div class='kicker'>{kick}</div><h2>{h2}</h2><div class='lead'>{lead}</div><p>{intro}</p>{st_html}</div>"

    def feature(name, josa, left_t, left, right, so):
        rows = "".join(
            f"<tr{' class=\"me\"' if b == name else ''}><td class='nw'>ANATOMY {b}</td><td class='nw'>{w}</td><td style='text-align:left'>{d}</td></tr>"
            for b, w, d in SERIES_ROADMAP)
        li = lambda xs: "".join(f"<li>{x}</li>" for x in xs)
        return f"""<div class='vpage'><div class='kicker'>WHAT MAKES IT DIFFERENT</div><h2>{name}{josa} 이런 점이 다릅니다</h2>
  <div class='two'><div><h3 class='u'>{left_t}</h3><ul>{li(left)}</ul></div><div><h3 class='j'>ANATOMY {name.upper()}{josa}</h3><ul>{li(right)}</ul></div></div>
  <div class='so'><b>그래서 ANATOMY {name.upper()}{josa}</b><ul>{li(so)}</ul></div>
  <div class='step' style='margin-top:16px'><h3><span>SERIES</span>ANATOMY 시리즈, 언제 무엇을 보나요?</h3>
  <table class='series-tbl'><tr><th>교재</th><th>언제</th><th>무엇을</th></tr>{rows}</table></div>
  <div class='quote'>"우리 학교 시험지를 가장 많이 본 사람이 만든 교재."<span>— {acad} PQRS영T</span></div></div>"""

    def cover(hl, pill, en, ko, meta):
        return f"""
<div class='vcover'><div class='bar'></div><div class='inner'>
  <div class='top'><span>PQRS YOUNG ENGLISH CURRICULUM SERIES</span><span>ANATOMY SERIES</span></div>
  <div>
    <div class='pill'>{pill}</div>
    <h1>ANATOMY <span>{hl}</span></h1>
    <div class='en'>{en}</div>
    <div class='ko'>{ko}</div>
    <div class='rule'></div>
    <div class='meta'>{meta}</div>
    <div class='credit'>Created &amp; Designed by <b>PQRS Young</b> &times; {acad}</div>
  </div>
  <div class='bottom'>
    <div><b style='color:#0f172a;font-size:13px'>PQRS Young &times; {acad}</b><br>High School English Exam &amp; CSAT Preparation</div>
    <div class='student'><div><span>SCHOOL</span><span style='font-weight:500;color:#64748b'>{sch}</span></div><div><span>GRADE / CLASS</span><span></span></div><div><span>STUDENT NAME</span><span></span></div></div>
  </div>
</div></div>"""

    def back(hl, line):
        ser = " &middot; ".join(f"<b>{x}</b>" if x == hl else x for x in ["One-Shot", "Self-Note", "Core", "Voca", "Vision"])
        return f"""
<div class='vcover blank filler'></div>
<div class='vcover copy'><div class='inner'><div class='cbox'>
  <div class='t'>ANATOMY {hl.upper()} &mdash; CREATED BY PQRS YOUNG</div>
  <div class='k'>본 교재의 모든 저작권은 제작자인 <b>PQRS영쌤</b>에게 있으며, 저작권자의 사전 서면 동의 없는 무단 복제, 배포, 전송, 스캔 및 도용 시 저작권법에 의거하여 민·형사상의 엄중한 법적 처벌을 받을 수 있습니다.</div>
  <div class='e'>Copyright &copy; PQRS Young ({acad}). All rights reserved.<br>Unauthorized reproduction, distribution, or plagiarism of this material is strictly prohibited and subject to legal prosecution.</div>
</div></div></div>
<div class='vcover back' style='break-after:auto;page-break-after:auto'><div class='bar'></div><div class='inner'>
  <h1>ANATOMY <span>{hl}</span></h1>
  <div class='line'>{line}</div>
  <div class='series'>Anatomy {ser}</div>
</div><div class='foot2'><span>장기동 {acad} · PQRS영T</span><span>031-991-1199</span></div></div>"""

    if book_type == "victory":
        line = stat_line([(f"지문 {np_}개", np_), (f"문장 {s['sentences']}개 구조 분석", s["sentences"]), (f"어법 POINT {s['points']}곳", s["points"])])
        front = cover("One-Shot", f"SELF-STUDY REFERENCE &bull; {np_} PASSAGES",
                      "Intensive Syntax Analysis &amp; Direct Comprehension System",
                      "수업 전·후 혼자 정리하는 문장별 구조 분석 · 직독직해 · 문법 참고서", line) + howto([
            ("수업 전: 배울 지문을 한판승으로 먼저 읽고 오세요.",
             "문장마다 끊어 읽기와 직독직해가 달려 있어요. 문장 구조를 미리 알고 수업을 들으면 훨씬 잘 따라갈 수 있어요. 수업이 빠르게 느껴지는 학생일수록 꼭 예습하세요."),
            ("'반드시 알아야 할 핵심 키워드'와 'Sentence Vocab'은 수업 전에 외우세요.",
             "단어를 알고 들어가면 수업 시간에는 구조와 문법에만 집중할 수 있어요."),
            ("수업 후: '+ 어법 POINT'와 'Grammar Point'로 그날 배운 문법을 혼자 정리하세요.",
             "표시된 문장마다 왜 그 형태인지 한 줄로 설명할 수 있으면 돼요. 막히는 문장은 표시해 두고 다음 수업 때 질문하세요."),
            ("마지막으로 '영쌤의 한판승 코칭'과 '한판승 핵심 요약'으로 글 전체를 정리하세요.",
             "요약을 가리고 이 글이 무슨 말을 하는지 두 문장으로 말할 수 있으면 그 지문은 끝난 거예요."),
        ]) + made("HOW ANATOMY ONE-SHOT WAS MADE", "ANATOMY ONE-SHOT은 이렇게 만들었습니다",
            '"수업을 따라가기 어려운 학생도, 혼자서 지문을 끝까지 이해할 수 있게."',
            "수업 시간에 처음 보는 문장을 그 자리에서 분석하기는 쉽지 않습니다. 한판승은 문장 하나하나의 구조와 문법을 미리 풀어 둔 참고서입니다. 수업 전에는 예습으로 수업을 따라갈 준비를 하고, 수업 후에는 배운 구문과 문법을 혼자 다시 정리할 수 있게 만들었습니다.",
            [("시험 범위 원문 지문 정리", [f"교과서·모의고사 시험 범위 지문 <b>{np_}개</b>를 문장 단위로 나눴습니다."]),
             ("문장 하나하나 구조 분석", [f"<b>{s['sentences']}개 문장</b>마다 끊어 읽기와 직독직해, 문장 어휘를 달아 혼자 읽어도 이해되게 했습니다." if s["sentences"] else
                                     "문장마다 끊어 읽기와 직독직해, 문장 어휘를 달아 혼자 읽어도 이해되게 했습니다."]),
             ("어법 출제 자리 표시", [f"수일치·관계사·분사·준동사처럼 내신에 자주 나오는 어법 자리{f' <b>{s['points']}곳</b>' if s['points'] else ''}에 '+ 어법 POINT'를 표시했습니다."]),
             ("한 줄 요약과 코칭", ["지문마다 핵심 요약과 '영쌤의 한판승 코칭'을 넣어, 글 전체 흐름까지 정리되게 했습니다."])]
        ) + feature("One-Shot", "은", "보통의 해설서는",
            ["해석과 문법 설명이 따로 떨어져 있어 앞뒤로 넘겨 봐야 합니다.", "설명이 어려워 혼자 예습하기 힘듭니다."],
            ["문장 바로 아래에 끊어 읽기·직독직해·어휘·어법이 함께 있습니다.", "혼자 읽어도 이해되게 문장 단위로 풀어 썼습니다.", "어법이 나올 자리를 문장 옆에 직접 표시했습니다."],
            ["예습과 복습 모두에 쓰는 참고서로 만들었습니다.", "모든 문장에 직독직해를 달았습니다.", "어법 출제 자리를 문장 단위로 표시했습니다."])
        return front, back("One-Shot", line)

    if book_type == "anatomy":
        line = stat_line([(f"지문 {np_}개", np_), (f"OX 퀴즈 {s['quizzes']}문항", s["quizzes"]), (f"문법 Point {s['grammar']}개", s["grammar"])])
        front = cover("Self-Note", f"IN-CLASS EDITION &bull; {np_} PASSAGES",
                      "Comprehensive Text Dissection, Grammar &amp; Structure Workbook",
                      "수업 시간에 시험에 나올 것을 직접 필기하고 복습하는 노트", line) + howto([
            ("Self-Note는 수업 시간에 펴 놓고 쓰는 교재예요.",
             "수업 시작 전에 ANATOMY PREVIEW의 '영쌤의 쉬운 지문 설명'과 Vocabulary만 가볍게 훑어 두세요."),
            ("수업 중: 선생님이 '시험에 나온다'고 짚는 것을 SELF-NOTE 여백에 직접 적으세요.",
             "어법 자리, 바꿔 쓸 수 있는 표현, 주제문처럼 시험에 나올 것을 너만의 손필기로 채우는 쪽이에요. 깔끔함보다 '내가 알아볼 수 있게'가 중요해요."),
            ("수업 후: REVIEW-NOTE의 '영쌤의 필기노트'와 내 필기를 비교하세요.",
             "빠뜨린 내용은 다른 색 펜으로 채우고, 따라 쓰며 복습하세요."),
            ("마지막: STRUCTURE &amp; QUIZ로 확인하세요.",
             "글의 흐름을 보고 OX 퀴즈 5문항을 풀어요. 틀린 문항은 '주요 문법 Point 3'과 '지문 해석'에서 이유를 찾으세요."),
        ]) + made("HOW ANATOMY SELF-NOTE WAS MADE", "ANATOMY SELF-NOTE는 이렇게 만들었습니다",
            '"시험에 나올 것을 내 손으로 적고, 내 필기로 복습하는 노트."',
            "필기를 받아 적기만 하면 시험 때 무엇이 중요한지 다시 찾아야 합니다. Self-Note는 수업 시간에 시험 포인트를 스스로 필기하고, 수업 후에 영쌤 필기와 비교하고, 퀴즈로 확인하는 순서를 지문마다 반복하게 만들어 수업 한 번으로 지문이 내 것이 되게 했습니다.",
            [("지문마다 4단계 구성", ["PREVIEW · SELF-NOTE · REVIEW-NOTE · STRUCTURE &amp; QUIZ를 같은 순서로 배치해, 어느 지문이든 같은 방법으로 공부할 수 있게 했습니다."]),
             ("쉬운 설명과 동의어 어휘", ["어려운 지문도 '영쌤의 쉬운 지문 설명'으로 먼저 큰 그림을 잡게 하고, 어휘는 동의어 뜻까지 함께 실었습니다."]),
             ("영쌤의 필기노트", ["수업 시간에 실제로 짚는 내용을 필기노트로 정리해, 내 필기와 비교할 기준을 만들었습니다."]),
             ("OX 퀴즈와 문법 Point", [f"지문마다 OX 퀴즈 5문항{f'(전체 <b>{s['quizzes']}문항</b>)' if s['quizzes'] else ''}과 주요 문법 Point 3개로 이해했는지 바로 확인하게 했습니다."])]
        ) + feature("Self-Note", "는", "보통의 필기 노트는",
            ["원문만 있고 무엇을 적어야 할지 알려 주지 않습니다.", "필기가 맞았는지 확인할 방법이 없습니다."],
            ["수업 중 필기 → 수업 후 비교 → 퀴즈 순서가 정해져 있습니다.", "영쌤의 필기노트로 내 필기를 바로 점검합니다.", "OX 퀴즈로 이해했는지 그 자리에서 확인합니다."],
            ["모든 지문을 같은 4단계로 구성했습니다.", "지문마다 OX 퀴즈 5문항을 넣었습니다.", "필기 공간을 넉넉히 두어 너만의 노트가 되게 했습니다."])
        return front, back("Self-Note", line)

    if book_type == "core":
        ne = s["exams"]
        yrs = f"{s['years'][0]}~{s['years'][1]}년 " if s.get("years") else ""
        line = stat_line([(f"기출 시험지 {ne}회분 분석", ne), (f"지문 {np_}개", np_), (f"선지 예상 문장 {s['sentences']}개", s["sentences"]), (f"T/F 선지 {s['tf']}개", s["tf"])])
        front = cover("Core", f"PARAPHRASING EDITION &bull; {np_} PASSAGES",
                      "Essential Paraphrasing, Topics, Main Ideas &amp; Summary Master",
                      "기출 시험지 전수 분석 — 선지에 나올 주제·제목·요지·요약문 패러프레이징", line) + howto([
            ("수업이 끝난 날, 그날 배운 지문의 Core를 바로 보며 복습하세요.",
             "이 글의 주제를 먼저 내 말로 한 줄 써 보고, '출제 1순위 영문 주제 3선'과 비교하세요. 내 문장과 어떤 단어가 다른지 보는 것이 핵심이에요."),
            ("제목·요지 문장의 단어가 원문 어느 표현을 바꾼 것인지 연결하세요.",
             "학교 시험 선택지는 원문 단어를 그대로 쓰지 않아요. 바뀐 표현을 알아보는 연습이에요."),
            ("요약문 빈칸 (A)(B)는 보기 전에 직접 골라 보세요.", "고른 근거가 되는 원문 문장에 밑줄을 그으세요."),
            ("T/F 선지는 틀린 선지가 어디서 틀렸는지 표시하세요.", "'패러프레이징 핵심 어휘'에서 문맥에 맞는 단어를 고르며 마무리하세요."),
            ("시험 직전에는 Core만 다시 훑으세요.",
             "지문마다 주제·제목·요지·요약문이 한곳에 모여 있어서, 시험 전 짧은 시간에 전 범위 핵심 내용을 다시 볼 수 있어요."),
        ]) + made("HOW ANATOMY CORE WAS MADE", "ANATOMY CORE는 이렇게 만들었습니다",
            '"실제 기출 시험지를 전수 분석해, 선지에 나올 가능성이 가장 높은 문장만."',
            "주제·제목·요지·요약문 문제의 선택지는 원문 표현을 상위 개념어와 다른 구조로 바꿔 씁니다. 학교마다 바꾸는 방식도 다릅니다. ANATOMY CORE는 먼저 실제 기출 시험지를 전수 분석해 출제 유형과 선지를 바꿔 쓰는 방식을 정리하고, 그 기준으로 우리 학교 시험지 선지에 등장할 가능성이 가장 높은 문장들을 골라 실었습니다. 수업이 끝난 뒤에는 복습 교재로, 시험 직전에는 전 범위 핵심 정리 교재로 쓰도록 지문마다 핵심 내용을 한곳에 모았습니다.",
            [("실제 기출 시험지 전수 분석", [f"운양고·제일고 {yrs}중간·기말 시험지{f' <b>{ne}회분</b>' if ne else ''}의 주제·제목·요지·요약문·&lt;보기&gt;형 문항과 선지를 하나하나 분석했습니다."]),
             ("출제 유형 마스터", ["원문의 어떤 표현을 어떤 상위 개념어와 구조로 바꾸는지, 학교별로 자주 쓰는 출제 유형과 선지 패턴을 정리해 출제 기준으로 삼았습니다."]),
             ("선지에 나올 가능성이 가장 높은 문장 수록", ["그 기준으로 지문마다 영문 주제 3개, 제목 3개, 요지·주장 문장, 요약문 3개와 빈칸 (A)(B)를 만들었습니다. 실제 시험지 선지에 등장할 가능성이 가장 높은 문장들입니다."]),
             ("&lt;보기&gt;형 T/F 선지와 어휘 연동", [f"패러프레이징한 진위 선지{f' <b>{s['tf']}개</b>' if s['tf'] else ''}를 순서를 섞어 싣고, 바꿔 쓴 어휘는 ANATOMY VOCA 단어장에도 자동으로 들어가게 했습니다."])]
        ) + feature("Core", "는", "보통의 요약 자료는",
            ["주제·요지를 한 가지 문장으로만 알려 줍니다.", "우리 학교 시험이 선지를 어떻게 바꾸는지와 상관없이 만들어집니다."],
            ["실제 기출 시험지 전수 분석으로 출제 유형을 정리했습니다.", "선지에 나올 가능성이 가장 높은 문장을 3가지 이상 표현으로 보여 줍니다.", "T/F 선지로 바뀐 표현이 맞는지 직접 판단합니다."],
            ["기출 시험지를 전수 분석해 출제 기준을 세웠습니다.", "주제·제목·요약문을 지문마다 3개씩 실었습니다.", "수업 후 복습과 시험 직전 정리에 모두 쓰게 만들었습니다."])
        return front, back("Core", line)

    if book_type == "past_voca":
        ne, nw = s["exams"], s["words"]
        line = stat_line([(f"실제 기출 시험지 {ne}회분", ne), (f"본문·선지 어휘 {nw}개", nw)])
        front = cover("Voca", f"PAST EXAM EDITION &bull; {ne} EXAMS",
                      "Actual Past Exam Paper Vocabulary &amp; Choices Master",
                      "학교별 실제 기출 시험지 본문·선지 출제 어휘 완벽 마스터", line) + howto([
            ("시험지 단위로 묶여 있어요. 한 회분씩 끊어서 외우세요.", "시험지가 바뀔 때마다 새 쪽에서 시작해요. 한 번에 한 회분이 목표예요."),
            ("본문뿐 아니라 선지에 나온 단어도 꼭 외우세요.", "선지 어휘는 다음 시험에도 바뀐 표현으로 다시 나와요."),
            ("[유의어/대체]·[반의어/함정]을 함께 보세요.", "우리 학교 출제자가 즐겨 쓰는 단어 바꾸기 방식이 보여요."),
            ("암기체크 두 칸(□ □)으로 두 번 외우세요.", "두 칸이 다 차지 않은 단어만 시험 전날 다시 보세요."),
        ]) + made("HOW IT WAS MADE", "기출 어휘 단어장은 이렇게 만들었습니다",
            '"우리 학교 출제자가 실제로 쓴 단어."',
            "같은 지문이라도 학교마다 시험지에 쓰는 단어가 다릅니다. 기출 어휘 단어장은 우리 학교 실제 시험지의 본문과 선택지에 쓰인 단어를 그대로 모아, 출제자가 즐겨 쓰는 어휘를 미리 익히게 했습니다.",
            [("실제 기출 시험지 수집", [f"{sch + ' ' if sch else ''}실제 시험지{f' <b>{ne}회분</b>' if ne else ''}을 모았습니다."]),
             ("본문·선지 어휘 추출", [f"본문과 선택지에 쓰인 어휘{f' <b>{nw}개</b>' if nw else ''}를 시험지별로 정리했습니다."]),
             ("문맥 의미·유의어·반의어", ["시험지 속 뜻과 함께 유의어·반의어를 붙였습니다."]),
             ("시험지 단위 구성", ["시험지가 바뀔 때만 쪽을 넘겨, 한 회분씩 끊어 외우기 좋게 했습니다."])])
        # 기출 어휘는 시안대로 '이런 점이 다릅니다' 쪽이 없어 앞쪽이 3쪽 → 본문이 오른쪽 면(5쪽)에서 시작하도록 빈 쪽 하나를 넣음
        return front + "<div class='vcover blank'></div>", back("Voca", line)

    # voca (시험 범위 지문 단어장)
    line = stat_line([(f"지문 {np_}개", np_), (f"지문 핵심 단어 {s['key_words']}개", s["key_words"]),
                      (f"CORE 단어 {s['core_words']}개", s["core_words"]), (f"유의어·반의어 {s['related']:,}개", s["related"])])
    front = cover("Voca", f"EXAM-RANGE VOCA EDITION &bull; {np_} PASSAGES",
                  "Contextual Vocabulary, Idioms, Synonyms &amp; Antonyms Master",
                  "시험 범위 지문으로 만든, 출제될 수 있는 모든 관련 단어 마스터", line) + howto([
        ("이 단어장의 단어는 두 종류예요.",
         "<b>지문 핵심 단어</b>는 시험 범위 지문 본문에 실제로 쓰인 단어예요. <b>CORE 단어</b>(색이 다른 줄)는 주제·제목·요지·요약문 선택지에서 원문 표현을 바꿔 쓸 때 나오는 단어예요. 둘 다 외워야 해요."),
        ("단어는 사전 뜻이 아니라 '문맥 의미'로 외우세요.", "같은 단어도 지문마다 뜻이 달라요. 이 교재의 뜻은 시험 범위 지문 속 뜻이에요."),
        ("[유의어/대체]는 선택지에 바뀌어 나올 단어, [반의어/함정]은 낱말 문제의 함정이에요.",
         "원문 단어 대신 유의어가 선택지에 나오고, 문맥상 낱말 문제에서는 반의어로 바꿔 내요."),
        ("암기체크 두 칸(□ □)으로 두 번 외우세요.", "1회독에 첫 칸, 시험 3일 전 2회독에 둘째 칸. 두 칸이 다 차지 않은 단어만 마지막에 다시 보세요."),
    ]) + made("HOW ANATOMY VOCA WAS MADE", "ANATOMY VOCA는 이렇게 만들었습니다",
        '"시험 범위 지문으로, 실제로 출제될 수 있는 관련 단어를 모두."',
        "두꺼운 단어책의 단어 대부분은 이번 시험에 나오지 않습니다. 반대로 시험 범위 지문과 관련된 단어는 하나만 몰라도 문제를 틀릴 수 있습니다. ANATOMY VOCA는 이번 시험 범위에 쓰인 지문을 가지고, 지문에 나온 단어는 물론 선택지에서 바꿔 쓸 수 있는 단어와 유의어·반의어까지 실제로 출제될 수 있는 관련 단어를 모두 담았습니다.",
        [("지문 핵심 단어", [f"시험 범위 지문 {np_}개 본문에서 해석과 어법·낱말 문제에 꼭 필요한 단어·숙어{f' <b>{s['key_words']}개</b>' if s['key_words'] else ''}를 골랐습니다. 단어마다 어느 지문에서 나왔는지 출처를 적었습니다."]),
         ("CORE 단어", [f"주제·제목·요지·요약문 선택지는 원문 단어를 상위 개념어로 바꿔 씁니다. ANATOMY CORE에서 이렇게 바꿔 쓴 단어{f' <b>{s['core_words']}개</b>' if s['core_words'] else ''}를 따로 표시해 함께 실었습니다."]),
         ("유의어와 함정 반의어", [f"단어마다 선택지에 바뀌어 나올 유의어와 낱말 문제 함정으로 쓰일 반의어{f'(<b>{s['related']:,}개</b>)' if s['related'] else ''}를 붙였습니다."]),
         ("문맥 의미와 한글 발음", ["사전 뜻이 아니라 그 지문 속 뜻을 적고, 읽기 어려운 단어는 한글 발음을 달았습니다."])]
    ) + feature("Voca", "는", "보통의 단어장은",
        ["시험 범위와 상관없는 단어가 대부분입니다.", "지문에 나온 단어만 있고, 선택지에 바뀌어 나올 단어는 없습니다."],
        ["시험 범위 지문으로 만들었습니다.", "지문 핵심 단어와 CORE 단어를 함께 담았습니다.", "유의어·함정 반의어까지 출제될 수 있는 관련 단어를 모두 외웁니다."],
        ["지문 핵심 단어와 CORE 단어를 구분해 표시했습니다.", "단어마다 출처 지문과 문맥 의미를 적었습니다.", "암기체크 두 칸으로 두 번 확인하게 했습니다."])
    return front, back("Voca", line)


# -----------------------------
# 2-C. CORE 어휘 <-> VOCA 단어장 병합 유틸리티
# -----------------------------
def merge_core_vocab_into_voca_item(core_data: Dict, voca_data: Dict) -> int:
    if not core_data or not voca_data:
        return 0
    voca_list = voca_data.get("voca_list", [])
    existing_words = {str(item.get("word", "")).strip().lower() for item in voca_list}
    added_count = 0
    for cv in core_data.get("core_vocab", []):
        cw = str(cv.get("word", "")).strip()
        if not cw:
            continue
        if cw.lower() in existing_words:
            for item in voca_list:
                if str(item.get("word", "")).strip().lower() == cw.lower():
                    item["category"] = "★CORE 패러프레이징"
            continue
        voca_list.append({
            "word": cw,
            "pos": cv.get("pos", "n/v/a"),
            "pronunciation": cv.get("pronunciation", ""),
            "meaning": cv.get("meaning", ""),
            "category": "★CORE 패러프레이징",
            "synonyms": cv.get("synonyms", []),
            "antonyms": cv.get("antonyms", [])
        })
        existing_words.add(cw.lower())
        added_count += 1
    voca_data["voca_list"] = voca_list
    return added_count


# -----------------------------
# 2-D. 엑셀(.xlsx) 및 분리형 CSV 2종(표제어 통합 / 유의어·반의어 전용) 생성기
# -----------------------------
def generate_voca_main_csv_bytes(selected_passages: List[Dict]) -> bytes:
    """파일 1: 각 지문의 지문핵심 + ★CORE 패러프레이징(또는 기출 표제어)만 합친 CSV"""
    rows = []
    word_no = 1
    for p in selected_passages:
        lbl = p.get("label", "")
        voca_list = p.get("data", {}).get("voca_list", [])
        for item in voca_list:
            w = str(item.get("word", "")).strip()
            if not w:
                continue
            pos = str(item.get("pos", "")).strip()
            pron = str(item.get("pronunciation", "")).strip()
            mean = str(item.get("meaning", "")).strip()
            cat = str(item.get("category", "")).strip()
            cls_label = cat if ("본문" in cat or "선지" in cat or "번" in cat) else lbl
            full_mean = f"[{pos}] {mean}" if pos else mean

            rows.append({
                "분류": cls_label,
                "단어번호": word_no,
                "영어단어": w,
                "발음기호": str(item.get("ipa", "")).strip(),
                "뜻": full_mean,
                "한글발음": pron,
                "예문": str(item.get("example", "")).strip()
            })
            word_no += 1

    columns = ["분류", "단어번호", "영어단어", "발음기호", "뜻", "한글발음", "예문"]
    df = pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def generate_voca_rel_csv_bytes(selected_passages: List[Dict]) -> bytes:
    """파일 2: 각 지문의 유의어(Synonyms)와 반의어(Antonyms)만 따로 모아놓은 CSV"""
    rows = []
    word_no = 1
    for p in selected_passages:
        lbl = p.get("label", "")
        voca_list = p.get("data", {}).get("voca_list", [])
        for item in voca_list:
            w = str(item.get("word", "")).strip()
            if not w:
                continue
            cat = str(item.get("category", "")).strip()
            cls_label = cat if ("본문" in cat or "선지" in cat or "번" in cat) else lbl

            for syn in item.get("synonyms", []):
                sw = str(syn.get("word", "")).strip()
                if not sw:
                    continue
                rows.append({
                    "분류": f"{cls_label} [유의어]",
                    "단어번호": word_no,
                    "영어단어": sw,
                    "발음기호": "",
                    "뜻": f"[유의어: {w}] {str(syn.get('meaning', '')).strip()}",
                    "한글발음": str(syn.get("pronunciation", "")).strip(),
                    "예문": ""
                })
                word_no += 1

            for ant in item.get("antonyms", []):
                aw = str(ant.get("word", "")).strip()
                if not aw:
                    continue
                rows.append({
                    "분류": f"{cls_label} [반의어]",
                    "단어번호": word_no,
                    "영어단어": aw,
                    "발음기호": "",
                    "뜻": f"[반의어: {w}] {str(ant.get('meaning', '')).strip()}",
                    "한글발음": str(ant.get("pronunciation", "")).strip(),
                    "예문": ""
                })
                word_no += 1

    columns = ["분류", "단어번호", "영어단어", "발음기호", "뜻", "한글발음", "예문"]
    df = pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def generate_voca_excel_bytes(selected_passages: List[Dict]) -> bytes:
    rows_3col = []
    rows_detail = []
    rows_test = []
    row_types_3col = []

    for p in selected_passages:
        lbl = p.get("label", "")
        voca_list = p.get("data", {}).get("voca_list", [])
        if not voca_list:
            continue

        rows_3col.append({
            "영어 단어": f"■ {lbl} 핵심 어휘",
            "한글 발음": "",
            "한국어 뜻": ""
        })
        row_types_3col.append("passage_banner")

        for item in voca_list:
            w = str(item.get("word", "")).strip()
            pos = str(item.get("pos", "")).strip()
            pron = str(item.get("pronunciation", "")).strip()
            mean = str(item.get("meaning", "")).strip()
            cat = str(item.get("category", "지문핵심")).strip()
            is_highlight = any(k in cat.upper() for k in ["CORE", "패러", "요약", "주제", "선지"])

            tag_prefix = f"[{cat}] " if cat else ""
            full_mean = f"{tag_prefix}[{pos}] {mean}" if pos else f"{tag_prefix}{mean}"

            rows_3col.append({
                "영어 단어": w,
                "한글 발음": pron,
                "한국어 뜻": full_mean
            })
            row_types_3col.append("main_core" if is_highlight else "main")

            rows_detail.append({
                "시험지/지문": lbl,
                "상세 출처 (문항·위치)": cat,
                "구분": "★ 표제어",
                "품사": pos,
                "영어 단어": w,
                "한글 발음": pron,
                "한국어 뜻": mean
            })
            rows_test.append({
                "시험지/지문": lbl,
                "구분": f"★ 표제어({cat})" if cat else "★ 표제어",
                "영어 단어": w,
                "한글 발음": pron,
                "한국어 뜻 (시험빈칸)": ""
            })

            for syn in item.get("synonyms", []):
                sw = str(syn.get("word", "")).strip()
                spron = str(syn.get("pronunciation", "")).strip()
                smean = str(syn.get("meaning", "")).strip()
                if not sw:
                    continue
                rows_3col.append({
                    "영어 단어": f"  └ [유의어] {sw}",
                    "한글 발음": spron,
                    "한국어 뜻": smean
                })
                row_types_3col.append("syn")

                rows_detail.append({
                    "시험지/지문": lbl,
                    "상세 출처 (문항·위치)": cat,
                    "구분": "└ 유의어",
                    "품사": pos,
                    "영어 단어": sw,
                    "한글 발음": spron,
                    "한국어 뜻": smean
                })
                rows_test.append({
                    "시험지/지문": lbl,
                    "구분": "└ 유의어",
                    "영어 단어": sw,
                    "한글 발음": spron,
                    "한국어 뜻 (시험빈칸)": ""
                })

            for ant in item.get("antonyms", []):
                aw = str(ant.get("word", "")).strip()
                apron = str(ant.get("pronunciation", "")).strip()
                amean = str(ant.get("meaning", "")).strip()
                if not aw:
                    continue
                rows_3col.append({
                    "영어 단어": f"  └ [반의어/함정] {aw}",
                    "한글 발음": apron,
                    "한국어 뜻": amean
                })
                row_types_3col.append("ant")

                rows_detail.append({
                    "시험지/지문": lbl,
                    "상세 출처 (문항·위치)": cat,
                    "구분": "└ 반의어/함정",
                    "품사": pos,
                    "영어 단어": aw,
                    "한글 발음": apron,
                    "한국어 뜻": amean
                })
                rows_test.append({
                    "시험지/지문": lbl,
                    "구분": "└ 반의어/함정",
                    "영어 단어": aw,
                    "한글 발음": apron,
                    "한국어 뜻 (시험빈칸)": ""
                })

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_3col = pd.DataFrame(rows_3col) if rows_3col else pd.DataFrame(columns=["영어 단어", "한글 발음", "한국어 뜻"])
        df_detail = pd.DataFrame(rows_detail) if rows_detail else pd.DataFrame(columns=["시험지/지문", "상세 출처 (문항·위치)", "구분", "품사", "영어 단어", "한글 발음", "한국어 뜻"])
        df_test = pd.DataFrame(rows_test) if rows_test else pd.DataFrame(columns=["시험지/지문", "구분", "영어 단어", "한글 발음", "한국어 뜻 (시험빈칸)"])

        df_3col.to_excel(writer, sheet_name="3열_단어_발음_뜻", index=False)
        df_detail.to_excel(writer, sheet_name="상세_출처_트랙포함", index=False)
        df_test.to_excel(writer, sheet_name="학생용_단어테스트지", index=False)

        try:
            from openpyxl.styles import PatternFill, Font, Alignment, Border, Side

            wb = writer.book
            thin_border = Border(
                left=Side(style='thin', color='D9D9D9'),
                right=Side(style='thin', color='D9D9D9'),
                top=Side(style='thin', color='D9D9D9'),
                bottom=Side(style='thin', color='D9D9D9')
            )
            header_fill = PatternFill(start_color="0047AB", end_color="0047AB", fill_type="solid")
            header_font = Font(name="맑은 고딕", size=11, bold=True, color="FFFFFF")
            banner_fill = PatternFill(start_color="1E293B", end_color="1E293B", fill_type="solid")
            banner_font = Font(name="맑은 고딕", size=11, bold=True, color="38BDF8")
            main_fill = PatternFill(start_color="F8FAFC", end_color="F8FAFC", fill_type="solid")
            main_font = Font(name="맑은 고딕", size=11, bold=True, color="0F172A")
            main_core_fill = PatternFill(start_color="FEF3C7", end_color="FEF3C7", fill_type="solid")
            main_core_font = Font(name="맑은 고딕", size=11, bold=True, color="92400E")
            syn_fill = PatternFill(start_color="EFF6FF", end_color="EFF6FF", fill_type="solid")
            syn_font = Font(name="맑은 고딕", size=10, bold=False, color="1D4ED8")
            ant_fill = PatternFill(start_color="FFF1F2", end_color="FFF1F2", fill_type="solid")
            ant_font = Font(name="맑은 고딕", size=10, bold=False, color="BE123C")

            ws1 = wb["3열_단어_발음_뜻"]
            ws1.column_dimensions["A"].width = 36
            ws1.column_dimensions["B"].width = 24
            ws1.column_dimensions["C"].width = 48
            for cell in ws1[1]:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal="center", vertical="center")

            for idx, r_type in enumerate(row_types_3col, start=2):
                row_cells = ws1[idx]
                for c in row_cells:
                    c.border = thin_border
                    c.alignment = Alignment(vertical="center")
                if r_type == "passage_banner":
                    ws1.merge_cells(start_row=idx, start_column=1, end_row=idx, end_column=3)
                    row_cells[0].fill = banner_fill
                    row_cells[0].font = banner_font
                elif r_type == "main_core":
                    for c in row_cells:
                        c.fill = main_core_fill
                        c.font = main_core_font
                elif r_type == "main":
                    for c in row_cells:
                        c.fill = main_fill
                        c.font = main_font
                elif r_type == "syn":
                    for c in row_cells:
                        c.fill = syn_fill
                        c.font = syn_font
                elif r_type == "ant":
                    for c in row_cells:
                        c.fill = ant_fill
                        c.font = ant_font

            for sname, widths in [
                ("상세_출처_트랙포함", {"A": 26, "B": 30, "C": 14, "D": 10, "E": 28, "F": 22, "G": 40}),
                ("학생용_단어테스트지", {"A": 26, "B": 28, "C": 28, "D": 22, "E": 40}),
            ]:
                ws = wb[sname]
                for col_letter, w_val in widths.items():
                    ws.column_dimensions[col_letter].width = w_val
                for cell in ws[1]:
                    cell.fill = header_fill
                    cell.font = header_font
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                for row in ws.iter_rows(min_row=2):
                    row_str = " ".join([str(c.value or "") for c in row[:3]])
                    for c in row:
                        c.border = thin_border
                        c.alignment = Alignment(vertical="center")
                        if "유의어" in row_str:
                            c.fill, c.font = syn_fill, syn_font
                        elif "반의어" in row_str:
                            c.fill, c.font = ant_fill, ant_font
                        elif "CORE" in row_str or "선지" in row_str:
                            c.fill, c.font = main_core_fill, main_core_font
                        else:
                            c.fill, c.font = main_fill, main_font
        except Exception:
            pass

    return output.getvalue()


# -----------------------------
# 2-E. HTML 스마트 스튜디오 컨트롤 바
# -----------------------------
def get_studio_toolbar_html(mode_type: str = "standard") -> str:
    extra_btns = ""
    if mode_type in ["voca", "past_voca"]:
        extra_btns = """
        <button class="studio-btn" style="background:#0284c7; border-color:#38bdf8;" onclick="setVocaTestMode('study')">📖 학습용 표준</button>
        <button class="studio-btn" style="background:#0d9488; border-color:#2dd4bf;" onclick="setVocaTestMode('hide-meaning')">📝 뜻 시험지 전환</button>
        <button class="studio-btn" style="background:#7c3aed; border-color:#a78bfa;" onclick="setVocaTestMode('hide-word')">✍️ 영어 시험지 전환</button>
        """
    elif mode_type == "core":
        extra_btns = """
        <button class="studio-btn" style="background:#0284c7; border-color:#38bdf8;" onclick="setCoreTestMode('study')">📖 해설·정답 표시 모드</button>
        <button class="studio-btn" style="background:#0d9488; border-color:#2dd4bf;" onclick="setCoreTestMode('test')">📝 요약문(A)(B) 양자택일 & T/F 실전 시험지 모드</button>
        """

    return f"""
<style>
.studio-toolbar {{
    position: sticky; top: 0; z-index: 99999; background: #0f172a; color: white;
    padding: 10px 20px; display: flex; flex-wrap: wrap; justify-content: space-between;
    align-items: center; gap: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.25); font-family: 'Noto Sans KR', sans-serif;
}}
.studio-toolbar .st-title {{ font-weight: 900; font-size: 14px; color: #38bdf8; display: flex; align-items: center; gap: 8px; }}
.studio-toolbar .st-actions {{ display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }}
.studio-btn {{
    background: #1e293b; color: #f8fafc; border: 1px solid #475569; padding: 6px 11px;
    border-radius: 6px; font-size: 12px; font-weight: 700; cursor: pointer; transition: all 0.15s; font-family: inherit;
}}
.studio-btn:hover {{ background: #334155; border-color: #94a3b8; }}
.studio-btn.primary {{ background: #2563eb; border-color: #3b82f6; }}
.studio-btn.active-edit {{ background: #dc2626; border-color: #f87171; }}
.studio-drawer {{
    display: none; background: #1e293b; color: #f1f5f9; padding: 12px 20px;
    border-bottom: 2px solid #334155; max-height: 240px; overflow-y: auto; font-family: 'Noto Sans KR', sans-serif;
}}
.studio-drawer.open {{ display: block; }}
.drawer-item {{
    display: flex; justify-content: space-between; align-items: center; padding: 6px 10px;
    margin-bottom: 4px; background: #0f172a; border-radius: 6px; font-size: 12px; border: 1px solid #334155;
}}
.drawer-item-btns button {{
    background: #334155; color: white; border: none; border-radius: 4px;
    padding: 3px 8px; margin-left: 4px; cursor: pointer; font-size: 11px;
}}
[contenteditable="true"] {{ outline: 1.5px dashed #ef4444 !important; background-color: rgba(254, 226, 226, 0.12) !important; }}
@media print {{
    .no-print, .studio-toolbar, .studio-drawer, .voca-del-btn {{ display: none !important; }}
    [contenteditable="true"] {{ outline: none !important; background-color: transparent !important; }}
}}
</style>
<nav class="studio-toolbar no-print">
    <div class="st-title"><span>⚡ 4pm 실시간 교재 스튜디오 (표지·사용법·소개·저작권·뒤표지 자동 탑재)</span></div>
    <div class="st-actions">
        {extra_btns}
        <label title="양면 인쇄 때 뒤표지가 맨 뒷면에 오게 하려면 전체 쪽수가 짝수여야 해요" style="font-size:12px; color:#cbd5e1; cursor:pointer;"><input type="checkbox" onchange="document.body.classList.toggle('addblank', this.checked)"> 인쇄 미리보기 쪽수가 홀수면 체크 → 뒤표지 앞에 빈 쪽 넣기</label>
        <button class="studio-btn" onclick="toggleOrderDrawer()">📋 지문/시험지 순서 변경 / 제외</button>
        <button id="btn-edit-mode" class="studio-btn" onclick="toggleEditMode()">✏️ 화면 텍스트 직접 수정: OFF</button>
        <button class="studio-btn" onclick="downloadEditedHtml()">💾 단행본 전체 HTML 저장</button>
        <button class="studio-btn primary" onclick="window.print()">🖨️ 즉시 인쇄 / PDF 저장</button>
    </div>
</nav>
<div id="studio-order-drawer" class="studio-drawer no-print">
    <div style="font-size:12px; font-weight:700; margin-bottom:8px; color:#38bdf8;">
        💡 화살표를 눌러 본문 순서를 실시간으로 바꾸거나 특정 항목을 제외할 수 있습니다. (앞표지·사용법·소개·저작권·뒤표지는 항상 고정됩니다)
    </div>
    <div id="drawer-list"></div>
</div>
<script>
let isEditMode = false;
function toggleEditMode() {{
    isEditMode = !isEditMode;
    const container = document.getElementById('book-wrapper-all');
    const btn = document.getElementById('btn-edit-mode');
    if (container) container.contentEditable = isEditMode ? "true" : "false";
    if (isEditMode) {{
        btn.innerText = "✏️ 화면 텍스트 직접 수정: ON (클릭해 수정)";
        btn.classList.add("active-edit");
    }} else {{
        btn.innerText = "✏️ 화면 텍스트 직접 수정: OFF";
        btn.classList.remove("active-edit");
        if (typeof fitAllPages === "function") setTimeout(fitAllPages, 100);
    }}
}}
function setVocaTestMode(mode) {{
    const container = document.getElementById('book-container');
    if (!container) return;
    container.classList.remove('mode-hide-meaning', 'mode-hide-word');
    if (mode === 'hide-meaning') container.classList.add('mode-hide-meaning');
    if (mode === 'hide-word') container.classList.add('mode-hide-word');
}}
function setCoreTestMode(mode) {{
    const container = document.getElementById('book-container');
    if (!container) return;
    if (mode === 'test') container.classList.add('core-test-mode');
    else container.classList.remove('core-test-mode');
}}
function removeVocaRow(btnEl) {{
    const tr = btnEl.closest('tr');
    if (tr) {{
        const tbody = tr.parentElement;
        tr.remove();
        if (tbody) {{
            Array.from(tbody.querySelectorAll('.voca-idx-num')).forEach((el, idx) => {{ el.innerText = (idx + 1); }});
        }}
    }}
}}
function getPassageGroups() {{
    const container = document.getElementById('book-container');
    if (!container) return [];
    const children = Array.from(container.children);
    const groups = [];
    const map = new Map();
    children.forEach(el => {{
        const pid = el.getAttribute('data-passage-id') || 'unknown';
        const plabel = el.getAttribute('data-passage-label') || pid;
        if (!map.has(pid)) {{
            const g = {{ id: pid, label: plabel, elements: [] }};
            map.set(pid, g);
            groups.push(g);
        }}
        map.get(pid).elements.push(el);
    }});
    return groups;
}}
function renderDrawerList() {{
    const listEl = document.getElementById('drawer-list');
    if (!listEl) return;
    const groups = getPassageGroups();
    listEl.innerHTML = groups.map((g, idx) => `
        <div class="drawer-item">
            <span><b>${{idx + 1}}.</b> ${{g.label}}</span>
            <div class="drawer-item-btns">
                <button onclick="movePassageGroup(${{idx}}, -1)" ${{idx === 0 ? 'disabled style="opacity:0.3"' : ''}}>⬆️ 위로</button>
                <button onclick="movePassageGroup(${{idx}}, 1)" ${{idx === groups.length - 1 ? 'disabled style="opacity:0.3"' : ''}}>⬇️ 아래로</button>
                <button onclick="removePassageGroup('${{g.id}}')" style="background:#991b1b;">❌ 제외</button>
            </div>
        </div>
    `).join('');
}}
function toggleOrderDrawer() {{
    const drawer = document.getElementById('studio-order-drawer');
    drawer.classList.toggle('open');
    if (drawer.classList.contains('open')) renderDrawerList();
}}
function movePassageGroup(idx, dir) {{
    const groups = getPassageGroups();
    const targetIdx = idx + dir;
    if (targetIdx < 0 || targetIdx >= groups.length) return;
    const temp = groups[idx];
    groups[idx] = groups[targetIdx];
    groups[targetIdx] = temp;
    const container = document.getElementById('book-container');
    groups.forEach(g => g.elements.forEach(el => container.appendChild(el)));
    renderDrawerList();
}}
function removePassageGroup(pid) {{
    const groups = getPassageGroups();
    const target = groups.find(g => g.id === pid);
    if (target) {{
        target.elements.forEach(el => el.remove());
        renderDrawerList();
    }}
}}
function downloadEditedHtml() {{
    if (isEditMode) toggleEditMode();
    const docClone = document.documentElement.cloneNode(true);
    const blob = new Blob(["<!DOCTYPE html>\\n" + docClone.outerHTML], {{type: "text/html;charset=utf-8"}});
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = document.title ? document.title.replace(/\\s+/g, '_') + "_Booklet.html" : "Anatomy_Booklet.html";
    a.click();
}}
</script>
"""


# -----------------------------
# 3-A. [한판승: Anatomy One-Shot] 서식 및 렌더링
# -----------------------------
def build_css_victory(h_img, f_img, e_img):
    return f"""<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;700;900&display=swap');
    @page {{ size: A4; margin: 12mm 15mm; }} 
    body {{ font-family: 'Noto Sans KR', sans-serif; overflow-wrap: anywhere;background: #f0f2f5; margin: 0; padding: 0; -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }}
    {BOOK_COVER_CSS}
    .passage-wrapper {{ width: 210mm; box-sizing: border-box; padding: 15mm 18mm; margin: 10mm auto; background: white; box-shadow: 0 0 10px rgba(0,0,0,0.1); page-break-after: always; }}
    @media print {{ body {{ background: white; }} .passage-wrapper {{ width: 100%; margin: 0; padding: 0; box-shadow: none; border: none; }} }}
    .master-table {{ width: 100%; border-collapse: collapse; }}
    .master-table thead {{ display: table-header-group; }}
    .master-table tfoot {{ display: table-footer-group; }}
    .master-table > tbody > tr > td {{ padding: 0; }}
    .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 3px solid #0047AB; padding-bottom: 8px; margin-bottom: 12px; }}
    .logo-area {{ display: flex; align-items: center; gap: 10px; font-weight: 900; font-size: 22px; color: #0047AB; }}
    .h-img {{ width: 40px; height: 40px; background: url('{h_img}') center/cover; border-radius: 50%; border: 2px solid #0047AB; }}
    .footer {{ text-align: center; font-size: 10px; color: #999; border-top: 1px solid #eee; padding-top: 8px; margin-top: 15px; display: flex; justify-content: center; align-items: center; gap: 8px; }}
    .f-img {{ width: 22px; height: 22px; background: url('{f_img}') center/cover; border-radius: 50%; }}
    .top-layout {{ display: flex; gap: 15px; margin-bottom: 12px; }}
    .visual-box {{ flex: 0 0 220px; height: 160px; border-radius: 10px; background: #f8f9fa; display: flex; align-items: center; justify-content: center; overflow: hidden; border: 1px solid #eee; }}
    .visual-box img {{ max-width: 100%; max-height: 100%; object-fit: contain; }}
    .summary-box {{ flex: 1; background: #0047AB; color: white; padding: 15px; border-radius: 10px; display: flex; align-items: center; font-weight: 700; font-size: 14.5px; line-height: 1.55; }}
    .keyword-section {{ background: #fff4e5; border: 1px solid #f5c27a; border-radius: 10px; padding: 8px 12px; margin-bottom: 12px; }}
    .keyword-title {{ color: #9a5a00; font-weight: 900; font-size: 12px; margin-bottom: 5px; }}
    .keyword-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 5px 10px; }}
    .keyword-item {{ font-size: 10.5px; color: #333; }}
    .keyword-word {{ font-weight: 800; color: #0047AB; }}
    .analysis-table {{ width: 100%; border-collapse: collapse; }} 
    .analysis-table tr {{ page-break-inside: avoid; break-inside: avoid; }} 
    .analysis-table td {{ padding-bottom: 15px; }} 
    .eng-box {{ padding: 10px 14px 15px 14px; font-weight: 700; color: #1e293b; background: #fdfdfd; border-top: 1px solid #dee2e6; border-left: 1px solid #dee2e6; border-right: 1px solid #dee2e6; border-top-left-radius: 8px; border-top-right-radius: 8px; border-bottom: 1px dashed #e2e8f0; font-size: 13px; line-height: 2.3; word-break: keep-all; }}
    ruby.syntax-ruby {{ ruby-position: under; ruby-align: center; margin: 0 2px; }}
    ruby.syntax-ruby rt {{ font-weight: 900; font-size: 0.7em; line-height: 1; margin-top: 2px; white-space: nowrap; overflow-wrap: normal; word-break: keep-all; }}
    .gp-num, .sent-no {{ white-space: nowrap; overflow-wrap: normal; }}
    .st-S span.syntax-text {{ border-bottom: 2px solid #e11d48; padding-bottom: 1px; }} .st-S rt {{ color: #e11d48; }}
    .st-V span.syntax-text {{ border-bottom: 2px solid #2563eb; padding-bottom: 1px; }} .st-V rt {{ color: #2563eb; }}
    .st-O span.syntax-text {{ border-bottom: 2px solid #16a34a; padding-bottom: 1px; }} .st-O rt {{ color: #16a34a; }}
    .st-C span.syntax-text {{ border-bottom: 2px solid #ea580c; padding-bottom: 1px; }} .st-C rt {{ color: #ea580c; }}
    .st-M span.syntax-text {{ border-bottom: 2px dashed #9333ea; padding-bottom: 1px; }} .st-M rt {{ color: #9333ea; }}
    .inline-vocab-box {{ padding: 10px 14px; background: #fafafa; border-left: 1px solid #dee2e6; border-right: 1px solid #dee2e6; border-bottom: 1px dashed #e2e8f0; font-size: 11.5px; line-height: 1.5; color: #444; }}
    .trans-box {{ padding: 10px 14px; color: #475569; background: #ffffff; border-left: 1px solid #dee2e6; border-right: 1px solid #dee2e6; border-bottom: 1px dashed #e2e8f0; font-size: 12.5px; line-height: 1.6; font-style: normal; word-break: keep-all; }}
    .inline-grammar-box {{ padding: 10px 14px; background: #fefce8; color: #854d0e; border-left: 1px solid #dee2e6; border-right: 1px solid #dee2e6; border-bottom: 1px dashed #e2e8f0; font-size: 12px; line-height: 1.5; }}
    .ys-box {{ padding: 10px 14px; background: #eef4ff; color: #0047AB; font-weight: 500; border-left: 4px solid #FF8C00 !important; border-right: 1px solid #dee2e6; border-bottom: 1px solid #dee2e6; border-bottom-left-radius: 8px; border-bottom-right-radius: 8px; font-size: 12.5px; line-height: 1.6; word-break: keep-all; }}
    .ys-header {{ display: flex; align-items: center; gap: 6px; margin-bottom: 6px; font-weight: 900; color: #FF8C00; font-size: 11px; }}
    .ys-img {{ width: 22px; height: 22px; background: url('{e_img}') center/cover; border-radius: 50%; border: 1px solid #FF8C00; }}
    /* 어법 위치 번호 표시 + 오른쪽 어법 POINT 칸 */
    .sent-row {{ display: flex; gap: 10px; align-items: flex-start; }}
    .sent-main {{ flex: 1; min-width: 0; }}
    .sent-no {{ display: inline-flex; align-items: center; justify-content: center; width: 20px; height: 20px; border-radius: 50%; background: #0047AB; color: white; font-size: 11px; font-weight: 900; margin-right: 6px; vertical-align: 2px; }}
    .essay-badge {{ display: inline-block; background: #e11d48; color: white; font-size: 9.5px; font-weight: 900; padding: 1px 7px; border-radius: 4px; margin-bottom: 4px; line-height: 1.6; }}
    .eng-box.essay {{ border: 1.5px solid #e11d48; border-bottom: 1px dashed #e2e8f0; }}
    /* 형광펜은 글자 아래쪽만 칠해 윗줄의 (S)(V) 표시를 덮지 않게 하고, rt는 항상 형광펜 위에 그림 */
    .gp-mark {{ background: linear-gradient(transparent 40%, #ffe58a 40%, #ffe58a 92%, transparent 92%); padding: 0 1px; -webkit-box-decoration-break: clone; box-decoration-break: clone; }}
    .eng-box ruby.syntax-ruby rt {{ position: relative; z-index: 2; }}
    .gp-num {{ display: inline-flex; align-items: center; justify-content: center; width: 15px; height: 15px; border-radius: 50%; background: #FF8C00; color: white; font-size: 9.5px; font-weight: 900; margin-right: 3px; line-height: 1; vertical-align: 1px; }}
    .gp-panel {{ flex: 0 0 58mm; box-sizing: border-box; background: #f8fbff; border: 1px solid #d6e2f5; border-radius: 8px; padding: 8px 10px; }}
    .gp-title {{ display: inline-block; background: #0047AB; color: white; font-size: 10.5px; font-weight: 900; padding: 2px 9px; border-radius: 4px; margin-bottom: 6px; }}
    .gp-item {{ font-size: 10.5px; line-height: 1.5; color: #334155; margin-bottom: 6px; word-break: keep-all; }}
    .gp-item:last-child {{ margin-bottom: 0; }}
    .gp-item b {{ color: #b45309; }}
    @media print {{ .gp-panel {{ flex-basis: 52mm; }} }}
    </style>"""

GP_MARK_RE = re.compile(r'\{(\d{1,2}):(.*?)\}', re.DOTALL)

def strip_grammar_marks(text: str) -> str:
    return GP_MARK_RE.sub(r'\2', str(text or ""))

def highlight_syntax(text):
    """[어구](S) 구문 표시를 ruby로 바꾸고, {번호:표현} 어법 마커를 번호 하이라이트로 바꿈"""
    raw = str(text or "")

    def keep_or_drop(m):
        # 대괄호 경계를 가로지르는 마커는 HTML이 깨지므로 표시만 지우고 단어는 남김
        inner = m.group(2)
        if inner.count("[") != inner.count("]"):
            return inner
        return "\x00" + m.group(1) + "\x01" + inner + "\x02"
    raw = GP_MARK_RE.sub(keep_or_drop, raw)

    text = esc(raw)
    pattern = r'\[(.*?)\]\s*\(([SVOCM])\)'
    replacement = r'<ruby class="syntax-ruby st-\2"><span class="syntax-text">\1</span><rt>(\2)</rt></ruby>'
    text = re.sub(pattern, replacement, text, flags=re.DOTALL)
    text = re.sub(r'\x00(\d+)\x01(.*?)\x02', r'<span class="gp-mark"><span class="gp-num">\1</span>\2</span>', text, flags=re.DOTALL)
    return text.replace("\x00", "").replace("\x01", "").replace("\x02", "")

def render_victory(p):
    data = p["data"]
    analysis_list = data.get("analysis", [])
    img_url = get_image_base64(p)
    img_html = f'<img src="{img_url}">' if img_url else '<div style="color:#ccc;">No Image</div>'
    kw_html = "".join([
        f'<div class="keyword-item"><span class="keyword-word">{esc(kw.get("word",""))}</span> ({esc(kw.get("meaning",""))}){(" : " + esc(kw.get("synonyms",""))) if str(kw.get("synonyms","")).strip() else ""}</div>'
        for kw in data.get("keywords", [])[:8]
    ])
    top_section = f"""
    <div class="top-layout">
        <div class="visual-box">{img_html}</div>
        <div class="summary-box">🎯 한판승 핵심 요약:<br>{esc(data.get('summary', ''))}</div>
    </div>
    <div class="keyword-section">
        <div class="keyword-title">📍 반드시 알아야 할 핵심 키워드</div>
        <div class="keyword-grid">{kw_html}</div>
    </div>
    """
    rows_html = ""
    for sent_idx, row in enumerate(analysis_list, start=1):
        eng_highlighted = highlight_syntax(row.get('eng', ''))
        vocab_list = row.get("sentence_vocab", [])
        vocab_html = ""
        if vocab_list:
            v_items = [
                f"<span style='white-space: nowrap;'>• <b>{esc(v.get('word', ''))}</b> ({esc(v.get('pronunciation', ''))}) [{esc(v.get('pos', ''))}] {esc(v.get('meaning', ''))}</span>"
                for v in vocab_list
            ]
            vocab_html = f"""<div class="inline-vocab-box">
                <div style="font-weight: 700; margin-bottom: 5px; color: #333;">📖 Sentence Vocab</div>
                <div style="display: flex; flex-wrap: wrap; gap: 4px 15px;">{"".join(v_items)}</div>
            </div>"""
        # 새 형식: 문장 속 번호 마커 + 오른쪽 어법 POINT 칸 / 예전 데이터: 기존 Grammar Point 상자
        points = [g for g in row.get("grammar_points", []) or [] if g.get("title") or g.get("explain")]
        grammar_html = ""
        panel_html = ""
        if points:
            items = "".join(
                f'<div class="gp-item"><span class="gp-num">{esc(g.get("no", i))}</span><b>{esc(g.get("title", ""))}</b> : {esc(g.get("explain", ""))}</div>'
                for i, g in enumerate(points, start=1)
            )
            panel_html = f'<div class="gp-panel"><div class="gp-title">+ 어법 POINT</div>{items}</div>'
        elif row.get("grammar"):
            grammar_html = f'<div class="inline-grammar-box"><b>💡 Grammar Point</b> : {esc(row.get("grammar"))}</div>'

        is_essay = bool(row.get("essay_candidate"))
        essay_reason = str(row.get("essay_reason", "") or "").strip()
        essay_badge = (f'<div class="essay-badge">서술형 출제 가능{(" · " + esc(essay_reason)) if essay_reason else ""}</div>') if is_essay else ""

        rows_html += f"""
        <tr>
            <td>
                <div class="sent-row">
                    <div class="sent-main">
                        <div class="eng-box{' essay' if is_essay else ''}">{essay_badge}<div><span class="sent-no">{sent_idx}</span>{eng_highlighted}</div></div>
                        {vocab_html}
                        <div class="trans-box"><span style="color:#64748b; font-weight:900; margin-right:5px;">K.</span>{esc(row.get('kor', ''))}</div>
                        {grammar_html}
                        <div class="ys-box"><div class="ys-header"><div class="ys-img"></div>영쌤의 한판승 코칭</div>{esc(row.get('ys', ''))}</div>
                    </div>
                    {panel_html}
                </div>
            </td>
        </tr>"""
    label_text = esc(p['label'])
    pid = esc(p.get('id', label_text))
    academy = esc(p.get('academy_name', '4pm 입시영어'))

    return f"""
    <div class="passage-wrapper" data-passage-id="{pid}" data-passage-label="{label_text}">
        <table class="master-table">
            <thead>
                <tr><td>
                    <div class="header">
                        <div class="logo-area"><div class="h-img"></div>ANATOMY 한판승</div>
                        <div style="text-align:right; font-size:11px;"><b>{academy}</b><br>{label_text}</div>
                    </div>
                </td></tr>
            </thead>
            <tfoot>
                <tr><td>
                    <div class="footer">
                        <div class="f-img"></div>{academy} | 시험 직전 최후의 승자, 한판승
                    </div>
                </td></tr>
            </tfoot>
            <tbody>
                <tr><td>
                    {top_section}
                    <table class="analysis-table">
                        <tbody>{rows_html}</tbody>
                    </table>
                </td></tr>
            </tbody>
        </table>
    </div>"""

def wrap_full_html_victory(body_html, css, front_html="", back_html=""):
    return f"""<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8"><title>Anatomy_One_Shot</title>{css}</head><body>{get_studio_toolbar_html('standard')}<div id="book-wrapper-all">{front_html}<main id="book-container">{body_html}</main>{back_html}</div></body></html>"""


# -----------------------------
# 3-B. [ANATOMY: Anatomy Self-Note] 서식 및 렌더링
# -----------------------------
def build_css_anatomy(h_logo, f_logo, e_logo):
    return f"""<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;700;900&display=swap');
    @page {{ size: A4; margin: 0; }}
    body {{ font-family: 'Noto Sans KR', sans-serif; overflow-wrap: anywhere;margin: 0; padding: 0; background: #f4f7fa; color: #333; -webkit-print-color-adjust: exact; print-color-adjust: exact; counter-reset: pagenum; }}
    {BOOK_COVER_CSS}
    #book-container {{ counter-reset: pagenum; }}
    .page {{ --scale: 1; width: 210mm; height: 297mm; padding: calc(15mm * var(--scale)) calc(18mm * var(--scale)) calc(12mm * var(--scale)) calc(18mm * var(--scale)); margin: 10mm auto; background: white; box-shadow: 0 0 15px rgba(0,0,0,0.10); box-sizing: border-box; overflow: hidden; display: flex; flex-direction: column; page-break-after: always; position: relative; counter-increment: pagenum; }}
    .header {{ display: flex; justify-content: space-between; align-items: flex-end; border-bottom: 4px solid #0047AB; padding-bottom: calc(12px * var(--scale)); margin-bottom: calc(18px * var(--scale)); flex: 0 0 auto; }}
    .logo-area {{ color: #0047AB; font-weight: 900; font-size: calc(30px * var(--scale)); display: flex; align-items: center; gap: calc(12px * var(--scale)); }}
    .header-char {{ width: calc(50px * var(--scale)); height: calc(50px * var(--scale)); background: url('{h_logo}') center/cover; border-radius: 50%; border: 2px solid #0047AB; flex-shrink: 0; }}
    .info-area {{ text-align: right; font-size: calc(14px * var(--scale)); color: #555; line-height: 1.5; }}
    .page-content {{ flex: 1 1 auto; min-height: 0; overflow: hidden; display: flex; flex-direction: column; gap: calc(8px * var(--scale)); }}
    .footer {{ flex: 0 0 auto; margin-top: calc(8px * var(--scale)); text-align: center; font-size: calc(11px * var(--scale)); color: #999; border-top: 1px solid #eee; padding-top: calc(8px * var(--scale)); display: flex; justify-content: center; align-items: center; gap: calc(8px * var(--scale)); }}
    .footer-char {{ width: calc(25px * var(--scale)); height: calc(25px * var(--scale)); background: url('{f_logo}') center/cover; border-radius: 50%; flex-shrink: 0; }}
    .section-title {{ font-size: calc(16px * var(--scale)); font-weight: bold; color: #0047AB; border-left: 6px solid #0047AB; padding-left: calc(10px * var(--scale)); margin: calc(4px * var(--scale)) 0; flex: 0 0 auto; }}
    .visual-container {{ width: 100%; min-height: calc(250px * var(--scale)); border: 2px dashed #0047AB; border-radius: calc(15px * var(--scale)); display: flex; align-items: center; justify-content: center; background: #fcfcfc; padding: calc(12px * var(--scale)); box-sizing: border-box; overflow: hidden; flex: 0 0 auto; }}
    .visual-image {{ width: 100%; height: 100%; max-width: 100%; max-height: calc(220px * var(--scale)); object-fit: contain; border-radius: calc(12px * var(--scale)); box-shadow: 0 4px 14px rgba(0,0,0,0.08); background: white; display: block; }}
    .title-area {{ text-align: center; margin-bottom: calc(4px * var(--scale)); flex: 0 0 auto; }}
    .title-en {{ font-size: calc(26px * var(--scale)); font-weight: 900; line-height: 1.3; }}
    .title-ko {{ font-size: calc(18px * var(--scale)); color: #666; line-height: 1.4; margin-top: calc(4px * var(--scale)); }}
    .topic-box {{ background: #0047AB; color: white; padding: calc(18px * var(--scale)); border-radius: calc(12px * var(--scale)); font-weight: bold; line-height: 1.65; font-size: calc(14.5px * var(--scale)); flex: 0 0 auto; }}
    .vocab-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: calc(8px * var(--scale)); flex: 1 1 auto; min-height: 0; overflow: hidden; }}
    .vocab-item {{ background: #f8f9fa; border: 1px solid #dee2e6; border-radius: calc(8px * var(--scale)); padding: calc(10px * var(--scale)) calc(12px * var(--scale)); overflow: hidden; min-height: 0; display: flex; align-items: stretch; justify-content: stretch; }}
    .vocab-fit {{ width: 100%; height: 100%; overflow: hidden; font-size: calc(12.5px * var(--scale)); line-height: 1.45; }}
    .vocab-word {{ font-weight: 900; color: #0047AB; display: block; margin-bottom: calc(2px * var(--scale)); font-size: 1em; line-height: 1.35; word-break: break-word; }}
    .vocab-syn {{ font-size: 0.88em; color: #666; display: block; line-height: 1.4; word-break: break-word; }}
    .passage-box {{ background: #fff; border: 2px solid #555; padding: calc(22px * var(--scale)); border-radius: calc(12px * var(--scale)); text-align: left; overflow: hidden; flex: 1 1 auto; min-height: 0; }}
    .passage-fit {{ width: 100%; height: 100%; overflow: hidden; font-size: calc(12px * var(--scale)); font-weight: 800; line-height: 2.8 !important; word-break: keep-all; }}
    .passage-meta-badge {{ display: inline-block; margin-bottom: calc(5px * var(--scale)); padding: calc(2px * var(--scale)) calc(4px * var(--scale)); background: #fff4e5; border: 1px solid #f5c27a; border-radius: 999px; color: #9a5a00; font-weight: 900; font-size: calc(5px * var(--scale)); letter-spacing: 0; }}
    .trans-content {{ font-size: calc(8px * var(--scale)); line-height: 1.75; color: #444; border-left: 5px solid #ddd; padding-left: calc(15px * var(--scale)); white-space: pre-line; overflow: hidden; flex: 1 1 auto; min-height: 0; }}
    .grammar-box {{ background: #fff9c4; padding: calc(18px * var(--scale)); border-radius: calc(12px * var(--scale)); border: 1px solid #f9e79f; font-size: calc(8px * var(--scale)); line-height: 1.75; overflow: hidden; flex: 0 0 auto; }}
    .easy-box {{ display: flex; gap: calc(10px * var(--scale)); background: #e8f0fe; padding: calc(10px * var(--scale)); border-radius: calc(12px * var(--scale)); border: 1px solid #b3d1ff; flex: 0 0 auto; }}
    .easy-char {{ width: calc(55px * var(--scale)); height: calc(55px * var(--scale)); background: url('{e_logo}') center/cover; border-radius: 12px; flex-shrink: 0; border: 2px solid white; }}
    .easy-text {{ font-size: calc(13px * var(--scale)); color: #0047AB; line-height: 1.55; overflow: hidden; }}
    .structure-box {{ text-align: center; padding: calc(10px * var(--scale)); border: 2px dashed #0047AB; border-radius: calc(15px * var(--scale)); font-weight: bold; color: #0047AB; background: #f9fbff; line-height: 1.6; white-space: pre-line; font-size: calc(12.5px * var(--scale)); overflow: hidden; flex: 0 0 auto; }}
    .quiz-container {{ background: #f8f9fa; padding: calc(16px * var(--scale)); border-radius: calc(12px * var(--scale)); border: 1px solid #dee2e6; overflow: hidden; flex: 1 1 auto; min-height: 0; }}
    .quiz-row {{ display: flex; justify-content: space-between; align-items: stretch; border-bottom: 1px solid #eee; padding: calc(6px * var(--scale)) 0; gap: calc(12px * var(--scale)); overflow: hidden; min-height: 0; }}
    .quiz-text {{ flex: 1; min-width: 0; overflow: hidden; font-size: calc(12px * var(--scale)); line-height: 1.45; }}
    .quiz-en {{ font-weight: 700; color: #222; margin-bottom: calc(4px * var(--scale)); line-height: 1.45; font-size: 1em; word-break: break-word; }}
    .quiz-ko {{ color: #4a4a4a; font-size: 0.92em; line-height: 1.45; word-break: break-word; }}
    .quiz-choice {{ white-space: nowrap; font-weight: bold; color: #0047AB; margin-left: calc(10px * var(--scale)); font-size: calc(12px * var(--scale)); letter-spacing: 1px; flex-shrink: 0; align-self: center; }}
    .quiz-answer-key {{ margin-top: calc(8px * var(--scale)); text-align: right; font-size: calc(10px * var(--scale)); color: #bbb; letter-spacing: 1px; flex: 0 0 auto; }}
    /* SELF-NOTE 2단 구성: 왼쪽 지문·해석 + 오른쪽 한줄 요약·필기노트·어휘 (기존 파란 상자 스타일 유지) */
    .sn-cols {{ flex: 1 1 auto; min-height: 0; display: grid; grid-template-columns: minmax(0, 1fr) calc(56mm * var(--scale)); gap: calc(14px * var(--scale)); }}
    .sn-left, .sn-right {{ display: flex; flex-direction: column; gap: calc(8px * var(--scale)); min-height: 0; overflow: hidden; }}
    .sn-qtag {{ color: #9a5a00; background: #fff4e5; border: 1px solid #f5c27a; border-radius: 999px; font-weight: 900; font-size: 0.75em; padding: 1px 9px; margin-left: 10px; vertical-align: 1px; white-space: nowrap; }}
    .sn-passage-fit {{ font-weight: 800; word-break: keep-all; line-height: 2.6 !important; }}
    /* 제목 줄은 한 줄 유지: 시험 정보가 길면 끝을 …로 줄임 */
    .sn-left .section-title {{ display: flex; align-items: center; white-space: nowrap; min-width: 0; }}
    .sn-left .section-title .sn-qtag {{ font-size: 0.68em; padding: 1px 7px; margin-left: 6px; min-width: 0; overflow: hidden; text-overflow: ellipsis; }}
    .sn-num {{ display: inline-flex; align-items: center; justify-content: center; min-width: 1.3em; height: 1.3em; padding: 0 0.15em; box-sizing: border-box; border-radius: 999px; background: #0047AB; color: white; font-size: 0.72em; font-weight: 900; line-height: 1; margin-right: 0.3em; vertical-align: 0.12em; letter-spacing: -0.03em; }}
    .sn-trans {{ flex: 0 0 auto; font-size: calc(9.5px * var(--scale)); line-height: 1.7; color: #444; border-left: 5px solid #ddd; padding-left: calc(12px * var(--scale)); word-break: keep-all; }}
    .sn-trans .sn-num {{ background: #64748b; }}
    .sn-summary {{ flex: 0 0 auto; background: #0047AB; color: white; padding: calc(12px * var(--scale)); border-radius: calc(12px * var(--scale)); font-weight: bold; line-height: 1.6; font-size: calc(12px * var(--scale)); word-break: keep-all; }}
    .sn-note {{ flex: 1 1 auto; min-height: calc(120px * var(--scale)); border: 2px dashed #0047AB; border-radius: calc(12px * var(--scale)); background-color: #fcfcfc; background-image: repeating-linear-gradient(to bottom, transparent 0, transparent calc(26px * var(--scale) - 1px), #e2e8f0 calc(26px * var(--scale) - 1px), #e2e8f0 calc(26px * var(--scale))); }}
    .sn-vocab {{ flex: 0 0 auto; background: #f8f9fa; border: 1px solid #dee2e6; border-radius: calc(8px * var(--scale)); padding: calc(8px * var(--scale)) calc(10px * var(--scale)); display: grid; grid-template-columns: auto 1fr; gap: calc(3px * var(--scale)) calc(8px * var(--scale)); font-size: calc(10px * var(--scale)); line-height: 1.4; }}
    .sn-vocab b {{ color: #0047AB; font-weight: 900; }}
    .auto-fit-box {{ overflow: hidden; }}
    .auto-fit-target {{ display: block; width: 100%; }}
    @media print {{ .page {{ margin: 0; box-shadow: none; border: none; page-break-after: always; }} }}
    .page-num {{ position: absolute; bottom: 12mm; font-size: 11px; font-weight: 900; color: #999; }}
    .page-num::after {{ content: "- " counter(pagenum) " -"; }}
    #book-container > .page:nth-child(odd) .page-num {{ right: 18mm; }}
    #book-container > .page:nth-child(even) .page-num {{ left: 18mm; }}
</style>"""

FIT_SCRIPT = """
<script>
function innerHeight(box) {
    // 상자 안쪽 여백(padding)을 뺀 실제 글자 공간 높이
    const cs = getComputedStyle(box);
    return box.clientHeight - parseFloat(cs.paddingTop || 0) - parseFloat(cs.paddingBottom || 0);
}
function fitTextBox(box) {
    const target = box.querySelector('.auto-fit-target');
    if (!target) return;
    const max = parseFloat(target.dataset.max || "12.5");
    const min = parseFloat(target.dataset.min || "8");
    // grow-lh 상자는 가장 좁은 줄 간격에서 글자 크기를 맞추고, 남는 공간은 growLineHeight가 줄 간격으로 채움
    const growBox = box.classList.contains('grow-lh');
    const setLh = (size, ratio) => {
        if (growBox) target.style.setProperty('line-height', box.dataset.minLh || '2.3', 'important');
        else target.style.lineHeight = (size * ratio) + "px";
    };
    let size = max, guard = 0;
    target.style.fontSize = size + "px";
    setLh(size, 1.40);
    while ((target.scrollHeight > innerHeight(box) || target.scrollWidth > box.clientWidth) && size > min && guard < 120) {
        size -= 0.2; target.style.fontSize = size + "px"; setLh(size, 1.36); guard++;
    }
}
function fitOnePage(page) {
    let scale = 1.00;
    const content = page.querySelector('.page-content');
    if (!content) return;
    page.style.setProperty('--scale', scale.toFixed(2));
    let guard = 0;
    while (content.scrollHeight > content.clientHeight && scale > 0.72 && guard < 20) {
        scale -= 0.03; page.style.setProperty('--scale', scale.toFixed(2)); guard++;
    }
    while (content.scrollHeight > content.clientHeight && scale > 0.64 && guard < 40) {
        scale -= 0.01; page.style.setProperty('--scale', scale.toFixed(2)); guard++;
    }
}
function growLineHeight(box) {
    // 지문이 짧아 아래가 비면 줄 간격을 넓혀 칸을 채움 (필기 공간 확보)
    const target = box.querySelector('.auto-fit-target');
    if (!target) return;
    const minLh = parseFloat(box.dataset.minLh || "2.3"), maxLh = parseFloat(box.dataset.maxLh || "2.9");
    let lh = minLh;
    target.style.setProperty('line-height', lh, 'important');
    while (lh + 0.05 <= maxLh) {
        target.style.setProperty('line-height', (lh + 0.05).toFixed(2), 'important');
        if (target.scrollHeight > innerHeight(box)) { target.style.setProperty('line-height', lh.toFixed(2), 'important'); break; }
        lh += 0.05;
    }
}
function fitAllPages() {
    const pages = document.querySelectorAll('.fit-page');
    pages.forEach(p => p.style.setProperty('--scale', '1'));
    document.querySelectorAll('.grow-lh').forEach(b => { const t = b.querySelector('.auto-fit-target'); if (t) t.style.setProperty('line-height', b.dataset.minLh || '2.3', 'important'); });
    document.querySelectorAll('.auto-fit-box').forEach(b => fitTextBox(b));
    pages.forEach(p => fitOnePage(p));
    document.querySelectorAll('.auto-fit-box').forEach(b => fitTextBox(b));
    pages.forEach(p => fitOnePage(p));
    document.querySelectorAll('.grow-lh').forEach(b => growLineHeight(b));
}
window.addEventListener('load', function() {
    setTimeout(fitAllPages, 120);
    if (document.fonts && document.fonts.ready) { document.fonts.ready.then(() => { setTimeout(fitAllPages, 120); setTimeout(fitAllPages, 260); setTimeout(fitAllPages, 420); }); }
});
window.addEventListener('resize', () => setTimeout(fitAllPages, 120));
</script>
"""

def wrap_full_html_anatomy(body_html, css, front_html="", back_html=""):
    return f"""<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8"><title>Anatomy_Self_Note</title>{css}</head><body>{get_studio_toolbar_html('standard')}<div id="book-wrapper-all">{front_html}<main id="book-container">{body_html}</main>{back_html}</div>{FIT_SCRIPT}</body></html>"""

def format_passage_for_display(body_text: str) -> str:
    if not body_text:
        return ""
    raw = str(body_text).strip().replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [re.sub(r'\s+', ' ', para).strip() for para in re.split(r'\n\s*\n', raw) if para.strip()]
    return "<br><br>".join([esc(p) for p in paragraphs])

def circled_num(n: int) -> str:
    """문장 번호 동그라미. ❶ 같은 원문자는 20까지만 있어서 CSS(.sn-num)로 그려 모든 번호를 같은 모양으로 맞춤"""
    return f'<span class="sn-num">{n}</span>'

def split_passage_sentences(body_text: str) -> List[str]:
    """줄바꿈(인사말·서명 등)과 문장부호 기준으로 문장을 나눔 (AI 문장 목록이 없거나 원문과 다를 때 사용)"""
    out = []
    for line in re.split(r'\n+', str(body_text or "").replace("\r", "")):
        line = re.sub(r'\s+', ' ', line).strip()
        if not line:
            continue
        # 문장부호(.!?) 바로 뒤 또는 문장부호+닫는 따옴표/괄호 뒤에서 나눔 (look-behind는 고정 길이만 가능해 두 경우로 나눔)
        pieces = [s.strip() for s in re.split(r'(?:(?<=[.!?])|(?<=[.!?]["”’)]))\s+(?=["“(]?[A-Z0-9])', line) if s.strip()]
        # Mr. / Ms. / Dr. 같은 약어 뒤에서 잘린 조각은 다음 조각과 다시 붙임
        merged = []
        for piece in pieces:
            if merged and re.search(r'\b(?:Mr|Mrs|Ms|Dr|Prof|St|Jr|Sr|vs|e\.g|i\.e|U\.S)\.$', merged[-1]):
                merged[-1] += " " + piece
            else:
                merged.append(piece)
        out.extend(merged)
    return out

def anatomy_sentence_pairs(p: Dict) -> (List[str], List[str]):
    """(영어 문장 목록, 해석 목록). AI가 준 문장이 원문과 같을 때만 그 문장·해석을 쓰고, 아니면 원문을 직접 나눔"""
    body = p.get("body_text", "")
    ai = [s for s in p.get("data", {}).get("sentences", []) or [] if s.get("eng")]
    norm = lambda t: re.sub(r'\s+', '', str(t))
    if ai and norm("".join(s["eng"] for s in ai)) == norm(body):
        return [s["eng"].strip() for s in ai], [s.get("kor", "").strip() for s in ai]
    return split_passage_sentences(body), []

def sentence_line_breaks(body_text: str, sents: List[str]) -> List[bool]:
    """각 문장 뒤에 원문 줄바꿈이 있는지 (편지의 인사말·맺음말·서명을 원문처럼 줄 바꿔 보여주기 위함)"""
    flat = str(body_text or "").replace("\r", "")
    breaks, cursor = [], 0
    for s in sents:
        # 공백 차이를 무시하고 원문에서 문장 위치를 찾음
        pat = r'\s+'.join(re.escape(w) for w in s.split())
        m = re.compile(pat).search(flat, cursor) if pat else None
        if not m:
            breaks.append(False)
            continue
        cursor = m.end()
        breaks.append(bool(re.match(r'[ \t]*\n', flat[cursor:])))
    return breaks

def numbered_sentences_html(sents: List[str], breaks: List[bool]) -> str:
    parts = []
    for i, s in enumerate(sents, start=1):
        sep = "<br>" if i - 1 < len(breaks) and breaks[i - 1] else " "
        parts.append(f'{circled_num(i)}{esc(s)}{sep if i < len(sents) else ""}')
    return "".join(parts)

def render_anatomy(p):
    data = p["data"]
    img_url = get_image_base64(p)
    img_html = f'<img class="visual-image" src="{img_url}">' if img_url else '<div class="visual-empty"></div>'
    academy = esc(p.get('academy_name', '4pm 입시영어'))
    school = esc(p.get('school_name', ''))
    label_text = esc(p["label"])
    pid = esc(p.get("id", label_text))
    body_html = format_passage_for_display(p.get('body_text', ''))

    vocab_html = "".join([
        f"""<div class="vocab-item auto-fit-box"><div class="vocab-fit auto-fit-target" data-max="12.5" data-min="8.2"><span class="vocab-word">{esc(v.get("word",""))} ({esc(v.get("meaning",""))})</span><span class="vocab-syn">{esc(v.get("synonyms_text",""))}</span></div></div>"""
        for v in data.get("vocab", [])
    ])
    grammar_html = "<br>".join([
        f"<b>{i+1}. {esc(g.get('title',''))}</b>: {esc(g.get('explanation',''))}"
        for i, g in enumerate(data.get("grammar_points", []))
    ])
    quiz_html = "".join([
        f"""<div class="quiz-row auto-fit-box"><div class="quiz-text auto-fit-target" data-max="12" data-min="8.0"><div class="quiz-en">{esc(q.get("english_statement",""))}</div><div class="quiz-ko">{esc(q.get("korean_statement",""))}</div></div><span class="quiz-choice">( O / X )</span></div>"""
        for q in data.get("quizzes", [])
    ])
    answers = "ANS: " + " ".join([f"{i+1}.{q.get('answer','')}" for i, q in enumerate(data.get("quizzes", []))])
    footer_left  = f"{academy} | 내신과 수능 1등급을 위한 필수선택 영어는 영목쌤"
    footer_right = f"{academy} | 영목쌤"
    data_attr = f'data-passage-id="{pid}" data-passage-label="{label_text}"'

    p1 = f"""<div class="page fit-page" {data_attr}><div class="header"><div class="logo-area"><div class="header-char"></div>ANATOMY PREVIEW</div><div class="info-area"><b>{academy}</b><br>{label_text}</div></div><div class="page-content"><div class="visual-container">{img_html}</div><div class="title-area"><div class="title-en">{esc(data.get("english_title",""))}</div><div class="title-ko">{esc(data.get("korean_title",""))}</div></div><div class="section-title">Topic</div><div class="topic-box">{esc(data.get("topic",""))}</div><div class="section-title">Vocabulary Focus (동의어 뜻 병기)</div><div class="vocab-grid">{vocab_html}</div></div><div class="footer"><div class="footer-char"></div>{footer_left}</div><div class="page-num"></div></div>"""
    # ---- 2쪽 SELF-NOTE (왼쪽 번호 매긴 지문·해석 + 오른쪽 한줄 요약·필기노트·어휘) ----
    eng_sents, kor_sents = anatomy_sentence_pairs(p)
    numbered_passage = numbered_sentences_html(eng_sents, sentence_line_breaks(p.get("body_text", ""), eng_sents))
    if kor_sents:
        trans_html = " ".join(f'{circled_num(i)}{esc(k)}' for i, k in enumerate(kor_sents, start=1))
    else:
        trans_html = nl2br(data.get("translation", ""))
    # 시험 정보(예: 2026년 10월 고1 18번)는 지문 상자 대신 제목 옆에 작게 표시
    exam_tag = f'<span class="sn-qtag">{esc(clean_label(p.get("label", "")).replace("[", "").replace("]", ""))}</span>'
    one_line = esc(data.get("one_line_summary") or data.get("topic", ""))
    sn_vocab = "".join(f'<b>{esc(v.get("word", ""))}</b><span>{esc(v.get("meaning", ""))}</span>' for v in data.get("vocab", []))

    p2 = f"""<div class="page fit-page" {data_attr}><div class="header"><div class="logo-area"><div class="header-char"></div>ANATOMY SELF-NOTE</div><div class="info-area"><b>{academy}</b><br>셀프노트: 너만의 손필기</div></div><div class="page-content"><div class="sn-cols"><div class="sn-left"><div class="section-title">■ Original Text (강의용 필기 노트){exam_tag}</div><div class="passage-box sn-passage auto-fit-box grow-lh" data-min-lh="2.2" data-max-lh="3.6"><div class="sn-passage-fit auto-fit-target" data-max="12.5" data-min="8.0"><div>{numbered_passage}</div></div></div><div class="section-title">■ 지문 해석</div><div class="sn-trans">{trans_html}</div></div><div class="sn-right"><div class="section-title">영쌤의 한줄 요약</div><div class="sn-summary">{one_line}</div><div class="section-title">영쌤의 필기노트</div><div class="sn-note"></div><div class="section-title">Vocabulary</div><div class="sn-vocab">{sn_vocab}</div></div></div></div><div class="footer"><div class="footer-char"></div>{footer_right}</div><div class="page-num"></div></div>"""
    p3 = f"""<div class="page fit-page" {data_attr}><div class="header"><div class="logo-area"><div class="header-char"></div>ANATOMY REVIEW-NOTE</div><div class="info-area"><b>{academy}</b><br>리뷰노트: 따라 쓰며 복습!</div></div><div class="page-content"><div class="section-title">■ Original Text (너만의 복습 노트){exam_tag}</div><div class="passage-box auto-fit-box grow-lh" style="border-color:#0047AB;" data-min-lh="2.3" data-max-lh="3.4"><div class="passage-fit auto-fit-target" data-max="14.5" data-min="8.0"><div class="passage-body">{body_html}</div></div></div><div class="section-title">■ 주요 문법 Point 3 정리</div><div class="grammar-box">{grammar_html}</div></div><div class="footer"><div class="footer-char"></div>{footer_left}</div><div class="page-num"></div></div>"""
    p4 = f"""<div class="page fit-page" {data_attr}><div class="header"><div class="logo-area"><div class="header-char"></div>STRUCTURE & QUIZ</div><div class="info-area"><b>{academy}</b><br>최종 확인 | {school}</div></div><div class="page-content"><div class="section-title">■ 영쌤의 쉬운 지문 설명</div><div class="easy-box"><div class="easy-char"></div><div class="easy-text">{nl2br(data.get('easy_explanation',''))}</div></div><div class="section-title">■ 글의 흐름 (Structure)</div><div class="structure-box">{nl2br(data.get("structure_flow",""))}</div><div class="section-title">■ ANATOMY OX Quiz (5문항)</div><div class="quiz-container">{quiz_html}</div><div class="quiz-answer-key">{esc(answers)}</div></div><div class="footer"><div class="footer-char"></div>{footer_right}</div><div class="page-num"></div></div>"""
    return p1 + p2 + p3 + p4


# -----------------------------
# 3-C. [💎 ANATOMY CORE] 서식 및 렌더링
# -----------------------------
def build_css_core(h_img, f_img, e_img):
    return f"""<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700;900&display=swap');
    @page {{ size: A4; margin: 12mm 14mm; }}
    body {{ font-family: 'Noto Sans KR', sans-serif; overflow-wrap: anywhere;background: #f0f2f5; margin: 0; padding: 0; color: #1e293b; -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }}
    {BOOK_COVER_CSS}
    .core-wrapper {{ width: 210mm; box-sizing: border-box; padding: 13mm 15mm; margin: 10mm auto; background: white; box-shadow: 0 0 12px rgba(0,0,0,0.1); page-break-after: always; }}
    @media print {{ body {{ background: white; }} .core-wrapper {{ width: 100%; margin: 0; padding: 0; box-shadow: none; border: none; }} }}
    .core-master-table {{ width: 100%; border-collapse: collapse; }}
    .core-master-table thead {{ display: table-header-group; }}
    .core-master-table tfoot {{ display: table-footer-group; }}
    .core-header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 3px solid #0047AB; padding-bottom: 8px; margin-bottom: 10px; }}
    .core-logo-area {{ display: flex; align-items: center; gap: 10px; font-weight: 900; font-size: 22px; color: #0047AB; }}
    .core-h-img {{ width: 40px; height: 40px; background: url('{h_img}') center/cover; border-radius: 50%; border: 2px solid #0047AB; }}
    .core-footer {{ text-align: center; font-size: 10px; color: #94a3b8; border-top: 1px solid #e2e8f0; padding-top: 8px; margin-top: 12px; display: flex; justify-content: center; align-items: center; gap: 8px; }}
    .core-f-img {{ width: 22px; height: 22px; background: url('{f_img}') center/cover; border-radius: 50%; }}
    .core-sec-title {{ font-size: 13px; font-weight: 900; color: #0047AB; background: #eff6ff; border-left: 5px solid #0047AB; padding: 5px 10px; margin: 10px 0 6px 0; border-radius: 0 6px 6px 0; display: flex; justify-content: space-between; align-items: center; }}
    .core-grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 8px; }}
    .core-card {{ border: 1px solid #cbd5e1; border-radius: 8px; padding: 8px 10px; background: #ffffff; page-break-inside: avoid; break-inside: avoid; }}
    .core-card-title {{ font-size: 11px; font-weight: 900; color: #1e40af; margin-bottom: 5px; border-bottom: 1px dashed #cbd5e1; padding-bottom: 3px; }}
    .core-item {{ margin-bottom: 6px; font-size: 11px; line-height: 1.45; }}
    .core-item:last-child {{ margin-bottom: 0; }}
    .core-en {{ font-weight: 800; color: #0f172a; font-size: 11.5px; }}
    .core-ko {{ color: #475569; font-size: 10.5px; margin-top: 1px; }}
    .core-badge {{ display: inline-block; background: #dbeafe; color: #1d4ed8; font-weight: 800; font-size: 9.5px; padding: 1px 5px; border-radius: 4px; margin-right: 4px; }}
    .summary-card {{ border: 1.5px solid #93c5fd; background: #f8fafc; border-radius: 8px; padding: 9px 12px; margin-bottom: 7px; page-break-inside: avoid; break-inside: avoid; }}
    .summary-en {{ font-size: 12px; font-weight: 800; color: #0f172a; line-height: 1.65; margin-bottom: 4px; }}
    .summary-ko {{ font-size: 11px; color: #475569; margin-bottom: 6px; }}
    .ab-choice-pill {{ display: inline-block; background: #eff6ff; border: 1.5px solid #3b82f6; color: #1e293b; font-weight: 800; padding: 0px 7px; border-radius: 5px; margin: 0 2px; }}
    .ab-correct-opt {{ color: #0047AB; font-weight: 900; text-decoration: underline; text-underline-offset: 3px; }}
    .ab-wrong-opt {{ color: #64748b; font-weight: 700; }}
    .summary-ans-bar {{ display: flex; flex-wrap: wrap; gap: 6px 14px; background: #fffbeb; border: 1px dashed #fcd34d; border-radius: 6px; padding: 5px 9px; font-size: 10.8px; color: #78350f; }}
    .tf-row {{ border: 1px solid #e2e8f0; border-radius: 6px; padding: 6px 10px; margin-bottom: 5px; display: flex; justify-content: space-between; align-items: center; gap: 10px; background: #fcfcfd; page-break-inside: avoid; break-inside: avoid; }}
    .tf-ans-badge {{ font-weight: 900; font-size: 11px; padding: 2px 8px; border-radius: 999px; white-space: nowrap; }}
    .tf-O {{ background: #dcfce7; color: #15803d; border: 1px solid #86efac; }}
    .tf-X {{ background: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; }}
    .core-voca-strip {{ display: grid; grid-template-columns: 1fr 1fr; gap: 5px 10px; background: #fff4e5; border: 1px solid #f5c27a; border-radius: 8px; padding: 8px 10px; font-size: 10.5px; page-break-inside: avoid; break-inside: avoid; }}
    #book-container.core-test-mode .ab-choice-pill {{ background: #ffffff !important; border-color: #475569 !important; }}
    #book-container.core-test-mode .ab-correct-opt,
    #book-container.core-test-mode .ab-wrong-opt {{ color: #0f172a !important; font-weight: 800 !important; text-decoration: none !important; }}
    #book-container.core-test-mode .summary-ans-bar {{ display: none !important; }}
    #book-container.core-test-mode .tf-ans-reveal {{ display: none !important; }}
    #book-container.core-test-mode .tf-ans-badge.tf-O,
    #book-container.core-test-mode .tf-ans-badge.tf-X {{
        background: #f8fafc !important; color: #334155 !important; border: 1px solid #cbd5e1 !important;
    }}
    </style>"""

def extract_trap_word_fallback(syn_str: str, default_word: str = "opposite") -> str:
    if not syn_str:
        return default_word
    m = re.search(r'(?:\[함정\]|반의어\s*[:\-]?|/)\s*([A-Za-z\-]+)', str(syn_str))
    if m:
        return m.group(1).strip()
    words = re.findall(r'[A-Za-z\-]{3,}', str(syn_str))
    return words[-1] if words else default_word

def format_summary_dual_choice(eng_text: str, blank_a: str, dist_a: str, blank_b: str, dist_b: str, seed_key: str) -> str:
    escaped = esc(eng_text)
    rng = random.Random(seed_key)

    def make_pill(tag: str, ans_w: str, trap_w: str) -> str:
        ans_clean = esc(ans_w.strip())
        trap_clean = esc(trap_w.strip()) if trap_w and trap_w.strip() else "opposite"
        ans_span = f'<span class="ab-correct-opt">{ans_clean}</span>'
        trap_span = f'<span class="ab-wrong-opt">{trap_clean}</span>'
        opts = [ans_span, trap_span]
        rng.shuffle(opts)
        return f'<span class="ab-choice-pill">({tag}) [ {opts[0]} / {opts[1]} ]</span>'

    if blank_a:
        pill_a = make_pill("A", blank_a, dist_a)
        pat_a = r'\(\s*A\s*\)\s*' + re.escape(esc(blank_a))
        if re.search(pat_a, escaped, flags=re.IGNORECASE):
            escaped = re.sub(pat_a, pill_a, escaped, count=1, flags=re.IGNORECASE)
        else:
            escaped = re.sub(r'\(\s*A\s*\)', pill_a, escaped, count=1)

    if blank_b:
        pill_b = make_pill("B", blank_b, dist_b)
        pat_b = r'\(\s*B\s*\)\s*' + re.escape(esc(blank_b))
        if re.search(pat_b, escaped, flags=re.IGNORECASE):
            escaped = re.sub(pat_b, pill_b, escaped, count=1, flags=re.IGNORECASE)
        else:
            escaped = re.sub(r'\(\s*B\s*\)', pill_b, escaped, count=1)

    return escaped

def render_core(p: Dict) -> str:
    data = p.get("data", {})
    label_text = esc(p.get("label", ""))
    pid = esc(p.get("id", label_text))
    academy = esc(p.get("academy_name", "포피엠 입시 영어"))
    school = esc(p.get("school_name", ""))

    topics_html = "".join([
        f'<div class="core-item"><div class="core-en"><span class="core-badge">Topic {i}</span>{esc(t.get("eng", ""))}</div><div class="core-ko">↳ {esc(t.get("kor", ""))}</div></div>'
        for i, t in enumerate(data.get("topics", []), start=1)
    ])
    titles_html = "".join([
        f'<div class="core-item"><div class="core-en"><span class="core-badge">{esc(t.get("style", f"Title {i}"))}</span>{esc(t.get("eng", ""))}</div><div class="core-ko">↳ {esc(t.get("kor", ""))}</div></div>'
        for i, t in enumerate(data.get("titles", []), start=1)
    ])
    claims_html = "".join([
        f'<div class="core-item" style="border-bottom:1px dashed #e2e8f0; padding-bottom:5px;"><div class="core-en"><span class="core-badge" style="background:#e0e7ff; color:#3730a3;">요지·주장 {i}</span>{esc(c.get("eng", ""))}</div><div class="core-ko">↳ {esc(c.get("kor", ""))}</div></div>'
        for i, c in enumerate(data.get("main_ideas", []), start=1)
    ])

    summaries_html = ""
    for i, s in enumerate(data.get("summaries", []), start=1):
        ba = s.get("blank_a", "")
        bb = s.get("blank_b", "")
        sa_syn = s.get("synonyms_a", "")
        sb_syn = s.get("synonyms_b", "")
        dist_a = s.get("distractor_a") or extract_trap_word_fallback(sa_syn, "superficial")
        dist_b = s.get("distractor_b") or extract_trap_word_fallback(sb_syn, "passivity")

        eng_hl = format_summary_dual_choice(s.get("eng", ""), ba, dist_a, bb, dist_b, seed_key=f"{pid}_sum_{i}")
        summaries_html += f"""
        <div class="summary-card">
            <div class="summary-en"><span class="core-badge" style="background:#0047AB; color:white;">Summary {i}</span> {eng_hl}</div>
            <div class="summary-ko">해석: {esc(s.get('kor', ''))}</div>
            <div class="summary-ans-bar">
                <div>🎯 <b>(A) 정답: {esc(ba)}</b> vs 오답: {esc(dist_a)}{f' <span style="color:#475569;">({esc(sa_syn)})</span>' if str(sa_syn).strip() else ""}</div>
                <div>🎯 <b>(B) 정답: {esc(bb)}</b> vs 오답: {esc(dist_b)}{f' <span style="color:#475569;">({esc(sb_syn)})</span>' if str(sb_syn).strip() else ""}</div>
            </div>
        </div>"""

    tf_list = list(data.get("paraphrased_tf", []))
    rng_tf = random.Random(f"{pid}_tf_shuffle")
    rng_tf.shuffle(tf_list)

    tf_html = ""
    tf_ans_summary = []
    for i, tf in enumerate(tf_list, start=1):
        ans = tf.get("answer", "O")
        badge_cls = "tf-O" if ans == "O" else "tf-X"
        tf_ans_summary.append(f"{i}.{ans}")
        tf_html += f"""
        <div class="tf-row">
            <div style="flex:1;">
                <div class="core-en"><b>{i}.</b> {esc(tf.get('eng', ''))}</div>
                <div class="core-ko">↳ {esc(tf.get('kor', ''))} <span class="tf-ans-reveal" style="color:#b45309; font-weight:700;">| 💡 출제포인트: {esc(tf.get('point', ''))}</span></div>
            </div>
            <div class="tf-ans-badge {badge_cls}"><span class="tf-ans-reveal">정답: {esc(ans)} | </span>( O / X )</div>
        </div>"""

    cv_items = []
    for cv in data.get("core_vocab", [])[:10]:
        syn_txt = ", ".join([f"{s.get('word','')}({s.get('meaning','')})" for s in cv.get("synonyms", [])[:2] if s.get("word")])
        ant_txt = ", ".join([f"{a.get('word','')}({a.get('meaning','')})" for a in cv.get("antonyms", [])[:1] if a.get("word")])
        rel_str = " | ".join(x for x in ((f"유의: {syn_txt}" if syn_txt else ""), (f"반의: {ant_txt}" if ant_txt else "")) if x)
        cv_items.append(
            f"<div>• <b style='color:#0047AB;'>{esc(cv.get('word',''))}</b> [{esc(cv.get('pronunciation',''))}] "
            f"{esc(cv.get('meaning',''))} <span style='color:#64748b; font-size:9.8px;'>({esc(rel_str)})</span></div>"
        )
    cv_html = f'<div class="core-voca-strip">{"".join(cv_items)}</div>' if cv_items else ""

    return f"""
    <div class="core-wrapper" data-passage-id="{pid}" data-passage-label="{label_text}">
        <table class="core-master-table">
            <thead>
                <tr><td>
                    <div class="core-header">
                        <div class="core-logo-area"><div class="core-h-img"></div>PQRS영쌤 | ANATOMY CORE</div>
                        <div style="text-align:right; font-size:11px;"><b>{academy}</b> {('| ' + school) if school else ''}<br><span style="color:#0047AB; font-weight:900;">{label_text}</span></div>
                    </div>
                </td></tr>
            </thead>
            <tfoot>
                <tr><td>
                    <div class="core-footer">
                        <div class="core-f-img"></div>{academy} | 내신 1등급 결정타: 주제·제목·요지·요약문 패러프레이징 완벽 마스터
                    </div>
                </td></tr>
            </tfoot>
            <tbody>
                <tr><td>
                    <div class="core-grid-2">
                        <div class="core-card">
                            <div class="core-card-title">📌 출제 1순위 영문 주제 (Topic Paraphrasing 3선)</div>
                            {topics_html}
                        </div>
                        <div class="core-card">
                            <div class="core-card-title">🏷️ 내신 빈출 영문 제목 (Title Variations 3선)</div>
                            {titles_html}
                        </div>
                    </div>
                    <div class="core-sec-title">
                        <span>💡 출제 유력 영문 요지 & 주장문 (Main Idea & Claim Paraphrasing)</span>
                        <span style="font-size:10.5px; font-weight:500; color:#475569;">원문 문장을 상위 개념어로 재구성한 선지 대비용 문장</span>
                    </div>
                    <div class="core-card" style="margin-bottom:8px;">{claims_html}</div>
                    <div class="core-sec-title">
                        <span>🎯 실전 요약문 (Summary 3선) & 빈칸 (A), (B) 양자택일 어휘 추론</span>
                        <span style="font-size:10.5px; font-weight:500; color:#475569;">문맥에 맞는 알맞은 패러프레이징 어휘를 고르시오</span>
                    </div>
                    {summaries_html}
                    <div class="core-sec-title">
                        <span>🔍 학교 내신 &lt;보기&gt;형 패러프레이징 진위(T/F) 선지 분석 (순서 랜덤 배치)</span>
                        <span class="tf-ans-reveal" style="font-size:10.5px; font-weight:700; color:#1e40af;">정답: {' / '.join(tf_ans_summary)}</span>
                    </div>
                    <div style="margin-bottom:8px;">{tf_html}</div>
                    <div class="core-sec-title" style="background:#fffbeb; border-left-color:#d97706; color:#92400e;">
                        <span>✨ CORE 패러프레이징 핵심 어휘 (ANATOMY VOCA 자동 연동 어휘)</span>
                    </div>
                    {cv_html}
                </td></tr>
            </tbody>
        </table>
    </div>"""

def wrap_full_html_core(body_html, css, front_html="", back_html=""):
    return f"""<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8"><title>Anatomy_Core</title>{css}</head><body>{get_studio_toolbar_html('core')}<div id="book-wrapper-all">{front_html}<main id="book-container">{body_html}</main>{back_html}</div></body></html>"""


# -----------------------------
# 3-D. [📖 ANATOMY VOCA & 기출 VOCA] 서식 및 시험지별 연속 렌더링
# -----------------------------
def build_css_voca(h_img, f_img):
    return f"""<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700;900&display=swap');
    @page {{ size: A4; margin: 12mm 14mm; }}
    body {{ font-family: 'Noto Sans KR', sans-serif; overflow-wrap: anywhere;background: #f0f2f5; margin: 0; padding: 0; color: #1e293b; -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }}
    {BOOK_COVER_CSS}
    .voca-wrapper {{ width: 210mm; box-sizing: border-box; padding: 14mm 16mm; margin: 10mm auto; background: white; box-shadow: 0 0 12px rgba(0,0,0,0.1); page-break-after: always; break-after: page; }}
    @media print {{ body {{ background: white; }} .voca-wrapper {{ width: 100%; margin: 0; padding: 0; box-shadow: none; border: none; page-break-after: always; break-after: page; }} }}
    .voca-master-table {{ width: 100%; border-collapse: collapse; }}
    .voca-master-table thead {{ display: table-header-group; }}
    .voca-master-table tfoot {{ display: table-footer-group; }}
    .voca-header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 3px solid #0047AB; padding-bottom: 8px; margin-bottom: 12px; }}
    .voca-logo-area {{ display: flex; align-items: center; gap: 10px; font-weight: 900; font-size: 23px; color: #0047AB; letter-spacing: -0.5px; }}
    .voca-h-img {{ width: 40px; height: 40px; background: url('{h_img}') center/cover; border-radius: 50%; border: 2px solid #0047AB; }}
    .voca-footer {{ text-align: center; font-size: 10px; color: #94a3b8; border-top: 1px solid #e2e8f0; padding-top: 8px; margin-top: 14px; display: flex; justify-content: center; align-items: center; gap: 8px; }}
    .voca-f-img {{ width: 22px; height: 22px; background: url('{f_img}') center/cover; border-radius: 50%; }}
    .voca-subbar {{ display: flex; justify-content: space-between; align-items: center; background: #f8fafc; border: 1px solid #cbd5e1; border-left: 5px solid #0047AB; border-radius: 6px; padding: 8px 14px; margin-bottom: 12px; font-size: 12px; }}
    .voca-table {{ width: 100%; border-collapse: collapse; table-layout: fixed; }}
    .voca-table thead {{ display: table-header-group; }}
    .voca-table th {{ background: #0047AB; color: white; font-size: 11.5px; font-weight: 800; padding: 7px 8px; border: 1px solid #003580; text-align: center; }}
    .voca-table tr {{ page-break-inside: avoid; break-inside: avoid; }}
    .voca-table td {{ border: 1px solid #cbd5e1; padding: 7px 9px; vertical-align: middle; font-size: 11.5px; line-height: 1.45; word-break: keep-all; }}
    .voca-table tbody tr:nth-child(even) {{ background: #f8fafc; }}
    .voca-table tbody tr.core-para-row {{ background: #fffbeb; }}
    .voca-idx-cell {{ text-align: center; font-weight: 900; color: #64748b; font-size: 11px; position: relative; }}
    .voca-word-main {{ font-weight: 900; font-size: 14px; color: #0f172a; display: block; margin-bottom: 2px; }}
    .voca-pron-badge {{ display: inline-block; font-size: 10.5px; color: #0369a1; background: #e0f2fe; padding: 1px 6px; border-radius: 4px; font-weight: 700; margin-right: 4px; }}
    .voca-pos-badge {{ display: inline-block; font-size: 10px; color: #475569; background: #e2e8f0; padding: 1px 5px; border-radius: 4px; font-weight: 800; margin-right: 4px; }}
    .voca-cat-core {{ display: inline-block; font-size: 9.5px; color: #92400e; background: #fef3c7; border: 1px solid #f59e0b; padding: 0px 5px; border-radius: 4px; font-weight: 900; margin-top: 2px; }}
    .voca-cat-text {{ display: inline-block; font-size: 9.5px; color: #1e40af; background: #dbeafe; padding: 0px 5px; border-radius: 4px; font-weight: 800; margin-top: 2px; }}
    .voca-meaning-cell {{ font-weight: 700; color: #1e293b; font-size: 12.5px; }}
    .syn-row-chip {{ display: block; background: #eff6ff; border: 1px solid #bfdbfe; color: #1e40af; border-radius: 5px; padding: 3px 7px; margin-bottom: 3px; font-size: 10.8px; }}
    .ant-row-chip {{ display: block; background: #fff1f2; border: 1px solid #fecdd3; color: #9f1239; border-radius: 5px; padding: 3px 7px; margin-bottom: 3px; font-size: 10.8px; }}
    .chip-tag-syn {{ font-weight: 900; color: #2563eb; margin-right: 4px; }}
    .chip-tag-ant {{ font-weight: 900; color: #e11d48; margin-right: 4px; }}
    .chip-pron {{ color: #64748b; font-size: 10px; margin: 0 3px; }}
    .voca-del-btn {{ background: #fee2e2; color: #dc2626; border: 1px solid #fca5a5; border-radius: 4px; font-size: 9px; padding: 1px 4px; cursor: pointer; margin-top: 3px; }}
    .voca-del-btn:hover {{ background: #ef4444; color: white; }}
    #book-container.mode-hide-meaning .voca-meaning-text,
    #book-container.mode-hide-meaning .chip-meaning-text {{ visibility: hidden; }}
    #book-container.mode-hide-word .voca-word-main,
    #book-container.mode-hide-word .chip-word-text {{ visibility: hidden; }}
    </style>"""

def render_voca(p):
    data = p.get("data", {})
    voca_list = data.get("voca_list", [])
    label_text = esc(p.get("label", ""))
    pid = esc(p.get("id", label_text))
    academy = esc(p.get("academy_name", "포피엠 입시 영어"))
    school = esc(p.get("school_name", ""))

    rows_html = ""
    for idx, item in enumerate(voca_list, start=1):
        w = esc(item.get("word", ""))
        pos = esc(item.get("pos", ""))
        pron = esc(item.get("pronunciation", ""))
        mean = esc(item.get("meaning", ""))
        cat = str(item.get("category", "지문핵심")).strip()
        is_core = any(k in cat.upper() for k in ["CORE", "패러", "요약", "주제", "선지"])
        cat_badge = f'<span class="voca-cat-core">{esc(cat)}</span>' if is_core else f'<span class="voca-cat-text">{esc(cat)}</span>'
        row_cls = 'class="core-para-row"' if is_core else ""

        syn_chips = []
        for s in item.get("synonyms", []):
            sw = esc(s.get("word", ""))
            spron = esc(s.get("pronunciation", ""))
            smean = esc(s.get("meaning", ""))
            if sw:
                pron_part = f'<span class="chip-pron">[{spron}]</span>' if spron else ""
                syn_chips.append(
                    f'<div class="syn-row-chip"><span class="chip-tag-syn">[유의어/대체]</span>'
                    f'<b class="chip-word-text">{sw}</b>{pron_part} : '
                    f'<span class="chip-meaning-text">{smean}</span></div>'
                )

        ant_chips = []
        for a in item.get("antonyms", []):
            aw = esc(a.get("word", ""))
            apron = esc(a.get("pronunciation", ""))
            amean = esc(a.get("meaning", ""))
            if aw:
                pron_part = f'<span class="chip-pron">[{apron}]</span>' if apron else ""
                ant_chips.append(
                    f'<div class="ant-row-chip"><span class="chip-tag-ant">[반의어/함정]</span>'
                    f'<b class="chip-word-text">{aw}</b>{pron_part} : '
                    f'<span class="chip-meaning-text">{amean}</span></div>'
                )

        rel_html = "".join(syn_chips + ant_chips)
        if not rel_html:
            rel_html = '<span style="color:#cbd5e1; font-size:10.5px;">-</span>'

        rows_html += f"""
        <tr {row_cls}>
            <td class="voca-idx-cell">
                <div class="voca-idx-num">{idx}</div>
                <button class="voca-del-btn no-print" onclick="removeVocaRow(this)" title="이 단어 제외">✕</button>
            </td>
            <td>
                <span class="voca-word-main">{w}</span>
                {f'<span class="voca-pron-badge">[{pron}]</span>' if pron else ''}
                <span class="voca-pos-badge">{pos}</span><br>
                {cat_badge}
            </td>
            <td class="voca-meaning-cell">
                <span class="voca-meaning-text">{mean}</span>
            </td>
            <td>{rel_html}</td>
            <td style="text-align:center; color:#cbd5e1; font-size:13px;">□ &nbsp; □</td>
        </tr>"""

    return f"""
    <div class="voca-wrapper" data-passage-id="{pid}" data-passage-label="{label_text}">
        <table class="voca-master-table">
            <thead>
                <tr><td>
                    <div class="voca-header">
                        <div class="voca-logo-area"><div class="voca-h-img"></div>PQRS영쌤 | ANATOMY VOCA</div>
                        <div style="text-align:right; font-size:11.5px;"><b>{academy}</b> {('| ' + school) if school else ''}<br><span style="color:#0047AB; font-weight:800;">{label_text}</span></div>
                    </div>
                </td></tr>
            </thead>
            <tfoot>
                <tr><td>
                    <div class="voca-footer">
                        <div class="voca-f-img"></div>포피엠 입시 영어 | 지문·선지 핵심 어휘 + 패러프레이징 유의어·반의어 통합 단어장
                    </div>
                </td></tr>
            </tfoot>
            <tbody>
                <tr><td>
                    <div class="voca-subbar">
                        <div>📌 <b>출처 시험지/지문:</b> {label_text} (총 <b>{len(voca_list)}</b>개 핵심 어휘 연속 수록)</div>
                        <div>이름: ________________ &nbsp;&nbsp;|&nbsp;&nbsp; 점수: ______ / {len(voca_list)}</div>
                    </div>
                    <table class="voca-table">
                        <colgroup>
                            <col style="width: 6%;">
                            <col style="width: 28%;">
                            <col style="width: 22%;">
                            <col style="width: 36%;">
                            <col style="width: 8%;">
                        </colgroup>
                        <thead>
                            <tr>
                                <th>No.</th>
                                <th>단어 / 숙어 [한글발음] &amp; 문항출처</th>
                                <th>문맥 및 패러프레이징 핵심 의미</th>
                                <th>문맥 유의어 (Synonyms) &amp; 내신 함정 반의어 (Antonyms)</th>
                                <th>암기체크</th>
                            </tr>
                        </thead>
                        <tbody>{rows_html}</tbody>
                    </table>
                </td></tr>
            </tbody>
        </table>
    </div>"""

def render_past_voca_grouped_by_exam(sel_passages: List[Dict]) -> str:
    """기출 어휘 렌더링 시 동일한 시험지에 속한 항목들은 한 시험지 테이블에 연달아 출력하고, 다음 시험지로 넘어갈 때만 페이지를 분리"""
    consolidated, _ = consolidate_past_voca_records(sel_passages)
    return "".join([render_voca(p) for p in consolidated])

def wrap_full_html_voca(body_html, css, front_html="", back_html=""):
    return f"""<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8"><title>Anatomy_Voca</title>{css}</head><body>{get_studio_toolbar_html('voca')}<div id="book-wrapper-all">{front_html}<main id="book-container">{body_html}</main>{back_html}</div></body></html>"""


# -----------------------------
# 4. AI 분석 엔진 (한판승 & ANATOMY & ANATOMY CORE & ANATOMY VOCA)
# -----------------------------
SCHEMA_VICTORY = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "keywords": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"word": {"type": "string"}, "meaning": {"type": "string"}, "synonyms": {"type": "string"}},
                "required": ["word", "meaning", "synonyms"]
            }
        },
        "analysis": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "eng": {"type": "string"},
                    "sentence_vocab": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"word": {"type": "string"}, "pronunciation": {"type": "string"}, "pos": {"type": "string"}, "meaning": {"type": "string"}}
                        }
                    },
                    "kor": {"type": "string"},
                    "grammar": {"type": "string"},
                    "grammar_points": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "no": {"type": "integer", "description": "eng 안의 {번호:표현} 마커 번호"},
                                "title": {"type": "string", "description": "어법 항목 이름 (예: 명사절 접속사 that)"},
                                "explain": {"type": "string", "description": "구조 + 짧은 해석 (예: show [타동사] + [목적어 명사절] : \"~을 보여준다\")"}
                            },
                            "required": ["no", "title", "explain"]
                        }
                    },
                    "essay_candidate": {"type": "boolean", "description": "학교 서술형 기출 DNA에 비추어 서술형으로 나올 가능성이 가장 높은 문장이면 true (지문당 보통 1문장)"},
                    "essay_reason": {"type": "string", "description": "essay_candidate가 true일 때만: '문장 역할 · 핵심 문법 · 예상 형식' (예: 주제문 · 관계대명사 what · 조건 영작). 아니면 빈 문자열"},
                    "ys": {"type": "string"}
                },
                "required": ["eng", "kor", "ys", "grammar_points", "essay_candidate", "essay_reason"]
            }
        }
    },
    "required": ["summary", "keywords", "analysis"]
}

SCHEMA_ANATOMY = {
    "type": "object",
    "properties": {
        "english_title": {"type": "string"}, "korean_title": {"type": "string"}, "topic": {"type": "string"},
        "vocab": {"type": "array", "items": {"type": "object", "properties": {"word": {"type": "string"}, "meaning": {"type": "string"}, "synonyms_text": {"type": "string"}}, "required": ["word", "meaning", "synonyms_text"]}},
        "translation": {"type": "string"},
        "grammar_points": {"type": "array", "items": {"type": "object", "properties": {"title": {"type": "string"}, "explanation": {"type": "string"}}}},
        "easy_explanation": {"type": "string"}, "structure_flow": {"type": "string"},
        "quizzes": {"type": "array", "items": {"type": "object", "properties": {"english_statement": {"type": "string"}, "korean_statement": {"type": "string"}, "answer": {"type": "string", "enum": ["O", "X"]}}}},
        "visual_guide": {"type": "string"},
        "one_line_summary": {"type": "string", "description": "지문 전체를 한국어 한 줄(40자 안팎)로 요약"},
        "sentences": {
            "type": "array",
            "items": {"type": "object", "properties": {"eng": {"type": "string"}, "kor": {"type": "string"}}, "required": ["eng", "kor"]}
        }
    },
    "required": ["english_title", "korean_title", "topic", "vocab", "translation", "grammar_points", "easy_explanation", "structure_flow", "quizzes", "visual_guide", "one_line_summary", "sentences"]
}

SCHEMA_CORE = {
    "type": "object",
    "properties": {
        "topics": {"type": "array", "items": {"type": "object", "properties": {"eng": {"type": "string"}, "kor": {"type": "string"}}, "required": ["eng", "kor"]}},
        "titles": {"type": "array", "items": {"type": "object", "properties": {"style": {"type": "string"}, "eng": {"type": "string"}, "kor": {"type": "string"}}, "required": ["style", "eng", "kor"]}},
        "main_ideas": {"type": "array", "items": {"type": "object", "properties": {"eng": {"type": "string"}, "kor": {"type": "string"}}, "required": ["eng", "kor"]}},
        "summaries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "eng": {"type": "string"}, "kor": {"type": "string"},
                    "blank_a": {"type": "string"}, "distractor_a": {"type": "string"},
                    "blank_b": {"type": "string"}, "distractor_b": {"type": "string"},
                    "synonyms_a": {"type": "string"}, "synonyms_b": {"type": "string"}
                },
                "required": ["eng", "kor", "blank_a", "distractor_a", "blank_b", "distractor_b", "synonyms_a", "synonyms_b"]
            }
        },
        "paraphrased_tf": {"type": "array", "items": {"type": "object", "properties": {"eng": {"type": "string"}, "kor": {"type": "string"}, "answer": {"type": "string", "enum": ["O", "X"]}, "point": {"type": "string"}}, "required": ["eng", "kor", "answer", "point"]}},
        "core_vocab": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "word": {"type": "string"}, "pos": {"type": "string"}, "pronunciation": {"type": "string"}, "meaning": {"type": "string"},
                    "synonyms": {"type": "array", "items": {"type": "object", "properties": {"word": {"type": "string"}, "pronunciation": {"type": "string"}, "meaning": {"type": "string"}}, "required": ["word", "pronunciation", "meaning"]}},
                    "antonyms": {"type": "array", "items": {"type": "object", "properties": {"word": {"type": "string"}, "pronunciation": {"type": "string"}, "meaning": {"type": "string"}}, "required": ["word", "pronunciation", "meaning"]}}
                },
                "required": ["word", "pos", "pronunciation", "meaning", "synonyms", "antonyms"]
            }
        }
    },
    "required": ["topics", "titles", "main_ideas", "summaries", "paraphrased_tf", "core_vocab"]
}

SCHEMA_VOCA = {
    "type": "object",
    "properties": {
        "voca_list": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "word": {"type": "string"}, "pos": {"type": "string"}, "pronunciation": {"type": "string"}, "meaning": {"type": "string"}, "category": {"type": "string"},
                    "synonyms": {"type": "array", "items": {"type": "object", "properties": {"word": {"type": "string"}, "pronunciation": {"type": "string"}, "meaning": {"type": "string"}}, "required": ["word", "pronunciation", "meaning"]}},
                    "antonyms": {"type": "array", "items": {"type": "object", "properties": {"word": {"type": "string"}, "pronunciation": {"type": "string"}, "meaning": {"type": "string"}}, "required": ["word", "pronunciation", "meaning"]}}
                },
                "required": ["word", "pos", "pronunciation", "meaning", "category", "synonyms", "antonyms"]
            }
        }
    },
    "required": ["voca_list"]
}

SCHEMA_PAST_PDF_VOCA = {
    "type": "object",
    "properties": {
        "school_name": {"type": "string", "description": "학교명 (예: 운양고, 제일고)"},
        "exam_year_term": {"type": "string", "description": "연도 및 학기 (예: 26년 1학기 중간, 26년 1학기 기말)"},
        "grade": {"type": "string", "description": "학년 (예: 고1, 고2, 고3)"},
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question_number": {"type": "string", "description": "문항 번호 (예: 1번, 18번, 21번, 논술형1 등)"},
                    "words": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "word": {"type": "string"},
                                "pos": {"type": "string"},
                                "pronunciation": {"type": "string"},
                                "meaning": {"type": "string"},
                                "source_part": {"type": "string", "description": "'기출본문' 또는 '기출선지'"},
                                "synonyms": {"type": "array", "items": {"type": "object", "properties": {"word": {"type": "string"}, "pronunciation": {"type": "string"}, "meaning": {"type": "string"}}, "required": ["word", "pronunciation", "meaning"]}},
                                "antonyms": {"type": "array", "items": {"type": "object", "properties": {"word": {"type": "string"}, "pronunciation": {"type": "string"}, "meaning": {"type": "string"}}, "required": ["word", "pronunciation", "meaning"]}}
                            },
                            "required": ["word", "pos", "pronunciation", "meaning", "source_part", "synonyms", "antonyms"]
                        }
                    }
                },
                "required": ["question_number", "words"]
            }
        }
    },
    "required": ["school_name", "exam_year_term", "grade", "questions"]
}

# 유의어 규칙 (2026-11-10): 개수를 꼭 채우라고 하면 상위어·연상어가 섞여 나옴 (enzyme ≈ protein, perishable ≈ degradable)
#   → 시험 선택지에서 바꿔 써도 뜻이 같은 말만, 확실하지 않으면 빼게 함. 교재·PPT·단어장 생성 프롬프트에 함께 씀
SYN_RULE = ("유의어는 시험 선택지에서 그 단어 자리에 바꿔 넣어도 문장 뜻이 그대로인 말(같은 품사·같은 뜻)만 쓴다. "
            "상위어·하위어(enzyme '효소' → protein '단백질'), 관련어·연상어(perishable '상하기 쉬운' → degradable '분해되는'), "
            "뜻이 더 넓거나 좁은 말은 쓰지 않는다. 확실하지 않으면 개수를 채우지 말고 빼며, 맞는 말이 없으면 비워 둔다.")

VICTORY_PROMPT = """너는 한국 고등학생 영어 내신용 '한판승' 교재 편집자다. 지문을 분석해 JSON만 반환하라.
1. summary: 반드시 '한글(한국어)'로 작성된 핵심 요약 2문장.
2. keywords: 중요 단어 8개. synonyms는 "동의어1(뜻), 동의어2(뜻)" 형식으로 최대 2개. """ + SYN_RULE + """ 맞는 동의어가 없으면 빈 문자열.
3. analysis: 모든 문장 분석.
   - eng: 문장에 끊어 읽기 표시(/)를 삽입해라. 단, 종속절의 자잘한 주어/동사가 아닌 **문장 전체의 메인 주어(S), 본동사(V), 메인 목적어(O), 메인 보어(C)**만 표시하고, 관계대명사절 등 나머지는 묶어서 수식어구(M)로 처리해라.
   ★(중요) 표시 형태는 반드시 해당 어구를 대괄호로 묶고 바로 뒤에 괄호로 기호를 붙여라.
   (예시: [Factors](S) / [such as team chemistry, which is related to how well people get along,](M) / [will likely remain](V) / [difficult to measure](C).)
   ★(어법 위치 표시) 내신 어법·서술형에 나올 만한 핵심 표현 1~4개를 eng 안에서 {번호:표현} 형태로 감싸라. 번호는 문장마다 1부터 시작한다.
     · 마커는 반드시 하나의 [ ](X) 대괄호 안쪽에만 넣거나, 대괄호 묶음 전체를 바깥에서 감싸야 한다. 대괄호 경계를 가로지르면 안 된다.
     · 마커 안의 단어는 원문 그대로 두고, 마커를 지우면 원래 eng와 똑같아야 한다.
     (예시: [Many real-world situations](S) / [{1:show that}](V) [strategies {2:based on self-interest} can {3:lead to bad outcomes}](O) / [for the individual {4:as well as the group}](M).)
   - grammar_points: eng의 마커 번호마다 하나씩, 번호 순서대로 작성.
     · title: 어법 항목 이름 (예: "명사절 접속사 that", "과거분사 (명사 수식)", "A as well as B")
     · explain: 구조를 [ ]로 짧게 보여주고 콜론 뒤에 짧은 해석 (예: "show [타동사] + [목적어로 쓰인 명사절 접속사 that] : \\"전략이 ~할 수 있다는 것을 보여준다\\"")
   - essay_candidate: 아래 [학교 서술형 출제 DNA]에 비추어 실제 내신 서술형(조건 영작·배열·어법 수정 등)으로 나올 가능성이 **가장 높은** 문장만 true.
     · 지문 전체에서 보통 1문장만 true. 문장이 14개 이상인 긴 지문만 최대 2문장. 기준에 확실히 맞는 문장이 없으면 모두 false로 둔다.
     · 학생이 이 문장 하나만 확실히 외우면 되도록, 가장 출제 가능성이 높은 문장을 골라라.
   - essay_reason: essay_candidate가 true인 문장에만 "문장 역할 · 핵심 문법 · 예상 형식"을 30자 이내로 쓴다 (예: "주제문 · 관계대명사 what · 조건 영작"). false인 문장은 빈 문자열.
   [학교 서술형 출제 DNA]
{essay_dna}
   - sentence_vocab: 해당 문장에서 사용된 주요 단어들을 추출. 유의어 없이 딱 문장에 쓰인 의미만 작성. (word: 영어단어, pronunciation: 한글발음기호 예:나이스, pos: 품사 약어(n, v, a, ad 등), meaning: 뜻)
   - kor: 직독직해.
   - grammar: 빈 문자열로 둔다. (어법 설명은 grammar_points에 작성)
   - ys: 친근한 구어체 설명(~야, ~지!, ~거거든!), 해석된 한글 문장을 쉽고 간략한 예를 들어서 이해하기 쉽게 다른말로 설명해줘.
     ★ ys는 반드시 2~3문장, 150자 이내로 쓰고 같은 말을 반복하지 마라. (그 문장 하나에 대한 설명이며 지문 전체 요약이 아니다)
지문: \"\"\"{text}\"\"\""""

ANATOMY_PROMPT = """너는 한국 고등학생용 영어 내신 교재를 만드는 최상급 편집자다. 지문을 분석해 반드시 JSON만 반환하라.
지문 정보: {label}
[반환 규칙]
1. vocab: 정확히 10개 추출. synonyms_text는 "동의어1(뜻), 동의어2(뜻)" 형식으로 최대 2개. """ + SYN_RULE + """ 맞는 동의어가 없으면 빈 문자열.
2. translation: 지문 전체의 자연스러운 한국어 해석.
3. grammar_points: 핵심 문법 3개 (title, explanation).
4. structure_flow: 반드시 '➔'를 사용한 도식형으로 작성.
5. quizzes: 5개의 OX퀴즈. answer는 'O' 또는 'X'만 사용.
6. one_line_summary: 지문 전체를 한국어 한 줄(40자 안팎)로 요약. (예: "기상 악화로 인한 현장체험학습 장소 변경 안내")
7. sentences: 지문을 문장 단위로 나눈 목록.
   - eng: 원문 문장을 한 글자도 바꾸지 말고 그대로 옮긴다. 모든 eng를 이어 붙이면 원문 전체와 같아야 한다.
   - kor: 그 문장의 자연스러운 한국어 해석.
   - 편지의 인사말(Dear ~,), 맺음말(Sincerely,), 서명 줄도 각각 한 문장으로 취급한다.
지문: \"\"\"{text}\"\"\""""

# AI가 가끔 같은 글자를 수천 번 반복하는 결과를 내는데(예: '구구구구…'), 그대로 저장하면 교재 페이지가 옆으로 넓어져 인쇄가 작아짐
DEGENERATE_REPEAT_RE = re.compile(r"(.{1,6}?)\1{14,}", re.S)

# 칸별 최대 글자 수: 저장된 교재 737개 실측 상위 1% 값의 약 4~6배. 띄어쓰기를 넣어 가며 비슷한 문장을 끝없이 되풀이하는 오류를 잡기 위함
# (예: 정상 코칭은 99%가 125자 이하인데 오류 코칭은 4,500~33,000자)
FIELD_MAX_CHARS = {
    "ys": 700, "kor": 800, "eng": 1500, "grammar": 700, "explain": 600, "summary": 900, "title": 300,
    "meaning": 300, "synonyms": 400, "synonyms_text": 400, "topic": 800, "easy_explanation": 1800,
    "structure_flow": 1200, "explanation": 1000, "english_statement": 700, "korean_statement": 500,
    "point": 1500, "one_line_summary": 300, "passage": 6000,
}
FIELD_MAX_DEFAULT = 4000  # 위에 없는 칸 (translation처럼 원래 긴 칸은 제외)
FIELD_NO_LIMIT = {"translation", "visual_guide", "text", "restored_text", "body_text"}

def find_degenerate_text(obj, path: str = "") -> str:
    """반복 오류가 있는 칸의 위치를 반환 (없으면 빈 문자열). 교재에 쓰이지 않는 visual_guide는 검사하지 않음."""
    if isinstance(obj, str):
        field = path.split(".")[-1] if path else ""
        if field.isdigit():  # 목록 안의 문자열이면 목록 이름으로 판단
            parts = [p for p in path.split(".") if not p.isdigit()]
            field = parts[-1] if parts else ""
        if field not in FIELD_NO_LIMIT and len(obj) > FIELD_MAX_CHARS.get(field, FIELD_MAX_DEFAULT):
            return path or "text"
        if max((len(w) for w in obj.split()), default=0) > 60:
            return path or "text"
        m = DEGENERATE_REPEAT_RE.search(obj)
        if m and re.search(r"\w", m.group(1)):
            return path or "text"
        return ""
    if isinstance(obj, dict):
        items = ((k, v) for k, v in obj.items() if k != "visual_guide")
    elif isinstance(obj, list):
        items = ((str(i), v) for i, v in enumerate(obj))
    else:
        return ""
    for k, v in items:
        hit = find_degenerate_text(v, f"{path}.{k}" if path else k)
        if hit:
            return hit
    return ""

# 서술형 출제 DNA가 아직 없을 때 쓰는 기본 기준
ESSAY_DNA_FALLBACK = """- 글의 주제문·요지·결론 문장을 우선한다.
- 관계사(what, 관계부사), 분사구문, 가주어/진주어, 간접의문문, 강조구문, 도치, 수동태처럼 영작·배열로 묻기 좋은 구문이 들어 있는 문장을 우선한다.
- 단순 예시·세부 사항 문장은 고르지 않는다."""

def essay_flag_limit(sentence_count: int) -> int:
    """서술형 출제 가능 표시 개수: 보통 1문장, 14문장 이상 긴 지문은 2문장"""
    return 2 if sentence_count >= 14 else 1

def limit_essay_flags(data: Dict) -> Dict:
    """AI가 서술형 표시를 너무 많이 달아도 지문당 허용 개수까지만 남김 (이유가 적힌 문장을 우선)"""
    rows = data.get("analysis", []) or []
    flagged = [i for i, r in enumerate(rows) if r.get("essay_candidate")]
    keep = sorted(flagged, key=lambda i: (not rows[i].get("essay_reason"), i))[:essay_flag_limit(len(rows))]
    for i, r in enumerate(rows):
        r["essay_candidate"] = i in keep
        if i not in keep:
            r["essay_reason"] = ""
    return data

def analyze_passage(label, text, model_name, target_type="victory", core_context: Dict = None, max_retries=5,
                    dna_rules: str = None, notify=None, gen_client=None, essay_rules: str = None):
    """notify(msg, level), gen_client: 백그라운드 생성 작업(스레드)에서 st.* 없이 호출할 때 사용"""
    if dna_rules is None:
        dna_rules = st.session_state.exam_dna.get("rules_text", DEFAULT_EXAM_DNA["rules_text"])
    if notify is None:
        notify = lambda msg, level="info": (st.error if level == "error" else st.toast)(msg)
    gen_client = gen_client or client

    if target_type == "victory":
        if essay_rules is None:
            essay_rules = st.session_state.exam_dna.get("essay_rules_text", "")
        prompt = VICTORY_PROMPT.replace("{essay_dna}", essay_rules or ESSAY_DNA_FALLBACK).replace("{text}", text)
        schema = SCHEMA_VICTORY

    elif target_type == "anatomy":
        prompt = ANATOMY_PROMPT.replace("{label}", label).replace("{text}", text)
        schema = SCHEMA_ANATOMY

    elif target_type == "core":
        prompt = f"""너는 대한민국 고등학교 영어 내신 변형 문제 및 패러프레이징 최고 전문가다.
아래의 [학교 실제 기출 출제 DNA]를 철저히 적용하여, 이 지문에서 출제될 가능성이 가장 높은 다중 주제문, 제목, 요지/주장문, 요약문(빈칸 A/B 양자택일 포함), <보기> 진위형 선지, 그리고 여기에 사용된 패러프레이징 핵심 어휘를 JSON으로 반환하라.

{dna_rules}

[세부 생성 규칙]
1. topics (영문 주제문 3선): 원문 표현을 그대로 복사하지 말고 상위 개념어(Meta-vocabulary)와 명사구 구문을 활용해 서로 다른 각도에서 3개 작성 (eng, kor).
2. titles (영문 제목 3선): 1번(style="콜론(:) 핵심형"), 2번(style="의문문(?) 탐구형"), 3번(style="대조/비유 압축형").
3. main_ideas (영문 요지·주장문 3선): 완결된 학술 영문 평서문 3개로 패러프레이징 작성 (eng, kor).
4. summaries (실전 요약문 3선 & 양자택일 어휘):
   - 서로 다른 문장 구조로 지문 전체를 요약한 영문 문장 3개 작성.
   - 반드시 영문 요약문(eng) 안에 `(A) 정답단어` 와 `(B) 정답단어` 형태로 표기할 것!
   - blank_a, blank_b에는 각각 (A), (B)의 정확한 정답 영어 단어를 적고, distractor_a, distractor_b에는 각각 (A), (B) 자리에 들어갈 매력적인 오답 영어 단어(반의어 또는 철자 유사 함정 어휘 1단어, 품사 일치 필수)를 적을 것.
   - synonyms_a, synonyms_b에는 각각 정답 단어의 동의어(최대 2개)와 오답 단어의 뜻을 "동의어1(뜻), 동의어2(뜻) / [함정]오답단어(뜻)" 형태로 작성할 것. 동의어는 아래 [유의어 규칙]에 맞는 것만 쓰고, 없으면 "[함정]오답단어(뜻)"만 쓴다.
5. paraphrased_tf (내신 <보기>형 패러프레이징 진위 문장 4선):
   - 2개는 O(True: 참 진술), 2개는 X(False: 매력적인 오답 진술). 단, O와 X의 배치 순서는 예측할 수 없게 섞어서 작성하라.
6. core_vocab (위 주제/제목/요지/요약문/진위문에서 사용된 핵심 패러프레이징 어휘 8~10개):
   - pronunciation은 반드시 '한글 발음'으로만 적고, synonyms(최대 2개, [유의어 규칙]에 맞는 것만 — 없으면 빈 배열)와 antonyms(1~2개)도 각각 word, pronunciation(한글발음), meaning(뜻)을 완비할 것.
[유의어 규칙] {SYN_RULE}

지문 정보: {label}
지문: \"\"\"{text}\"\"\""""
        schema = SCHEMA_CORE

    else:
        core_ref_str = ""
        if core_context:
            core_ref_str = f"""
[★중요 연동 지시: 이 지문의 ANATOMY CORE에서 추출된 핵심 패러프레이징 어휘 목록]
{json.dumps(core_context.get('core_vocab', []), ensure_ascii=False)}
위 ANATOMY CORE 어휘들을 누락 없이 단어장에 포함시키고, 해당 단어들의 category 필드는 반드시 "★CORE 패러프레이징"으로 설정하라!
"""
        prompt = f"""너는 대한민국 고등학교 내신 및 수능 영어 어휘 전문 편집자다.
아래의 [학교 실제 기출 출제 DNA]를 반영하여, 지문 속 내신 변형 1순위 어휘/숙어와 주제·요약문 패러프레이징 핵심 어휘를 총 15~22개 엄선해 JSON으로 반환하라.

{dna_rules}

[절대 준수 규칙]
1. 초등 기초 단어 제외: be동사, 관사(a, an, the), 대명사, 단순 전치사, beautiful, happy, make, good 같은 기초 단어 제외.
2. 2-Track 어휘 구성:
   - Track 1 (`category`: "지문핵심"): 지문에 등장한 중고등 핵심 어휘 및 숙어 10~14개.
   - Track 2 (`category`: "★CORE 패러프레이징"): 주제문, 제목, 요지문, 요약문 빈칸(A)/(B)용 상위 개념어 및 핵심 요약 어휘 5~8개.
3. 각 항목 세부 규칙:
   - word: 영어 단어 또는 숙어 / pos: 품사 약어 / pronunciation: 한글 발음(IPA 금지, 한글로만 표기!) / meaning: 한국어 뜻
   - synonyms: 유의어/대체 표현 최대 3개 (word, pronunciation, meaning 필수). {SYN_RULE} 맞는 말이 없으면 빈 배열.
   - antonyms: 반의어 또는 철자 유사 혼동어 1~2개 (word, pronunciation, meaning 필수).

지문 정보: {label}{core_ref_str}
지문: \"\"\"{text}\"\"\""""
        schema = SCHEMA_VOCA

    # 지문 앞까지(규칙·출제 DNA·서술형 DNA)는 지문마다 똑같으므로 따로 보내 두 번째 지문부터 캐시로 읽게 함 (GPT·Claude·Gemini 모두)
    #   Self-Note(anatomy)는 앞부분이 짧고 지문 이름이 앞에 있어 캐시 이득이 없음
    cut = (prompt.find("지문 정보:") if "지문 정보:" in prompt else prompt.rfind('지문: """')) if target_type != "anatomy" else -1
    contents = [prompt[:cut], prompt[cut:]] if cut > 0 else prompt
    for attempt in range(max_retries):
        try:
            result = llm_json(model_name, contents, schema, 0.3, gen_client, cache_first=cut > 0)
            bad_field = find_degenerate_text(result)
            if bad_field:
                # 반복 오류 결과는 저장하지 않고 다시 생성
                if attempt < max_retries - 1:
                    notify(f"⚠️ AI 결과에 반복 오류가 있어 다시 생성합니다 ({label} - {target_type}, {bad_field}) ({attempt + 1}/{max_retries})", "info")
                    time.sleep(5)
                    continue
                notify(f"❌ 반복 오류가 계속되어 생성하지 못했습니다 ({label} - {target_type}, {bad_field})", "error")
                return None
            if target_type == "victory":
                result = limit_essay_flags(result)
            return result
        except Exception as e:
            error_str = str(e).lower()
            if any(k in error_str for k in ["503", "high demand", "429", "unavailable", "quota", "overloaded", "timeout", "timed out", "connection", "500", "internal"]):
                if attempt < max_retries - 1:
                    wait_time = 10 + (attempt * 10)
                    notify(f"⏳ 서버 대기 감지 ({label} - {target_type}). {wait_time}초 후 재시도... ({attempt + 1}/{max_retries})", "info")
                    time.sleep(wait_time)
                    continue
            notify(f"❌ 오류 발생 ({label} - {target_type}): {e}", "error")
            return None
    return None


# -----------------------------
# 4-B. 원본 JSON 영구 수정 에디터 (5개 라이브러리 전체 지원)
# -----------------------------
def _cell(v, default: str = "") -> str:
    """표 편집기 칸 값 → 글자. 비운 칸(None·NaN)은 'None'·'nan' 글자가 아니라 빈 글자(또는 default)로"""
    try:
        if v is None or (not isinstance(v, (list, dict)) and pd.isna(v)):
            return default
    except (TypeError, ValueError):
        pass
    t = str(v).strip()
    return default if not t or t.lower() in ("nan", "none") else t

def _df_records(df) -> List[Dict]:
    """표 편집기 → 기록 목록. 비운 칸은 빈 글자로, 모든 칸이 빈 줄(+로 추가만 한 줄)은 버림"""
    out = []
    for rec in df.to_dict(orient="records"):
        clean = {}
        for k, v in rec.items():
            if hasattr(v, "item") and not isinstance(v, (list, dict, str)):
                v = v.item()  # numpy 숫자·참거짓 → 파이썬 값
            clean[k] = v if isinstance(v, (bool, int, float, list, dict)) and not (isinstance(v, float) and pd.isna(v)) else _cell(v)
        if any(v not in ("", None, False) for v in clean.values()):
            out.append(clean)
    return out

def parse_rel_string(rel_str: str) -> List[Dict]:
    res_list = []
    if not rel_str or pd.isna(rel_str):
        return res_list
    chunks = [c.strip() for c in re.split(r',\s*(?![^\(\[]*[\)\]])', str(rel_str)) if c.strip()]
    for ch in chunks:
        m = re.match(r'^([^\[\(]+?)(?:\[(.*?)\])?\s*(?:\((.*?)\))?$', ch.strip())
        if m:
            res_list.append({
                "word": (m.group(1) or "").strip(),
                "pronunciation": (m.group(2) or "").strip(),
                "meaning": (m.group(3) or "").strip()
            })
        else:
            res_list.append({"word": ch, "pronunciation": "", "meaning": ""})
    return res_list

def render_permanent_json_editor(prefix: str, target_type: str, file_path: str, selected_passages: List[Dict], all_saved: List[Dict]):
    if not selected_passages:
        return

    with st.expander("🛠️ 선택 지문/시험지 원본 데이터 영구 수정 (수정 후 클라우드/JSON 파일에 영구 반영)", expanded=False):
        passage_map = {f"{p['label']} (ID: {p['id'][:6]})": p for p in selected_passages}
        chosen_key = st.selectbox("✏️ 수정할 항목을 선택하세요", options=list(passage_map.keys()), key=f"{prefix}_edit_select")
        target_p = passage_map[chosen_key]
        pid = target_p["id"]
        pdata = target_p.get("data", {})

        new_label = st.text_input("📌 시험지/지문 라벨명", value=target_p.get("label", ""), key=f"{prefix}_elbl_{pid}")

        if target_type in ["voca", "past_voca"]:
            st.markdown("##### 📖 단어장 목록 편집 (셀 더블클릭 수정, 하단 `+`로 행 추가, 좌측 체크 후 삭제 가능)")
            st.caption("💡 유의어와 반의어는 `단어[한글발음](뜻), 단어2[한글발음](뜻)` 형식으로 입력하면 구조가 그대로 보존됩니다.")

            rows_for_editor = []
            for item in pdata.get("voca_list", []):
                syn_str = ", ".join([
                    f"{s.get('word','')}[{s.get('pronunciation','')}]({s.get('meaning','')})"
                    for s in item.get("synonyms", []) if s.get("word")
                ])
                ant_str = ", ".join([
                    f"{a.get('word','')}[{a.get('pronunciation','')}]({a.get('meaning','')})"
                    for a in item.get("antonyms", []) if a.get("word")
                ])
                rows_for_editor.append({
                    "출제 트랙/문항출처": item.get("category", "지문핵심"),
                    "영어 단어/숙어": item.get("word", ""),
                    "품사": item.get("pos", ""),
                    "한글 발음": item.get("pronunciation", ""),
                    "한국어 뜻": item.get("meaning", ""),
                    "유의어 (단어[발음](뜻))": syn_str,
                    "반의어 (단어[발음](뜻))": ant_str,
                })

            edited_df = st.data_editor(
                pd.DataFrame(rows_for_editor) if rows_for_editor else pd.DataFrame(columns=["출제 트랙/문항출처", "영어 단어/숙어", "품사", "한글 발음", "한국어 뜻", "유의어 (단어[발음](뜻))", "반의어 (단어[발음](뜻))"]),
                num_rows="dynamic",
                use_container_width=True,
                key=f"{prefix}_voca_df_{pid}"
            )

            if st.button("💾 단어장 수정사항 영구 저장", type="primary", key=f"{prefix}_save_json_{pid}"):
                new_voca_list = []
                for _, r in edited_df.iterrows():
                    w_val = _cell(r.get("영어 단어/숙어"))
                    if not w_val:
                        continue
                    new_voca_list.append({
                        "category": _cell(r.get("출제 트랙/문항출처"), "지문핵심"),
                        "word": w_val,
                        "pos": _cell(r.get("품사")),
                        "pronunciation": _cell(r.get("한글 발음")),
                        "meaning": _cell(r.get("한국어 뜻")),
                        "synonyms": parse_rel_string(r.get("유의어 (단어[발음](뜻))", "")),
                        "antonyms": parse_rel_string(r.get("반의어 (단어[발음](뜻))", ""))
                    })
                target_p["label"] = clean_label(new_label)
                target_p["data"]["voca_list"] = new_voca_list
                save_to_file(file_path, all_saved)
                st.success(f"✅ [{target_p['label']}] 단어장 데이터가 영구 업데이트되었습니다!")

        elif target_type == "core":
            st.markdown("##### 📌 영문 주제 (Topics) 편집")
            top_df = st.data_editor(pd.DataFrame(pdata.get("topics", [])), num_rows="dynamic", use_container_width=True, key=f"{prefix}_ctop_{pid}")
            st.markdown("##### 🏷️ 영문 제목 (Titles) 편집")
            tit_df = st.data_editor(pd.DataFrame(pdata.get("titles", [])), num_rows="dynamic", use_container_width=True, key=f"{prefix}_ctit_{pid}")
            st.markdown("##### 💡 영문 요지·주장 (Main Ideas) 편집")
            mi_df = st.data_editor(pd.DataFrame(pdata.get("main_ideas", [])), num_rows="dynamic", use_container_width=True, key=f"{prefix}_cmi_{pid}")
            st.markdown("##### 🎯 실전 요약문 (Summaries & 빈칸 A/B 양자택일) 편집")
            sum_df = st.data_editor(pd.DataFrame(pdata.get("summaries", [])), num_rows="dynamic", use_container_width=True, key=f"{prefix}_csum_{pid}")
            st.markdown("##### 🔍 <보기>형 진위 진술문 (Paraphrased T/F) 편집")
            tf_df = st.data_editor(pd.DataFrame(pdata.get("paraphrased_tf", [])), num_rows="dynamic", use_container_width=True, key=f"{prefix}_ctf_{pid}")

            if st.button("💾 ANATOMY CORE 수정사항 영구 저장", type="primary", key=f"{prefix}_save_json_{pid}"):
                target_p["label"] = clean_label(new_label)
                pdata["topics"] = _df_records(top_df)
                pdata["titles"] = _df_records(tit_df)
                pdata["main_ideas"] = _df_records(mi_df)
                pdata["summaries"] = _df_records(sum_df)
                pdata["paraphrased_tf"] = _df_records(tf_df)
                save_to_file(file_path, all_saved)
                st.success(f"✅ [{target_p['label']}] ANATOMY CORE 데이터가 영구 업데이트되었습니다!")

        elif target_type == "victory":
            new_sum = st.text_area("🎯 한판승 핵심 요약", value=pdata.get("summary", ""), height=80, key=f"{prefix}_esum_{pid}")
            kw_df = st.data_editor(pd.DataFrame(pdata.get("keywords", [])), num_rows="dynamic", use_container_width=True, key=f"{prefix}_ekw_{pid}")
            st.markdown("##### 📝 문장별 구문/해석/코칭 수정")
            ana_rows = [{"eng": row.get("eng", ""), "kor": row.get("kor", ""), "grammar": row.get("grammar", ""), "ys": row.get("ys", "")} for row in pdata.get("analysis", [])]
            ana_df = st.data_editor(pd.DataFrame(ana_rows), use_container_width=True, key=f"{prefix}_eana_{pid}")

            if st.button("💾 한판승 수정사항 영구 저장", type="primary", key=f"{prefix}_save_json_{pid}"):
                target_p["label"] = clean_label(new_label)
                pdata["summary"] = new_sum
                pdata["keywords"] = _df_records(kw_df)
                orig_analysis = pdata.get("analysis", [])
                updated_analysis = []
                for idx, r in ana_df.iterrows():
                    try:
                        orig_row = orig_analysis[int(idx)] if 0 <= int(idx) < len(orig_analysis) else {}
                    except (TypeError, ValueError):
                        orig_row = {}
                    cells = {k: _cell(r.get(k)) for k in ("eng", "kor", "grammar", "ys")}
                    if not any(cells.values()):
                        continue  # +로 추가만 하고 비워 둔 줄
                    updated_analysis.append({
                        **cells,
                        "sentence_vocab": orig_row.get("sentence_vocab", []),
                        "grammar_points": orig_row.get("grammar_points", []),
                        "essay_candidate": orig_row.get("essay_candidate", False),
                        "essay_reason": orig_row.get("essay_reason", "")
                    })
                pdata["analysis"] = updated_analysis
                save_to_file(file_path, all_saved)
                st.success(f"✅ [{target_p['label']}] 한판승 데이터가 영구 업데이트되었습니다!")

        elif target_type == "anatomy":
            c1, c2 = st.columns(2)
            new_en_t = c1.text_input("영문 제목", value=pdata.get("english_title", ""), key=f"{prefix}_eent_{pid}")
            new_ko_t = c2.text_input("한글 제목", value=pdata.get("korean_title", ""), key=f"{prefix}_ekot_{pid}")
            new_topic = st.text_area("Topic", value=pdata.get("topic", ""), height=65, key=f"{prefix}_etop_{pid}")
            new_trans = st.text_area("지문 해석", value=pdata.get("translation", ""), height=130, key=f"{prefix}_etr_{pid}")
            new_easy = st.text_area("영쌤의 쉬운 지문 설명", value=pdata.get("easy_explanation", ""), height=90, key=f"{prefix}_eeasy_{pid}")
            new_flow = st.text_area("글의 흐름 (Structure)", value=pdata.get("structure_flow", ""), height=70, key=f"{prefix}_eflow_{pid}")
            v_df = st.data_editor(pd.DataFrame(pdata.get("vocab", [])), num_rows="dynamic", use_container_width=True, key=f"{prefix}_evoc_{pid}")
            q_df = st.data_editor(pd.DataFrame(pdata.get("quizzes", [])), num_rows="dynamic", use_container_width=True, key=f"{prefix}_eq_{pid}")

            if st.button("💾 ANATOMY 수정사항 영구 저장", type="primary", key=f"{prefix}_save_json_{pid}"):
                target_p["label"] = clean_label(new_label)
                pdata["english_title"], pdata["korean_title"] = new_en_t, new_ko_t
                pdata["topic"], pdata["translation"] = new_topic, new_trans
                pdata["easy_explanation"], pdata["structure_flow"] = new_easy, new_flow
                pdata["vocab"] = _df_records(v_df)
                pdata["quizzes"] = _df_records(q_df)
                save_to_file(file_path, all_saved)
                st.success(f"✅ [{target_p['label']}] ANATOMY 데이터가 영구 업데이트되었습니다!")


# -----------------------------
# 5. 초고속 라이브러리 관리 컴포넌트 (@st.fragment 적용)
# -----------------------------
def parse_number_query(query_str: str) -> List[int]:
    result = []
    if not query_str.strip():
        return result
    parts = re.split(r'[,;\s]+', query_str.strip())
    for part in parts:
        if not part:
            continue
        if "-" in part or "~" in part:
            bounds = re.split(r'[\-~]', part)
            if len(bounds) == 2 and bounds[0].isdigit() and bounds[1].isdigit():
                start, end = int(bounds[0]), int(bounds[1])
                step = 1 if start <= end else -1
                result.extend(range(start, end + step, step))
        elif part.isdigit():
            result.append(int(part))
    return result

# -----------------------------
# 4-Z. 수업용 PPT (AI 사용 안 함 · 토큰 0)
#   - 만들어 둔 교재 자료로 칠판 빔용 PPT를 만듦 (검은 배경, 4:3, 메인 영어 32pt)
#   - 제목: Self-Note 영어/한글 제목 · 문장·해석·단어: 한판승 · 요점 정리: 한판승 핵심 요약
#   - 오른쪽 아래 10pt: 문장에 나온 단어의 유의어 (Voca 단어장 + 한판승 키워드), 발표자 노트에도 함께
#   - 모든 글은 맑은 고딕 글자 폭으로 줄바꿈을 계산해 슬라이드 밖으로 넘치지 않게 크기를 정함
#     (서버에는 맑은 고딕이 없어서 글자 폭 표를 코드에 넣어 둠)
# -----------------------------
PPT_ASSET_DIR = os.path.join(BASE_DIR, "ppt_assets")
PPT_IMG_HEADER = os.path.join(PPT_ASSET_DIR, "header.png")
PPT_IMG_TITLE = os.path.join(PPT_ASSET_DIR, "title.png")
PPT_IMG_SUMMARY = os.path.join(PPT_ASSET_DIR, "summary.png")
PPT_SLIDE_W, PPT_SLIDE_H = 10.0, 7.5   # 4:3 → 4:3 빔에서 화면을 꽉 채움
PPT_BODY_MAX_PT = 32                   # 메인 영어 문장 기본 크기 (긴 문장만 넘치지 않게 줄어듦)
PPT_MARGIN_X = 0.4
PPT_CONTENT_W = PPT_SLIDE_W - 2 * PPT_MARGIN_X
PPT_LINE_PITCH = 1.35   # 글자 크기 대비 한 줄 높이 (넉넉하게 잡아 넘침 방지)
PPT_BOX_PAD_IN = 0.06   # 텍스트 상자 좌우 여백

# 맑은 고딕(보통/굵게) 글자 폭 (글자 크기 100 기준). 한글·한자 등 그 밖의 글자는 100
_PPT_ASCII_W = {
    False: [35, 29, 40, 61, 55, 84, 82, 23, 30, 30, 42, 70, 22, 41, 22, 40, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 22, 22, 70, 70, 70, 46,
            98, 66, 58, 63, 72, 52, 50, 70, 72, 27, 36, 59, 48, 92, 77, 77, 57, 77, 61, 54, 53, 70, 63, 95, 60, 56, 58, 30, 76, 30, 70, 43,
            27, 52, 60, 47, 60, 54, 32, 60, 58, 25, 25, 51, 25, 88, 58, 60, 60, 60, 35, 43, 34, 58, 49, 74, 46, 49, 46, 30, 24, 30, 70],
    True: [35, 32, 48, 61, 58, 88, 86, 28, 36, 36, 46, 72, 26, 41, 26, 44, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 26, 26, 72, 72, 72, 45,
           98, 70, 64, 64, 75, 54, 52, 72, 77, 31, 43, 65, 51, 96, 80, 78, 61, 78, 65, 57, 58, 73, 67, 101, 65, 61, 61, 36, 76, 36, 72, 43,
           31, 54, 63, 49, 63, 55, 37, 63, 61, 27, 28, 56, 27, 92, 61, 62, 63, 63, 39, 46, 38, 61, 54, 80, 54, 54, 48, 36, 31, 36, 72],
}
_PPT_SPECIAL_W = {
    False: {"·": 22, "•": 42, "…": 74, "‘": 23, "’": 23, "“": 38, "”": 38, "—": 103, "–": 51, "≈": 66},
    True: {"·": 26, "•": 42, "…": 88, "‘": 28, "’": 28, "“": 47, "”": 47, "—": 103, "–": 51, "≈": 66},
}

def _ppt_char_w(ch: str, bold: bool) -> float:
    o = ord(ch)
    if 32 <= o <= 126:
        return _PPT_ASCII_W[bold][o - 32]
    if o >= 0x1F000:  # 이모지는 실제 화면에서 더 넓게 그려지므로 넉넉하게
        return 125
    return _PPT_SPECIAL_W[bold].get(ch, 100)

def _ppt_text_w_pt(s: str, pt: float, bold: bool) -> float:
    return sum(_ppt_char_w(ch, bold) for ch in s) * pt / 100.0

def _ppt_count_lines(text: str, width_in: float, pt: float, bold: bool) -> int:
    """글자 폭으로 단어 단위 줄바꿈했을 때의 줄 수"""
    width_pt = max(1.0, (width_in - 2 * PPT_BOX_PAD_IN) * 72 * 0.97)
    total = 0
    for para in str(text).split("\n"):
        lines, cur = 1, ""
        for w in para.split(" "):
            cand = (cur + " " + w) if cur else w
            if _ppt_text_w_pt(cand, pt, bold) <= width_pt:
                cur = cand
                continue
            if cur:
                lines += 1
            # 한 단어가 한 줄보다 길면 글자 단위로 나눔
            cur = ""
            for ch in w:
                if _ppt_text_w_pt(cur + ch, pt, bold) > width_pt and cur:
                    lines += 1
                    cur = ch
                else:
                    cur += ch
        total += lines
    return max(1, total)

@functools.lru_cache(maxsize=50000)
def ppt_text_height_in(text: str, width_in: float, pt: float, bold: bool, spacing: float, space_before_pt: float = 0) -> float:
    return _ppt_count_lines(text, width_in, pt, bold) * pt * PPT_LINE_PITCH * spacing / 72.0 + space_before_pt / 72.0

def ppt_fit_pt(text: str, width_in: float, height_in: float, max_pt: int, min_pt: int, bold: bool, spacing: float):
    """높이 안에 들어가는 가장 큰 글자 크기와 그때 높이. 최소 크기로도 안 들어가면 (None, 높이)"""
    for pt in range(max_pt, min_pt - 1, -1):
        h = ppt_text_height_in(text, width_in, pt, bold, spacing)
        if h <= height_in:
            return pt, h
    return None, ppt_text_height_in(text, width_in, min_pt, bold, spacing)

def ppt_clean_eng(s: str) -> str:
    """한판승 문장에서 [어구](S) 표시, {1:어법} 표시, 끊어 읽기 / 를 지운 깨끗한 원문
    괄호 안 괄호([[it](S) ...](M)), S·V·O·C·M 밖의 이름(V1, O2, IO, S+V …), 이름 없는 [ ], 번호 없는 { }까지 모두 지움"""
    s = str(s or "")
    prev = None
    while prev != s:  # {1:...} 안에 {2:...}가 든 경우: 안쪽부터
        prev, s = s, re.sub(r"\{\d{1,2}:([^{}]*)\}", r"\1", s)
    s = re.sub(r"\{\d{1,2}:|[{}]", "", s)
    s = re.sub(r"\]\s*\(\s*(?:[A-Z][A-Za-z0-9+_/'’\-]{0,9}|conj)\s*\)", "]", s)  # ](S) ](V1) ](S+V) ](IO) …
    s = re.sub(r"(?<=[A-Za-z])\((?:[SVOCM][0-9]?|IO|DO)\)", "", s)  # 괄호 없이 붙은 이름: us(O)
    s = re.sub(r"[\[\]]", "", s)
    s = re.sub(r"\s*/\s*", " ", s)
    s = re.sub(r"\s+([,.;:!?’”)])", r"\1", s)
    s = re.sub(r"\s+([\"'])(?=\s|$|[,.;:!?])", r"\1", s)  # 닫는 따옴표 앞 띄어쓰기만 지움 (여는 따옴표는 앞 낱말과 띄움)
    s = re.sub(r"\s+((?:['’](?:s|re|ve|d|ll|m|t)|n['’]t)\b)", r"\1", s)  # it 's → it's
    return re.sub(r"\s{2,}", " ", s).strip()

def ppt_clean_kor(s: str) -> str:
    return re.sub(r"\s{2,}", " ", re.sub(r"\s*/\s*", " ", str(s or ""))).strip()

def ppt_note_terms(sentence: str, voca_items, keywords) -> list:
    """문장에 나온 단어의 유의어 목록 (Voca 단어장 → 한판승 핵심 키워드 순)"""
    low = " " + re.sub(r"[^a-z' -]", " ", sentence.lower()) + " "
    terms, seen = [], set()
    def in_sentence(word):
        w = re.sub(r"[^a-z' -]", " ", str(word).lower()).strip()
        if len(w) <= 2:
            return False
        # 그대로 있거나, 한 단어일 때 어미만 다른 형태(예: decide → decided)도 인정
        return ((" " + w + " ") in low or (" " + w + "s ") in low or (" " + w + "es ") in low
                or (" " not in w and len(w) > 4 and (" " + w[:-1]) in low))
    for it in voca_items or []:
        w = str(it.get("word", "")).strip()
        syns = [str(s.get("word", "")).strip() for s in (it.get("synonyms") or []) if isinstance(s, dict) and s.get("word")]
        if w and syns and w.lower() not in seen and in_sentence(w):
            terms.append(f"{w} ≈ {', '.join(syns[:3])}")
            seen.add(w.lower())
    for kw in keywords or []:
        w = str(kw.get("word", "")).strip()
        syn = re.sub(r"\([^)]*\)", "", str(kw.get("synonyms", ""))).strip(" ,")
        if w and syn and w.lower() not in seen and in_sentence(w):
            terms.append(f"{w} ≈ {syn}")
            seen.add(w.lower())
    return terms

def ppt_header_label(label: str) -> str:
    """머리말에는 [라벨]로 감싸 넣으므로 '[2026년 10월 고1] 20번' -> '2026년 10월 고1 20번'"""
    m = re.match(r"^\[([^\]]+)\]\s*(.*)$", str(label))
    return f"{m.group(1)} {m.group(2)}".strip() if m else str(label).strip()

def ppt_norm_sent(s) -> str:
    """문장 비교용: 표시·문장부호·대소문자를 뺀 글"""
    return re.sub(r"[^a-z0-9]+", " ", ppt_clean_eng(s).lower()).strip()

def ppt_key_tag(row: Dict, topic: Dict = None) -> str:
    """한판승 문장이 핵심 문장이면 형광펜 위에 붙일 이름표, 아니면 빈 글
    핵심 문장 = 서술형 후보로 표시된 문장, 또는 AI가 고른 주제문(data['topic_sentence'], 문장 글로 찾음)"""
    if row.get("essay_candidate"):
        role = str(row.get("essay_reason", "")).split("·")[0].strip()
        return "★ 핵심 문장" + (f" · {role}" if role else "")
    if topic and topic.get("eng") and ppt_norm_sent(row.get("eng", "")) == ppt_norm_sent(topic["eng"]):
        return f"★ 핵심 문장 · {str(topic.get('role') or '주제문').strip()}"
    return ""

def ppt_needs_topic(vdata: Dict) -> bool:
    """서술형 후보 표시도, AI가 고른 주제문도 없는 한판승 지문"""
    rows = vdata.get("analysis") or []
    return bool(rows) and not any(r.get("essay_candidate") for r in rows) and not (vdata.get("topic_sentence") or {}).get("eng")

def ppt_core_summary_pick(cp: Dict):
    """PPT에 넣을 Core 요약문: (A)·(B) 빈칸이 제대로 있는 첫 번째 것 → (seed, 요약문) 또는 ("", None)
    seed는 Core 교재와 같게 해서 고르는 단어 순서가 교재와 같음"""
    pid = esc(cp.get("id", esc(cp.get("label", ""))))
    for i, ps in enumerate((cp.get("data") or {}).get("summaries") or [], start=1):
        if isinstance(ps, dict) and ppt_core_summary_parts(ps, False, f"{pid}_sum_{i}"):
            return f"{pid}_sum_{i}", ps
    return "", None

def ppt_needs_glossary(cp: Dict) -> bool:
    """PPT 요약문의 단어 뜻(data['summary_glossary'])이 없거나, 요약문이 바뀌어 다시 만들어야 하는 Core 지문"""
    _, ps = ppt_core_summary_pick(cp)
    if ps is None:
        return False
    g = (cp.get("data") or {}).get("summary_glossary") or {}
    return not g.get("words") or ppt_norm_sent(g.get("eng", "")) != ppt_norm_sent(ps.get("eng", ""))

def ppt_summary_vocab(cdata: Dict, ps: Dict):
    """요약문 단어 뜻 → (정답 [(A, 단어, 뜻)], 그 밖의 단어 [(단어, 뜻)])
    그 밖의 단어에서는 정답·함정 단어를 뺌 (고르기 슬라이드에서 답이 드러나지 않게). 요약문에 실제로 있는 단어만"""
    eng = " " + re.sub(r"[^a-z'-]+", " ", str(ps.get("eng", "")).lower()) + " "
    words = [w for w in ((cdata.get("summary_glossary") or {}).get("words") or []) if isinstance(w, dict)]
    dic = {}
    for w in words + [v for v in (cdata.get("core_vocab") or []) if isinstance(v, dict)]:
        k = str(w.get("word", "")).strip().lower()
        if k and str(w.get("meaning", "")).strip():
            dic.setdefault(k, str(w["meaning"]).strip())
    choice = {str(ps.get(k, "")).strip().lower() for k in ("blank_a", "blank_b", "distractor_a", "distractor_b")}
    answers = [(t, str(ps.get(f"blank_{t.lower()}", "")).strip(), dic.get(str(ps.get(f"blank_{t.lower()}", "")).strip().lower(), ""))
               for t in ("A", "B") if str(ps.get(f"blank_{t.lower()}", "")).strip()]
    others, seen = [], set()
    for w in words:
        k = str(w.get("word", "")).strip().lower()
        m = str(w.get("meaning", "")).strip()
        if not k or not m or k in choice or k in seen or (" " + re.sub(r"[^a-z'-]+", " ", k).strip() + " ") not in eng:
            continue
        seen.add(k)
        others.append((str(w["word"]).strip(), m))
    return answers, others

# ---- 수업용 PPT 빈칸 채우기 (AI): 핵심 문장 표시가 없는 지문의 주제문, PPT 요약문의 단어 뜻 ----
#   PPT 만들기 자체는 AI를 쓰지 않음. 이 버튼으로 한 번 채워 저장해 두면 그다음부터는 토큰 0
SCHEMA_PPT_TOPIC = {
    "type": "object",
    "properties": {"results": {"type": "array", "items": {"type": "object", "properties": {
        "id": {"type": "string"}, "no": {"type": "integer"}, "role": {"type": "string"}}, "required": ["id", "no", "role"]}}},
    "required": ["results"]
}
SCHEMA_PPT_GLOSSARY = {
    "type": "object",
    "properties": {"results": {"type": "array", "items": {"type": "object", "properties": {
        "id": {"type": "string"},
        "words": {"type": "array", "items": {"type": "object", "properties": {"word": {"type": "string"}, "meaning": {"type": "string"}}, "required": ["word", "meaning"]}}},
        "required": ["id", "words"]}}},
    "required": ["results"]
}
PPT_TOPIC_PROMPT = """너는 고등 영어 내신 강사다. 아래 지문마다 글의 주제문을 한 문장 골라 번호(no)로 답하라.
- 주제문: 글의 주제·요지를 가장 직접 드러내는 문장. 드러난 주제문이 없으면 요지·결론·주장에 가장 가까운 문장.
- role: 고른 문장의 역할을 한국어 2~6자로 (예: 주제문, 요지, 결론/주장, 핵심 주장).
- id는 지문 앞 [ ] 안의 값을 그대로 쓴다. 지문마다 하나씩 빠짐없이 답한다."""
PPT_GLOSSARY_PROMPT = """너는 고등 영어 내신 강사다. 아래 영어 요약문마다 고등학생이 모르거나 헷갈릴 만한 단어·숙어를 골라 문맥에 맞는 한국어 뜻을 달아라.
- '보기'에 있는 (A)·(B) 정답 단어와 함정 단어는 반드시 넣는다.
- 그 밖에 요약문에서 어려운 단어·숙어 4~10개. a, the, people, school처럼 쉬운 단어는 뺀다.
- word는 요약문에 나온 철자·형태 그대로 (예: solicits, amplified). meaning은 문맥에 맞는 짧은 한국어 뜻 (예: 요청하다, 증폭된).
- id는 요약문 앞 [ ] 안의 값을 그대로 쓴다. 요약문마다 하나씩 빠짐없이 답한다."""
PPT_FILL_CHUNK = 8  # 한 번에 보내는 지문 수

def ppt_fill_cost_krw(model_spec: str, n_topic: int, n_gloss: int) -> float:
    """AI로 채우기 예상 비용(원). 생각(추론) 토큰까지 넉넉히 잡은 대략값"""
    _, model = split_model_spec(model_spec)
    calls = math.ceil(n_topic / PPT_FILL_CHUNK) + math.ceil(n_gloss / PPT_FILL_CHUNK)
    inp = n_topic * 450 + n_gloss * 150 + calls * 350
    out = n_topic * 120 + n_gloss * 350 + calls * 1500
    return ai_cost_usd(model, inp, 0, 0, out) * 1400

def ppt_fill_missing(model_spec: str, vic_items: List[Dict], core_items: List[Dict], logs: List[str]) -> (int, int):
    """vic_items 중 핵심 문장이 없는 지문에 주제문(data['topic_sentence']),
    core_items 중 단어 뜻이 없는 지문에 PPT 요약문 단어 뜻(data['summary_glossary'])을 채움 (받은 dict를 바로 고침, 저장은 부르는 쪽)
    → (주제문 채운 수, 단어 뜻 채운 수)"""
    from concurrent.futures import ThreadPoolExecutor
    tp = [v for v in vic_items if ppt_needs_topic(v.get("data") or {})]
    gl = [c for c in core_items if ppt_needs_glossary(c)]

    def run_topic(chunk):
        body, numbered = [], []
        for k, v in enumerate(chunk):
            rows = v["data"].get("analysis") or []
            lines = [(i, ppt_clean_eng(r.get("eng", ""))) for i, r in enumerate(rows, start=1)]
            numbered.append({i: s for i, s in lines if s})
            body.append(f"[{k}]\n" + "\n".join(f"{i}. {s}" for i, s in lines if s))
        res = llm_json(model_spec, [PPT_TOPIC_PROMPT, "\n\n".join(body)], SCHEMA_PPT_TOPIC, cache_first=True)
        n = 0
        for r in res.get("results") or []:
            try:
                k, no = int(str(r.get("id", "")).strip("[] ")), int(r.get("no"))
            except (TypeError, ValueError):
                continue
            if 0 <= k < len(chunk) and no in numbered[k]:
                chunk[k]["data"]["topic_sentence"] = {"eng": numbered[k][no], "role": str(r.get("role") or "주제문").strip()[:12] or "주제문"}
                n += 1
        return n

    def run_gloss(chunk):
        body, picked = [], []
        for k, c in enumerate(chunk):
            _, ps = ppt_core_summary_pick(c)
            picked.append(ps)
            opts = "  ".join(f"({t}) {ps.get(f'blank_{t.lower()}', '')} / {ps.get(f'distractor_{t.lower()}', '')}" for t in ("A", "B"))
            one_line = " ".join(str(ps.get("eng", "")).split())
            body.append(f"[{k}] {one_line}\n보기: {opts}")
        res = llm_json(model_spec, [PPT_GLOSSARY_PROMPT, "\n\n".join(body)], SCHEMA_PPT_GLOSSARY, cache_first=True)
        n = 0
        for r in res.get("results") or []:
            try:
                k = int(str(r.get("id", "")).strip("[] "))
            except (TypeError, ValueError):
                continue
            words = [{"word": str(w.get("word", "")).strip(), "meaning": str(w.get("meaning", "")).strip()}
                     for w in (r.get("words") or []) if isinstance(w, dict) and str(w.get("word", "")).strip() and str(w.get("meaning", "")).strip()]
            if 0 <= k < len(chunk) and words:
                chunk[k].setdefault("data", {})["summary_glossary"] = {"eng": str(picked[k].get("eng", "")), "words": words}
                n += 1
        return n

    jobs = [(run_topic, tp[i:i + PPT_FILL_CHUNK]) for i in range(0, len(tp), PPT_FILL_CHUNK)]
    jobs += [(run_gloss, gl[i:i + PPT_FILL_CHUNK]) for i in range(0, len(gl), PPT_FILL_CHUNK)]
    n_topic = n_gloss = 0
    if not jobs:
        return 0, 0
    with ThreadPoolExecutor(max_workers=min(4, len(jobs))) as pool:
        futs = [(fn, chunk, pool.submit(fn, chunk)) for fn, chunk in jobs]
        for fn, chunk, f in futs:
            try:
                n = f.result()
            except Exception as e:
                logs.append(f"⚠️ {'주제문' if fn is run_topic else '단어 뜻'} {len(chunk)}개 묶음 실패: {e}")
                continue
            if fn is run_topic:
                n_topic += n
            else:
                n_gloss += n
    return n_topic, n_gloss

def ppt_core_summary_parts(ps: Dict, reveal: bool, seed_key: str) -> list:
    """Core 요약문 한 개 → PPT 문단 조각 [(글, 굵게, 밑줄)]
    reveal=False: (A) [ 정답 / 함정 ] 고르기, True: (A) 정답. 고르는 순서는 Core 교재와 같음 (같은 seed)"""
    eng = re.sub(r"\s+", " ", str(ps.get("eng", ""))).strip()
    if not eng:
        return []
    rng = random.Random(seed_key)
    spans = []
    for tag, default_trap in (("A", "superficial"), ("B", "passivity")):
        ans = str(ps.get(f"blank_{tag.lower()}", "")).strip()
        if not ans:
            continue
        trap = str(ps.get(f"distractor_{tag.lower()}") or extract_trap_word_fallback(ps.get(f"synonyms_{tag.lower()}", ""), default_trap)).strip() or "opposite"
        opts = [ans, trap]
        rng.shuffle(opts)
        m = (re.search(r"\(\s*" + tag + r"\s*\)\s*" + re.escape(ans), eng, flags=re.IGNORECASE)
             or re.search(r"\(\s*" + tag + r"\s*\)", eng))
        if m:
            spans.append((m.start(), m.end(), f"({tag}) {ans}" if reveal else f"({tag}) [ {opts[0]} / {opts[1]} ]"))
    spans.sort()
    if not spans or any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
        return []
    parts, pos = [], 0  # 글은 모두 굵게 (빔에서 잘 보이게), 고르는 자리·정답은 밑줄까지
    for s, e, txt in spans:
        parts += [(eng[pos:s], True, False), (txt, True, True)]
        pos = e
    parts.append((eng[pos:], True, False))
    return [p for p in parts if p[0]]

def build_class_ppt(passages: List[Dict], anatomy_list: List[Dict], voca_list: List[Dict], school: str, test_name: str, core_list: List[Dict] = None):
    """선택한 한판승 지문으로 수업용 PPT를 만들어 (파일 bytes, 슬라이드 수)를 돌려줌"""
    from pptx import Presentation  # PPT를 만들 때만 불러와 앱 시작을 느리게 하지 않음
    from pptx.util import Pt, Inches
    from pptx.enum.text import MSO_AUTO_SIZE, PP_ALIGN
    from pptx.dml.color import RGBColor
    from pptx.oxml.ns import qn
    from pptx.oxml.xmlchemy import OxmlElement

    YELLOW_TITLE, YELLOW_VOCAB = RGBColor(255, 215, 0), RGBColor(255, 255, 102)
    WHITE, NOTE_GRAY, HEAD_GRAY = RGBColor(255, 255, 255), RGBColor(150, 150, 150), RGBColor(200, 200, 200)
    BLACK = RGBColor(0, 0, 0)

    def new_slide():
        slide = prs.slides.add_slide(layout)
        fill = slide.background.fill
        fill.solid()
        fill.fore_color.rgb = RGBColor(0, 0, 0)
        return slide

    def picture(slide, path, left, top, width):
        if os.path.exists(path):
            try:
                slide.shapes.add_picture(path, Inches(left), Inches(top), width=Inches(width))
            except Exception:
                pass

    def textbox(slide, left, top, width, height):
        tf = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height)).text_frame
        tf.word_wrap = True
        tf.auto_size = MSO_AUTO_SIZE.NONE
        tf.margin_left = tf.margin_right = Inches(PPT_BOX_PAD_IN)
        tf.margin_top = tf.margin_bottom = 0
        return tf

    def white_marker(run):
        """형광펜처럼 글자 줄에만 흰색을 칠함 (PowerPoint 2019·365의 '텍스트 강조 색').
        강조 색을 못 그리는 옛 프로그램에서도 글자가 보이도록 흰 테두리를 함께 넣음 (흰 형광펜 위에서는 티 안 남)"""
        rPr = run._r.get_or_add_rPr()
        ln = OxmlElement("a:ln")
        ln.set("w", "9525")
        ln_fill = OxmlElement("a:solidFill")
        ln_clr = OxmlElement("a:srgbClr")
        ln_clr.set("val", "FFFFFF")
        ln_fill.append(ln_clr)
        ln.append(ln_fill)
        rPr.insert(0, ln)  # a:ln은 rPr의 맨 앞이어야 함
        hl = OxmlElement("a:highlight")
        hl_clr = OxmlElement("a:srgbClr")
        hl_clr.set("val", "FFFFFF")
        hl.append(hl_clr)
        rPr.find(qn("a:solidFill")).addnext(hl)  # a:highlight는 글자 색(solidFill) 바로 뒤

    def rich_paragraph(tf, parts, size, color, first=False, spacing=1.6, before=0):
        """한 문단 안에 굵게·밑줄을 섞어 씀. parts = [(글, 굵게, 밑줄)]"""
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        p.line_spacing = spacing
        if before > 0:
            p.space_before = Pt(before)
        for text, bold, underline in parts:
            if not text:
                continue
            run = p.add_run()
            run.text = text
            run.font.size = Pt(size)
            run.font.bold = bold
            run.font.underline = underline
            run.font.color.rgb = color

    def paragraph(tf, text, size, color, first=False, spacing=1.85, before=0, bold=True, align=None, marker=False):
        # 글자 색을 글자마다 직접 넣어 PowerPoint에서 검정으로 풀리지 않게 함
        lines = [ln.strip() for ln in str(text).split("\n") if ln.strip()] or [""]
        for i, line in enumerate(lines):
            p = tf.paragraphs[0] if (first and i == 0) else tf.add_paragraph()
            p.line_spacing = spacing
            if i == 0 and before > 0:
                p.space_before = Pt(before)
            if align is not None:
                p.alignment = align
            run = p.add_run()
            run.text = line
            run.font.size = Pt(size)
            run.font.bold = bold
            run.font.color.rgb = color
            if marker:
                white_marker(run)

    def header(slide, label):
        text_left = PPT_MARGIN_X
        if os.path.exists(PPT_IMG_HEADER):
            picture(slide, PPT_IMG_HEADER, PPT_MARGIN_X, 0.15, 0.5)
            text_left = PPT_MARGIN_X + 0.62
        w = PPT_SLIDE_W - PPT_MARGIN_X - text_left
        text = " ".join(t for t in (school, test_name) if t) + f" 대비_낱낱이 해부하다! [{ppt_header_label(label)}]"
        pt, h = ppt_fit_pt(text, w, 0.5, 15, 10, True, 1.2)
        tf = textbox(slide, text_left, 0.15, w, max(h, 0.3))
        paragraph(tf, text, pt or 10, HEAD_GRAY, first=True, spacing=1.2)

    def title_slide(label, eng_title, kor_title):
        slide = new_slide()
        header(slide, label)
        has_img = os.path.exists(PPT_IMG_TITLE)
        width, box_h = (PPT_CONTENT_W - 1.4 if has_img else PPT_CONTENT_W), 5.8
        parts = [(f"📌 {label}", -4, 0, 1.5), (f"\"{eng_title}\"", 0, 20, 1.6), (kor_title, -2, 18, 1.6)]
        chosen = 14
        for pt in range(PPT_BODY_MAX_PT + 4, 13, -1):
            if sum(ppt_text_height_in(t, width, max(12, pt + d), True, sp, b) for t, d, b, sp in parts if t) <= box_h:
                chosen = pt
                break
        tf = textbox(slide, PPT_MARGIN_X, 1.2, width, box_h)
        paragraph(tf, parts[0][0], max(12, chosen - 4), YELLOW_TITLE, first=True, spacing=1.5)
        paragraph(tf, parts[1][0], chosen, WHITE, spacing=1.6, before=20)
        if kor_title:
            paragraph(tf, kor_title, max(12, chosen - 2), YELLOW_TITLE, spacing=1.6, before=18)
        if has_img:
            picture(slide, PPT_IMG_TITLE, PPT_SLIDE_W - PPT_MARGIN_X - 1.1, 1.4, 1.1)

    def sentence_slide(label, sent, trans, vocabs, note_terms, key_tag=""):
        """key_tag가 있으면 핵심 문장: 영어 문장에 흰 형광펜을 긋고 글씨는 검게 (배경은 그대로 검정)"""
        slide = new_slide()
        header(slide, label)
        left, width, top0, bottom = PPT_MARGIN_X, PPT_CONTENT_W, 0.8, PPT_SLIDE_H - 0.1
        tag_pt = 14
        band_extra = ppt_text_height_in(key_tag, width, tag_pt, True, 1.2) if key_tag else 0.0

        # 오른쪽 아래 10pt 단어·유의어 (최대 3줄, 넘치는 단어는 발표자 노트에만)
        note_w, note_text, note_h = 4.2, "", 0.0
        shown = []
        for t in note_terms:
            if _ppt_count_lines("💡 " + "  ·  ".join(shown + [t]), note_w, 10, False) <= 3:
                shown.append(t)
        if shown:
            note_text = "💡 " + "  ·  ".join(shown)
            note_h = ppt_text_height_in(note_text, note_w, 10, False, 1.0) + 0.02
        content_bottom = bottom - (note_h + 0.06 if note_h else 0)

        vocab_lines = [f"• {v}" for v in vocabs]
        two_cols = len(vocab_lines) >= 3 or sum(len(v) for v in vocab_lines) > 95
        half = math.ceil(len(vocab_lines) / 2)
        cols = [vocab_lines[:half], vocab_lines[half:]] if two_cols else [vocab_lines]
        col_w = (width - 0.2) / 2 if two_cols else width

        def vocab_height(pt):
            return max((ppt_text_height_in("\n".join(c), col_w, pt, True, 1.2) for c in cols if c), default=0.0)

        # 본문 32pt를 지키고 단어 17pt → 10pt, 해석 → 9pt를 먼저 줄임. 그래도 안 들어가면 본문을 줄임
        plan = None
        for body_pt in range(PPT_BODY_MAX_PT, 13, -1):
            body_h = ppt_text_height_in(sent, width, body_pt, True, 1.85) + band_extra
            for vocab_pt in range(17, 9, -1):
                v_h = vocab_height(vocab_pt) if vocab_lines else 0.0
                for trans_pt in range(vocab_pt, 8, -1):
                    t_h = ppt_text_height_in(trans, width, trans_pt, False, 1.2) if trans else 0.0
                    if top0 + body_h + (0.12 + v_h if v_h else 0) + (0.08 + t_h if t_h else 0) <= content_bottom:
                        plan = (body_pt, body_h, vocab_pt, v_h, trans_pt, t_h)
                        break
                if plan:
                    break
            if plan:
                break
        if plan is None:  # 아주 긴 문장: 단어를 빼서라도 본문·해석을 지킴 (단어는 발표자 노트에 남음)
            vocab_lines, cols = [], [[]]
            plan = (14, ppt_text_height_in(sent, width, 14, True, 1.85) + band_extra, 12, 0.0, 9,
                    ppt_text_height_in(trans, width, 9, False, 1.2) if trans else 0.0)
        body_pt, body_h, vocab_pt, v_h, trans_pt, t_h = plan

        if key_tag:
            b_tf = textbox(slide, left, top0, width, body_h)
            paragraph(b_tf, key_tag, tag_pt, WHITE, first=True, spacing=1.2)
            paragraph(b_tf, sent, body_pt, BLACK, spacing=1.85, marker=True)
        else:
            paragraph(textbox(slide, left, top0, width, body_h), sent, body_pt, WHITE, first=True, spacing=1.85)
        y = top0 + body_h
        if vocab_lines and v_h:
            # 단어는 가능하면 4.4인치 아래부터 (자리가 모자라면 본문 바로 아래)
            rest = (0.08 + t_h) if t_h else 0
            vocab_top = max(y + 0.12, 4.4) if max(y + 0.12, 4.4) + v_h + rest <= content_bottom else y + 0.12
            positions = [(left, col_w), (left + col_w + 0.2, col_w)] if two_cols else [(left, width)]
            for col, (c_left, c_w) in zip(cols, positions):
                if col:
                    v_tf = textbox(slide, c_left, vocab_top, c_w, v_h)
                    for i, line in enumerate(col):
                        paragraph(v_tf, line, vocab_pt, YELLOW_VOCAB, first=(i == 0), spacing=1.2)
            y = vocab_top + v_h
        if trans and t_h:
            paragraph(textbox(slide, left, y + 0.08, width, t_h), trans, trans_pt, WHITE, first=True, spacing=1.2, bold=False)
        if note_text:
            n_tf = textbox(slide, PPT_SLIDE_W - 0.3 - note_w, bottom - note_h, note_w, note_h)
            paragraph(n_tf, note_text, 10, NOTE_GRAY, first=True, spacing=1.0, bold=False, align=PP_ALIGN.RIGHT)

        # 발표자 노트: 단어·유의어 전체와 해석 (슬라이드 쇼 발표자 보기에서만 보임)
        notes = []
        if key_tag:
            notes.append("[핵심 문장] " + key_tag)
        if vocabs:
            notes.append("[문장 단어] " + " / ".join(vocabs))
        if note_terms:
            notes.append("[유의어] " + " / ".join(note_terms))
        if trans:
            notes.append("[해석] " + trans)
        if notes:
            slide.notes_slide.notes_text_frame.text = "\n".join(notes)

    def summary_slide(label, summary_text):
        slide = new_slide()
        header(slide, label)
        if os.path.exists(PPT_IMG_SUMMARY):
            picture(slide, PPT_IMG_SUMMARY, PPT_MARGIN_X, 1.2, 1.0)
            s_left, s_w = PPT_MARGIN_X + 1.25, PPT_CONTENT_W - 1.25
        else:
            s_left, s_w = PPT_MARGIN_X, PPT_CONTENT_W
        box_h, head, chosen = 5.9, "📌 영쌤의 요점 정리", 14
        for pt in range(PPT_BODY_MAX_PT, 13, -1):
            if ppt_text_height_in(head, s_w, pt, True, 1.5) + ppt_text_height_in(summary_text, s_w, pt, True, 1.7, 20) <= box_h:
                chosen = pt
                break
        tf = textbox(slide, s_left, 1.1, s_w, box_h)
        paragraph(tf, head, chosen, YELLOW_TITLE, first=True, spacing=1.5)
        paragraph(tf, summary_text, chosen, WHITE, spacing=1.7, before=20)

    def summary_layout(blocks, vocab_lines, width, box_h, max_pt, min_pt):
        """blocks = [(글, 크기 차이, 줄간격, 위 여백pt)] (모두 굵게) + 그 아래 단어 목록(4개 이상이면 두 줄로)이
        box_h에 다 들어가는 가장 큰 기준 크기 → (기준 크기, 글 높이, 단어 크기, 단어 높이, 단어 칸들, 칸 폭)"""
        two = len(vocab_lines) >= 4
        half = math.ceil(len(vocab_lines) / 2)
        cols = [vocab_lines[:half], vocab_lines[half:]] if two else [vocab_lines]
        col_w = (width - 0.2) / 2 if two else width
        plan = None
        for pt in range(max_pt, min_pt - 1, -1):
            h = sum(ppt_text_height_in(t, width, max(10, pt + d), True, sp, bf) for t, d, sp, bf in blocks if t)
            v_pt = max(12, min(20, pt - 8))
            v_h = max((ppt_text_height_in("\n".join(c), col_w, v_pt, True, 1.2) for c in cols if c), default=0.0)
            plan = (pt, h, v_pt, v_h, cols, col_w)
            if h + (0.15 + v_h if vocab_lines else 0) <= box_h:
                break
        return plan

    def vocab_columns(slide, cols, col_w, top, v_pt, v_h):
        for c_i, col in enumerate(cols):
            if col:
                v_tf = textbox(slide, PPT_MARGIN_X + c_i * (col_w + 0.2), top, col_w, v_h)
                for i, line in enumerate(col):
                    paragraph(v_tf, line, v_pt, YELLOW_VOCAB, first=(i == 0), spacing=1.2)

    def core_summary_slides(label, cdata, ps, seed_key):
        """Core 요약문 하나로 (A)·(B) 단어 고르기 슬라이드 + 정답 슬라이드. 글은 모두 굵게 (빔에서 잘 보이게)
        고르기: 정답·함정을 뺀 나머지 단어 뜻 · 정답: 정답 단어 뜻 + 나머지 단어 뜻"""
        q = ppt_core_summary_parts(ps, False, seed_key)
        a = ppt_core_summary_parts(ps, True, seed_key)
        if not q:
            return
        answers, others = ppt_summary_vocab(cdata, ps)
        other_lines = [f"• {w}: {m}" for w, m in others]
        width, top, box_h = PPT_CONTENT_W, 0.95, PPT_SLIDE_H - 0.95 - 0.3
        ask, kor = "(A), (B)에 들어갈 알맞은 말을 고르세요.", ppt_clean_kor(ps.get("kor", ""))

        slide = new_slide()
        header(slide, label)
        head = "📝 요약문 완성 · 알맞은 말 고르기"
        q_text = "".join(t for t, _, _ in q)
        pt, h, v_pt, v_h, cols, col_w = summary_layout([(head, -4, 1.3, 0), (ask, -10, 1.2, 8), (q_text, 0, 1.7, 22)],
                                                       other_lines, width, box_h, 30, 14)
        tf = textbox(slide, PPT_MARGIN_X, top, width, h)
        paragraph(tf, head, max(10, pt - 4), WHITE, first=True, spacing=1.3)
        paragraph(tf, ask, max(10, pt - 10), WHITE, spacing=1.2, before=8)
        rich_paragraph(tf, q, pt, WHITE, spacing=1.7, before=22)
        if other_lines:
            vocab_columns(slide, cols, col_w, top + h + 0.15, v_pt, v_h)
        slide.notes_slide.notes_text_frame.text = (f"[정답] (A) {ps.get('blank_a', '')}  (B) {ps.get('blank_b', '')}"
                                                   + (f"\n[해석] {kor}" if kor else ""))

        slide = new_slide()
        header(slide, label)
        head = "✅ 정답 확인"
        ans_text = "\n".join(f"({t}) {w}" + (f": {m}" if m else "") for t, w, m in answers)
        a_text = "".join(t for t, _, _ in a)
        pt, h, v_pt, v_h, cols, col_w = summary_layout([(head, -4, 1.3, 0), (a_text, 0, 1.7, 18), (ans_text, -2, 1.3, 14), (kor, -8, 1.3, 14)],
                                                       other_lines, width, box_h, 30, 12)
        tf = textbox(slide, PPT_MARGIN_X, top, width, h)
        paragraph(tf, head, max(10, pt - 4), WHITE, first=True, spacing=1.3)
        rich_paragraph(tf, a, pt, WHITE, spacing=1.7, before=18)
        paragraph(tf, ans_text, max(10, pt - 2), WHITE, spacing=1.3, before=14)
        if kor:
            paragraph(tf, kor, max(10, pt - 8), WHITE, spacing=1.3, before=14)
        if other_lines:
            vocab_columns(slide, cols, col_w, top + h + 0.15, v_pt, v_h)
        syn = [f"({k}) {ppt_clean_kor(ps.get(f'synonyms_{k.lower()}', ''))}" for k in ("A", "B") if ps.get(f"synonyms_{k.lower()}")]
        if syn:
            slide.notes_slide.notes_text_frame.text = "[유의어·함정] " + "\n".join(syn)

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(PPT_SLIDE_W), Inches(PPT_SLIDE_H)
    layout = prs.slide_layouts[6]
    anatomy = {p["label"]: p for p in anatomy_list}
    voca = {p["label"]: p for p in voca_list}
    core = {p["label"]: p for p in (core_list or [])}
    for v in passages:
        label = v.get("label", "")
        vdata, adata = v.get("data", {}), anatomy.get(label, {}).get("data", {})
        title_slide(label, str(adata.get("english_title") or "Reading Comprehension").strip(), str(adata.get("korean_title") or "").strip())
        voca_items = voca.get(label, {}).get("data", {}).get("voca_list", [])
        topic = vdata.get("topic_sentence") if isinstance(vdata.get("topic_sentence"), dict) else None
        for row in vdata.get("analysis", []):
            sent = ppt_clean_eng(row.get("eng", ""))
            if not sent:
                continue
            vocabs = [f"{str(x.get('word', '')).strip()}: {str(x.get('meaning', '')).strip()}" for x in (row.get("sentence_vocab") or []) if x.get("word")]
            sentence_slide(label, sent, ppt_clean_kor(row.get("kor", "")), vocabs, ppt_note_terms(sent, voca_items, vdata.get("keywords", [])),
                           key_tag=ppt_key_tag(row, topic))
        summary_slide(label, str(vdata.get("summary", "")).strip())
        cp = core.get(label)
        if cp:
            seed_key, ps = ppt_core_summary_pick(cp)
            if ps:
                core_summary_slides(label, cp.get("data") or {}, ps, seed_key)
    stream = io.BytesIO()
    prs.save(stream)
    return stream.getvalue(), len(prs.slides)

def render_class_ppt_section(sel_passages: List[Dict], school_name: str, text_model: str = ""):
    """한판승 탭: 고른 지문으로 수업용 PPT 만들기"""
    with st.expander(f"🎬 선택한 {len(sel_passages)}개 지문으로 수업용 PPT 만들기 (AI 사용 안 함 · 토큰 0)"):
        st.caption("검은 배경 4:3 슬라이드 · 영어 문장 32pt(긴 문장만 줄어듦) · 제목은 Self-Note, 요점 정리는 한판승에서 가져옵니다. "
                   "오른쪽 아래 10pt 유의어와 발표자 노트도 함께 들어갑니다. "
                   "핵심 문장(주제문)은 흰 형광펜에 검은 글씨로 표시하고, 끝에 Core 요약문 (A)·(B) 고르기와 정답 슬라이드가 붙습니다.")

        # 핵심 문장 표시가 없는 지문 · 요약문 단어 뜻이 없는 지문은 AI로 한 번 채워 저장 (그다음부터는 토큰 0)
        core_by = {p["label"]: p for p in (st.session_state.get("saved_c") or [])}
        need_t = [p for p in sel_passages if ppt_needs_topic(p.get("data") or {})]
        need_g = [core_by[p["label"]] for p in sel_passages if p.get("label") in core_by and ppt_needs_glossary(core_by[p["label"]])]
        if (need_t or need_g) and text_model:
            parts = ([f"핵심 문장 표시가 없는 지문 {len(need_t)}개"] if need_t else []) + ([f"요약문 단어 뜻이 없는 지문 {len(need_g)}개"] if need_g else [])
            st.info(f"🖍️ {' · '.join(parts)}가 있어요. AI로 한 번 채워 저장하면 PPT에 형광펜과 단어 뜻이 들어가요.")
            won = ppt_fill_cost_krw(text_model, len(need_t), len(need_g))
            if st.button(f"🤖 AI로 채워 저장 (예상 약 {won:,.0f}원 · {ai_model_label(text_model)})", key="ppt_fill_btn"):
                logs = []
                with st.spinner("주제문과 요약문 단어 뜻을 고르는 중이에요..."):
                    n_t, n_g = ppt_fill_missing(text_model, need_t, need_g, logs)
                if n_t:
                    save_to_file(VICTORY_FILE, st.session_state.saved_v)
                if n_g:
                    save_to_file(CORE_FILE, st.session_state.saved_c)
                st.session_state.pop("ppt_file", None)  # 예전에 만든 PPT는 새 내용이 없으니 지움
                for line in logs:
                    st.warning(line)
                st.success(f"✅ 주제문 {n_t}개 · 요약문 단어 뜻 {n_g}개를 채워 저장했어요. PPT를 다시 만들어 주세요.")
        p_col1, p_col2, p_col3 = st.columns([2, 2, 1.6])
        ppt_school = p_col1.text_input("🏫 머리말 학교명", value=school_name, key="ppt_school")
        ppt_test = p_col2.text_input("📝 머리말 시험명", value="", placeholder="예: 고1 10모", key="ppt_test")
        if p_col3.button("🎬 PPT 만들기", type="primary", use_container_width=True, key="ppt_make_btn"):
            with st.spinner("PPT를 만드는 중입니다..."):
                data, n_slides = build_class_ppt(sel_passages, st.session_state.saved_a, st.session_state.saved_w, ppt_school.strip(), ppt_test.strip(),
                                                 st.session_state.get("saved_c") or [])
            # 파일 이름은 머리말 시험명 그대로 (파일 이름에 못 쓰는 글자만 지움, 비어 있으면 날짜로)
            file_stem = re.sub(r'[\\/:*?"<>|]', "", ppt_test).strip() or f"수업용_PPT_{time.strftime('%m%d_%H%M')}"
            st.session_state.ppt_file = (data, n_slides, len(sel_passages), f"{file_stem}.pptx")
        if st.session_state.get("ppt_file"):
            data, n_slides, n_passages, file_name = st.session_state.ppt_file
            st.download_button(f"📥 수업용 PPT 다운로드 (지문 {n_passages}개 · 슬라이드 {n_slides}장)", data,
                               file_name,
                               "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                               use_container_width=True, key="ppt_dl_btn")

@st.fragment
def render_library_manager(prefix: str, title_name: str, file_path: str, css_str: str, target_type: str, text_model: str, api_delay: int, academy_name: str, school_name: str, h_url: str, f_url: str):
    saved_key = {"victory": "saved_v", "anatomy": "saved_a", "core": "saved_c", "voca": "saved_w", "past_voca": "saved_past_voca"}[target_type]
    sel_key = {"victory": "v_selected", "anatomy": "a_selected", "core": "c_selected", "voca": "w_selected", "past_voca": "past_v_selected"}[target_type]
    chk_prefix = {"victory": "vchk_", "anatomy": "achk_", "core": "cchk_", "voca": "wchk_", "past_voca": "past_vchk_"}[target_type]

    saved_list: List[Dict] = st.session_state[saved_key]
    selected_ids: List[str] = st.session_state[sel_key]

    if target_type in ["core", "voca"]:
        existing_labels = {p["label"] for p in saved_list}
        source_pool = {}
        for p in st.session_state.saved_v + st.session_state.saved_a + st.session_state.saved_c:
            lbl = p.get("label", "")
            if lbl and lbl not in existing_labels and p.get("body_text"):
                source_pool[lbl] = p

        core_by_label = {p["label"]: p["data"] for p in st.session_state.saved_c if p.get("label") and p.get("data")}

        if source_pool or (target_type == "voca" and core_by_label):
            with st.container(border=True):
                col_sync1, col_sync2 = st.columns([3.5, 2])
                if source_pool:
                    col_sync1.markdown(
                        f"💡 **엑셀 재업로드 불필요!** 기존 저장된 지문 중 **{title_name} 데이터가 아직 없는 지문이 `{len(source_pool)}개`** 있습니다."
                    )
                    # 비용이 드는 일괄 생성이라 바로 시작하지 않고 예상 비용을 보여 준 뒤 한 번 더 확인
                    with col_sync2.popover(f"🔄 누락된 {len(source_pool)}개 지문 {title_name} 자동 생성", use_container_width=True):
                        st.caption(book_cost_line(text_model, {target_type: len(source_pool)}))
                        if st.button("생성 시작", type="primary", use_container_width=True, key=f"sync_{prefix}_btn"):
                            rows = [{"label": sp["label"], "text": sp["body_text"], "school": sp.get("school_name", ""), "img_path": sp.get("image_path", "")}
                                    for sp in source_pool.values()]
                            n_added = enqueue_generation(rows, [target_type], text_model, api_delay, academy_name, school_name)
                            st.session_state.gen_flash = f"✅ {n_added}개 지문의 {title_name} 생성을 작업에 추가했습니다. 창을 닫아도 서버에서 계속 진행됩니다."
                            st.rerun()

                if target_type == "voca" and core_by_label and saved_list:
                    if st.button("🔗 이미 생성된 [ANATOMY CORE]의 주제·요약 패러프레이징 어휘들을 현재 VOCA 단어장에 일괄 병합", use_container_width=True, key="merge_core_to_voca_btn"):
                        total_merged = 0
                        for vp in saved_list:
                            lbl = vp.get("label", "")
                            if lbl in core_by_label:
                                total_merged += merge_core_vocab_into_voca_item(core_by_label[lbl], vp.get("data", {}))
                        save_to_file(VOCA_FILE, saved_list)
                        st.success(f"✅ 총 {total_merged}개의 ANATOMY CORE 패러프레이징 핵심 어휘가 단어장에 추가·병합되었습니다!")

    if not saved_list:
        st.info(f"저장된 {title_name} 데이터가 없습니다. 상단에서 파일을 업로드해 생성해주세요.")
        return

    id_to_item = {p["id"]: p for p in saved_list}
    shares_selection = target_type in SHARED_SELECTION_TYPES
    label_to_id = {p["label"]: p["id"] for p in saved_list}
    if shares_selection:
        # 다른 교재 탭에서 고른 지문도 같은 이름이면 선택된 상태로 (이 교재에 없는 지문은 건너뜀)
        selected_ids[:] = [label_to_id[l] for l in st.session_state.shared_sel_labels if l in label_to_id]
    else:
        selected_ids[:] = [sid for sid in selected_ids if sid in id_to_item]
    selected_set = set(selected_ids)

    def sync_shared_selection():
        """이 탭에서 바뀐 선택을 공유 목록에 반영 (이 교재에 없는 지문의 선택은 그대로 유지)"""
        if not shares_selection:
            return
        mine = [id_to_item[sid]["label"] for sid in selected_ids if sid in id_to_item]
        others = [l for l in st.session_state.shared_sel_labels if l not in label_to_id]
        st.session_state.shared_sel_labels = mine + others

    if target_type == "past_voca":
        q_c1, q_c2, q_c3 = st.columns(3)
        if q_c1.button("🏫 운양고 기출 시험지 전체 일괄 선택", use_container_width=True, key="past_sel_unyang"):
            selected_ids[:] = [p["id"] for p in saved_list if "운양" in p.get("label", "") or "운양" in p.get("school_name", "")]
            selected_set = set(selected_ids)
        if q_c2.button("🏫 제일고 기출 시험지 전체 일괄 선택", use_container_width=True, key="past_sel_jeil"):
            selected_ids[:] = [p["id"] for p in saved_list if "제일" in p.get("label", "") or "제일" in p.get("school_name", "")]
            selected_set = set(selected_ids)
        if q_c3.button("📚 저장된 모든 학교 기출 시험지 전체 선택", use_container_width=True, key="past_sel_all_schools"):
            selected_ids[:] = [p["id"] for p in saved_list]
            selected_set = set(selected_ids)

    groups = sorted(list({extract_group_prefix(p["label"]) for p in saved_list}), key=natural_sort_key)
    f_col1, f_col2, f_col3, f_col4 = st.columns([2, 2.5, 2.5, 1.5])
    
    with f_col1:
        selected_group = st.selectbox("📁 학교/출처 그룹 필터", options=["전체 보기"] + groups, key=f"{prefix}_group_filter")
    with f_col2:
        search_kw = st.text_input(f"🔍 {title_name} 키워드 검색 (띄어쓰기로 다중 검색)", placeholder="예: 운양고 26년 1학기", key=f"{prefix}_search")
    with f_col3:
        num_query = st.text_input("⚡ 번호 퀵 선택", placeholder="예: 1, 2 (또는 문항 번호)", key=f"{prefix}_num_query")
    with f_col4:
        page_limit_opt = st.selectbox("📄 화면 표시 개수", options=[60, 120, 240, "전체"], index=0, key=f"{prefix}_page_limit")

    keywords = [k.lower() for k in search_kw.split() if k.strip()]
    display_items = []
    for p in saved_list:
        if selected_group != "전체 보기" and extract_group_prefix(p["label"]) != selected_group:
            continue
        lbl_lower = p["label"].lower()
        if keywords and not all(k in lbl_lower for k in keywords):
            continue
        display_items.append(p)

    b_col1, b_col2, b_col3, b_col4, b_col5 = st.columns([1.5, 1.5, 1.5, 1.5, 1.5])

    if b_col1.button("✅ 검색 결과 전체 선택", use_container_width=True, key=f"{prefix}_sel_all"):
        for p in display_items:
            pid = p["id"]
            if pid not in selected_set:
                selected_ids.append(pid)
                selected_set.add(pid)

    if b_col2.button("⚡ 입력한 번호만 선택 추가", use_container_width=True, key=f"{prefix}_apply_nums"):
        target_nums = parse_number_query(num_query)
        if target_nums:
            for t_num in target_nums:
                for p in display_items:
                    if extract_question_number(p["label"]) == t_num:
                        pid = p["id"]
                        if pid not in selected_set:
                            selected_ids.append(pid)
                            selected_set.add(pid)
        else:
            st.toast("💡 먼저 '번호 퀵 선택' 칸에 숫자를 입력해주세요.")

    if b_col3.button("🔢 선택 항목 이름/번호순 정렬", use_container_width=True, key=f"{prefix}_sort_sel"):
        selected_ids.sort(key=lambda sid: natural_sort_key(id_to_item[sid]))

    if b_col4.button("⬜ 전체 선택 해제", use_container_width=True, key=f"{prefix}_desel_all"):
        selected_ids.clear()
        selected_set.clear()
        if shares_selection:
            st.session_state.shared_sel_labels = []  # 모든 교재 탭의 선택을 함께 해제

    # 영구 삭제는 실수 방지를 위해 한 번 더 확인
    with b_col5.popover(f"🗑️ 선택 항목 삭제 ({len(selected_ids)})", use_container_width=True, disabled=not selected_ids):
        st.warning(f"선택한 {len(selected_ids)}개를 {title_name}에서 영구 삭제합니다. 되돌릴 수 없습니다.")
        if st.button("삭제 확정", type="primary", use_container_width=True, key=f"{prefix}_del"):
            st.session_state[saved_key] = [p for p in saved_list if p["id"] not in selected_set]
            selected_ids.clear()
            save_to_file(file_path, st.session_state[saved_key])
            st.rerun()

    sync_shared_selection()

    total_filtered = len(display_items)
    if page_limit_opt == "전체" or total_filtered <= int(page_limit_opt):
        paged_items = display_items
        cur_page_info = f"총 {len(saved_list)}개 중 {total_filtered}개 표시"
    else:
        limit = int(page_limit_opt)
        total_pages = max(1, (total_filtered + limit - 1) // limit)
        pg_col1, pg_col2 = st.columns([4, 1])
        with pg_col2:
            cur_page = st.number_input("페이지 이동", min_value=1, max_value=total_pages, value=1, step=1, key=f"{prefix}_pg")
        start_idx = (cur_page - 1) * limit
        paged_items = display_items[start_idx:start_idx + limit]
        cur_page_info = f"총 {len(saved_list)}개 중 {total_filtered}개 검색됨 ({cur_page}/{total_pages} 페이지 표시 중)"

    st.caption(f"📊 {cur_page_info} | 🎯 **현재 선택된 시험지/지문: {len(selected_ids)}개**")

    def on_card_toggle(pid: str):
        # 카드 버튼을 누를 때마다 선택/해제 (누른 순서가 곧 출력 순서)
        if shares_selection:
            lbl = id_to_item[pid]["label"]
            shared = st.session_state.shared_sel_labels
            if lbl in shared:
                shared.remove(lbl)
            else:
                shared.append(lbl)
            return
        sel = st.session_state[sel_key]
        if pid in sel:
            sel.remove(pid)
        else:
            sel.append(pid)

    order_map = {sid: idx + 1 for idx, sid in enumerate(selected_ids)}

    st.caption("👆 카드를 누르면 선택되고, 다시 누르면 해제됩니다. 누른 순서대로 출력됩니다.")
    with st.container(height=400, border=True):
        cols = st.columns(6)
        for i, p in enumerate(paged_items):
            pid = p["id"]
            is_sel = pid in order_map
            sub_info = ""
            if target_type in ["voca", "past_voca"]:
                sub_info = f" · {len(p.get('data', {}).get('voca_list', []))}단어"
            elif target_type == "core":
                sub_info = f" · 요약 {len(p.get('data', {}).get('summaries', []))}선"
            card_label = f"{order_map[pid]}. {p['label']}{sub_info}" if is_sel else f"{p['label']}{sub_info}"
            with cols[i % 6]:
                st.button(
                    md_safe(card_label),
                    key=f"{chk_prefix}card_{pid}",
                    type="primary" if is_sel else "secondary",
                    use_container_width=True,
                    on_click=on_card_toggle,
                    args=(pid,)
                )

    if not selected_ids:
        st.info("👆 위 목록에서 출력하거나 다운로드할 시험지/지문을 선택해주세요.")
    else:
        sel_passages = [id_to_item[sid] for sid in selected_ids if sid in id_to_item]
        preview_limit = 15
        order_preview = " ➔ ".join([f"**[{i+1}]** {md_safe(p['label'])}" for i, p in enumerate(sel_passages[:preview_limit])])
        if len(sel_passages) > preview_limit:
            order_preview += f" ... 외 {len(sel_passages) - preview_limit}개"
        st.markdown(f"🛒 **출력 순서 (총 {len(sel_passages)}개):** {order_preview}")

        render_permanent_json_editor(prefix, target_type, file_path, sel_passages, saved_list)

        if target_type == "victory":
            render_class_ppt_section(sel_passages, school_name, text_model)

        if target_type == "voca":
            sel_passages = render_voca_choice_option(prefix, sel_passages, text_model)

        if target_type in ["voca", "past_voca"]:
            r_col1, r_col2, r_col3, r_col4, r_col5 = st.columns([1.8, 1.8, 2.1, 2.1, 2.2])
        else:
            r_col1, r_col2 = st.columns([2.4, 5.6])
            r_col3, r_col4, r_col5 = None, None, None

        render_clicked = r_col1.button(
            f"👁️ 선택한 {title_name} 단행본 렌더링 ({len(sel_passages)}개)",
            type="primary",
            use_container_width=True,
            key=f"{prefix}_render_btn"
        )

        if target_type in ["voca", "past_voca"] and r_col2 is not None and r_col3 is not None and r_col4 is not None:
            excel_bytes = generate_voca_excel_bytes(sel_passages)
            r_col2.download_button(
                label="📊 선택 단어장 엑셀(.xlsx)",
                data=excel_bytes,
                file_name=f"{title_name.replace(' ', '_')}_통합단어장.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key=f"{prefix}_excel_dl_btn"
            )
            csv_main_bytes = generate_voca_main_csv_bytes(sel_passages)
            r_col3.download_button(
                label="📄 표제어(핵심+CORE) CSV",
                data=csv_main_bytes,
                file_name=f"{title_name.replace(' ', '_')}_표제어_통합.csv",
                mime="text/csv",
                use_container_width=True,
                key=f"{prefix}_csv_main_dl_btn"
            )
            csv_rel_bytes = generate_voca_rel_csv_bytes(sel_passages)
            r_col4.download_button(
                label="📄 유의어·반의어 전용 CSV",
                data=csv_rel_bytes,
                file_name=f"{title_name.replace(' ', '_')}_유의어_반의어.csv",
                mime="text/csv",
                use_container_width=True,
                key=f"{prefix}_csv_rel_dl_btn"
            )

        if render_clicked:
            with st.spinner(f"{title_name} 단행본(표지·사용법·소개·본문·저작권·뒤표지) 교재를 렌더링 중입니다..."):
                front_html, back_html = build_booklet_front_and_back(
                    target_type, h_url, f_url, academy_name, school_name, sel_passages
                )
                front_html = f"<style>{QB_BOOK_CSS}</style>" + front_html
                if target_type == "victory":
                    body_str = "".join([render_victory(p) for p in sel_passages])
                    final_html = wrap_full_html_victory(body_str, css_str, front_html, back_html)
                    file_name = "Anatomy_One-Shot_Booklet.html"
                elif target_type == "anatomy":
                    body_str = "".join([render_anatomy(p) for p in sel_passages])
                    final_html = wrap_full_html_anatomy(body_str, css_str, front_html, back_html)
                    file_name = "Anatomy_Self-Note_Booklet.html"
                elif target_type == "core":
                    body_str = "".join([render_core(p) for p in sel_passages])
                    final_html = wrap_full_html_core(body_str, css_str, front_html, back_html)
                    file_name = "Anatomy_Core_Booklet.html"
                elif target_type == "past_voca":
                    # 시험지 단위로 연속 출력되고 다음 시험지로 바뀔 때만 페이지 넘김
                    body_str = render_past_voca_grouped_by_exam(sel_passages)
                    final_html = wrap_full_html_voca(body_str, css_str, front_html, back_html)
                    file_name = f"{title_name.replace(' ', '_')}_Booklet.html"
                else:
                    body_str = "".join([render_voca(p) for p in sel_passages])
                    final_html = wrap_full_html_voca(body_str, css_str, front_html, back_html)
                    file_name = f"{title_name.replace(' ', '_')}_Booklet.html"

                dl_target_col = r_col5 if (target_type in ["voca", "past_voca"] and r_col5 is not None) else r_col2
                with dl_target_col:
                    st.download_button(
                        f"📥 단행본 HTML 다운로드 ({title_name})",
                        final_html,
                        file_name,
                        "text/html",
                        use_container_width=True,
                        key=f"{prefix}_dl_btn"
                    )
                components.html(final_html, height=850, scrolling=True)


# -----------------------------
# 5-A. 백그라운드 교재 생성 작업 (작업 목록을 DB에 저장하고 서버 스레드가 하나씩 처리)
#   - 브라우저를 닫거나 연결이 끊겨도 서버에서 계속 진행
#   - 오류가 나면 기다렸다가 다시 시도, 그래도 안 되면 '실패'로 표시하고 다음 지문으로 넘어감
#   - '생성 중단'은 지금 처리 중인 지문을 끝낸 뒤 멈추고, '다시 시작'은 남은 지문부터 이어서 진행
#   - 서버가 재시작돼도 작업 목록이 남아 있어 앱을 다시 열면 자동으로 이어감
# -----------------------------
GEN_JOB_FILE = os.path.join(BASE_DIR, "gen_job.json")
GEN_TYPES = [
    ("victory", VICTORY_FILE, "Anatomy One-Shot"),
    ("anatomy", ANATOMY_FILE, "Anatomy Self-Note"),
    ("core", CORE_FILE, "Anatomy Core"),
    ("voca", VOCA_FILE, "Anatomy Voca"),
]
GEN_TYPE_NAMES = {t: name for t, _, name in GEN_TYPES}
GEN_RETRY_WAITS = [60, 180, 600]  # 분석 실패 시 1분 → 3분 → 10분 기다렸다가 다시 시도
GEN_WORKER_LEASE_SEC = 1200  # 작업 담당 서버가 이 시간 동안 소식이 없으면 다른 서버가 넘겨받음

# 교재·원문처럼 큰 문서는 생성 작업(서버 스레드)도 화면과 같은 서버 기억(_lib_mem_cache)을 씀:
#   저장 시각만 먼저 읽어 그대로면 내용을 다시 내려받지 않음.
#   (예전에는 지문 하나 만들 때마다 '이미 있나' 확인·저장 직전에 교재 문서 전체를 여러 번 받아서 DB 전송량이 컸음)
#   스레드에서 st.cache_resource를 부르지 않도록, 화면이 실행될 때 같은 dict를 꺼내 둠.
_BIG_DOC_KEYS = {filepath_to_doc_key(f) for f in (VICTORY_FILE, ANATOMY_FILE, CORE_FILE, VOCA_FILE, PAST_VOCA_FILE, MOCK_FILE)} | {"syn_review", "gichul_full"}  # AI 검수 기록(약 1MB)도 바뀌었을 때만 다시 받음
_SHARED_DOC_CACHE = _lib_mem_cache()

def _store_get(sb, filepath, default=None):
    """스레드에서도 쓸 수 있는 DB 읽기 (st.* 를 쓰지 않음)"""
    doc_key = filepath_to_doc_key(filepath)
    if sb is not None:
        if doc_key in _BIG_DOC_KEYS:
            hit = _SHARED_DOC_CACHE.get(doc_key)
            if hit and hit[0]:
                res = sb.table("app_storage").select("updated_at").eq("doc_key", doc_key).execute()
                if res.data and _ver_key(res.data[0].get("updated_at")) == _ver_key(hit[0]):
                    return json.loads(hit[1])
            res = sb.table("app_storage").select("content,updated_at").eq("doc_key", doc_key).execute()
            if res.data:
                content = res.data[0]["content"]
                if content is not None:
                    _SHARED_DOC_CACHE[doc_key] = (res.data[0].get("updated_at"), json.dumps(content, ensure_ascii=False))
                return content
            return default
        res = sb.table("app_storage").select("content").eq("doc_key", doc_key).execute()
        if res.data:
            return res.data[0]["content"]
        return default
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return default

def _store_put(sb, filepath, content):
    doc_key = filepath_to_doc_key(filepath)
    if isinstance(content, list):
        content.sort(key=natural_sort_key)
    if sb is not None:
        ver = _doc_upsert(sb, doc_key, content)
        if doc_key in _BIG_DOC_KEYS:
            # 방금 저장한 내용을 서버 기억에도 → 화면이 1분마다 새로 불러올 때도 다시 내려받지 않음
            _SHARED_DOC_CACHE[doc_key] = (ver, json.dumps(content, ensure_ascii=False))
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(content, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def job_item_model(item: Dict, opts: Dict) -> str:
    """작업 1건에 쓸 모델: 작업 전체 모델(opts) 기준, '가벼운 모델' 표시가 있으면 같은 제공사의 가벼운 모델
    (예전 작업은 건마다 모델이 고정돼 있었음 → 작업 모델과 다르게 적힌 건만 가벼운 모델로 봄)"""
    base = opts.get("model", "")
    light = item.get("light", bool(item.get("model")) and item.get("model") != base and item.get("difficulty") in ("하", "중"))
    return light_model_for(base) if light else base

def _job_log(job: Dict, msg: str):
    job.setdefault("log", []).append(f"{time.strftime('%m-%d %H:%M:%S')} {msg}")
    job["log"] = job["log"][-60:]

def _job_update_retry(sb, fn, waits=(5, 15, 30)):
    """작업 문서 쓰기가 잠깐 실패해도 쓰기만 몇 번 더 시도 (예전엔 바로 1분 쉬고 처음부터 다시 해서 같은 지문을 또 만들거나 반값 묶음을 또 냈음)"""
    for i in range(len(waits) + 1):
        try:
            return _job_update(sb, fn)
        except Exception:
            if i == len(waits):
                raise
            time.sleep(waits[i])

def _job_update(sb, fn):
    """작업 문서를 새로 읽어 fn으로 고친 뒤 저장 (화면에서 누른 중단·추가가 덮어써지지 않게 매번 새로 읽음)"""
    job = _store_get(sb, GEN_JOB_FILE) or {}
    fn(job)
    _job_slim(job)
    job["updated_at"] = time.time()
    _store_put(sb, GEN_JOB_FILE, job)
    return job

# 작업 문서를 가볍게: 같은 지문 본문을 건마다 복사하지 않고 job["texts"]에 한 번만 두고 건에는 이름표(text_key)만 둠
#   (변형문제는 지문 1개 × 유형 15 × 난이도 2 = 30건이 같은 본문을 쓰는데, 작업 문서는 처리 중 계속 읽고 쓰므로 DB 전송량이 컸음)
#   AI에 보내는 지문 글자는 예전과 똑같음 → 프롬프트 캐시 영향 없음 · 예전 방식(건에 text가 있는 작업)도 그대로 처리
def _job_text_key(job: Dict, text: str) -> str:
    k = hashlib.md5(str(text).encode("utf-8")).hexdigest()[:12]
    job.setdefault("texts", {})[k] = str(text)
    return k

def job_item_full(job: Dict, it: Dict) -> Dict:
    """처리할 때 쓸 작업 1건 (지문 본문·같은 단원 원문을 채운 사본)"""
    out = dict(it)
    texts = job.get("texts") or {}
    if "text" not in out:
        out["text"] = texts.get(it.get("text_key", ""), "")
    if "unit_other" not in out and it.get("unit_key"):
        out["unit_other"] = texts.get(it["unit_key"], "")
    return out

def _job_slim(job: Dict):
    """끝난 건은 본문을 지우고, 남은 건이 아무도 안 쓰는 본문은 작업 문서에서 뺌 (실패한 건은 다시 시도할 수 있게 남김)"""
    items = job.get("items") or []
    for it in items:
        if it.get("state") == "done":
            it.pop("text", None)
            it.pop("unit_other", None)
            it.pop("plans", None)
    if job.get("texts"):
        used = {k for it in items if it.get("state") != "done" for k in (it.get("text_key"), it.get("unit_key")) if k}
        job["texts"] = {k: v for k, v in job["texts"].items() if k in used}

def _generate_one_type(sb, gclient, item: Dict, t: str, opts: Dict, dna_rules: str, logs: List[str]) -> (bool, str):
    """지문 하나의 교재 1종을 생성해 저장. 이미 있으면 건너뜀. (동시 실행되므로 각자 받은 DB 연결만 사용)"""
    fpath, tname = next((fp, nm) for tt, fp, nm in GEN_TYPES if tt == t)
    lbl, text = item["label"], item["text"]
    note = lambda msg, level="info": logs.append(msg)
    if any(p.get("label") == lbl for p in (_store_get(sb, fpath, []) or [])):
        return True, ""
    core_ctx = None
    if t == "voca":
        core_ctx = next((p.get("data") for p in (_store_get(sb, CORE_FILE, []) or []) if p.get("label") == lbl), None)

    data = None
    for round_idx in range(len(GEN_RETRY_WAITS) + 1):
        time.sleep(opts.get("api_delay", 3))
        data = analyze_passage(lbl, text, opts["model"], t, core_context=core_ctx,
                               dna_rules=dna_rules, notify=note, gen_client=gclient, essay_rules=opts.get("essay_rules", ""))
        if data:
            break
        if round_idx < len(GEN_RETRY_WAITS):
            wait = GEN_RETRY_WAITS[round_idx]
            logs.append(f"⏳ {tname} 생성 실패 → {wait // 60}분 뒤 다시 시도 ({round_idx + 1}/{len(GEN_RETRY_WAITS)})")
            time.sleep(wait)
    if not data:
        return False, f"{tname} 생성 실패"

    if t == "voca" and core_ctx:
        merge_core_vocab_into_voca_item(core_ctx, data)
    record = {"id": str(uuid.uuid4()), "label": lbl, "data": data, "body_text": text,
              "academy_name": opts.get("academy_name", "포피엠 입시 영어")}
    if t in ("victory", "anatomy"):
        # 작업을 넣은 뒤에 원문 라이브러리에서 이미지를 만들었을 수 있으므로, 저장 직전에 최신 이미지를 다시 확인
        img_path = item.get("img_path", "")
        if not img_path:
            src = next((m for m in (_store_get(sb, MOCK_FILE, []) or []) if m.get("label") == lbl), None)
            img_path = (src or {}).get("image_path", "")
        record.update({"image_data_url": "", "image_path": img_path})
    if t != "victory":
        record["school_name"] = item.get("school") or opts.get("school_name", "")
    lib = _store_get(sb, fpath, []) or []
    if not any(p.get("label") == lbl for p in lib):
        lib.append(record)
        _store_put(sb, fpath, lib)
    if t == "core":
        # 이미 만들어진 Voca 단어장이 있으면 Core 패러프레이징 어휘를 병합
        voca = _store_get(sb, VOCA_FILE, []) or []
        if sum(merge_core_vocab_into_voca_item(data, vp.get("data", {})) for vp in voca if vp.get("label") == lbl):
            _store_put(sb, VOCA_FILE, voca)
    logs.append(f"✅ {tname} 완료: {lbl}")
    return True, ""

def _generate_item(sb_factory, gclient, item: Dict, opts: Dict, dna_rules: str, logs: List[str], heartbeat=lambda: None) -> (bool, str):
    """지문 하나에 대해 요청된 교재를 생성.
    한판승·Self-Note·Core는 서로 독립이라 동시에 만들고, Voca는 Core 어휘를 넣어야 하므로 Core가 끝난 뒤 만듦.
    sb_factory: 동시 작업마다 DB 연결을 따로 열기 위한 함수 (연결 하나를 여러 작업이 같이 쓰면 끊길 수 있음)
    heartbeat: 기다리는 동안 몇 분마다 '아직 처리 중'임을 기록해 다른 서버가 넘겨받지 않게 함"""
    from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
    wanted = item.get("types", [])

    def run_parallel(type_list):
        errors = []
        if not type_list:
            return errors
        with ThreadPoolExecutor(max_workers=len(type_list)) as pool:
            futures = {pool.submit(_generate_one_type, sb_factory(), gclient, item, t, opts, dna_rules, logs): t for t in type_list}
            pending = set(futures)
            while pending:
                heartbeat()
                done, pending = wait(pending, timeout=240, return_when=FIRST_COMPLETED)
                for fut in done:
                    try:
                        ok, err = fut.result()
                    except Exception as e:
                        ok, err = False, f"{GEN_TYPE_NAMES[futures[fut]]} 처리 중 오류: {str(e)[:150]}"
                    if not ok:
                        errors.append(err)
        return errors

    # 엑셀 탭에서 'AI 이미지 생성'을 켠 지문: 한판승·Self-Note에 넣을 이미지를 먼저 만듦 (실패해도 교재는 이미지 없이 계속 생성)
    sb_check = sb_factory() if item.get("gen_image") and not item.get("img_path") else None
    needs_picture = [t for t, fp, _ in GEN_TYPES if t in ("victory", "anatomy") and t in wanted
                     and not any(p.get("label") == item["label"] for p in (_store_get(sb_check, fp, []) or []))] if item.get("gen_image") and not item.get("img_path") else []
    if needs_picture:
        heartbeat()
        raw, img_err = generate_passage_image(item["text"], [], item.get("image_model") or IMAGE_MODELS[0], item.get("image_style", ""))
        img_path = save_booklet_image(raw, f"xl_{hashlib.md5(item['label'].encode()).hexdigest()[:10]}", sb_client=sb_factory(), quiet=True) if raw else ""
        if img_path:
            item["img_path"] = img_path
            logs.append(f"🎨 이미지 생성 완료: {item['label']}")
        else:
            logs.append(f"⚠️ 이미지 생성 실패({(img_err or '저장 실패')[:80]}) → 이미지 없이 교재 생성: {item['label']}")

    errors = run_parallel([t for t in ("victory", "anatomy", "core") if t in wanted])
    core_failed = "core" in wanted and any(GEN_TYPE_NAMES["core"] in e for e in errors)
    if "voca" in wanted and not core_failed:
        errors += run_parallel(["voca"])
    elif "voca" in wanted:
        errors.append("Anatomy Voca: Core 실패로 다음 재시도 때 함께 생성")
    return (False, " / ".join(errors)) if errors else (True, "")

def gen_worker_main(api_key: str, sb_url: str, sb_key: str):
    """서버 스레드: 작업 목록에서 대기 중인 지문을 하나씩 처리"""
    sb = create_client(sb_url, sb_key) if (sb_url and sb_key and create_client is not None and IS_CLOUD_DB) else None
    gclient = genai.Client(api_key=api_key) if api_key else None
    dna = _store_get(sb, DNA_FILE, {}) or {}
    dna_rules = dna.get("rules_text") or DEFAULT_EXAM_DNA["rules_text"]
    essay_rules = dna.get("essay_rules_text", "")
    yt_points = qb_yt_points(dna)
    # PC 앱과 배포된 앱이 동시에 열려 있어도 한 곳에서만 처리하도록 '작업 담당자'를 기록 (20분 동안 소식 없으면 넘겨받음)
    my_id = str(uuid.uuid4())
    def claim(j):
        j["worker"] = {"id": my_id, "ts": time.time()}
    # DB가 길게 끊겨 작업 문서에 못 쓴 변경(지문 완료·반값 제출 기록)을 기억해 두고 다음 바퀴에 먼저 씀
    #   → 이미 만든 지문을 또 만들거나, 이미 낸 반값 묶음을 또 내서 비용이 두 번 나가지 않게
    unsaved = []
    def write(fn):
        try:
            return _job_update_retry(sb, fn)
        except Exception:
            unsaved.append(fn)
            raise
    last_poll = 0.0
    last_err = ""
    while True:
        try:
            if unsaved:
                _job_update(sb, unsaved[0])
                unsaved.pop(0)
                continue
            job = _store_get(sb, GEN_JOB_FILE) or {}
            lease = job.get("worker") or {}
            if lease.get("id") not in (None, my_id) and time.time() - float(lease.get("ts", 0)) < GEN_WORKER_LEASE_SEC:
                return  # 다른 서버가 처리 중
            if job.get("status") == "stopping":
                _job_update(sb, lambda j: (j.update(status="stopped", current="", worker=None), _job_log(j, "⏸️ 요청대로 멈췄습니다. '다시 시작'을 누르면 남은 지문부터 이어서 진행합니다.")))
                return
            if job.get("status") != "running":
                if lease.get("id") == my_id:
                    _job_update(sb, lambda j: j.update(worker=None))
                return
            all_items = job.get("items", [])
            sb_factory = (lambda: create_client(sb_url, sb_key)) if sb is not None else (lambda: None)
            # 반값 모드: 새로 들어온 건은 모아서 한 번에 제출 → 'waiting'. 결과는 몇 분마다 확인해서 들어온 것부터 검증·저장
            fresh = [it for it in all_items if it.get("state") == "pending" and it.get("batch") and it.get("kind") == "qbank"]
            if fresh:
                qb_batch_submit(sb, [job_item_full(job, it) for it in fresh[:QB_BATCH_MAX]], job.get("opts", {}), dna_rules, essay_rules, claim, yt_points,
                                write=write)
                continue
            pending = [it for it in all_items if it.get("state") == "pending"]
            waiting = [it for it in all_items if it.get("state") == "waiting"]
            if waiting and time.time() - last_poll > (QB_BATCH_POLL_SEC if not pending else QB_BATCH_POLL_BUSY_SEC):
                last_poll = time.time()
                qb_batch_poll(sb, gclient, job, claim, write=write)
                flush_ai_usage()
                continue
            if not pending and waiting:
                _job_update(sb, claim)  # 다른 서버가 넘겨받지 않게 담당 표시만 새로 고침
                time.sleep(max(5, QB_BATCH_POLL_SEC - (time.time() - last_poll)))
                continue
            if not pending:
                _job_update(sb, lambda j: (j.update(status="done", current="", worker=None), _job_log(j, "🎉 모든 작업을 마쳤습니다.")))
                return
            item = job_item_full(job, pending[0])
            item_model = job_item_model(item, job.get("opts", {}))
            _job_update(sb, lambda j: (j.update(current=item["label"], current_model=item_model), claim(j)))
            logs = []
            try:
                sb_factory = (lambda: create_client(sb_url, sb_key)) if sb is not None else (lambda: None)
                if item.get("kind") == "qbank":
                    ok, err = _generate_qbank_item(sb_factory, gclient, item, job.get("opts", {}), dna_rules, essay_rules, logs, yt_points)
                elif item.get("kind") == "trans":
                    ok, err = _trans_item(sb_factory, item, job.get("opts", {}), logs)
                else:
                    ok, err = _generate_item(sb_factory, gclient, item, {**job.get("opts", {}), "essay_rules": essay_rules}, dna_rules, logs,
                                             heartbeat=lambda: _job_update(sb, claim))
            except Exception as e:
                ok, err = False, f"처리 중 오류: {str(e)[:200]}"

            def finish(j):
                for it in j.get("items", []):
                    if it.get("id") == item["id"]:
                        it["state"] = "done" if ok else "failed"
                        it["error"] = err
                        # 만든 이미지를 기억해 두어 '실패한 지문 다시 시도' 때 이미지를 또 만들지 않게 함
                        if item.get("img_path"):
                            it["img_path"] = item["img_path"]
                for m in logs[-8:]:
                    _job_log(j, f"{m} · {ai_model_label(item_model)}" if m.startswith("✅") else m)
                if not ok:
                    _job_log(j, f"❌ {item['label']}: {err} (나중에 '실패한 지문 다시 시도'로 재시도할 수 있어요)")
                j["current"] = ""
                j["current_model"] = ""
                claim(j)
            write(finish)
            flush_ai_usage()
            last_err = ""
        except Exception as e:
            # DB 연결이 잠깐 끊기는 등 예상 못 한 오류: 1분 쉬었다가 같은 지점부터 다시
            #   예전엔 오류가 어디에도 안 남아 '지금 처리 중'에서 멈춘 것처럼만 보였음 → 서버 기록과 작업 기록에 남김 (같은 오류는 한 번만)
            msg = f"{type(e).__name__}: {str(e)[:160]}"
            print(f"[gen-worker] {msg}", flush=True)
            if msg != last_err:
                try:
                    _job_update(sb, lambda j, m=msg: _job_log(j, f"⚠️ 작업 처리 중 예상 못 한 오류 · 1분 뒤 같은 지점부터 다시 시도: {m}"))
                    last_err = msg  # 기록에 남긴 뒤에만 (못 남겼으면 다음에 다시 시도)
                except Exception:
                    pass
            time.sleep(60)

@st.cache_resource
def get_gen_runner() -> Dict:
    return {"thread": None, "lock": threading.Lock()}

def gen_thread_alive() -> bool:
    t = get_gen_runner()["thread"]
    return bool(t and t.is_alive())

def start_gen_thread() -> bool:
    runner = get_gen_runner()
    with runner["lock"]:
        if runner["thread"] and runner["thread"].is_alive():
            return False
        t = threading.Thread(target=gen_worker_main, args=(API_KEY, SUPABASE_URL, SUPABASE_KEY), daemon=True, name="gen-worker")
        t.start()
        runner["thread"] = t
        return True

def load_gen_job() -> Dict:
    """작업 상태 읽기: 저장 시각만 먼저 보고 바뀌었을 때만 내용을 내려받음
    (진행 중에는 5초마다·화면을 누를 때마다 읽어서, 예전에는 작업이 클수록 DB 전송량이 컸음)"""
    try:
        if not IS_CLOUD_DB:
            return _store_get(None, GEN_JOB_FILE) or {}
        doc_key = filepath_to_doc_key(GEN_JOB_FILE)
        cached = st.session_state.get("_gen_job_cache")
        if cached:
            ver = _doc_version(doc_key)
            if ver and _ver_key(cached[0]) == _ver_key(ver):
                return cached[1]
        res = supabase.table("app_storage").select("content,updated_at").eq("doc_key", doc_key).execute()
        job = (res.data[0].get("content") if res.data else None) or {}
        st.session_state["_gen_job_cache"] = ((res.data[0].get("updated_at") if res.data else None), job)
        return job
    except Exception:
        return {}

def enqueue_generation(rows: List[Dict], types_list: List[str], text_model: str, api_delay: int, academy_name: str, school_name: str) -> int:
    """rows: [{"label", "text", "school", "img_path"}] → 작업 목록에 추가하고 서버 작업을 시작. 추가된 지문 수를 반환."""
    sb = supabase if IS_CLOUD_DB else None
    added = {"n": 0}

    def add(job):
        if job.get("status") not in ("running", "stopping"):
            # 끝났거나 멈춘 작업에 새로 추가하면 남은 지문과 함께 다시 시작
            job["status"] = "running"
            job.setdefault("items", [])
            job["items"] = [it for it in job["items"] if it.get("state") in ("pending", "failed", "waiting")]
            job["created_at"] = time.time()
            job["worker"] = None
        job["opts"] = {"model": text_model, "api_delay": api_delay, "academy_name": academy_name, "school_name": school_name}
        queued = {(it["label"], tuple(it.get("types", []))) for it in job["items"] if it.get("state") == "pending"}
        for r in rows:
            key = (r["label"], tuple(types_list))
            if key in queued:
                continue
            job["items"].append({"id": str(uuid.uuid4()), "label": r["label"], "text_key": _job_text_key(job, r["text"]),
                                 "school": r.get("school", ""), "img_path": r.get("img_path", ""),
                                 "gen_image": bool(r.get("gen_image")), "image_style": r.get("image_style", ""),
                                 "image_model": r.get("image_model", ""),
                                 "types": list(types_list), "state": "pending", "error": ""})
            added["n"] += 1
        _job_log(job, f"➕ {added['n']}개 지문 추가 ({', '.join(GEN_TYPE_NAMES[t] for t in types_list)})")
    _job_update(sb, add)
    start_gen_thread()
    return added["n"]

def reload_libraries_from_db():
    """교재 목록 다시 불러오기 (DB에서 바뀐 교재만 실제로 내려받고, 그대로인 교재는 서버에 기억해 둔 것을 씀)"""
    st.session_state.saved_v = load_from_file(VICTORY_FILE, keep=st.session_state.get("saved_v"))
    st.session_state.saved_a = load_from_file(ANATOMY_FILE, keep=st.session_state.get("saved_a"))
    st.session_state.saved_c = load_from_file(CORE_FILE, keep=st.session_state.get("saved_c"))
    st.session_state.saved_w = load_from_file(VOCA_FILE, keep=st.session_state.get("saved_w"))
    st.session_state.saved_mock = load_from_file(MOCK_FILE, keep=st.session_state.get("saved_mock"))

def _gen_job_counts(job: Dict) -> Dict:
    items = job.get("items", [])
    return {s: sum(1 for it in items if it.get("state") == s) for s in ("pending", "done", "failed", "waiting")} | {"total": len(items)}

def _render_gen_job_body(job: Dict):
    c = _gen_job_counts(job)
    status = job.get("status", "")
    status_text = {
        "running": "🟢 생성 중 · 창을 닫아도 서버에서 계속 진행됩니다",
        "stopping": "🟡 멈추는 중 · 지금 지문을 끝낸 뒤 멈춥니다",
        "stopped": "⏸️ 멈춤 · '다시 시작'을 누르면 남은 지문부터 이어서 진행합니다",
        "done": "✅ 완료",
    }.get(status, status)
    finished = c["done"] + c["failed"]
    if status == "running" and c["waiting"] and not c["pending"]:
        status_text = "🟢 반값 모드 결과 기다리는 중 · 보통 몇 분~몇 시간, 길면 24시간 · 서버가 2분마다 확인해서 들어온 것부터 저장해요"
    st.markdown(f"**⚙️ 교재 생성 작업** — {status_text}")
    st.progress(finished / max(1, c["total"]), text=f"{finished}/{c['total']}건 처리 · 실패 {c['failed']}개 · 남음 {c['pending']}개"
                + (f" · 💸 반값 결과 대기 {c['waiting']}개" if c["waiting"] else ""))
    if job.get("current"):
        st.caption(f"⏳ 지금 처리 중: {job['current']}" + (f" · 🤖 {ai_model_label(job['current_model'])}" if job.get("current_model") else ""))
    elif job.get("opts", {}).get("model") and status in ("running", "stopping"):
        st.caption(f"🤖 {ai_model_label(job['opts']['model'])}")
    cur_model = globals().get("text_model", "")
    job_model = job.get("opts", {}).get("model", "")

    def use_model(j, spec):
        j.setdefault("opts", {})["model"] = spec
        for it in j.get("items", []):
            if it.get("state") == "pending" and "light" not in it:
                it["light"] = bool(it.get("model")) and it.get("model") != job_model and it.get("difficulty") in ("하", "중")
            it.pop("model", None) if it.get("state") == "pending" else None
        _job_log(j, f"🤖 남은 작업 모델 → {ai_model_label(spec)}")

    if c["pending"] and cur_model and job_model and cur_model != job_model:
        w1, w2 = st.columns([3, 1.2], vertical_alignment="center")
        w1.warning(f"남은 작업은 **{ai_model_label(job_model)}** 로 진행돼요. 사이드바에서 고른 모델은 **{ai_model_label(cur_model)}** 이에요.")
        if w2.button("남은 작업을 사이드바 모델로", use_container_width=True, key="gen_use_model"):
            _job_update(supabase if IS_CLOUD_DB else None, lambda j: use_model(j, cur_model))
            st.rerun()

    # 서버가 다시 켜지는 등으로 담당 서버가 사라지면 20분(GEN_WORKER_LEASE_SEC) 뒤에 자동으로 이어받음 → 8분 넘게 소식이 없으면 바로 이어받기 버튼
    lease_age = time.time() - float((job.get("worker") or {}).get("ts", time.time()))
    #   (이 서버의 스레드는 화면이 그려질 때마다 새로 떴다가 다른 서버 표시를 보고 바로 끝나므로 '살아 있음'으로는 판단하지 않음)
    if status == "running" and c["pending"] and lease_age > 480:
        k1, k2 = st.columns([3, 1.2], vertical_alignment="center")
        k1.warning(f"작업을 처리하던 서버 소식이 {int(lease_age // 60)}분째 없어요 (서버가 다시 켜졌을 수 있어요). "
                   f"{max(0, int((GEN_WORKER_LEASE_SEC - lease_age) // 60))}분 뒤에는 자동으로 이어서 하고, 지금 바로 이어서 하려면 오른쪽을 누르세요. 끝난 건은 그대로 저장돼 있어요.")
        if k2.button("▶️ 지금 이어서 진행", type="primary", use_container_width=True, key="gen_takeover"):
            _job_update(supabase if IS_CLOUD_DB else None, lambda j: (j.update(worker=None), _job_log(j, "▶️ 멈춘 작업을 이어서 진행")))
            start_gen_thread()
            st.rerun()
    b1, b2, b3, b4 = st.columns(4)
    if status == "running":
        if b1.button("⏸️ 생성 중단 (지금 지문 끝내고 멈춤)", use_container_width=True, key="gen_stop"):
            _job_update(supabase if IS_CLOUD_DB else None, lambda j: (j.update(status="stopping"), _job_log(j, "⏸️ 중단 요청 — 지금 지문을 끝낸 뒤 멈춥니다.")))
            st.rerun()
    elif status in ("stopped", "stopping") and (c["pending"] or c["waiting"]):
        if b1.button("▶️ 다시 시작", type="primary", use_container_width=True, key="gen_resume"):
            _job_update(supabase if IS_CLOUD_DB else None,
                        lambda j: (use_model(j, cur_model) if cur_model and cur_model != job_model else None, j.update(status="running"), _job_log(j, "▶️ 다시 시작")))
            start_gen_thread()
            st.rerun()
    if c["failed"] and status != "stopping":
        if b2.button(f"🔁 실패한 {c['failed']}개 다시 시도", use_container_width=True, key="gen_retry_failed"):
            def retry(j):
                for it in j.get("items", []):
                    if it.get("state") == "failed":
                        it["state"], it["error"] = "pending", ""
                j["status"] = "running"
                _job_log(j, "🔁 실패한 지문 다시 시도")
            _job_update(supabase if IS_CLOUD_DB else None, retry)
            start_gen_thread()
            st.rerun()
    if b3.button("🔄 교재 목록 새로고침", use_container_width=True, key="gen_reload_libs"):
        reload_libraries_from_db()
        st.rerun()
    if status in ("done", "stopped") and b4.button("🧹 작업 기록 지우기", use_container_width=True, key="gen_clear", disabled=bool(c["waiting"]),
                                                   help="반값 결과를 기다리는 작업이 있으면 지울 수 없어요 (지우면 이미 낸 비용의 결과를 못 받아요)" if c["waiting"] else None):
        _store_put(supabase if IS_CLOUD_DB else None, GEN_JOB_FILE, {})
        st.rerun()

    with st.expander("📜 작업 기록 / 실패한 지문", expanded=False):
        failed = [it for it in job.get("items", []) if it.get("state") == "failed"]
        for it in failed:
            st.caption(f"❌ {it['label']} — {it.get('error', '')}")
        st.code("\n".join(reversed(job.get("log", [])[-30:])) or "기록 없음", language=None)

def _gen_job_live_body(slow: bool):
    job = load_gen_job()
    c = _gen_job_counts(job)
    # 지문이 완료될 때마다 화면의 교재 목록을 DB에서 다시 불러와, 화면이 오래된 목록으로 덮어쓰지 않게 함
    # (변형문제는 교재 목록과 상관없으므로 세지 않음)
    done_books = sum(1 for it in job.get("items", []) if it.get("state") == "done" and it.get("kind") not in ("qbank", "trans"))
    seen = st.session_state.get("gen_seen_done")
    if seen is not None and seen != done_books:
        # 지문이 빨리 끝날 때 몇 초마다 교재 전체를 다시 받지 않게 1분에 한 번까지만 (작업이 끝나면 바로)
        #  — 저장할 때 최신 목록과 합쳐 저장하므로, 잠깐 오래된 목록이 보여도 지문이 사라지지 않음
        if job.get("status") not in ("running", "stopping") or time.time() - st.session_state.get("_gen_lib_reload_ts", 0) > 60:
            reload_libraries_from_db()
            st.session_state["_gen_lib_reload_ts"] = time.time()
            st.session_state.gen_seen_done = done_books
    else:
        st.session_state.gen_seen_done = done_books
    if job.get("status") in ("running", "stopping") and not gen_thread_alive():
        start_gen_thread()  # 서버가 재시작된 경우 자동으로 이어서 진행
    _render_gen_job_body(job)
    if job.get("status") not in ("running", "stopping") or slow != bool(c["waiting"] and not c["pending"]):
        st.rerun()  # 끝나면 자동 새로고침을 멈추고 일반 화면으로 (반값 결과만 기다릴 때는 느린 새로고침으로 바꿈)

@st.fragment(run_every=5)
def render_gen_job_live():
    """작업이 진행 중일 때만 5초마다 상태를 새로 읽어 보여줌"""
    _gen_job_live_body(False)

@st.fragment(run_every=60)
def render_gen_job_slow():
    """반값 결과만 기다릴 때는 1분마다만 읽음 (몇 시간 걸릴 수 있어서 DB 읽기를 줄임)"""
    _gen_job_live_body(True)

def render_gen_job_panel():
    flash = st.session_state.pop("gen_flash", None)
    if flash:
        st.success(flash)
    job = load_gen_job()
    if not job.get("items"):
        return
    with st.container(border=True):
        c = _gen_job_counts(job)
        if job.get("status") in ("running", "stopping"):
            (render_gen_job_slow if c["waiting"] and not c["pending"] else render_gen_job_live)()
        else:
            _render_gen_job_body(job)


# -----------------------------
# 5-Q. 변형문제 은행 & 시험지 만들기
#   - 원문 라이브러리(모의고사 번호·교과서 단락)마다 유형·난이도별 변형문제를 계속 쌓음
#   - 문제는 Supabase 'question_bank' 표에 한 문제씩 따로 저장 (앱을 열 때 불러오지 않아 다른 탭 속도에 영향 없음)
#   - 생성은 교재 생성과 같은 백그라운드 작업 목록에서 하나씩 처리 (창을 닫아도 계속)
#   - 시험지: 고른 번호에서 골고루 뽑아 원하는 개수·유형·난이도로 구성, 실제 학교 기출이면 출처 표시
# -----------------------------
QB_TABLE = "question_bank"
QB_FILE = os.path.join(BASE_DIR, "question_bank.json")   # Supabase가 없을 때만 쓰는 로컬 저장
GICHUL_FULL_FILE = os.path.join(BASE_DIR, "gichul_full.json")  # 실제 기출 시험지 전 문항 (발문·지문·선지·정답) — 2026-11-10부터, 학교·학년별 분석용
GICHUL_FILE = os.path.join(BASE_DIR, "gichul_map.json")   # 실제 기출 시험지 문항 → 원문 라이브러리 지문 연결
QB_LEVELS = ["최상", "상", "중", "하"]
QB_SCHOOLS = ["운양고", "제일고"]
# 변형문제 비용 추정 (원, 1달러 1,400원): 작업 1건 = 고정(공통 규칙 약 6,800토큰은 캐시 + 지문·조건 약 1,400토큰) + 문제 수 × (출력 + 정답 검증)
#   2026-11-04 GPT-6.1 Terra 165건(건당 2문제) 실제 $2.71과 맞춘 값 · 문제당 출력 토큰은 모델마다 실측 (Claude·Gemini는 생각 토큰 포함)
QB_OUT_TOKENS = {"opus": 2300, "fable": 2300, "sonnet": 2200, "haiku": 900, "gpt": 750, "pro": 2500, "flash-lite": 1200, "flash": 1900}

def qb_job_cost_krw(model_spec: str, n_q: int, verify: bool = True, batch: bool = False) -> float:
    """batch: 반값 모드 (생성만 절반, 정답 검증은 결과가 들어온 뒤 바로 하므로 그대로)"""
    name = split_model_spec(model_spec)[1].lower()
    p_in, p_cache, p_out = ai_price(model_spec)
    out = next((v for k, v in QB_OUT_TOKENS.items() if k in name), 1200)
    gen = 6800 * p_cache + 1400 * p_in + n_q * out * p_out
    ver = (400 * p_in + n_q * (600 * p_in + 50 * p_out)) if verify else 0
    return (gen * (0.5 if batch else 1) + ver) / 1_000_000 * 1400

# 교재 생성 비용 추정 (원, 1달러 1,400원) — 지문 1개·교재 1종의 대략 토큰 (2026-11 저장된 교재 220개 글자 수 ÷ 2로 잼)
#   (공통 앞부분: 규칙·출제 DNA, 두 번째 지문부터 캐시로 읽음 / 지문·나머지 입력 / 출력) · 생각 토큰은 변형문제 실측 비율(QB_OUT_TOKENS)로 더함
BOOK_TOKENS = {"victory": (5500, 500, 3600), "anatomy": (0, 700, 2400), "core": (12800, 500, 3900), "voca": (12800, 900, 3300)}

def book_cost_krw(model_spec: str, counts: Dict) -> float:
    """counts: {교재 종류: 새로 만들 지문 수}"""
    name = split_model_spec(model_spec)[1].lower()
    p_in, p_cache, p_out = ai_price(model_spec)
    think = next((v for k, v in QB_OUT_TOKENS.items() if k in name), 1200) / QB_OUT_TOKENS["gpt"]
    usd = 0.0
    for t, n in counts.items():
        c, i, o = BOOK_TOKENS.get(t, (0, 1000, 3000))
        usd += n * (c * p_cache + i * p_in + o * think * p_out) / 1_000_000
    return usd * 1400

def book_new_counts(labels: List[str], types_list: List[str]) -> Dict:
    """고른 지문 중 교재 종류별로 아직 없는 지문 수 (이미 있는 교재는 생성 작업이 건너뜀 → 비용 없음)"""
    have = {"victory": st.session_state.saved_v, "anatomy": st.session_state.saved_a,
            "core": st.session_state.saved_c, "voca": st.session_state.saved_w}
    out = {}
    for t in types_list:
        names = {p.get("label") for p in have.get(t, [])}
        out[t] = sum(1 for l in dict.fromkeys(labels) if l not in names)
    return out

def book_cost_line(model_spec: str, counts: Dict, extra: str = "") -> str:
    if not sum(counts.values()):
        return "✅ 고른 지문은 이미 교재가 다 있어요 (눌러도 새로 만들지 않아 비용 없음)"
    parts = " · ".join(f"{GEN_TYPE_NAMES[t].replace('Anatomy ', '')} {n}개" for t, n in counts.items() if n)
    return (f"🧾 새로 만들 교재: {parts} · 🤖 {ai_model_label(model_spec)} · 예상 비용 약 {book_cost_krw(model_spec, counts):,.0f}원"
            f"{extra} (대략 · 이미 있는 교재는 건너뛰어요)")

QB_SETUP_SQL = """create table if not exists public.question_bank (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  passage_label text not null,
  exam_key text,
  number text,
  q_type text not null,
  difficulty text not null,
  school text,
  content jsonb not null
);
create index if not exists question_bank_label_idx on public.question_bank (passage_label);
create index if not exists question_bank_label_created_idx on public.question_bank (passage_label, created_at, id);
alter table public.question_bank enable row level security;
create policy "qb select" on public.question_bank for select using (true);
create policy "qb insert" on public.question_bank for insert with check (true);
create policy "qb update" on public.question_bank for update using (true);
create policy "qb delete" on public.question_bank for delete using (true);"""

# 유형별 문제 형식 (학교 시험에 나오는 유형)
QB_TYPES = {
    "주제": "stem '다음 글의 주제로 가장 적절한 것은?'. choices는 영어 명사구 5개. 정답은 지문 핵심을 상위 개념어로 재진술.",
    "제목": "stem '다음 글의 제목으로 가장 적절한 것은?'. choices는 영어 제목 5개(콜론형·의문형 등 섞기).",
    "요지·주장": "stem '다음 글의 요지로 가장 적절한 것은?' 또는 '필자가 주장하는 바로 가장 적절한 것은?'. 하·중은 한국어 선택지, 상·최상은 영어 재진술 선택지.",
    "내용 일치·불일치": "stem '윗글의 내용과 일치하는(일치하지 않는) 것은?'. 하·중은 한국어 선택지, 상·최상은 영어 재진술 선택지. 숫자·비교·원인/결과 함정 활용.",
    "<보기> 일치 조합형": "box에 ⓐ~ⓖ(최상은 ⓐ~ⓗ) 진술을 줄바꿈으로 나열. stem '윗글의 내용과 일치하는 것만을 &lt;보기&gt;에서 있는 대로 고른 것은?'. choices는 'ⓐ, ⓒ, ⓔ' 같은 조합 5개.",
    "빈칸": "passage의 핵심 표현 자리를 '__________'로 비우고 stem '다음 빈칸에 들어갈 말로 가장 적절한 것은?'. choices는 영어 어구 5개.",
    "순서": "given에 주어진 글(첫 부분), passage에 (A)(B)(C) 세 덩어리를 섞어 줄바꿈으로 구분(각 덩어리 앞에 (A) 등 표시). stem '주어진 글 다음에 이어질 글의 순서로 가장 적절한 것은?'. choices는 '(A) - (C) - (B)' 형식 5개.",
    "삽입": "given에 빼낸 문장, passage의 문장 사이에 ( ① )~( ⑤ ) 표시. stem '글의 흐름으로 보아, 주어진 문장이 들어가기에 가장 적절한 곳은?'. choices는 빈 문자열 5개.",
    "무관한 문장": "passage 중간 문장 다섯 개 앞에 ①~⑤를 붙이고 그중 하나는 흐름과 무관한 새 문장으로 바꿔 넣음. stem '다음 글에서 전체 흐름과 관계 없는 문장은?'. choices는 빈 문자열 5개.",
    "요약문 (A)(B)": "box에 영어 요약문 한 문장(빈칸 (A), (B) 포함). stem '다음 글의 내용을 한 문장으로 요약하고자 한다. 빈칸 (A), (B)에 들어갈 말로 가장 적절한 것은?'. choices는 'word1 …… word2' 형식 5개.",
    "어법 (조합형)": "passage에 ⓐ~ⓖ 밑줄 7개(<u>…</u> 앞에 ⓐ 등). 틀린 것 2~4개는 원문 형태를 실제로 틀린 형태로 바꿔 passage에 써 넣는다(예: 원문 is → passage에는 ⓐ <u>are</u>). 바꾸지 않은 밑줄은 원문 그대로라 옳은 것. stem '밑줄 친 ⓐ~ⓖ 중, 어법상 틀린 것만을 있는 대로 고른 것은?'(또는 옳은 것만). choices는 조합 5개.",
    "어법 (단일)": "passage에 ①~⑤ 밑줄 5개, 그중 1개만 원문을 실제로 틀린 형태로 바꿔 써 넣는다(나머지 4개는 원문 그대로). stem '다음 글의 밑줄 친 부분 중, 어법상 틀린 것은?'. choices는 빈 문자열 5개.",
    "낱말 (문맥상)": "passage에 ①~⑤ 밑줄 낱말 5개, 그중 하나를 문맥과 반대되는 낱말로 바꿈. stem '다음 글의 밑줄 친 부분 중, 문맥상 낱말의 쓰임이 적절하지 않은 것은?'. choices는 빈 문자열 5개.",
    "함축 의미": "passage의 비유적 어구 하나에 밑줄. stem '밑줄 친 부분이 다음 글에서 의미하는 바로 가장 적절한 것은?'. choices는 영어 5개.",
    "추론 (추론 가능·불가)": "stem '윗글을 읽고 추론할 수 있는(없는) 것은?' 또는 box에 짧은 상황 글(<보기>)을 새로 써 주고 stem '윗글 필자의 관점에서 <보기>의 상황을 설명한 것으로 가장 적절한 것은?'. choices는 영어 문장 5개(하·중은 한국어 가능). 정답·오답 모두 지문에 그대로 쓰이지 않은 추론 문장으로 쓰고, 오답은 지문이 말하지 않은 비교·인과를 지어내거나 범위를 넓히거나 주체를 바꾼 것.",
    "서술형 (어법 수정)": "passage의 원문 2~3곳을 실제로 틀린 형태로 바꿔 써 넣고(밑줄 없이 또는 ⓐ~ⓗ 밑줄 중 일부), stem에 '찾아 바르게 고쳐 쓰시오' 조건. choices는 빈 배열, answer에 '틀린 표현 → 고친 표현'을 모두 적음.",
    "서술형 (조건 영작)": "box에 <조건>(사용할 단어·어형 변화·단어 수)을 줄바꿈으로. passage의 핵심 문장 한 곳을 빈칸 또는 우리말로 바꿈. choices는 빈 배열, answer에 모범 답안 영어 문장.",
}
QB_TYPE_NAMES = list(QB_TYPES)
QB_ESSAY_TYPES = {t for t in QB_TYPE_NAMES if t.startswith("서술형")}
QB_MARK_TYPES = {"삽입", "무관한 문장", "어법 (단일)", "낱말 (문맥상)"}  # 선택지가 지문 속 ①~⑤ 표시인 유형

QB_SCHOOL_STYLE = {
    "운양고": "24문항·100점(선택형 18 + 논술형 4로 내기도 함), 배점 3.8~4.7(최근 1학년 2026 2학기 1차는 4.2~4.7). 선택지는 전부 긴 영어 재진술이고 5개 모두 지문 소재를 다뤄, 지문을 안 읽고 지울 수 있는 선지가 없다. 난이도는 지문 변형보다 선택지와 문항 설계에서 나온다. 어법 '틀린/옳은 것만 있는 대로 고른 것'(ⓐ~ⓖ 조합형), '<보기>에서 일치하는/일치하지 않는 것만 있는 대로'(ⓐ~ⓗ 진술), 순서·삽입·요약문(A)(B)·무관한 문장·낱말·빈칸·주제·제목·요지. 1학년 1학기는 한글 선택지 내용일치·우리말→영어 표현도 출제하고, 2학기는 영어 선택지가 기본. 추론형('윗글을 읽고 추론할 수 있는/없는 것', '필자의 관점에서 <보기> 상황 설명', 4.5~4.6점)도 낸다.",
    "제일고": "24문항, 배점 3.6~4.8. 영어 선택지가 길고 지문 재진술 어휘가 고급. 어법 조합형, 빈칸, 제목/주제, 내용 일치, 밑줄 의미, 문맥상 낱말, 요약문, 도치 구문(So+형용사, Not only) 자주 활용.",
}
QB_LEVEL_RULES = {
    "하": "단일 정답형 위주, 오답은 명백히 틀림, 지문 본문은 원문 그대로(선택지는 쉬운 말로 재진술).",
    "중": "오답 1~2개는 지문 키워드를 재사용한 매력적 오답.",
    "상": "조합형·<보기>형 포함, 숫자·more than/less than·원인/결과 뒤바꾸기 함정, 바로 뒤 단어가 미끼가 되는 낱말 문제, 지시어로 막히는 삽입 위치.",
    "최상": "상 + 선택지를 전부 재진술(paraphrase), <보기> 7~8개, 어법은 완료부정사·동격 that·분사구문 태·도치 등 고난도 포인트로만 구성.",
}
# 유형별 난이도 기준 (일반 기준만으로는 선택지가 없는 유형의 '최상'이 쉽게 나와서 유형마다 따로 정함)
QB_TYPE_LEVEL = {
    "무관한 문장": {
        "하": "끼워 넣는 문장은 소재부터 다른 문장.",
        "중": "같은 소재지만 글의 논지와 상관없는 곁가지 문장.",
        "상": "지문 핵심어를 2개 이상 그대로 쓰고 앞 문장과 자연스럽게 이어지는 듯하지만, 글의 주장과 방향이 다른 문장. 정답 문장만 혼자 튀면 안 된다.",
        "최상": "지문 핵심어와 연결어(Indeed, In fact, However 등)를 써서 앞뒤 문장과 표면상 매끄럽게 이어지고, 같은 소재의 일반론·반대 관점이라 주제문과 비교해야만 어긋남이 보이는 문장. 나머지 ①~⑤ 중에도 처음 보면 어색해 보이는 문장이 하나 있도록 원문 문장 위치를 고른다.",
    },
    "순서": {
        "하": "연결어·지시어 단서가 분명하게 남아 있음.",
        "중": "연결어 일부만 남기고 지시어로 판단.",
        "상": "this/such/these·대명사 같은 지시어와 논리 흐름으로만 판단, 오답 순서 두 개가 그럴듯함. 지문은 [문항별 출제 계획]의 순서 재구성을 따른다.",
        "최상": "덩어리 첫머리 연결어 단서를 최소화하고 길이를 비슷하게 하며, 두 가지 순서가 모두 그럴듯해 보이지만 해석해서 지시어·논리를 따라가야 하나만 맞게 나눈다. 지문은 [문항별 출제 계획]의 순서 재구성을 따른다.",
    },
    "삽입": {
        "하": "주어진 문장에 However, For example 같은 분명한 연결어.",
        "중": "지시어(this, they)로 들어갈 자리를 찾게.",
        "상": "주어진 문장이 들어가야 바로 뒤 문장의 지시어가 풀리는 자리, 그럴듯한 함정 자리 하나 포함. 주어진 문장·주변 문장은 [문항별 출제 계획]의 삽입 재구성을 따른다.",
        "최상": "주어진 문장에는 연결어를 두지 않아 내용 논리로만 판단하게 하고, 두 자리가 모두 들어갈 듯해 보이지만 해석해야 앞뒤 흐름이 한 곳에서만 끊기는 것이 보이게 만든다. 주어진 문장·주변 문장은 [문항별 출제 계획]의 삽입 재구성을 따른다.",
    },
    "빈칸": {
        "하": "정답은 원문 핵심 표현을 쉬운 말로 바꾼 것.",
        "중": "정답은 재진술, 오답 일부는 지문 단어를 그대로 써서 함정.",
        "상": "주제문 핵심을 추상어로 재진술한 정답, 반대 의미 오답과 지문 단어 재사용 오답 포함.",
        "최상": "역접·이중부정 뒤처럼 논리를 따라야 풀리는 자리에 빈칸, 정답·오답 모두 고급 재진술, 부분적으로만 맞는 오답 2개 이상.",
    },
    "요약문 (A)(B)": {
        "하": "(A)(B) 정답이 지문 단어 그대로(요약문 문장 자체는 재진술).",
        "중": "(A)(B) 중 하나는 재진술.",
        "상": "(A)(B) 모두 재진술, 오답 쌍은 하나만 틀린 것이 섞임.",
        "최상": "(A)(B) 모두 지문에 없는 상위 개념어, 오답 쌍 대부분이 한쪽만 틀려 끝까지 따져야 함.",
    },
    "주제": {"하": "정답에 지문 핵심어 포함.", "중": "정답은 재진술, 오답은 지문 단어를 쓴 함정.", "상": "너무 넓은/좁은 오답 포함, 정답은 상위 개념어.", "최상": "모든 선택지가 지문 소재를 다루며, 범위·초점만 다른 오답들 사이에서 정답을 골라야 함."},
    "제목": {"하": "정답에 지문 핵심어 포함.", "중": "비유적 제목 하나 포함.", "상": "비유·의문형 정답, 지문 단어를 쓴 함정 오답.", "최상": "비유적 정답을 글 전체 논지로 해석해야 하고, 오답은 지문 일부만 반영."},
    "요지·주장": {"하": "필자 주장이 직접 드러난 문장 기준.", "중": "예시 뒤에 숨은 주장.", "상": "통념 제시 후 반박 구조에서 진짜 주장을 고르게.", "최상": "통념·예시·반박이 섞여 있고, 오답이 통념이나 예시를 그대로 옮긴 것."},
    "내용 일치·불일치": {"하": "원문 내용을 쉬운 말로 짧게 바꿔 씀.", "중": "한두 단어만 바꾼 함정.", "상": "숫자·비교·원인/결과 뒤바꾸기 함정.", "최상": "모든 선택지를 재진술하고, 부분 부정·시간 순서·주체 바꾸기 함정을 섞음."},
    "<보기> 일치 조합형": {"하": "<보기> 5개, 원문에 가까운 진술.", "중": "<보기> 6개, 일부 재진술.", "상": "<보기> 7개, 숫자·인과 함정.", "최상": "<보기> 8개 모두 재진술, 맞는 진술 개수를 짐작할 수 없게 조합."},
    "추론 (추론 가능·불가)": {"하": "지문에 거의 그대로 나온 내용을 쉬운 말로.", "중": "한 문장만 보면 추론되는 내용.", "상": "두 문장 이상을 연결해야 추론되고, 오답 하나는 지문이 암시만 한 인과를 단정함.", "최상": "글 전체 논지를 새 상황에 적용해야 풀리고, 오답 2개 이상이 지문 단어를 쓰면서 비교·인과·범위만 어긋남."},
    "함축 의미": {"하": "비유가 바로 앞뒤에서 설명됨.", "중": "글 전체를 보면 알 수 있음.", "상": "비유 해석에 추론이 필요하고 문자 그대로 해석한 오답 포함.", "최상": "글의 결론과 연결해야 풀리는 비유, 오답은 부분적으로 맞는 해석."},
    "낱말 (문맥상)": {"하": "명백한 반의어 교체.", "중": "바로 앞뒤 문장과 모순.", "상": "단락 논리를 따라야 보이는 반의어, 혼동어 함정 포함.", "최상": "글 전체 논지와 비교해야 보이는 교체, 다른 밑줄 낱말도 처음 보면 어색해 보이게 고급 어휘 선택."},
}

# 실제 기출 지문을 원문과 비교해 본 학교별 지문 변형 방식 (운양고·제일고 실제 기출 72문항, 4단어 묶음이 원문과 같은 비율)
#   운양고: 빈칸 91~95%·순서 88%·삽입 본문 93%로 지문은 거의 원문 그대로, 대신 삽입의 '주어진 문장'을 다른 말로 바꿔 씀
#   제일고: 삽입 62~94%·무관한 문장 65~85%·빈칸 55~75%로 본문도 상당 부분 재진술, 순서는 주어진 글을 새로 써서 도입부를 바꾸기도 함(12%)
QB_SCHOOL_PASSAGE = {
    "운양고": "지문 본문은 원문 표현을 거의 그대로(약 90% 이상) 쓴다. 삽입 유형은 '주어진 문장'만 원문과 다른 표현으로 바꿔 쓴다(예: 'at three degrees of warming, or four' → 'when the temperature rises by three or four more degrees'). 하·중 순서 유형은 덩어리 문장을 원문 그대로 두고 나누는 위치와 연결어 단서로 난이도를 조절한다. 단, 상·최상 순서·삽입은 학생이 원문을 외우고 오므로 [문항별 출제 계획]의 재구성 규칙을 따른다(원문 그대로 자르기만 한 순서·삽입은 상 이상이 될 수 없다).",
    "제일고": "지문 본문 일부를 재진술한다(원문 표현은 약 60~85%만 남기고 동의어·구조 변경·연결어 교체). 상·최상일수록 재진술 비율을 높이되 최상도 원문 표현을 60% 가까이는 남기고(실제 기출 최저치), 순서 유형은 주어진 글을 같은 논지의 다른 도입 문장으로 새로 써서 원문을 외운 학생이 도입부만 보고 순서를 떠올리지 못하게 할 수 있다. 시험에 따라 지문 전체를 같은 내용의 다른 문장으로 새로 쓰기도 한다(제일고 26년 1학기 기말 고1: 원문 표현 5~14%) — 이 경우는 [문항별 출제 계획]을 따른다. 어느 경우든 뜻·논리·정보는 원문과 같게 유지한다.",
}
QB_PASSAGE_TYPES = {"순서", "삽입", "무관한 문장", "빈칸", "낱말 (문맥상)"}

# ---- 문항별 출제 계획 (파이썬이 문항마다 무작위로 정해 프롬프트 뒤쪽에 붙임 · AI 호출 추가 없음, 공통 앞부분 캐시 그대로) ----
# 2026-11-04 실제 기출 54문항(올바르게 연결된 것) 재측정: 영어 선택지·<보기>·요약문 문장은 두 학교 모두 원문 4단어 묶음이 0~6%만 남음
QB_CHOICE_TYPES = {"주제", "제목", "요지·주장", "내용 일치·불일치", "<보기> 일치 조합형", "빈칸", "요약문 (A)(B)", "함축 의미", "추론 (추론 가능·불가)"}
# 영어 선택지 하나의 단어 수 (2026-11-10 측정: 운양고·제일고 24~26년 고1·고2 시험지 30개, 'FINAL 단어 수 재기'로 문항마다 잰 영어 선지 단어 ÷ 5)
#   우리 변형문제는 같은 유형에서 실제보다 1.3~2배 길었고, '중'도 최상만큼 길었음 → 학교·학년·유형별로 이 길이에 맞춤 (난이도와 관계없이)
#   학년 자료가 모자란 칸(운양고1 함축 등)은 그 학교 24~26년 전체 평균. 추론은 내용 일치와 같게
QB_GRADES = ["고1", "고2"]
QB_CHOICE_LEN = {
    ("운양고", "고1"): {"주제": 9, "제목": 8, "요지·주장": 17, "내용 일치·불일치": 16, "빈칸": 7, "함축 의미": 10},
    ("운양고", "고2"): {"주제": 9, "제목": 9, "요지·주장": 19, "내용 일치·불일치": 13, "빈칸": 8, "함축 의미": 10},
    ("제일고", "고1"): {"주제": 10, "제목": 8, "요지·주장": 14, "내용 일치·불일치": 15, "빈칸": 7, "함축 의미": 10},
    ("제일고", "고2"): {"주제": 11, "제목": 8, "요지·주장": 17, "내용 일치·불일치": 17, "빈칸": 8, "함축 의미": 12},
    ("운양고", ""): {"주제": 9, "제목": 9, "요지·주장": 18, "내용 일치·불일치": 15, "빈칸": 8, "함축 의미": 10},
    ("제일고", ""): {"주제": 10, "제목": 8, "요지·주장": 16, "내용 일치·불일치": 16, "빈칸": 9, "함축 의미": 11},
}

def qb_passage_grade(p) -> str:
    """지문의 학년: 원문 라이브러리의 grade(모의고사) → 이름 '[2025년 10월 고1]' → 교과서 이름(공통영어 = 고1, 영어Ⅰ·Ⅱ = 고2). 모르면 ''"""
    label = str(p.get("label", "") if isinstance(p, dict) else p)
    g = str(p.get("grade", "") if isinstance(p, dict) else "").strip()
    if re.fullmatch(r"고[123]", g):
        return g
    m = re.search(r"고\s*([123])\s*\]", label) or re.search(r"(?<![가-힣])고([123])(?![0-9])", label)
    if m:
        return "고" + m.group(1)
    if "공통영어" in label:
        return "고1"
    if re.search(r"영어\s*(Ⅰ|Ⅱ|II|I|1|2)(?![0-9])", label):
        return "고2"
    return ""

def qb_choice_len_rule(school: str, grade: str, q_type: str, level: str) -> str:
    """[이번 출제]에 붙이는 선지 길이·어휘 기준 (영어 선지 유형만). 호출마다 다른 뒷부분이라 캐시는 그대로"""
    key_t = "내용 일치·불일치" if q_type == "추론 (추론 가능·불가)" else q_type
    tbl = (QB_CHOICE_LEN.get((school, grade)) or {}) if grade else {}
    src = f"{school} {grade}" if key_t in tbl else school
    n = tbl.get(key_t) or (QB_CHOICE_LEN.get((school, "")) or {}).get(key_t)
    if not n:
        return ""
    lo, hi = max(2, round(n * 0.6)), round(n * 1.4)
    out = (f"[선택지 길이 — {src} 실제 기출(24~26년) 측정] 영어 선택지 하나는 평균 약 {n}단어로 쓴다. 5개 모두 {lo}~{hi}단어 안에 두고, "
           f"가장 긴 선택지도 {hi}단어를 넘기지 않는다. 난이도는 길이가 아니라 재진술·함정 구조로 올린다(최상도 이 길이를 지킨다). "
           "실제 시험보다 선택지가 길면 학생이 '우리 학교 시험과 다르다'고 느낀다.")
    if level in ("상", "최상"):
        out += ("\n[선택지 어휘 — 실제 기출 측정] 실제 선택지는 고교 필수 단어 밖의 어려운 단어가 5단어에 1개 정도다. "
                "상위 개념어 재진술은 선택지마다 1~2개로 하고, 지문보다 어려운 단어를 늘어놓아 어렵게 만들지 않는다.")
    return out
QB_CHOICE_PARA = ("영어 선택지·<보기> 진술·요약문 문장은 난이도와 상관없이 원문 문장을 그대로 옮기지 않고 모두 다른 말로 재진술한다"
                  "(하·중은 쉬운 어휘로, 상·최상은 상위 개념어·구조 변경으로). 한국어 선택지는 직역이 아니라 의미 요약으로 쓴다.")
QB_OPPOSITE_TRAP = ("오답 {n}개는 지문 핵심어를 그대로 쓰면서 글의 주장·결론을 뒤집은 재진술로 만든다"
                    "(예: 'regardless of whether there is a reason' → 'only when you have something significant', 'Beyond Convenience' → 'Tech over Values', "
                    "원인↔결과·주체↔대상 뒤바꾸기). 겉보기에는 정답보다도 지문과 비슷해 보이게 쓴다.")
# 실제 기출 어법 선택지 개념 빈도 (QB_GRAMMAR_DNA와 같은 자료) — 문항마다 이 목록에서 무작위로 '꼭 묻는 개념'을 뽑음
QB_GRAMMAR_FREQ = {
    "운양고": {"수일치": 66, "분사·분사구문": 58, "관계대명사": 48, "능동/수동": 44, "5형식 목적격보어": 37, "동명사": 34, "명사절·동격 that": 30,
              "병렬": 29, "시제·가정법·조동사": 26, "형용사/부사·2형식 보어": 23, "관계부사 vs 관계대명사": 21, "도치·강조": 17, "대명사 지칭": 16,
              "가주어·가목적어": 13, "비교 구문": 10, "to부정사": 9, "한정사(each/every 등)": 9, "접속사 vs 전치사": 7, "복합관계사": 6},
    "제일고": {"분사·분사구문": 82, "관계대명사": 46, "명사절·동격 that": 35, "능동/수동": 34, "동명사": 33, "시제·가정법·조동사": 30, "병렬": 27,
              "수일치": 25, "형용사/부사·2형식 보어": 25, "대명사 지칭": 16, "접속사 vs 전치사": 16, "to부정사": 16, "5형식 목적격보어": 15,
              "관계부사 vs 관계대명사": 14, "도치·강조": 14, "가주어·가목적어": 13, "동사 자리 vs 준동사": 11, "자동사/타동사": 9},
}
QB_GRAMMAR_RE = {"수일치": r"수\s*일치", "분사·분사구문": r"분사", "관계대명사": r"관계대명사|선행사", "능동/수동": r"수동|능동",
                 "5형식 목적격보어": r"목적격\s*보어|5형식", "동명사": r"동명사", "명사절·동격 that": r"동격|명사절", "병렬": r"병렬",
                 "시제·가정법·조동사": r"시제|가정법|조동사", "형용사/부사·2형식 보어": r"형용사|2형식", "관계부사 vs 관계대명사": r"관계부사",
                 "도치·강조": r"도치|강조", "대명사 지칭": r"지칭|재귀", "가주어·가목적어": r"가주어|가목적어", "비교 구문": r"비교",
                 "to부정사": r"부정사", "한정사(each/every 등)": r"한정사|each|every", "접속사 vs 전치사": r"전치사", "복합관계사": r"복합관계",
                 "동사 자리 vs 준동사": r"준동사|동사 자리", "자동사/타동사": r"자동사|타동사"}
# '섞기'일 때 문장 단위 밑줄로 낼 확률 (운양고: 단어·구 위주에 문장형도 · 제일고: 문장 단위 밑줄이 많음)
QB_SENTENCE_RATE = {"운양고": {"어법 (단일)": 0.25, "어법 (조합형)": 0.3, "서술형 (어법 수정)": 0.2},
                    "제일고": {"어법 (단일)": 0.5, "어법 (조합형)": 0.35, "서술형 (어법 수정)": 0.5}}
QB_UNDERLINE_OPTS = {"mix": "섞기 (학교 기출 비율)", "word": "단어·구", "sentence": "문장 전체"}
# 고난도 어법 (2026-11-05 선생님 지시 · 운양고 2026 2학기 1차 7·8번 4.6~4.7점 방식): 문장 전체 밑줄 ⓐ~ⓔ(ⓕ) + '옳은 것만을 있는 대로'
#   '섞기'일 때 (학교, 난이도, 유형)별 문장 밑줄 확률 — 없는 조합은 위 QB_SENTENCE_RATE. '단어·구'를 고르면 그대로 단어 밑줄
QB_HARD_SENT_RATE = {("운양고", "최상", "어법 (조합형)"): 1.0, ("운양고", "상", "어법 (조합형)"): 0.7,
                     ("운양고", "최상", "어법 (단일)"): 0.6, ("운양고", "상", "어법 (단일)"): 0.4,
                     ("제일고", "최상", "어법 (조합형)"): 1.0, ("제일고", "최상", "어법 (단일)"): 0.6}
QB_RIGHT_ONLY_RATE = {("운양고", "최상"): 1.0, ("운양고", "상"): 0.7, ("제일고", "최상"): 0.7}
QB_SENT_GRAMMAR_RECIPE = ("밑줄 단위: 문장(고난도) — 밑줄은 5~6개(ⓐ~ⓔ 또는 ⓕ)로, 각 밑줄은 한 문장(또는 긴 절) 전체를 <u>…</u>로 감싼다(위 기준의 밑줄 개수보다 이 줄이 우선). "
                          "옳은 문장 중 1~2개는 원문 그대로인데 학생 눈에는 틀려 보이는 구조를 고른다(예: All you have to do is wait — is + 원형부정사, larger than that of Busan — 비교 대상 병렬, "
                          "The results of the study conducted by … show — 긴 수식어 뒤 복수 동사, is of great importance — of + 추상명사, 도치·강조, 가주어 it). "
                          "틀린 문장은 그 안의 한 곳만, 바로 옆 단어만 보면 맞아 보이는 자리(긴 삽입구 뒤 동사, 부분 표현 of 뒤 명사에 맞춘 수, 선행사 착각)에 오류를 넣는다. "
                          "조합형이면 옳은 문장을 2~4개로 해서 개수로 답을 추측할 수 없게 한다. 기본 빈출(are/is, hold/holding)만으로 구성하지 않는다. 지문 문장 수보다 밑줄을 많이 긋지 않는다. explanation에 옳은 문장 중 틀려 보이는 것이 왜 맞는지도 적는다.")
# 지문을 바꾸지 않는 유형은 AI가 지문을 다시 베껴 쓰지 않게 하고 프로그램이 원문을 넣음 (2026-11-04 측정: 이 유형들은 지문의 90~98%가 원문 그대로,
#   출력 토큰의 40~54%가 지문 베끼기였음). 함축은 밑줄 어구, 운양고 빈칸은 비울 어구만 받아 프로그램이 표시
QB_NOCOPY_TYPES = {"주제", "제목", "요지·주장", "내용 일치·불일치", "<보기> 일치 조합형", "요약문 (A)(B)", "함축 의미", "추론 (추론 가능·불가)"}
QB_NOCOPY_MARK = "지문 쓰기: passage에는 '원문' 두 글자만 쓴다"
QB_POS_TYPES = {"어법 (단일)", "낱말 (문맥상)", "무관한 문장"}  # 지문 속 ①~⑤가 선택지인데 정답 자리를 고를 수 있는 유형
# ---- '우리 학교 시험 같다'는 느낌을 주는 헷갈림 장치 (2026-11-05, 운양고 25년 2학기 중간 고1 100점 시험지에서 학생이 '어렵다'고 별표 친 문항 분석) ----
#   잘하는 학생이 헷갈린 이유는 대부분 "지문 단어가 보이면 맞다 / 바꿔 쓴 말이면 틀리다"는 습관을 역이용한 것
#   문항별 출제 계획(프롬프트 맨 뒤)에만 붙여서 캐시되는 공통 앞부분은 그대로 둠
QB_FEEL_RATE = {"하": 0.0, "중": 0.25, "상": 0.6, "최상": 0.9}
QB_FEEL_DEVICES = {
    "topic": ["헷갈림 장치(구체어 함정): 오답 하나는 글의 일부(특히 후반)에서 가장 눈에 띄는 구체적 용어 2~3개를 그대로 묶은 '부분 주제'로 써서 정답보다 더 정답처럼 보이게 하고, "
              "정답은 글 앞부분과 뒷부분을 함께 아우르는 상위 표현으로 쓴다(함정 배치 줄이 있으면 이 오답을 그 '낚는 선지' 자리에 둔다). "
              "실제 예: 조직 문화·갈등 관리 글에서 오답 'Comparing Mechanistic and Organic Structures'(후반 용어 그대로) vs 정답 'Cultural Diversity and Conflict Management within Organizations'."],
    "summary": ["헷갈림 장치(한 칸으로 안 갈림): (A) 선택지에 문맥상 둘 다 그럴듯한 비슷한 뜻의 단어 2개를 넣어 (A)만으로는 고를 수 없게 하고, 그중 하나는 (B)가 틀려 (B)까지 따져야 정답이 정해지게 한다. "
                "실제 예: 'at a disadvantage' 자리에 vulnerable(정답 쌍)과 disempowered(오답 쌍)를 함께 둠.",
                "헷갈림 장치(반쪽 지문 단어): 오답 하나는 한 칸은 지문 단어를 그대로 쓰고 다른 칸은 반대 뜻으로 써서, 익숙한 한 칸만 보고 고르게 만든다. 정답 쌍은 두 칸 모두 지문에 없는 동의어로 쓴다. "
                "실제 예: 'independent power … observe a scene' 글에서 오답 'subordinate …… observe' vs 정답 'autonomous …… inspect'."],
    "tf": ["헷갈림 장치(맞는 선지 낯설게 — 이 유형의 형식은 그대로 지킨다): 내용에 맞는 선택지(<보기> 유형이면 <보기> 진술)는 지문에 없는 상위어·구조 변경으로 바꿔 써서 처음 보면 틀려 보이게 하고, 틀린 것은 지문 단어를 그대로 쓰면서 핵심 한 단어만 뒤집는다"
           "(예: 'very different functions' → 'identical functions', 'When they die' → 'when they are alive'). 익숙한 단어가 보이면 맞다고 넘어가는 학생이 걸리게 한다.",
           "헷갈림 장치(암시만 된 내용 — 이 유형의 형식은 그대로 지킨다): 선택지(<보기> 유형이면 <보기> 진술) 하나는 지문이 암시만 하고 직접 말하지 않은 내용을 지문 단어로 그럴듯하게 써서, 지문에 나온 것처럼 착각하게 만든다"
           "(예: 산소가 수온 상승보다 빨리 줄었다는 글에서 '다른 원인'을 지문이 밝힌 것처럼 쓴 문장). 일치/불일치 판단 근거는 explanation에 밝힌다."],
    "blank": ["헷갈림 장치(암시 함정): 오답 하나는 지문이 암시만 하거나 한 예시에만 맞는 내용을 지문 단어로 써서 그럴듯하게 만들고, 정답은 빈칸 앞뒤의 역접·인과 논리를 따라가야만 확정되게 한다."],
    "insert": ["헷갈림 장치(가짜 자리): 주어진 문장의 단어가 바로 앞 문장의 단어와 겹쳐 표면적으로 이어져 보이는 가짜 자리를 하나 둔다. 진짜 자리는 주어진 문장 첫머리 연결어(So, This, Such 등)가 받는 내용과 "
               "바로 뒤 문장의 핵심어로만 확정되게 한다. 실제 예: 'So when you start to watch television … see the world as being shaped like television'이 'see things differently' 뒤에도, 'TV teaches you …' 앞에도 들어갈 듯해 보임."],
    "order": ["헷갈림 장치(두 순서 그럴듯): 덩어리 첫머리의 지시어(this, these, such, they)가 두 덩어리에 모두 걸리는 것처럼 보이게 하고, 정답 순서는 내용 논리(일반 → 예시 → 결론, 문제 → 해결)를 따라가야만 확정되게 한다."],
    "grammar": ["헷갈림 장치(같은 함정 두 번 + 긴 주어): 같은 어법 포인트(예: one of + 복수명사, 수일치, 관계사 what/that)를 밑줄 두 곳에 넣어 한 곳만 보고 판단하면 틀리게 하고, 맞게 쓴 밑줄에는 동명사구·삽입구가 붙은 긴 주어를 써서 틀려 보이게 한다. "
                "실제 예: 'one of the most famous non-great composer' / 'one of the world's great composer'(둘 다 틀림), 'Distributing food and other goods to neighborhood vendors allows …'(맞음), "
                "'The musical forms, or ideas expressed within them, were …'(맞음)."],
}
QB_FEEL_GROUP = {"주제": "topic", "제목": "topic", "요지·주장": "topic", "요약문 (A)(B)": "summary", "내용 일치·불일치": "tf", "<보기> 일치 조합형": "tf", "추론 (추론 가능·불가)": "tf",
                 "빈칸": "blank", "함축 의미": "blank", "삽입": "insert", "순서": "order", "어법 (단일)": "grammar", "어법 (조합형)": "grammar"}
QB_FEEL_EXPLAIN = "explanation 마지막 줄에 '헷갈리는 포인트: (끌리는 오답 번호·자리)가 끌리는 이유 → 실제로 틀린 이유'를 한 줄로 쓴다(학생이 '이래서 시험에서 틀렸구나'를 알 수 있게)."

QB_ORDER_RECIPE = ("순서 재구성(암기 무력화 — 학교와 관계없이 필수): 주어진 글과 (A)(B)(C)를 같은 뜻의 다른 문장으로 전면 재진술한다(원문 4단어 묶음이 거의 남지 않게). "
                   "원문의 한 부분(예시·결론 등)을 주어진 글로 앞당기거나 원인·이유를 뒤 덩어리로 보내 정답 순서가 원문 서술 순서와 달라지게 하고, 원문에서 바로 이어지던 순서(외운 학생이 떠올릴 순서)를 오답 선택지 하나로 넣는다. "
                   "확정 단서는 연결어(Then, And so, However)가 아니라 내용 지시어(these conclusions, such explanations)로만 두고 덩어리 길이를 비슷하게 한다. "
                   "원문 정보는 빠짐없이 담고 새 내용은 확실한 일반 상식만 쓰며, 다른 순서가 성립하지 않는지 직접 확인한다. explanation 끝에 '재구성: 무엇을 새로 쓰고 옮겼는지'를 적는다. "
                   "예: 사이먼의 계산 분산 글에서 변기 예시를 주어진 글로 앞당기고, 계산을 사람들에게 나눠 맡긴다는 설명을 그 이유(This is possible because …)로 뒤에 둠.")
QB_INSERT_RECIPE = ("삽입 재구성(암기 무력화 — 학교와 관계없이 필수): 주어진 문장은 원문 문장을 같은 뜻의 다른 표현으로 바꿔 쓰고 첫머리 연결어를 두지 않는다. "
                    "원문을 외운 학생이 떠올리는 '원래 자리'만으로 풀리지 않도록, 주어진 문장 앞뒤 2~3문장을 같은 뜻의 다른 문장으로 재진술하고 지시어·연결 표현도 바꿔서 "
                    "원문 위치 기억이 아니라 바뀐 문장의 지시어·논리로만 자리가 정해지게 한다(나머지 본문은 학교 기준대로, 원문에 없는 정보는 넣지 않는다). "
                    "가짜 자리 1개는 주어진 문장의 단어가 앞 문장 단어와 겹치는 곳에 두고, 진짜 자리는 바로 뒤 문장의 지시어로만 확정되게 한다. explanation에 '지문 재진술: 원문 표현 → 바꾼 표현'을 적는다.")
QB_BLANK_RECIPE = ("빈칸 함정(암기 무력화): 정답은 원문에 없는 상위어 재진술로 쓰고, 원문 표현을 그대로 옮긴 선지 하나는 오답으로 둔다(원문을 외운 학생이 끌리게). "
                   "오답 2개 이상은 부분적으로 맞게 쓴다(한 예시에만 맞음 · 지문이 암시만 함 · instead of/even though 논리를 거꾸로 따름). "
                   "빈칸은 주제문보다 역접·대조·이유처럼 논리를 따라가야 하는 자리에 우선 둔다.")

# 함정형 문항 비율(난이도별)과 함정형 문항의 정답 번호 가중치 (③·④·⑤ = 45%·40%·15%) — 출제DNA 탭 현황판도 이 값을 그대로 보여 줌
QB_TRAP_RATE = {"하": 0.0, "중": 0.4, "상": 0.65, "최상": 0.7}
QB_TRAP_ANS_POS = [2, 3, 4]
QB_TRAP_ANS_W = [0.45, 0.4, 0.15]

# 어법 '꼭 묻는 개념'을 학교 전체로 고르게: 목표 비율은 실제 기출 빈도의 제곱근(많이 나온 개념도 자주, 드문 개념도 꾸준히),
#   지금까지 만든 문제(문제 은행 + 이번 실행에서 뽑은 것)에서 목표보다 덜 나온 개념부터 채움
_QB_GRAMMAR_PICKS: Dict[str, Dict[str, float]] = {}

def qb_grammar_concepts(r: Dict) -> List[str]:
    """어법 문제 한 개가 다룬 개념 목록 (출제 포인트·해설 글에서 찾음)"""
    t = " ".join(str(x) for x in [(r.get("content") or {}).get("focus", ""), (r.get("content") or {}).get("explanation", "")])
    return [c for c, pat in QB_GRAMMAR_RE.items() if re.search(pat, t)]

def _qb_grammar_counter(school: str) -> Dict[str, float]:
    """학교별 개념 사용 횟수 (처음 한 번만 문제 은행에서 읽어 오고, 그 뒤로는 뽑을 때마다 더함)"""
    if school not in _QB_GRAMMAR_PICKS:
        cnt = {c: 0.0 for c in QB_GRAMMAR_FREQ.get(school, QB_GRAMMAR_FREQ["운양고"])}
        try:
            if IS_CLOUD_DB:
                res = supabase.table(QB_TABLE).select("content").eq("school", school).in_("q_type", sorted(QB_GRAMMAR_TYPES)).limit(3000).execute()
                for r in res.data or []:
                    for c in qb_grammar_concepts(r):
                        if c in cnt:
                            cnt[c] += 1
        except Exception:
            pass
        _QB_GRAMMAR_PICKS[school] = cnt
    return _QB_GRAMMAR_PICKS[school]

def qb_question_plans(q_type: str, level: str, school: str, count: int, existing: List[Dict], underline: str = "mix", rnd=None) -> List[List[str]]:
    """문항마다 다르게 낼 출제 장치를 무작위로 정함 → [[계획 문장, ...], ...] (문항 수만큼)"""
    rnd = rnd or random.Random()
    plans = [[] for _ in range(count)]
    hard = level in ("상", "최상")
    if q_type in QB_GRAMMAR_TYPES - {"서술형 (조건 영작)"}:
        freq = QB_GRAMMAR_FREQ.get(school, QB_GRAMMAR_FREQ["운양고"])
        used = {c: 0 for c in freq}
        for r in existing:  # 이 지문에서 이미 물은 개념은 덜 뽑히게
            for c in qb_grammar_concepts(r):
                if c in used:
                    used[c] += 1
        glob = _qb_grammar_counter(school)
        tw = {c: math.sqrt(freq[c]) for c in freq}
        tsum = sum(tw.values())
        k = 3 if q_type == "어법 (조합형)" else 2
        for pl in plans:
            pool = list(freq)
            pick = []
            for _ in range(k):
                total = sum(glob.values()) + 1
                # 목표 몫보다 얼마나 덜 나왔나(+) - 이 지문에서 이미 물었나 + 약간의 무작위 → 가장 큰 개념
                score = {c: tw[c] / tsum * total - glob.get(c, 0) - 3 * used[c] + rnd.random() * 1.5 for c in pool}
                c = max(pool, key=score.get)
                pick.append(c)
                pool.remove(c)
                used[c] += 1
                glob[c] = glob.get(c, 0) + 1
            pl.append("이 문항에 꼭 넣을 어법 개념(밑줄 포인트로, 틀린 곳이든 맞게 쓴 함정이든): " + ", ".join(pick)
                      + " — 나머지 밑줄도 이 개념들과 겹치지 않게 학교 기출 개념 목록에서 골고루 고른다(분사·관계사처럼 많이 나온 개념만 반복하지 않는다)")
            base = QB_HARD_SENT_RATE.get((school, level, q_type), QB_SENTENCE_RATE.get(school, {}).get(q_type, 0.3))
            rate = {"word": 0.0, "sentence": 1.0}.get(underline, base)
            if rnd.random() < rate:
                if q_type != "서술형 (어법 수정)" and hard and (school, level, q_type) in QB_HARD_SENT_RATE:
                    # 실제 운양고 고난도 방식: 문장 전체 밑줄 + '옳은 것만' + 원문이라 틀려 보이는 옳은 문장
                    right_only = q_type == "어법 (조합형)" and rnd.random() < QB_RIGHT_ONLY_RATE.get((school, level), 0.5)
                    pl.append(QB_SENT_GRAMMAR_RECIPE + (" stem은 '밑줄 친 ⓐ~ⓔ(ⓕ) 중, 어법상 옳은 것만을 있는 대로 고른 것은?'으로 한다." if right_only else ""))
                elif q_type == "서술형 (어법 수정)":
                    pl.append("밑줄 단위: 문장 — 밑줄 친 문장 ⓐ~ⓔ 중 어법상 틀린 문장 2개를 찾아 바르게 고쳐 쓰게 한다(각 문장은 <u>문장 전체</u>, 틀린 문장은 그 안의 한 곳만 틀림).")
                else:
                    pl.append("밑줄 단위: 문장 — 각 밑줄은 한 문장(또는 긴 절) 전체를 <u>…</u>로 감싼다. 틀린 문장은 그 안의 한 곳만 틀리게 바꾸고, 맞는 문장에도 함정 포인트를 둔다. "
                              "stem은 '밑줄 친 문장 중, 어법상 틀린(옳은) 것은?' 또는 조합형이면 '밑줄 친 ⓐ~ⓖ 중 어법상 옳은(틀린) 문장만을 있는 대로 고른 것은?'. 지문 문장 수보다 밑줄을 많이 긋지 않는다.")
            else:
                pl.append("밑줄 단위: 단어·구 — 각 밑줄은 판단할 단어나 짧은 구.")
    # 상·최상 순서·삽입·빈칸은 학교와 관계없이 '암기 무력화' (2026-11-05 선생님 지시: 학생은 원문을 외우고 옴 → 원문 그대로면 상 이상 불가)
    if q_type == "순서" and hard:
        for pl in plans:
            pl.append(QB_ORDER_RECIPE)
    if q_type == "삽입" and hard:
        for pl in plans:
            pl.append(QB_INSERT_RECIPE)
    if q_type == "빈칸" and hard:
        for pl in plans:
            pl.append(QB_BLANK_RECIPE)
    if school == "제일고" and hard and q_type != "순서":
        for pl in plans:
            if rnd.random() < 0.25:
                pl.append("지문 전면 재진술: 지문의 모든 문장을 같은 뜻의 다른 문장으로 새로 쓴다(원문 4단어 표현이 10~30%만 남게, 내용·논리·정보 순서는 원문과 같게). 원문을 외운 학생도 다시 읽어야 풀리게 한다.")
    if q_type in QB_CHOICE_TYPES:
        for pl in plans:
            if school == "제일고":
                n = 0 if (level == "하" and rnd.random() < 0.5) else rnd.choice([1, 2] if hard else [1])
            else:
                n = 1 if (level != "하" and rnd.random() < 0.5) else 0
            if n:
                pl.append(QB_OPPOSITE_TRAP.format(n=n))
    # 헷갈림 장치: 상·최상(중은 가끔)에서 유형에 맞는 장치 하나 + 해설에 '헷갈리는 포인트' 한 줄
    grp = QB_FEEL_DEVICES.get(QB_FEEL_GROUP.get(q_type, ""))
    if grp and level in ("상", "최상", "중"):
        for pl in plans:
            if rnd.random() < QB_FEEL_RATE.get(level, 0):
                pl.append(rnd.choice(grp))
                pl.append(QB_FEEL_EXPLAIN)
    # 정답 번호와 함정 배치 (2026-11-04 측정: 실제 기출은 정답이 ①~⑤에 고르게, 원문 단어를 쓴 가장 그럴듯한 오답이 정답보다 앞 번호인 경우가 운양고 62%·제일고 50%.
    #   AI에 맡기면 정답이 ③에 몰리고(제일고 상 65%) ⑤는 거의 안 나와서 파이썬이 정해 줌)
    if q_type in QB_CHOICE_TYPES or q_type in QB_POS_TYPES:
        trap_rate = QB_TRAP_RATE.get(level, 0.5)
        used = []
        for i, pl in enumerate(plans):
            use_trap = q_type in QB_CHOICE_TYPES and rnd.random() < trap_rate
            if use_trap:
                # 실제 내신 방식: 낚이는 선지를 앞(①·②)에, 정답은 뒤(③·④, 가끔 ⑤)에 완전히 다른 말로
                pos = rnd.choices(QB_TRAP_ANS_POS, weights=QB_TRAP_ANS_W)[0]
            else:
                # 함정 문항이 정답을 ③·④에 두므로, 나머지는 ①·②·⑤ 쪽으로 기울여 세트 전체 정답이 고르게
                free = [p for p in range(5) if p not in used] or list(range(5))
                pos = rnd.choices(free, weights=[2 if p in (0, 1, 4) else 1 for p in free])[0]
            used.append(pos)
            if q_type in QB_CHOICE_TYPES:
                line = f"정답 번호: {CIRCLED[pos]} (choices에서 정답을 {pos + 1}번째에 둔다)"
                if use_trap:
                    tp = rnd.choice([t for t in (0, 1) if t < pos])
                    line += (f". 함정 배치(실제 내신 방식): {CIRCLED[tp]}에는 지문의 핵심 단어·표현을 그대로 여러 개 가져와 언뜻 정답처럼 보이지만 "
                             f"범위·초점·논리(주장 뒤집기·부분만 맞음 등)가 어긋난 '낚는 선지'를 둔다. 정답 {CIRCLED[pos]}는 지문 단어를 거의 쓰지 않고 "
                             f"완전히 다른 말(상위 개념어·동의어·구조 변경)로 재진술해서, 지문 단어가 보이는 {CIRCLED[tp]}보다 덜 정답처럼 보이게 한다. "
                             f"앞에서부터 읽던 학생이 {CIRCLED[tp]}에서 멈추고 고르게 만드는 것이 목표다. 단, {CIRCLED[tp]}에도 지문 근거로 분명히 틀린 곳이 있어야 한다(복수 정답 금지)")
            else:
                line = f"정답 번호: {CIRCLED[pos]} (지문 속 ①~⑤ 중 {CIRCLED[pos]} 자리가 정답이 되게 만든다)"
            pl.append(line)
    if q_type in QB_NOCOPY_TYPES or (q_type == "빈칸" and school == "운양고"):
        for pl in plans:
            if any(x.startswith("지문 전면 재진술") for x in pl):
                continue  # 지문을 새로 쓰는 문항은 지금처럼 전부 씀
            if q_type == "함축 의미":
                pl.append(QB_NOCOPY_MARK + "(프로그램이 원문 지문을 넣음). target에 밑줄 칠 비유적 어구를 원문에서 글자 하나 다르지 않게 그대로 옮겨 적는다(프로그램이 그 어구에 밑줄을 그음).")
            elif q_type == "빈칸":
                pl.append(QB_NOCOPY_MARK + "(프로그램이 원문 지문을 넣음). target에 빈칸으로 비울 원문 어구를 원문에서 글자 하나 다르지 않게 그대로 옮겨 적는다(프로그램이 그 자리를 __________로 바꿈).")
            else:
                pl.append(QB_NOCOPY_MARK + " — 프로그램이 원문 지문을 그대로 넣으므로 지문을 옮겨 쓰지 않는다. target은 빈 문자열.")
    if q_type in QB_ONE_CHOICE_TYPES:
        # 2026-11-05: 운양고 내용 일치 최상이 <보기> 조합형과 섞여 나와서 (발문 두 개, 조합 + 문장 선지) 형식을 못 박음
        for pl in plans:
            pl.append(QB_ONE_CHOICE_RULE)
    return plans

# <보기>·ⓐ~ⓗ 조합형이 아닌 객관식 유형 (선택지 5개가 각각 한 문장)
QB_ONE_CHOICE_TYPES = {"주제", "제목", "요지·주장", "내용 일치·불일치", "빈칸", "함축 의미", "추론 (추론 가능·불가)"}
QB_ONE_CHOICE_RULE = "형식: 선택지 5개가 각각 완결된 한 문장(또는 어구)인 일반 객관식. box는 비우고 <보기>·ⓐ~ⓗ 진술이나 'ⓐ, ⓒ' 같은 조합 선택지는 쓰지 않는다(조합형은 '<보기> 일치 조합형' 유형에서만)."
_QB_COMBO_RE = re.compile(r"^\s*([ⓐ-ⓩ](?:(?:\s*(?:,|and|및|와|과)\s*)+[ⓐ-ⓩ])*)(.*)$", re.S)

def qb_fix_combo(q: Dict, q_type: str):
    """일반 객관식 유형인데 <보기> 조합형으로 나온 문항을 '<보기> 일치 조합형'으로 정리 (AI 호출 없음).
    → None: 조합형 아님(그대로) / '<보기> 일치 조합형': 정리함 / False: 정답이 하나로 안 정해져 못 씀"""
    if q_type not in QB_ONE_CHOICE_TYPES:
        return None
    ch = [re.sub(r"^\s*[①②③④⑤]\s*", "", str(c)) for c in (q.get("choices") or [])]
    ms = [_QB_COMBO_RE.match(c) for c in ch]
    if len(ch) != 5 or sum(bool(m) for m in ms) < 4:
        return None
    if not all(ms) or len(set(re.findall(r"[ⓐ-ⓩ]", str(q.get("box", ""))))) < 3:
        return False
    combos = [", ".join(re.findall(r"[ⓐ-ⓩ]", m.group(1))) for m in ms]
    if len(set(combos)) != 5:
        return False  # 조합이 겹치면 뒤에 붙은 문장으로만 갈리던 문항 → 정답이 하나로 안 정해짐
    stem = str(q.get("stem", ""))
    if not (re.search(r"있는 대로|모두 고른", stem) and "<보기>" in stem and stem.count("?") == 1 and stem.rstrip().endswith("?")):
        neg = bool(re.search(r"일치하지 않|않은 것|않는 것", re.sub(r"<[^>]+>", "", stem)))
        who = "(A)와 (B)의 내용과" if "(A)와 (B)" in stem else "윗글의 내용과"
        stem = f"{who} {'일치하지 않는' if neg else '일치하는'} 것만을 <보기>에서 있는 대로 고른 것은?"
    q["stem"], q["choices"] = stem, combos
    q["format_fix"] = "보기 조합형으로 정리"
    return "<보기> 일치 조합형"

def _qb_find_span(body: str, target: str):
    """원문에서 target 어구 위치 (띄어쓰기·대소문자 차이는 봐줌) → (시작, 끝) 또는 None"""
    t = re.sub(r"<[^>]+>", "", str(target or "")).strip().strip("\"'“”‘’")
    if len(t) < 2:
        return None
    i = body.find(t)
    if i >= 0:
        return i, i + len(t)
    m = re.search(r"\s+".join(re.escape(w) for w in t.split()), body, re.I)
    return m.span() if m else None

def qb_fill_passage(q: Dict, q_type: str, text: str, plan: List[str]) -> bool:
    """'원문'만 써 온 지문을 원문으로 채움 (함축은 밑줄, 빈칸은 빈칸 표시). 어구를 원문에서 못 찾으면 False(그 문항은 버림)"""
    tgt = q.pop("target", "")
    if not any(str(x).startswith(QB_NOCOPY_MARK) for x in plan or []):
        return True
    if len(re.sub(r"\s+", "", str(q.get("passage", "")))) > 60:
        return True  # AI가 그래도 지문을 써 왔으면 그대로 씀
    body = str(text or "").strip()
    if q_type in ("함축 의미", "빈칸"):
        span = _qb_find_span(body, tgt)
        if not span:
            return False
        a, b = span
        body = body[:a] + (f"<u>{body[a:b]}</u>" if q_type == "함축 의미" else "__________") + body[b:]
    q["passage"] = body
    return True

# 어법 조합형은 난이도마다 밑줄 개수·틀린 개수·묻는 방향을 따로 정함 (최상은 실제 내신처럼 '있는 대로 고른 것')
# 실제 운양고(29회)·제일고(25회) 기출 어법 선택지 전수 정리(PQRS영쌤 '어법·어휘 출제 총정리')에서 뽑은 개념별 빈도와 함정 패턴
QB_GRAMMAR_TYPES = {"어법 (조합형)", "어법 (단일)", "서술형 (어법 수정)", "서술형 (조건 영작)"}
QB_GRAMMAR_DNA = {
    "운양고": """- 형식: 'ⓐ~ⓕ(ⓖ) 중 어법상 틀린/옳은 것만 모두 고르기'가 대부분(단어·구 밑줄과 문장 단위 밑줄 모두), 다음으로 ①~⑤ 단일, (A)(B)(C) 네모 어법. 서술형은 '틀린 두 곳 찾아 고치기'·<보기> 배열 영작.
- 개념 빈도(524개 선택지): 수일치 66 > 분사·분사구문 58 > 관계대명사 48 > 능동/수동 44 > 5형식 목적격보어 37 > 동명사 34 > 명사절·동격 that 30 > 병렬 29 > 시제·가정법·조동사 26 > 형용사/부사·2형식 보어 23 > 관계부사 vs 관계대명사 21 > 도치·강조 17 > 대명사 지칭 16 > 가주어·가목적어 13 > 비교 10 > to부정사 9 > 한정사 9 > 접속사 vs 전치사 7 > 복합관계사 6.""",
    "제일고": """- 형식: '①~⑤ 중 어법상 틀린(옳은) 것 1개'와 문장 단위 밑줄이 많고, 'ⓐ~ⓗ 중 옳은/틀린 것만을 있는 대로', '옳은 표현의 개수' 형식도 출제. 서술형은 '틀린 문장 찾아 고쳐 쓰기'·조건 영작(관계대명사·가정법·비교 구문 지정).
- 개념 빈도(488개 선택지): 분사·분사구문 82 > 관계대명사 46 > 명사절·동격 that 35 > 능동/수동 34 > 동명사 33 > 시제·가정법·조동사 30 > 병렬 27 > 수일치 25 > 형용사/부사 25 > 대명사 지칭 16 > 접속사 vs 전치사 16 > to부정사 16 > 5형식 목적격보어 15 > 관계부사 14 > 도치·강조 14 > 가주어·가목적어 13 > 동사 자리 vs 준동사 11 > 자동사/타동사 9.""",
}
QB_GRAMMAR_TRAPS = """- 수일치: 긴 수식어·삽입구 뒤 핵심 주어, 동명사구·명사절 주어는 단수, one of + 복수명사 → 단수, a number of → 복수, 관계절 동사는 선행사에, 계속적 용법 which(앞 절)는 단수, 부분 표현 + of + 명사, each/every + 단수, the + 국적형용사 복수, 도치 구문의 뒤 주어.
- 관계사: 선행사가 있으면 what 불가(최다 함정), 계속적 용법 what 불가, 전치사 + that 불가, 뒤가 완전한 절이면 관계부사/전치사 + 관계대명사, 관계부사 뒤가 불완전하면 which/that, 사람 선행사 + which 오류, 주격 관계대명사 생략 불가.
- 명사절: fact/belief/claim/sign/evidence 뒤 동격 that ↔ which 오류, 완전한 절 that ↔ 불완전한 절 what, 간접의문문 어순.
- 분사: 감정 분사(satisfied/pleased/excited/interested/surprised ↔ -ing), 분사구문 능동·수동(Compared to, First expressed, Considered, Loaded), with + O + 분사, 접속사 없이 두 동사 → 분사구문, 명사 후치 수식 분사.
- 능동/수동: 자동사(happen, emerge, appear) 수동 불가, 목적어가 있으면 능동, be believed/expected to V, 수동 부정사 to be p.p.
- 5형식: help + O + (to) V, make/let/have + O + 원형, allow/force/order/expect/inspire/prompt/beg/compel + O + to V, 지각동사 + O + 원형/-ing, make + O + 형용사(부사 오류).
- 동명사: 전치사 + 동명사(instead of, by, without, devoted to, when it comes to), end up/keep/cannot help/stop -ing, be used to -ing ↔ used to V.
- 형용사/부사: 2형식 보어(feel/seem/look + 형용사), 과거분사·형용사를 꾸미는 부사, 동사 수식 부사.
- 병렬: and/or/but 앞뒤 시제·형태 일치, not only A but also B.
- 시제·가정법: If 생략 도치(Had you not ~), 가정법 과거완료 would have p.p., 과거보다 앞선 일 과거완료, 조동사 + 원형, 조건 부사절 현재시제.
- 도치·강조: 부정어(Not only/Rarely/Seldom/Only then) 문두 조동사 도치, So + 형용사 문두, Here + V + S, It ~ that 강조의 수일치.
- 기타: 가목적어 it 누락(make it possible for A to V), 대동사 do/be 구분, such a + 형용사 + 명사, every + 단수명사, -thing + 형용사, 접속사 vs 전치사(despite/due to/during + 명사 ↔ although/because/while + 절).
- 정답 구성: 실제 기출처럼 옳은 밑줄에도 위 포인트를 그대로 써서(맞게 쓴 함정) 틀린 것과 구별이 어렵게 하고, 원문을 고치지 않아도 되는 곳은 원문 그대로 둔다."""
QB_VOCAB_TRAPS = """- 운양고·제일고 낱말 문제는 대부분 원문 낱말을 반의어로 바꿈(rarely→frequently, disproves→proves, prevent→facilitate, decrease→increase, shrink→improve).
- 바로 뒤·앞 문장과 모순되게 바꾸는 방식이 가장 많음(예: 'The few times that they do occur' 앞의 frequently).
- 형태가 비슷한 혼동어(conservation↔conversion, adherence↔withdrawal, elicit↔eliminate, beneficence↔maleficence, interpersonal↔intrapersonal)도 자주 씀.
- ⓐ~ⓘ 중 '적절하지 않은 것만 모두 고르기'(2~3개), 영영풀이 일치/불일치, 네모 (A)(B)(C) 어휘 고르기도 출제."""

QB_GRAMMAR_LEVEL = {
    "하": "밑줄 5개(ⓐ~ⓔ), 틀린 것 1~2개. 수 일치·시제·품사(형용사/부사)·동사원형 같은 기본 포인트. '틀린 것만을 고른 것'.",
    "중": "밑줄 6개(ⓐ~ⓕ), 틀린 것 2~3개. 관계사·분사·to부정사/동명사 같은 빈출 포인트 섞기.",
    "상": "밑줄 7개(ⓐ~ⓖ), 틀린 것 2~4개. 빈출 포인트 + 고난도 1~2개. '틀린 것만을' 또는 '옳은 것만을' 중 하나.",
    "최상": "지문 전체에 고르게 밑줄 8개(ⓐ~ⓗ). stem은 '어법상 옳은 것만을 있는 대로 고른 것은?' 또는 '틀린 것만을 있는 대로 고른 것은?' 중 하나(문항마다 섞기). "
            "틀린 것 개수는 3~5개로 정해 개수로 답을 추측할 수 없게 하고, choices 5개는 2개·3개·4개·5개짜리 조합을 섞는다. "
            "포인트는 완료부정사, 동격 that vs 관계대명사, 관계부사 vs 관계대명사, 분사구문의 태, 도치 후 수 일치, 가정법, 병렬, 강조구문, 수동태 불가 동사 등 고난도 위주로 하고 "
            "바로 옆 단어만 보면 맞아 보이는 함정을 넣는다.",
}

SCHEMA_QBANK = {
    "type": "object",
    "properties": {"questions": {"type": "array", "items": {"type": "object", "properties": {
        "focus": {"type": "string", "description": "이 문항이 묻는 핵심 한 줄 (예: 'ⓑ 완료부정사 to have p.p. / ⓖ 동격 that')"},
        "stem": {"type": "string", "description": "발문"},
        "given": {"type": "string", "description": "주어진 글·문장 (순서·삽입 유형), 없으면 빈 문자열"},
        "passage": {"type": "string", "description": "문제용으로 표시한 지문. 단락은 줄바꿈"},
        "target": {"type": "string", "description": "[문항별 출제 계획]에서 쓰라고 한 경우만: 밑줄·빈칸으로 바꿀 원문 어구(글자 그대로). 아니면 빈 문자열"},
        "box": {"type": "string", "description": "<보기>·요약문·<조건> 박스 내용, 없으면 빈 문자열"},
        "choices": {"type": "array", "items": {"type": "string"}, "description": "선택지 5개 (번호 기호 없이), 서술형은 빈 배열"},
        "answer": {"type": "string", "description": "객관식은 ①~⑤ 중 하나, 서술형은 모범 답안"},
        "explanation": {"type": "array", "items": {"type": "string"}, "description": "한국어 해설 2~4줄"},
    }, "required": ["focus", "stem", "given", "passage", "target", "box", "choices", "answer", "explanation"]}}},
    "required": ["questions"],
}
SCHEMA_QB_SOLVE = {
    "type": "object",
    "properties": {"answers": {"type": "array", "items": {"type": "object", "properties": {
        "index": {"type": "integer"}, "answer": {"type": "string"}}, "required": ["index", "answer"]}}},
    "required": ["answers"],
}
# 상·최상을 만들 때는 정답 검증과 같은 호출에서 난이도도 판정 (추가 호출 없음, 문항당 출력 약 30토큰)
SCHEMA_QB_SOLVE_LEVEL = {
    "type": "object",
    "properties": {"answers": {"type": "array", "items": {"type": "object", "properties": {
        "index": {"type": "integer"}, "answer": {"type": "string"},
        "level": {"type": "string", "enum": ["최상", "상", "중"]},
        "why": {"type": "string", "description": "판정 이유 40자 이내"}}, "required": ["index", "answer", "level", "why"]}}},
    "required": ["answers"],
}
QB_LEVEL_JUDGE = """[난이도 판정 — 문제마다 level과 why]
학생은 시험 범위 원문을 완벽히 외우고 온 상위권 고등학생이다. 참고용 원문과 문제를 비교해 판정한다.
- '중': 아래 중 하나라도 해당 — (1) 원문 기억만으로 풀린다(원문을 그대로 자르거나 표시만 한 순서·삽입, 정답 위치·순서가 원문과 같고 그 주변도 원문 그대로인 경우) (2) 정답 선지가 원문 단어·표현을 거의 그대로 옮겼다(뜻만 같은 상위어·구조 변경 재진술이면 해당하지 않음) (3) 지문을 안 읽고도 지울 수 있는 오답이 2개 이상이다.
- '상': 위에 해당하지 않고, 지문을 다시 읽고 따져야 풀린다.
- '최상': 상위권 학생도 두 선지 사이에서 실제로 고민하거나 실수할 구조일 때만 — 지문 단어를 그대로 쓴 그럴듯한 오답 vs 낯설게 재진술한 정답, 한쪽 칸만 틀린 요약문 쌍, 부분적으로 맞는 빈칸 오답 2개 이상, 단어만 겹치는 가짜 삽입 자리, 외운 원문 순서가 오답인 재구성 순서, 맞게 쓴 함정이 섞인 문장 단위 '옳은 것만' 어법, 단락 논리로만 보이는 낱말 교체, 어휘 때문에 두 선지 사이에서 고민하게 되는 구조 등. 지문을 안 읽고 지울 수 있는 오답이 하나라도 있거나, 기본 빈출(are/is, hold/holding)만으로 된 어법 조합형이면 최상이 아니다(상).
- 어법·낱말 문제에는 위 (1)(2)를 적용하지 않는다(실제 시험도 원문을 바꿔 출제함). 대신 어법은 문장 단위 밑줄·맞게 쓴 함정(원문 그대로인데 틀려 보이는 옳은 문장)·바로 옆 단어만 보면 맞아 보이는 오류 자리·고난도 개념이 갖춰지면 '최상', 기본 빈출 위주면 '상', 오류가 한눈에 보이면 '중'. 낱말은 단락 논리를 따라야 보이는 교체면 '최상', 바로 앞뒤 문장과 모순이면 '상'.
- 엄격하게 판정한다. 애매하면 한 단계 낮게 준다."""
QB_LEVEL_ORDER = {"하": 0, "중": 1, "상": 2, "최상": 3}
CIRCLED = "①②③④⑤"

def qb_norm_answer(a: str) -> str:
    """'③', '3', '정답 ③' → '③' (못 찾으면 빈 문자열)"""
    s = str(a or "")
    m = re.search(r"[①②③④⑤]", s)
    if m:
        return m.group(0)
    m = re.search(r"(?<!\d)([1-5])(?!\d)", s)  # '3번'도 읽음 (\b는 한글 앞에서 끊기지 않아 '3번'을 못 읽었음)
    return CIRCLED[int(m.group(1)) - 1] if m else ""

# 순서 유형 선택지는 수능과 같은 고정 배열 ((A)로 시작하는 선지는 ① 하나, (A)-(B)-(C)는 없음)
QB_ORDER_CANON = ["ACB", "BAC", "BCA", "CAB", "CBA"]

def qb_fix_order(q: Dict, rnd=None, only_if_needed: bool = False) -> bool:
    """순서 문항의 덩어리 이름 (A)(B)(C)를 바꿔 달아 선택지를 수능식 고정 배열로 맞춤 (AI 호출 없음).
    정답 번호는 rnd로 고르게 정함. 해설에 번호·순서로 언급된 오답이 사라지지 않게 고름.
    only_if_needed=True면 이미 고정 배열인 문항은 그대로 둠. 형식을 못 읽으면 바꾸지 않고 False."""
    seqs = ["".join(re.findall(r"\(([ABC])\)", str(c))) for c in (q.get("choices") or [])]
    ans = str(q.get("answer", ""))
    if len(seqs) != 5 or len(set(seqs)) != 5 or any(sorted(s) != ["A", "B", "C"] for s in seqs) or len(ans) != 1 or ans not in CIRCLED:
        return False
    if only_if_needed and seqs == QB_ORDER_CANON:
        return True
    passage = str(q.get("passage", ""))
    parts = re.split(r"\(([ABC])\)[ \t]*", passage)
    labels = parts[1::2]
    if sorted(labels) != ["A", "B", "C"]:
        return False
    prefix, blocks = parts[0], {labels[i]: parts[2 + 2 * i].strip() for i in range(3)}
    expl = q.get("explanation")
    exp_text = " ".join(expl) if isinstance(expl, list) else str(expl or "")
    correct = seqs[CIRCLED.index(ans)]
    rnd = rnd or random.Random()
    cands = list(range(5))
    rnd.shuffle(cands)
    for p in cands:
        m = dict(zip(correct, QB_ORDER_CANON[p]))
        mapped = ["".join(m[ch] for ch in s) for s in seqs]
        lost = [j for j, s in enumerate(mapped) if s == "ABC"]
        if lost and (CIRCLED[lost[0]] in exp_text
                     or re.search(r"\(" + r"\)\s*[-–→]\s*\(".join(seqs[lost[0]]) + r"\)", exp_text)):
            continue  # 해설이 가리키는 오답이 없어지는 자리는 피함 (원래 빠졌던 순서가 (A)-(B)-(C)가 되는 자리는 항상 가능)
        break
    num = {CIRCLED[j]: CIRCLED[QB_ORDER_CANON.index(s)] for j, s in enumerate(mapped) if s in QB_ORDER_CANON}

    def relabel(s: str) -> str:
        s = re.sub(r"\(([ABC])\)", lambda x: f"({m[x.group(1)]})", str(s))
        return re.sub("[①②③④⑤]", lambda x: num.get(x.group(0), x.group(0)), s)

    new_blocks = {m[k]: v for k, v in blocks.items()}
    q["passage"] = (prefix.rstrip() + "\n" if prefix.strip() else "") + "\n".join(f"({k}) {new_blocks[k]}" for k in "ABC")
    q["choices"] = [f"({s[0]}) - ({s[1]}) - ({s[2]})" for s in QB_ORDER_CANON]
    q["answer"] = CIRCLED[p]
    if isinstance(expl, list):
        q["explanation"] = [relabel(e) for e in expl]
    elif expl:
        q["explanation"] = relabel(expl)
    if q.get("given"):
        q["given"] = re.sub(r"\(([ABC])\)", lambda x: f"({m[x.group(1)]})", str(q["given"]))
    return True

def _qb_call(gclient, model: str, prompt, schema: Dict, temperature: float, logs: List[str], what: str, cache_first: bool = False):
    for attempt in range(4):
        try:
            return llm_json(model, prompt, schema, temperature, gclient, cache_first=cache_first)
        except Exception as e:
            err_s = str(e).lower()
            if attempt < 3 and any(k in err_s for k in ["503", "high demand", "429", "unavailable", "quota", "overloaded", "timeout", "timed out", "connection", "500", "internal"]):
                time.sleep(15 + attempt * 15)
                continue
            logs.append(f"❌ {what}: {str(e)[:150]}")
            return None
    return None

def _qb_clean_dna(text: str) -> str:
    """출제 DNA 앞에 실수로 붙어 저장된 JSON 스키마 덩어리는 빼고 보냄 (토큰만 먹고 쓸모없음)"""
    return re.sub(r"^\s*\{[\s\S]*?\n\}\s*\n", "", str(text or "")).strip()

# ---- 영쌤 출제 포인트: 선생님이 계속 덧붙이는 '낚시' 출제 방식 (학교와 관계없이 적용) ----
#   출제 DNA 문서(exam_dna_rules)의 'yt_points' 목록에 저장. 이번 출제 부분(문항별 계획)에만 넣어서 공통 앞부분 캐시는 그대로.
QB_YT_SCOPES = {
    "전체 유형": None,
    "선택지형 (주제·제목·요지·일치·빈칸·요약·추론)": QB_CHOICE_TYPES,
    "일치·추론 (내용 일치·<보기>·추론)": {"내용 일치·불일치", "<보기> 일치 조합형", "추론 (추론 가능·불가)"},
    "주제·제목·요지": {"주제", "제목", "요지·주장"},
    "빈칸·요약문": {"빈칸", "요약문 (A)(B)"},
    "순서": {"순서"},
    "삽입": {"삽입"},
    "어법": QB_GRAMMAR_TYPES,
    "어법·낱말": QB_GRAMMAR_TYPES | {"낱말 (문맥상)"},
}
QB_YT_RATE = 0.5          # 평소: 문항마다 이 확률로 영쌤 포인트 하나 적용 ('영쌤 포인트 위주'를 켜면 모든 문항)
QB_YT_UNIT_MAX = 3000     # 교과서 같은 단원 다른 단락을 붙일 때 최대 글자 수
DEFAULT_YT_POINTS = [
    {"on": True, "scope": "선택지형 (주제·제목·요지·일치·빈칸·요약·추론)", "tb": False,
     "text": "①·② 선지는 본문에 나온 단어·구문을 그대로 많이 써서 정답처럼 보이게 만든 오답(비슷하게 생긴 단어로 살짝 바꾸거나 범위·논리를 어긋나게). "
             "정답은 ③·④·⑤ 중에 두고 본문 단어를 거의 안 쓴 완전한 패러프레이즈로 써서, 본문 문장을 외운 학생이 앞 번호의 익숙한 구문을 고르고 정답을 놓치게 한다. "
             "(예: 정답 we're influenced by sheer convenience → 낚는 선지 we are influenced by sheer coincidence)"},
    {"on": True, "scope": "선택지형 (주제·제목·요지·일치·빈칸·요약·추론)", "tb": False,
     "text": "선지 문장은 거의 다 본문 내용과 일치하고 단어도 본문 단어를 쓰되, 문장 후반부의 어려운 패러프레이즈 단어 하나만 뜻이 틀리게 바꿔 놓는다. "
             "앞부분만 보고 맞다고 판단한 학생이 일치·불일치나 주제·요지에서 헷갈리게 한다."},
    {"on": True, "scope": "순서", "tb": False,
     "text": "학생은 본문을 외워서 글의 마지막 부분(숫자, 마지막 문장이 끝나는 방식)을 기억하고, 그 내용이 있는 덩어리를 맨 끝으로 보내는 식으로 푼다. "
             "이를 역이용해 본문 내용을 논리적으로 약간 고치거나 내용을 덧붙여서, 기억 속 마지막 덩어리가 정답 순서에서는 끝이 아니게 만든다(정답은 지시어·논리로만 확정)."},
    {"on": True, "scope": "일치·추론 (내용 일치·<보기>·추론)", "tb": True,
     "text": "교과서 지문: 같은 단원의 다른 단락에는 있지만 이번 문제 지문(단락)에는 없는 내용으로 선지를 하나 만든다. "
             "교과서 전체를 공부한 학생이 '본 내용'이라 맞다고 고르게 하는 오답(지문에 없으므로 일치하지 않음/추론 불가)."},
    {"on": True, "scope": "어법", "tb": False,
     "text": "'있는 대로 고른 것' 조합형이면: 선지 하나는 정답 조합에 하나를 더 얹은 것으로 둔다(예: 정답 ⓐ, ⓓ, ⓔ / 오답 선지 ⓐ, ⓒ, ⓓ, ⓔ). "
             "아는 것 몇 개만으로는 못 풀고 밑줄 하나하나를 모두 정확히 판단해야 풀리게 한다."},
    {"on": True, "scope": "어법", "tb": False,
     "text": "어법 함정 장치(이 중 지문에 맞는 것을 골라 씀): 전치사+관계대명사에서 전치사를 절 뒤로 보내기(the house in which he lives → the house which he lives in, "
             "또는 전치사를 빼서 틀리게), enough 어순(old enough to ↔ enough old to), 후치수식해야 할 형용사(구)를 명사 앞에 두기(something important → important something, "
             "a box full of toys → a full of toys box), supply A with B의 A·B 자리 바꾸기(supply them with food ↔ supply food with them, provide·present·furnish도 같음)."},
    {"on": True, "scope": "어법", "tb": False,
     "text": "어법 함정 — 사소한 수일치 실수 유도(이 중 지문에 맞는 것 1~2개): one of 뒤 복수명사 자리에 단수명사(one of the reasons → one of the reason), "
             "every·each 뒤 단수명사 자리에 복수명사(every student → every students), 소유격·대명사의 수일치(가리키는 명사가 복수인데 their → its, them → it, "
             "또는 그 반대). 긴 수식어 때문에 가리키는 명사가 멀리 떨어진 자리를 고른다."},
    {"on": True, "scope": "어법", "tb": False,
     "text": "어법 함정 — 태·분사(이 중 지문에 맞는 것 1~2개): 현재완료 수동 ↔ 능동 바꾸기(has been reduced ↔ has reduced — 주어가 행위자인지 대상인지로 판단), "
             "대표 자동사를 수동으로(emerge·happen·occur·appear·disappear·arise·exist·consist of·result in → was emerged, is occurred 등), "
             "관용 분사 형태 바꾸기(compared to → comparing to, given that·provided that·considering 등도 같은 방식)."},
    {"on": True, "scope": "어법", "tb": False,
     "text": "어법 함정 — 비교급 강조·수량 형용사(이 중 지문에 맞는 것 1~2개): 비교급 강조는 much·still·far·even·a lot인데 very·more 등으로 바꾸기"
             "(much better → very better), 셀 수 없는 명사에 쓰는 much·little ↔ 셀 수 있는 복수명사에 쓰는 many·few 바꾸기(little information → few information, many people → much people)."},
    {"on": True, "scope": "어법", "tb": False,
     "text": "어법 함정 — 동사의 목적어·목적격보어(이 중 지문에 맞는 것 1~2개): 동명사만 목적어로 취하는 동사(enjoy·avoid·mind·finish·consider·deny·suggest·give up) / "
             "to부정사만 취하는 동사(want·decide·hope·plan·refuse·agree·expect·manage) / 둘 다 되는 동사(begin·start·like, remember·forget·stop은 뜻이 달라짐)의 목적어 형태 바꾸기, "
             "5형식 목적격보어 바꾸기(want·ask·allow·encourage·enable·force + O + to V ↔ 원형·-ing, 사역 make·let·have + O + 원형 ↔ to V, 지각동사 + O + 원형·분사, "
             "keep·find·leave + O + 형용사·분사)."},
    {"on": True, "scope": "어법", "tb": False,
     "text": "어법 함정 — 품사·가목적어·대동사(이 중 지문에 맞는 것 1~2개): 준동사(to V·동명사·분사)를 꾸미는 부사를 형용사로(to respond quick, by carefully reading → careful), "
             "형용사 보어 자리(seem·remain·become·stay·sound, 5형식 find·keep·make O 형용사)를 부사로(remain silently, found it easily), "
             "5형식 가목적어 it을 빼거나 다른 대명사로(make it possible to → make possible to / make this possible to), "
             "대동사를 앞 동사 종류·시제와 다르게(앞이 일반동사인데 is, 과거인데 does)."},
    {"on": True, "scope": "어법", "tb": False,
     "text": "어법 함정 — 문장 구조(이 중 지문에 맞는 것 1~2개): 병렬 구조에서 and·or·but 뒤 동사를 앞 동사 시제와 맞추지 않고 동명사·분사로(collected and analyzing), "
             "명령문이나 주절의 본동사를 동명사·분사로 바꿔 본동사를 없애기(Imagine → Imagining ~, 문장에 동사가 없게), "
             "가정법 if 생략 도치 바꾸기(Had I known → I had known / If had I known, Were it not for, Should you need), "
             "간접의문문 어순을 의문문 어순으로(I wonder what he wants → what does he want, how much it costs → how much does it cost)."},
    {"on": True, "scope": "어법", "tb": False,
     "text": "어법 함정 — 관계사(이 중 지문에 맞는 것 1~2개): 관계대명사 ↔ 관계부사(뒤가 불완전한데 where·when / 완전한데 which), "
             "전치사+관계대명사 ↔ 관계대명사만(in which → which, 또는 전치사를 다른 것으로), 접속사 that ↔ 선행사를 포함한 what(선행사 유무·뒤 절의 완전성으로 판단), "
             "사람 선행사의 who를 which로(또는 사물 선행사에 who), 콤마 뒤 that."},
]

def qb_is_yt(r: Dict) -> bool:
    """영쌤 출제 포인트로 낸 문항인지 (예전에 만든 문항은 출제 계획 기록으로 판단)"""
    c = r.get("content") or {}
    return bool(c.get("yt")) or any(str(x).startswith("영쌤 출제 포인트") for x in (c.get("plan") or []))

QB_YT_FILTER = ["섞기", "🎣 영쌤 포인트 문제만", "영쌤 포인트 빼기"]

def qb_yt_keep(r: Dict, mode: str) -> bool:
    return mode == QB_YT_FILTER[0] or (qb_is_yt(r) == (mode == QB_YT_FILTER[1]))

def qb_yt_points(dna: Dict = None) -> List[Dict]:
    """저장된 영쌤 출제 포인트 (처음이면 기본 4개)"""
    dna = dna if dna is not None else (st.session_state.get("exam_dna") or {})
    pts = dna.get("yt_points")
    return [dict(p) for p in (DEFAULT_YT_POINTS if pts is None else pts) if str(p.get("text", "")).strip()]

def qb_yt_applicable(points: List[Dict], q_type: str, textbook: bool) -> List[Dict]:
    out = []
    for p in points or []:
        scope = QB_YT_SCOPES.get(p.get("scope"), None) if p.get("scope") in QB_YT_SCOPES else None
        if scope and str(p.get("text", "")).lstrip().startswith(("어법 함정", "'있는 대로")):
            scope = scope & QB_GRAMMAR_TYPES  # 어법 함정은 낱말(문맥상) 문제에 붙이지 않음 (붙이면 어법 문제로 나옴)
        if p.get("on", True) and (scope is None or q_type in scope) and (textbook or not p.get("tb")):
            out.append(p)
    return out

def qb_unit_other(p: Dict, lib: List[Dict]) -> str:
    """교과서 지문이면 같은 단원 다른 단락 원문 (영쌤 포인트 '교과서만' 용)"""
    parts = set(qb_link_parts(p.get("label", ""))[1]) | {p.get("label", "")}
    ek = p.get("exam_key", "")
    out, n = [], 0
    for x in lib:
        if x.get("source_type") == "textbook" and ek and x.get("exam_key") == ek and x.get("label") not in parts and x.get("body_text"):
            s = f"[{x['label']}] {str(x['body_text']).strip()}"
            if n + len(s) > QB_YT_UNIT_MAX:
                s = s[:max(0, QB_YT_UNIT_MAX - n)]
            out.append(s)
            n += len(s)
            if n >= QB_YT_UNIT_MAX:
                break
    return "\n".join(out)

def qb_yt_plans(plans: List[List[str]], points: List[Dict], focus: bool, rnd=None) -> bool:
    """문항별 계획에 영쌤 포인트를 하나씩 넣음 (문항마다 다른 포인트부터) → 하나라도 넣었으면 True"""
    if not points:
        return False
    rnd = rnd or random.Random()
    order = list(range(len(points)))
    rnd.shuffle(order)
    used = False
    k = 0
    for pl in plans:
        if focus or rnd.random() < QB_YT_RATE:
            p = points[order[k % len(order)]]
            k += 1
            pl.append("영쌤 출제 포인트(선생님 지시 · 최우선 — 위 계획의 정답 번호 등과 부딪히면 이 포인트를 따름): " + str(p["text"]).strip())
            used = True
    return used

def qb_build_prompt(label: str, text: str, q_type: str, level: str, school: str, count: int,
                    avoid: List[str], dna_rules: str, essay_rules: str, existing: List[Dict] = None, underline: str = "mix",
                    yt_points: List[Dict] = None, yt_focus: bool = False, unit_other: str = "", grade: str = ""):
    """→ (공통 앞부분, 이번 출제 부분, 문항별 계획). grade: '고1'·'고2' (비우면 지문 이름으로 판단)
    프롬프트 앞부분(출제 DNA + 공통 규칙)은 모든 호출에서 글자 하나까지 같게 두어 캐시로 읽히게 하고,
    학교·유형·난이도·지문처럼 호출마다 다른 내용은 뒤에 붙임"""
    avoid_str = ""
    if avoid:
        avoid_str = "\n[이미 만든 문제의 출제 포인트 — 같은 포인트·같은 정답 구성을 피하고 다른 문장·다른 포인트로 출제]\n" + "\n".join(f"- {a}" for a in avoid[-25:])
    extra = f"\n[서술형 출제 DNA]\n{essay_rules or ESSAY_DNA_FALLBACK}\n" if q_type in QB_ESSAY_TYPES else ""
    if q_type in QB_GRAMMAR_TYPES:
        extra += f"\n[{school} 실제 기출 어법 출제 DNA — 이 빈도와 함정 패턴을 반영해 밑줄 포인트를 고른다]\n{QB_GRAMMAR_DNA.get(school, '')}\n{QB_GRAMMAR_TRAPS}\n"
    elif q_type == "낱말 (문맥상)":
        extra += f"\n[운양고·제일고 실제 기출 어휘 출제 패턴]\n{QB_VOCAB_TRAPS}\n"
    if q_type in QB_CHOICE_TYPES:
        extra += f"\n[선택지 재진술 — 실제 기출 측정]\n- {QB_CHOICE_PARA}\n"
    extra += qb_link_note(label, text)
    plans = qb_question_plans(q_type, level, school, count, existing or [], underline)
    yt = qb_yt_applicable(yt_points, q_type, bool(unit_other)) if level != "하" else []
    if qb_yt_plans(plans, yt, yt_focus) and unit_other and any("영쌤 출제 포인트" in x and p.get("tb") and str(p["text"]).strip() in x
                                                             for pl in plans for x in pl for p in yt):
        extra += ("\n[같은 단원의 다른 단락 — 문제 지문이 아님. 영쌤 포인트(교과서)의 '이번 지문에는 없는 내용' 선지를 만들 때만 참고]\n"
                  + unit_other + "\n")
    if any(plans):
        extra += "\n[문항별 출제 계획 — 문항마다 다르게, 반드시 따름]\n" + "\n".join(
            f"- 문항 {i}: " + " / ".join(pl) for i, pl in enumerate(plans, 1) if pl) + "\n"
    shared = f"""너는 운양고·제일고 영어 내신 출제위원이다. 맨 아래 [이번 출제] 조건대로 변형문제를 만들어 JSON으로 반환하라.

[운양고·제일고 실제 내신 출제 DNA]
{_qb_clean_dna(dna_rules)}

[공통 규칙]
- passage는 원문을 바탕으로 이 유형에 필요한 표시만 넣는다(밑줄 <u>…</u>, ⓐ~ⓖ, ①~⑤, 빈칸 __________, (A)(B)(C)). [이번 출제]의 학교별 지문 변형 기준에서 허용한 만큼만 재진술하고 나머지는 원문 그대로 둔다. 어느 경우든 요약·생략하지 않고 원문의 모든 문장(정보)을 담는다. 단락은 줄바꿈으로 구분한다.
- 재진술한 지문이면 explanation 마지막 줄에 '지문 재진술: 원문 표현 → 바꾼 표현' 예시를 1~2개 적는다.
- 원문에 없는 문장을 새로 지어 붙이지 않는다('무관한 문장' 유형의 한 문장과 [문항별 출제 계획]에서 허용한 순서 재구성만 예외). 지문이 짧아 밑줄 개수를 다 채우기 어려우면 밑줄 개수를 줄인다.
- [문항별 출제 계획]이 있으면 문항마다 그 계획을 그대로 따르고, 계획이 이 규칙들과 다르면 계획을 따른다.
- 어법·낱말 문제의 '틀린 곳'은 passage에 틀린 표현이 실제로 적혀 있어야 한다(원문 그대로 두고 틀렸다고 하지 않는다). 오류는 꼭 필요한 곳에만 넣고, explanation에 'ⓐ are → is (주어가 동명사구라 단수)'처럼 passage에 쓴 틀린 표현 → 올바른 표현을 밝힌다.
- 객관식은 choices 5개(①② 같은 번호 기호는 쓰지 않음), answer는 ①~⑤ 중 하나. 여러 문항이면 정답 번호가 한쪽으로 몰리지 않게 하고, 문항마다 묻는 핵심(어법 개념·문장·논리)을 다르게 한다.
- 서술형은 choices를 빈 배열로 두고 answer에 모범 답안을 쓴다.
- 정답이 하나로 확실히 정해지는지 직접 풀어 확인하고, 애매하면 고쳐서 낸다.
- 만든 뒤 스스로 점검: 상·최상인데 상위권 고등학생이 한눈에 정답을 고를 수 있으면(정답만 혼자 튀거나 오답이 명백히 틀림) 기준에 맞게 다시 어렵게 고친다.
- explanation은 한국어 2~4줄: 근거(영어 원문 일부) + 오답 이유.
- 서식 태그는 <u>, <b>만 쓴다."""
    grade = grade or qb_passage_grade(label)
    len_rule = qb_choice_len_rule(school, grade, q_type, level) if q_type in QB_CHOICE_TYPES else ""
    task = f"""[이번 출제] {school}{(" " + grade) if grade else ""} 스타일 · [{q_type}] 유형 · 난이도 [{level}] · {count}문항

[{school} 출제 스타일]
{QB_SCHOOL_STYLE.get(school, '')}
{len_rule}

[난이도 {level} 기준]
{QB_LEVEL_RULES[level]}
{("- 이 유형의 " + level + " 기준: " + QB_TYPE_LEVEL[q_type][level]) if q_type in QB_TYPE_LEVEL else ""}
{("- " + school + " 지문 변형 기준(실제 기출 분석): " + QB_SCHOOL_PASSAGE.get(school, "")) if q_type in QB_PASSAGE_TYPES else ""}

[{q_type} 유형 형식]
{QB_TYPES[q_type]}{chr(10) + "[이 난이도의 어법 조합형 기준] " + QB_GRAMMAR_LEVEL[level] if q_type == "어법 (조합형)" else ""}
{extra}{avoid_str}

지문 정보: {label}
지문: \"\"\"{text}\"\"\""""
    return shared, task, plans

def qb_generate(gclient, model: str, label: str, text: str, q_type: str, level: str, school: str, count: int,
                avoid: List[str], dna_rules: str, essay_rules: str, logs: List[str], existing: List[Dict] = None, underline: str = "mix",
                yt_points: List[Dict] = None, yt_focus: bool = False, unit_other: str = "", grade: str = "") -> List[Dict]:
    """변형문제 count개 생성 (형식이 맞지 않는 문항은 버림)"""
    shared, task, plans = qb_build_prompt(label, text, q_type, level, school, count, avoid, dna_rules, essay_rules, existing, underline,
                                          yt_points, yt_focus, unit_other, grade)
    out = _qb_call(gclient, model, [shared, task], SCHEMA_QBANK, 0.7, logs, f"{label} {q_type} {level} 생성", cache_first=True)
    return qb_parse(out, q_type, count, plans, text)

def qb_shape_ok(q: Dict, q_type: str) -> bool:
    """풀 수 없는 모양의 문항 거르기 (AI 호출 없음): 지문 속 ①~⑤ 빠짐, 같은 선지 두 번, 조합 선지에 없는 기호, 유형과 다른 발문"""
    p, stem = str(q.get("passage") or ""), str(q.get("stem") or "")
    if q_type in QB_MARK_TYPES and any(m not in p for m in CIRCLED[:5]):
        return False
    ch = [str(c).strip() for c in q.get("choices") or []]
    if q_type not in QB_MARK_TYPES and len(set(ch)) < len(ch):
        return False
    if q_type in ("<보기> 일치 조합형", "어법 (조합형)"):
        where = str(q.get("box") or "") if q_type.startswith("<보기>") else p
        if set(re.findall("[ⓐ-ⓗ]", " ".join(ch))) - set(re.findall("[ⓐ-ⓗ]", where)):
            return False
    if (q_type == "낱말 (문맥상)" and "어법" in stem) or (q_type.startswith("어법") and "문맥" in stem):
        return False
    return True

def qb_parse(out, q_type: str, count: int, plans: List[List[str]], text: str) -> List[Dict]:
    """AI 답에서 형식이 맞는 문항만 골라 냄 ('원문'만 써 온 지문은 원문으로 채움)"""
    good = []
    if not isinstance(out, dict):
        return good  # 형식이 다른 답 (예: 목록) → 이 건은 실패로
    for qi, q in enumerate((out.get("questions") or [])[:count]):
        if not isinstance(q, dict):
            continue
        plan = plans[qi] if qi < len(plans) else []
        if not qb_fill_passage(q, q_type, text, plan):
            continue
        if find_degenerate_text(q):
            continue
        if q_type in QB_ESSAY_TYPES:
            q["choices"] = []
            if not str(q.get("answer", "")).strip():
                continue
        else:
            q["answer"] = qb_norm_answer(q.get("answer"))
            q["choices"] = [""] * 5 if q_type in QB_MARK_TYPES else [str(c).strip() for c in (q.get("choices") or [])][:5]
            if not q["answer"] or len(q["choices"]) != 5 or (q_type not in QB_MARK_TYPES and not all(q["choices"])):
                continue
            fx = qb_fix_combo(q, q_type)  # <보기> 조합형으로 섞여 나온 문항 정리 (정답이 애매하면 버림)
            if fx is False:
                continue
            if fx:
                q["_q_type"] = fx
            if q_type == "순서":
                qb_fix_order(q)  # 수능식 고정 선택지 + 정답 번호 고르게 (정답 검증 전에 바꿔서 검증도 바뀐 문항으로)
            if not qb_shape_ok(q, q.get("_q_type") or q_type):
                continue
        if qi < len(plans) and plans[qi]:
            q["plan"] = plans[qi]   # 어떤 장치로 냈는지 남김 (나중에 골고루 나왔는지 확인용)
            yt = [x.split("): ", 1)[-1] for x in plans[qi] if x.startswith("영쌤 출제 포인트")]
            if yt:
                q["yt"] = yt[0][:300]  # 영쌤 출제 포인트로 낸 문항 표시 (시험지 🎣 표시·고르기용)
        good.append(q)
    return good

def qb_verify(gclient, model: str, text: str, questions: List[Dict], logs: List[str], level: str = "") -> List[bool]:
    """객관식 문항을 AI가 따로 풀어 정답이 같은지 확인 (서술형은 그대로 통과).
    level이 상·최상이면 같은 호출에서 난이도도 판정해 문항에 q['_judged'] = (판정, 이유)로 남김 (추가 호출 없음)"""
    idx = [i for i, q in enumerate(questions) if q.get("choices")]
    if not idx:
        return [True] * len(questions)
    blocks = []
    for k, i in enumerate(idx, 1):
        q = questions[i]
        ch = "\n".join(f"{CIRCLED[j]} {c}" for j, c in enumerate(q["choices"]))
        blocks.append(f"### 문제 {k}\n{q['stem']}\n" + (f"[주어진 글]\n{q['given']}\n" if q.get("given") else "")
                      + f"[지문]\n{q['passage']}\n" + (f"[박스]\n{q['box']}\n" if q.get("box") else "") + f"[선택지]\n{ch}")
    prompt = ("아래 영어 객관식 문제들을 각 지문을 근거로 직접 풀어라. 원래 글(참고용 원문)과 비교해 판단해도 된다. "
              "단, 문제의 지문·덩어리가 원문과 다르게 재진술·재구성되어 있으면 원문 순서·표현이 아니라 문제에 실린 글을 기준으로 푼다. "
              "각 문제 번호(index)와 정답(①~⑤ 중 하나)만 반환하라.\n\n[참고용 원문]\n" + text + "\n\n" + "\n\n".join(blocks))
    judge = level in ("상", "최상")
    if judge:
        prompt = prompt.replace("각 문제 번호(index)와 정답(①~⑤ 중 하나)만 반환하라.",
                                "각 문제 번호(index)와 정답(①~⑤ 중 하나), 아래 기준의 난이도 판정(level, why)을 반환하라.\n" + QB_LEVEL_JUDGE)
    out = _qb_call(gclient, model, prompt, SCHEMA_QB_SOLVE_LEVEL if judge else SCHEMA_QB_SOLVE, 0.0, logs, "정답 검증")
    if out is None:
        for q in questions:
            q["_unverified"] = True  # 검증 호출 실패는 탈락시키지 않고 '검증 안 됨'으로 저장 (예전엔 ✅ 검증됨으로 저장됐음)
        return [True] * len(questions)
    solved = {int(a.get("index", 0)): qb_norm_answer(a.get("answer")) for a in out.get("answers", [])}
    judged = {int(a.get("index", 0)): (a.get("level"), str(a.get("why", ""))[:80]) for a in out.get("answers", []) if a.get("level") in QB_LEVEL_ORDER}
    ok = [True] * len(questions)
    for k, i in enumerate(idx, 1):
        ok[i] = solved.get(k, "") == questions[i]["answer"]
        if judge and k in judged:
            questions[i]["_judged"] = judged[k]
    return ok

# ---- 문제 저장소 (Supabase 표, 없으면 로컬 파일) ----
def qb_table_status(sb) -> (bool, str):
    if sb is None:
        return True, ""
    try:
        sb.table(QB_TABLE).select("id").limit(1).execute()
        return True, ""
    except Exception as e:
        return False, str(e)[:200]

def _qb_local_load() -> List[Dict]:
    if os.path.exists(QB_FILE):
        with open(QB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def qb_insert(sb, rows: List[Dict]):
    if not rows:
        return
    if sb is not None:
        sb.table(QB_TABLE).insert(rows, **_UPSERT_MIN).execute()  # 넣은 문제를 다시 돌려받지 않음 (전송량 절약)
        return
    data = _qb_local_load() + [{**r, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")} for r in rows]
    with open(QB_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

def qb_fetch(sb, labels: List[str], cols: str = "*") -> List[Dict]:
    """고른 지문의 문제만 불러옴 (한 번에 1000개씩 나눠 읽음)"""
    labels = list(dict.fromkeys(labels))
    if not labels:
        return []
    if sb is None:
        keep = set(labels)
        return [r for r in _qb_local_load() if r.get("passage_label") in keep]
    out = []
    for i in range(0, len(labels), 40):
        chunk = labels[i:i + 40]
        start = 0
        while True:
            res = sb.table(QB_TABLE).select(cols).in_("passage_label", chunk).order("created_at").order("id").range(start, start + 999).execute()  # id까지 정렬해야 1000개씩 나눠 읽을 때 겹치거나 빠지지 않음 (한 번에 넣은 문제는 created_at이 같음)
            out += res.data or []
            if len(res.data or []) < 1000:
                break
            start += 1000
    return out

def qb_delete(sb, ids: List[str]):
    if not ids:
        return
    if sb is not None:
        for i in range(0, len(ids), 100):
            sb.table(QB_TABLE).delete().in_("id", ids[i:i + 100]).execute()
        return
    drop = set(ids)
    with open(QB_FILE, "w", encoding="utf-8") as f:
        json.dump([r for r in _qb_local_load() if r.get("id") not in drop], f, ensure_ascii=False)

def _generate_qbank_item(sb_factory, gclient, item: Dict, opts: Dict, dna_rules: str, essay_rules: str, logs: List[str],
                         yt_points: List[Dict] = None) -> (bool, str):
    """작업 목록의 변형문제 1건(지문·유형·난이도) 생성 → 정답 검증 → 문제 표에 저장"""
    sb = sb_factory()
    label, q_type, level, school = item["passage_label"], item["q_type"], item["difficulty"], item.get("school", "운양고")
    count = int(item.get("count", 2))
    model = job_item_model(item, opts)
    existing = [r for r in qb_fetch(sb, [label], "q_type,school,content") if r.get("q_type") == q_type]
    avoid = [str((r.get("content") or {}).get("focus", "")) for r in existing if (r.get("content") or {}).get("focus")]
    questions = []
    for round_idx in range(len(GEN_RETRY_WAITS) + 1):
        time.sleep(opts.get("api_delay", 3))
        questions = qb_generate(gclient, model, label, item["text"], q_type, level, school, count, avoid, dna_rules, essay_rules, logs,
                                existing, item.get("underline", "mix"), yt_points, bool(item.get("yt_focus")), item.get("unit_other", ""),
                                item.get("grade", ""))
        if questions:
            break
        if round_idx < len(GEN_RETRY_WAITS):
            time.sleep(GEN_RETRY_WAITS[round_idx])
    if not questions:
        return False, "변형문제 생성 실패"
    return qb_save_questions(sb, gclient, model, item, questions, logs, existing)

_QB_SIG_KEYS = ("stem", "given", "passage", "box", "choices", "answer", "target")

def qb_content_sig(c) -> str:
    """문제 내용이 같은지 비교하는 지문(指紋): 발문·지문·선지·정답이 모두 같으면 같은 문제 (해설·난이도 표시는 안 봄)"""
    c = c if isinstance(c, dict) else {}
    return json.dumps({k: c.get(k) for k in _QB_SIG_KEYS}, ensure_ascii=False, sort_keys=True)

def qb_dedupe_rows(rows: List[Dict]) -> List[Dict]:
    """같은 지문·유형·학교에서 내용이 완전히 같은 문제는 하나만 남김 (난이도 이름만 다르게 두 번 저장된 경우 등)
    → 한 시험지·클리닉에 같은 문제가 두 번 들어가지 않게. 내용을 안 읽은 줄은 그대로 둠"""
    seen, out = set(), []
    for r in rows:
        if not isinstance(r.get("content"), dict):
            out.append(r)
            continue
        key = (r.get("passage_label"), r.get("q_type"), r.get("school"), qb_content_sig(r["content"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out

def qb_save_questions(sb, gclient, model: str, item: Dict, questions: List[Dict], logs: List[str], existing: List[Dict] = None) -> (bool, str):
    """정답 검증 → 통과한 문제만 문제 표에 저장. 이미 있는 문제와 내용이 똑같은 문제는 저장하지 않음 (검증 비용도 아낌)"""
    label, q_type, level, school = item["passage_label"], item["q_type"], item["difficulty"], item.get("school", "운양고")
    if existing is None:
        try:
            existing = [r for r in qb_fetch(sb, [label], "q_type,school,content") if r.get("q_type") == q_type]
        except Exception:
            existing = []
    have = {qb_content_sig(r.get("content")) for r in existing if r.get("school", school) == school}
    uniq = []
    for q in questions:
        sig = qb_content_sig(q)
        if sig not in have:
            have.add(sig)
            uniq.append(q)
    n_same = len(questions) - len(uniq)
    if not uniq:
        logs.append(f"⏭️ {item['label']}: 새로 만든 문제가 이미 있는 문제와 똑같아 저장하지 않았어요 ({n_same}개)")
        return True, ""
    questions = uniq
    checks = qb_verify(gclient, model, item["text"], questions, logs, level) if item.get("verify", True) else [True] * len(questions)
    rows, lowered = [], 0
    for q, ok in zip(questions, checks):
        if not ok:
            continue
        q["verified"] = bool(item.get("verify", True)) and not q.pop("_unverified", False)
        qt = q.pop("_q_type", q_type)
        lv = level
        jl, why = q.pop("_judged", (None, ""))
        if jl and QB_LEVEL_ORDER.get(jl, 9) < QB_LEVEL_ORDER.get(level, 0):
            # 난이도 라벨 원칙: 기준에 못 미치면 낮춰서 저장 (요청 난이도·이유는 문항 안에 남김)
            lv, lowered = jl, lowered + 1
            q["level_req"], q["level_note"] = level, why
        rows.append({"id": str(uuid.uuid4()), "passage_label": label, "exam_key": item.get("exam_key", ""), "number": item.get("number", ""),
                     "q_type": qt, "difficulty": lv, "school": school, "content": q})
    dropped = len(questions) - len(rows)
    if not rows:
        return False, f"정답 검증을 통과한 문제가 없음 ({len(questions)}개 모두 탈락)"
    qb_insert(sb, rows)
    logs.append(f"✅ 변형문제 {len(rows)}개 저장: {item['label']}" + (f" (정답 검증 탈락 {dropped}개)" if dropped else "")
                + (f" · {level} 기준 미달 {lowered}개는 낮춰 저장" if lowered else "")
                + (f" · 이미 있는 문제와 똑같은 {n_same}개는 저장 안 함" if n_same else ""))
    return True, ""

# ---- 반값 모드 작업 처리 (서버 스레드에서) ----
QB_BATCH_MAX = 300           # 한 번에 묶어 내는 최대 건수
QB_BATCH_POLL_SEC = 120      # 반값 결과만 기다릴 때 확인 간격
QB_BATCH_POLL_BUSY_SEC = 600 # 일반 작업을 하는 중일 때 확인 간격
QB_BATCH_GIVEUP_SEC = 30 * 3600  # 이만큼 지나도 안 끝나면 실패로 (회사 약속은 24시간)

def qb_batch_submit(sb, items: List[Dict], opts: Dict, dna_rules: str, essay_rules: str, claim, yt_points: List[Dict] = None, write=None):
    """반값 모드 건들의 프롬프트를 만들어 모델별로 한 번에 제출 → 'waiting'. 제출이 안 되면 그 건들은 일반 모드로 바로 생성"""
    groups, upd, new_batches, notes = {}, {}, [], []
    for it in items:
        try:
            existing = [r for r in qb_fetch(sb, [it["passage_label"]], "q_type,content") if r.get("q_type") == it["q_type"]]
            avoid = [str((r.get("content") or {}).get("focus", "")) for r in existing if (r.get("content") or {}).get("focus")]
            shared, task, plans = qb_build_prompt(it["passage_label"], it["text"], it["q_type"], it["difficulty"], it.get("school", "운양고"),
                                                  int(it.get("count", 2)), avoid, dna_rules, essay_rules, existing, it.get("underline", "mix"),
                                                  yt_points, bool(it.get("yt_focus")), it.get("unit_other", ""), it.get("grade", ""))
            groups.setdefault(job_item_model(it, opts), []).append((it, shared, task, plans))
        except Exception as e:
            upd[it["id"]] = {"batch": False}
            notes.append(f"⚠️ 반값 모드 준비 실패 → 일반 모드로: {it['label']} ({str(e)[:80]})")
    for m, lst in groups.items():
        try:
            bid = llm_batch_submit(m, [(it["id"], (shared, task)) for it, shared, task, _ in lst], SCHEMA_QBANK, 0.7)
            now = time.time()
            new_batches.append({"id": bid, "model": m, "ts": now, "n": len(lst)})
            for it, _, _, plans in lst:
                upd[it["id"]] = {"state": "waiting", "batch_id": bid, "batch_model": m, "batch_ts": now, "plans": plans}
            notes.append(f"💸 반값 모드로 {len(lst)}건 제출 · {ai_model_label(m)} (보통 몇 분~몇 시간, 길면 24시간)")
        except Exception as e:
            for it, *_ in lst:
                upd[it["id"]] = {"batch": False}
            notes.append(f"⚠️ 반값 모드 제출 실패 → {len(lst)}건 일반 모드로 바로 생성 ({str(e)[:120]})")

    def apply(j):
        for it in j.get("items", []):
            if it.get("id") in upd and it.get("state") == "pending":
                it.update(upd[it["id"]])
        have = {b.get("id") for b in j.get("batches", [])}
        j.setdefault("batches", []).extend(b for b in new_batches if b["id"] not in have)
        for n in notes:
            _job_log(j, n)
        claim(j)
    (write or (lambda fn: _job_update_retry(sb, fn)))(apply)

def qb_batch_poll(sb, gclient, job: Dict, claim, write=None):
    """기다리는 반값 묶음마다 끝났는지 확인 → 끝난 묶음은 결과를 받아 건마다 정답 검증·저장"""
    waiting = [job_item_full(job, it) for it in job.get("items", []) if it.get("state") == "waiting"]
    by_batch = {}
    for it in waiting:
        by_batch.setdefault(it.get("batch_id", ""), []).append(it)
    write = write or (lambda fn: _job_update_retry(sb, fn))
    recs = {b.get("id"): b for b in job.get("batches", [])}
    for bid, its in by_batch.items():
        rec = recs.get(bid, {})
        model = its[0].get("batch_model") or rec.get("model", "")
        ts = float(rec.get("ts") or its[0].get("batch_ts") or 0)
        if not ts:
            # 제출 시각 기록이 없는 예전 묶음 → 지금부터 30시간을 셈 (기록이 없으면 영원히 기다렸음)
            ts = time.time()
            def add_rec(j, bid=bid, model=model, n=len(its), ts=ts):
                if not any(b.get("id") == bid for b in j.get("batches", [])):
                    j.setdefault("batches", []).append({"id": bid, "model": model, "ts": ts, "n": n})
                claim(j)
            write(add_rec)
        try:
            finished, res, note = llm_batch_collect(model, bid, [it["id"] for it in its], log_usage=not rec.get("logged"))
        except Exception as e:
            if time.time() - ts < QB_BATCH_GIVEUP_SEC:
                write(lambda j, e=e: (_job_log(j, f"⚠️ 반값 결과 확인 실패 (다음에 다시): {str(e)[:120]}"), claim(j)))
                continue
            finished, res, note = True, {}, f"30시간 동안 결과를 받지 못함 ({str(e)[:80]})"
        if not finished:
            if time.time() - ts < QB_BATCH_GIVEUP_SEC:
                continue
            res, note = {}, "30시간이 지나도 끝나지 않음"

        def mark_logged(j, bid=bid):
            for b in j.get("batches", []):
                if b.get("id") == bid:
                    b["logged"] = True
            claim(j)
        write(mark_logged)
        for it in its:
            logs = []
            out = res.get(it["id"])
            try:
                qs = qb_parse(out, it["q_type"], int(it.get("count", 2)), it.get("plans") or [], it["text"]) if out else []
                ok, err = qb_save_questions(sb, gclient, it.get("batch_model") or model, it, qs, logs) if qs else \
                    (False, f"반값 모드 결과가 없거나 형식이 맞지 않음 ({note})")
            except Exception as e:
                ok, err = False, f"반값 결과 처리 중 오류: {str(e)[:150]}"

            def finish(j, it=it, ok=ok, err=err, logs=logs):
                for x in j.get("items", []):
                    if x.get("id") == it["id"] and x.get("state") == "waiting":
                        x["state"], x["error"] = ("done" if ok else "failed"), err
                        for k in ("batch_id", "plans"):
                            x.pop(k, None)
                for m in logs[-4:]:
                    _job_log(j, f"{m} · 💸 반값" if m.startswith("✅") else m)
                if not ok:
                    _job_log(j, f"❌ {it['label']}: {err} (나중에 '실패한 지문 다시 시도'로 재시도할 수 있어요)")
                claim(j)
            write(finish)

        def drop(j, bid=bid):
            if not any(x.get("state") == "waiting" and x.get("batch_id") == bid for x in j.get("items", [])):
                j["batches"] = [b for b in j.get("batches", []) if b.get("id") != bid]
        write(drop)

# ---- 연결 지문: 실제 내신처럼 지문 여러 개를 묶어 한 문제로 ----
#   이어 붙이기: 교과서의 짧은 단락 2~4개를 한 지문으로 (예: 운양 25-2 기말 5·6번 = 4-1+4-2, 8·10번 = 4-3~4-5)
#   (A)(B) 함께: 같은 시험 범위의 비슷한 내용 두 지문을 (A)(B)로 싣고 두 글을 함께 묻기 (제일고·운양고 추론형)
#   문제는 'passage_label'에 묶은 이름(예: "[교재 4과] 4-1 + 4-2", "[교재 4과] 4-1 & 4-3")으로 저장
QB_LINK_JOIN = " + "
QB_LINK_PAIR = " & "
QB_PAIR_TYPES = ["추론 (추론 가능·불가)", "내용 일치·불일치", "<보기> 일치 조합형", "주제", "요지·주장", "제목"]
QB_LINK_MODES = {"따로": "지문마다 따로", "join": "🔗 이어 붙이기 (교과서 짧은 단락)", "pair": "🔗 (A)(B) 두 지문 함께"}

def qb_link_label(labels: List[str], mode: str) -> str:
    sep = QB_LINK_PAIR if mode == "pair" else QB_LINK_JOIN
    ms = [re.match(r"^(\[[^\]]+\]\s*)(.+)$", l) for l in labels]
    if all(ms) and len({m.group(1) for m in ms}) == 1:
        return ms[0].group(1) + sep.join(m.group(2) for m in ms)
    return sep.join(labels)

def qb_link_parts(label: str) -> (str, List[str]):
    """묶은 이름 → (방식 'join'/'pair', 원래 지문 이름들). 묶은 이름이 아니면 ('', [label])"""
    for sep, mode in ((QB_LINK_PAIR, "pair"), (QB_LINK_JOIN, "join")):
        if sep in str(label):
            parts = str(label).split(sep)
            m = re.match(r"^(\[[^\]]+\]\s*)", parts[0])
            pre = m.group(1) if m else ""
            return mode, [parts[0]] + [p if p.startswith("[") else pre + p for p in parts[1:]]
    return "", [label]

def qb_link_passage(ps: List[Dict], mode: str) -> Dict:
    """고른 지문들 → 문제 만들기에 쓸 가상 지문 하나"""
    bodies = [str(p.get("body_text", "")).strip() for p in ps]
    text = "\n".join(f"({'AB'[i]}) {b}" for i, b in enumerate(bodies)) if mode == "pair" else "\n".join(bodies)
    return {"label": qb_link_label([p["label"] for p in ps], mode), "body_text": text, "exam_key": ps[0].get("exam_key", ""), "number": ""}

# 교과서 짧은 단락 자동 이어 붙이기 (2026-11-05 선생님 요청: 실제 내신처럼 짧은 단락은 앞뒤 단락과 묶어 출제 — 예: 운양 25-2 기말 4-1(104단어)+4-2)
QB_SHORT_WORDS = 120   # 이보다 짧은 교과서 단락은 앞뒤 단락과 묶음
QB_JOIN_MIN = 150      # 묶은 지문이 이 길이가 될 때까지 (최대 3단락)
QB_JOIN_MAX = 320
QB_JOIN_TARGET = 230

def _qb_words(p: Dict) -> int:
    return len(str(p.get("body_text", "")).split())

def qb_auto_join(ps: List[Dict], lib: List[Dict]) -> (List[Dict], List[str]):
    """고른 지문 중 짧은 교과서 단락을 같은 단원의 바로 앞·뒤 단락(고른 것 중)과 묶음
    → (만들 지문 목록, 묶은 이름들). 짧은 단락은 묶은 지문으로만 내고, 묶인 긴 단락은 따로도 냄. 모의고사는 그대로."""
    chosen = {p["label"] for p in ps}
    out, joined, used = [], [], set()
    units = {}
    for x in sorted(lib, key=natural_sort_key):
        if x.get("source_type") == "textbook" and x.get("exam_key"):
            units.setdefault(x["exam_key"], []).append(x)
    short_done = set()
    for seq in units.values():
        for i, p in enumerate(seq):
            if p["label"] not in chosen or _qb_words(p) >= QB_SHORT_WORDS or p["label"] in used:
                continue
            grp = [i]
            while len(grp) < 3 and sum(_qb_words(seq[j]) for j in grp) < QB_JOIN_MIN:
                tot = sum(_qb_words(seq[j]) for j in grp)
                cands = [j for j in (grp[0] - 1, grp[-1] + 1) if 0 <= j < len(seq) and seq[j]["label"] in chosen
                         and seq[j]["label"] not in used and tot + _qb_words(seq[j]) <= QB_JOIN_MAX]
                if not cands:
                    break
                j = min(cands, key=lambda j: abs(tot + _qb_words(seq[j]) - QB_JOIN_TARGET))
                grp = sorted(grp + [j])
            if len(grp) < 2:
                continue  # 붙일 이웃 단락이 없으면 그대로 따로
            parts = [seq[j] for j in grp]
            used.update(x["label"] for x in parts)
            short_done.update(x["label"] for x in parts if _qb_words(x) < QB_SHORT_WORDS)
            jp = qb_link_passage(parts, "join")
            out.append(jp)
            joined.append(jp["label"])
    keep = [p for p in ps if p["label"] not in short_done]
    return keep + out, joined

def qb_link_note(label: str, text: str) -> str:
    """연결 지문이면 이번 출제 부분에 붙일 안내 (공통 앞부분은 그대로라 캐시에 영향 없음)"""
    mode, parts = qb_link_parts(label)
    if mode == "pair" and str(text).lstrip().startswith("(A)"):
        return ("\n[연결 지문 — (A)(B) 두 글]\n- 이 지문은 같은 시험 범위에서 내용이 비슷한 두 글 (A)(B)다. 실제 내신처럼 두 글을 함께 읽어야 풀리게 낸다.\n"
                "- stem은 두 글을 가리킨다(예: '(A)와 (B)를 읽고 추론할 수 없는 것은?', '(A) 필자의 관점에서 (B)의 내용을 설명한 것으로 가장 적절한 것은?', "
                "'(A)와 (B)의 공통 주제로 가장 적절한 것은?', '(A)와 (B)에 대한 설명으로 일치하지 않는 것은?').\n"
                "- passage에는 (A) 글과 (B) 글을 줄바꿈으로 나눠 앞에 (A)·(B)를 붙여 싣는다.\n"
                "- 선택지 5개 중 3개 이상은 두 글을 모두 따져야 판단되게 한다(한 글에만 맞는 내용, 두 글의 관계·공통점을 거꾸로 한 내용, 한 글의 주장을 다른 글의 사례에 잘못 적용한 내용 등).\n")
    if mode == "join" and len(parts) >= 2:
        return (f"\n[연결 지문 — 교과서 단락 {len(parts)}개]\n- 이 지문은 교과서에서 이어지는 단락 {len(parts)}개를 실제 내신처럼 한 지문으로 이어 붙인 것이다. "
                "단락 경계를 넘는 내용(앞 단락의 문제 제기 → 뒤 단락의 해결·사례 등)을 묻는 문항을 우선하고, 밑줄·표시는 지문 전체에 고르게 둔다.\n")
    return ""

@st.cache_data(ttl=60, show_spinner=False)
def qb_link_labels(_sb) -> List[str]:
    """문제 은행에 있는 연결 지문 이름 (가벼운 열 하나만 읽음)"""
    if _sb is None:
        found = {r.get("passage_label", "") for r in _qb_local_load()}
    else:
        found = set()
        for pat in (f"%{QB_LINK_JOIN}%", f"%{QB_LINK_PAIR}%"):
            start = 0
            while True:
                res = _sb.table(QB_TABLE).select("passage_label").like("passage_label", pat).range(start, start + 999).execute()
                found |= {r["passage_label"] for r in (res.data or [])}
                if len(res.data or []) < 1000:
                    break
                start += 1000
    return sorted(l for l in found if qb_link_parts(l)[0])

def enqueue_qbank(passages: List[Dict], q_types: List[str], levels: List[str], school: str, count: int, verify: bool,
                  text_model: str, api_delay: int, academy_name: str, school_name: str, light_easy: bool = False, underline: str = "mix",
                  batch: bool = False, yt_focus: bool = False, plan: List[tuple] = None, note: str = "", grade: str = "") -> int:
    """지문 × 유형 × 난이도 조합마다 작업 1건씩 추가 (light_easy: 하·중은 같은 제공사의 가벼운 모델로)
    grade: '고1'·'고2'로 고르면 모든 건에, 비우면 지문마다 원문 라이브러리 학년·지문 이름으로 (선지 길이 기준에 씀)"""
    sb = supabase if IS_CLOUD_DB else None
    added = {"n": 0}
    # 영쌤 포인트 중 '교과서만'이 켜져 있으면, 교과서 지문에는 같은 단원 다른 단락을 함께 넘김 (그 포인트가 쓰이는 유형만)
    tb_pts = [p for p in qb_yt_points() if p.get("on", True) and p.get("tb")]
    lib = st.session_state.get("saved_mock") or []
    lib_by = {x.get("label"): x for x in lib}

    def unit_for(p, t):
        if not tb_pts or not qb_yt_applicable(tb_pts, t, True):
            return ""
        parts = qb_link_parts(p["label"])[1]
        if not all((lib_by.get(x) or {}).get("source_type") == "textbook" for x in parts):
            return ""
        return qb_unit_other({**p, "exam_key": p.get("exam_key") or (lib_by.get(parts[0]) or {}).get("exam_key", "")}, lib)

    def add(job):
        if job.get("status") not in ("running", "stopping"):
            job["status"] = "running"
            job["items"] = [it for it in job.get("items", []) if it.get("state") in ("pending", "failed", "waiting")]
            job["created_at"] = time.time()
            job["worker"] = None
        old_model = (job.get("opts") or {}).get("model", "")
        for it in job.get("items", []):  # 예전 방식(건마다 모델 고정)으로 남은 작업도 새 모델을 따르게
            if it.get("state") == "pending" and "light" not in it:
                it["light"] = bool(it.get("model")) and it.get("model") != old_model and it.get("difficulty") in ("하", "중")
                it.pop("model", None)
        job["opts"] = {"model": text_model, "api_delay": api_delay, "academy_name": academy_name, "school_name": school_name}
        # 어법은 최상이면 단일 정답형 대신 '있는 대로 고른 것' 조합형으로 (같은 조합은 한 번만)
        combos = list(dict.fromkeys(("어법 (조합형)" if (t == "어법 (단일)" and lv == "최상") else t, lv) for t in q_types for lv in levels))
        # plan이 있으면 (지문, 유형, 난이도, 문제 수)만 정확히 추가 (주간 클리닉의 '모자란 문제 만들기')
        todo = ([(p, ("어법 (조합형)" if (t == "어법 (단일)" and lv == "최상") else t), lv, n) for p, t, lv, n in plan] if plan is not None
                else [(p, t, lv, count) for p in passages for t, lv in combos])
        for p, t, lv, cnt in todo:
            job["items"].append({"id": str(uuid.uuid4()), "kind": "qbank", "label": f"{p['label']} · {t} · {lv}",
                                 "passage_label": p["label"], "exam_key": p.get("exam_key", ""), "number": str(p.get("number", "")),
                                 "text_key": _job_text_key(job, p.get("body_text", "")), "q_type": t, "difficulty": lv, "school": school,
                                 "count": cnt, "verify": verify, "state": "pending", "error": "",
                                 "light": bool(light_easy and lv in ("하", "중")), "underline": underline, "batch": bool(batch),
                                 "yt_focus": bool(yt_focus), "unit_key": (_job_text_key(job, uo) if (uo := unit_for(p, t)) else ""),
                                 "grade": grade or qb_passage_grade(p)})
            added["n"] += 1
        _job_log(job, f"➕ 변형문제 {added['n']}건 추가 ({note or (school + ' · ' + ', '.join(levels) + f' · 건당 {count}문제')}{' · 💸 반값 모드' if batch else ''}"
                      f"{' · 🎣 영쌤 포인트 위주' if yt_focus else ''})")
    _job_update(sb, add)
    start_gen_thread()
    return added["n"]

# ---- 실제 기출 시험지 → 원문 라이브러리 지문 연결 + 실제 기출문제 수록 ----
#   - gichul_map 문서: 시험지마다 문항 번호·유형·배점·출처 지문 (가벼운 목록, '기출 이력' 표시에 씀)
#   - 원문 라이브러리 지문과 연결된 문항은 문제 전체를 question_bank 표에 difficulty='기출'로 수록 (시험지에 섞을 수 있음)
QB_GICHUL_LEVEL = "기출"

SCHEMA_GICHUL = {
    "type": "object",
    "properties": {
        "school": {"type": "string", "description": "운양고 또는 제일고"},
        "grade": {"type": "string", "description": "예: 고1"},
        "exam_name": {"type": "string", "description": "예: 25년 2학기 기말"},
        "questions": {"type": "array", "items": {"type": "object", "properties": {
            "no": {"type": "string", "description": "문항 번호 (예: 7, 서술형 2)"},
            "points": {"type": "string", "description": "배점 (보이면)"},
            "q_type": {"type": "string", "enum": QB_TYPE_NAMES + ["기타"]},
            "source_hint": {"type": "string", "description": "지문 출처 (예: 2025년 6월 고1 모의고사 21번, 공통영어1 능률(민) 3과 2단락). 모르면 지문 첫 문장 앞부분"},
            "matched_label": {"type": "string", "description": "[후보 지문 목록]에서 같은 지문의 label (없으면 빈 문자열)"},
            "stem": {"type": "string", "description": "발문 (배점 표시 제외)"},
            "given": {"type": "string", "description": "주어진 글·문장, 없으면 빈 문자열"},
            "passage": {"type": "string", "description": "시험지에 실린 지문 그대로 (밑줄·기호 유지)"},
            "box": {"type": "string", "description": "<보기>·요약문·<조건>, 없으면 빈 문자열"},
            "choices": {"type": "array", "items": {"type": "string"}},
            "answer": {"type": "string", "description": "직접 풀어 낸 정답 (①~⑤, 서술형은 모범 답안)"},
            "explanation": {"type": "array", "items": {"type": "string"}},
        }, "required": ["no", "points", "q_type", "source_hint", "matched_label", "stem", "given", "passage", "box", "choices", "answer", "explanation"]}},
    },
    "required": ["school", "grade", "exam_name", "questions"],
}

def gichul_candidates_text(mock_lib: List[Dict]) -> str:
    lines = []
    for p in sorted(mock_lib, key=natural_sort_key):
        words = re.sub(r"\s+", " ", str(p.get("body_text", ""))).split(" ")[:18]
        lines.append(f"{p['label']} :: {' '.join(words)}")
    return "\n".join(lines)

def gichul_extract(gclient, model: str, pdf_part, file_name: str, candidates: str) -> Dict:
    prompt = f"""이 PDF는 운양고 또는 제일고 영어 내신 시험지다 (파일명: {file_name}).
객관식·서술형 모든 문항마다 번호, 배점, 유형, 지문 출처를 정리하고, 문제 전체를 시험지 그대로 옮겨라 (기출 분석용으로 모든 문항을 보관한다).
- q_type은 주어진 목록에서 가장 가까운 것 (해당 없으면 '기타').
- 각 문항 지문을 아래 [후보 지문 목록]의 지문 첫 부분과 비교해, 같은 지문이면 matched_label에 그 label을 그대로 적는다 (일부 문장만 같아도 같은 지문). 없으면 빈 문자열.
- 모든 문항을 시험지 그대로 옮긴다: stem(발문 — 시험지 문구 그대로, '[4~6] 다음 글의 제목으로 가장 적절한 것을 고르시오.' 같은 묶음 머리말이면 그 발문을, 부정어 밑줄 <u>않은</u>도 유지), given(주어진 글·문장), passage(지문 — 밑줄은 <u>…</u>, ⓐ~ⓖ·①~⑤·빈칸·(A)(B)(C) 표시 유지, 단락은 줄바꿈), box(<보기>·요약문·<조건>), choices(선택지 5개, 번호 기호 없이. 지문 속 ①~⑤가 선택지인 유형은 빈 문자열 5개, 서술형은 빈 배열).
- 여러 문항이 지문 하나를 같이 쓰면 문항마다 passage를 똑같이 넣는다.
- answer: 시험지의 학생 필기·표시는 믿지 말고 지문으로 직접 풀어 ①~⑤ 중 하나를 적는다 (서술형은 모범 답안). explanation은 한국어 1~2줄.
- school, grade, exam_name은 표지·머리말·파일명으로 판단한다.
- 서식 태그는 <u>, <b>만 쓴다.

[후보 지문 목록] (label :: 지문 첫 부분)
{candidates}"""
    return _qb_call(gclient, model, [pdf_part, prompt], SCHEMA_GICHUL, 0.1, [], f"{file_name} 기출 분석")

def gichul_label_from_hint(hint: str, labels: set) -> str:
    """'2025년 6월 고1 모의고사 21번' → '[2025년 6월 고1] 21번' (원문 라이브러리에 있으면)"""
    m = re.search(r"(20\d{2}|\d{2})\s*년\s*(\d{1,2})\s*월\s*(고\s*\d).*?(\d{2}(?:\s*[~\-]\s*\d{2})?)\s*번", str(hint))
    if not m:
        return ""
    y = m.group(1) if len(m.group(1)) == 4 else "20" + m.group(1)
    num = re.sub(r"\s*[~\-]\s*", "~", m.group(4))
    cand = f"[{y}년 {int(m.group(2))}월 {m.group(3).replace(' ', '')}] {num}번"
    return cand if cand in labels else ""

def gichul_short(ex: Dict) -> str:
    """'운양고' '25년 2학기 기말' '고1' → '운양 25-2기말 고1' (기출 이력을 한 줄에 여러 개 적을 때)"""
    school = str(ex.get("school", "")).replace("김포", "").replace("고", "") or "?"
    name = str(ex.get("exam_name", ""))
    m = re.search(r"(\d{2})\s*년.*?(\d)\s*학기.*?(중간|기말)", name)
    name = f"{m.group(1)}-{m.group(2)}{m.group(3)}" if m else name
    return f"{school} {name} {ex.get('grade', '')}".strip()

def gichul_question_label(ex: Dict, q: Dict, labels: set) -> str:
    if q.get("link_off"):
        return ""
    return q.get("matched_label") if q.get("matched_label") in labels else gichul_label_from_hint(q.get("source_hint", ""), labels)

# 기출 문항 ↔ 원문 지문 연결 확인: 원문의 내용 단어(4글자 이상) 중 시험지 지문에 나온 비율
#   2026-11-04 측정: 바르게 연결된 문항 29~100%(제일고 전면 재진술도 29% 이상), 잘못 연결된 문항 4~13% → 20% 미만이면 연결하지 않음
QB_LINK_MIN = 0.2
_LINK_STOP = set("that this with from they their them have been were which what when there these those into also than more most such will would could should about other only some your".split())

def qb_link_share(exam_text: str, orig_text: str) -> float:
    def words(t):
        return {w for w in re.findall(r"[a-z]{4,}", re.sub(r"<[^>]+>", " ", str(t)).lower()) if w not in _LINK_STOP}
    wo = words(orig_text)
    return len(words(exam_text) & wo) / len(wo) if wo else 1.0

def gichul_check_links(res: Dict, mock_by_label: Dict[str, Dict]) -> int:
    """AI가 짝지은 지문과 내용이 거의 안 겹치는 문항은 연결을 끊음 (link_off) → 끊은 개수"""
    labels, n = set(mock_by_label), 0
    for q in res.get("questions", []):
        lbl = gichul_question_label({}, q, labels)
        body = f"{q.get('given', '')} {q.get('passage', '')}"
        if lbl and len(body.split()) > 40 and qb_link_share(body, mock_by_label[lbl].get("body_text", "")) < QB_LINK_MIN:
            q["link_off"] = True
            n += 1
    return n

def load_gichul_map() -> Dict:
    return _store_get(supabase if IS_CLOUD_DB else None, GICHUL_FILE, {}) or {}

def gichul_add_exam(exam_id: str, ex: Dict):
    """기출 시험지 하나를 DB 최신 기출 기록에 더해 바로 저장 (다른 창에서 그사이 더한 시험지도 지켜짐)"""
    sb_ = supabase if IS_CLOUD_DB else None
    gmap = _store_get(sb_, GICHUL_FILE, {}) or {}
    gmap.setdefault("exams", {})[exam_id] = ex
    gmap["updated_at"] = time.strftime("%Y-%m-%d %H:%M")
    _store_put(sb_, GICHUL_FILE, gmap)

def gichul_full_record(res: Dict, file_name: str, model: str) -> Dict:
    """기출 시험지 전 문항 (발문·지문·선지·정답) — 학교·학년별 선지 길이·어휘·발문 말투·난이도 분석용 (문제 은행과 따로 보관)"""
    meta = gichul_light(res, file_name)
    keep = ("no", "points", "q_type", "source_hint", "matched_label", "stem", "given", "passage", "box", "choices", "answer", "explanation")
    qs = [{k: (list(q.get(k) or []) if k in ("choices", "explanation") else q.get(k, "")) for k in keep} for q in res.get("questions", [])]
    return {k: meta[k] for k in ("school", "grade", "exam_name", "file", "display")} | {"questions": qs, "model": model, "at": time.strftime("%Y-%m-%d")}

def gichul_full_add(exam_id: str, rec: Dict):
    """전 문항 기록 하나를 DB 최신 문서에 더해 바로 저장"""
    sb_ = supabase if IS_CLOUD_DB else None
    doc = _store_get(sb_, GICHUL_FULL_FILE, {}) or {}
    doc.setdefault("exams", {})[exam_id] = rec
    doc["updated_at"] = time.strftime("%Y-%m-%d %H:%M")
    _store_put(sb_, GICHUL_FULL_FILE, doc)

def gichul_index(gmap: Dict, labels: set) -> Dict[str, List[Dict]]:
    """지문 label → 그 지문이 나온 실제 기출 문항 목록 (운양고·제일고 전부, 최신 시험 먼저)"""
    idx = {}
    for exam_id, ex in (gmap.get("exams") or {}).items():
        for q in ex.get("questions", []):
            lbl = gichul_question_label(ex, q, labels)
            if lbl:
                idx.setdefault(lbl, []).append({"exam_id": exam_id, "exam": ex.get("display", ""), "short": gichul_short(ex),
                                                "school": ex.get("school", ""), "no": str(q.get("no", "")),
                                                "q_type": q.get("q_type", ""), "points": q.get("points", "")})
    for hits in idx.values():
        hits.sort(key=lambda h: natural_sort_key({"label": h["short"].split(" ", 1)[-1]}), reverse=True)
    return idx

def gichul_norm_meta(school: str, grade: str, exam_name: str, file_name: str = "") -> (str, str, str):
    """학교·학년·시험 이름을 '운양고' '고2' '25년 2학기 기말' 형태로 통일 (파일명 '[운양2] 25년 2학기 기말.pdf'를 우선)"""
    src = f"{file_name} {school}"
    school = "운양고" if "운양" in src else ("제일고" if "제일" in src else str(school).strip())
    # '[운양2]'·'운양고1'·'… 고1.pdf'에서 학년을 읽음. '운양고 25년'·'제일고 2026'·'운양고 2학기'의 숫자는 학년이 아님
    #   (예전엔 '운양고 25년 2학기 기말 고1'을 고2로 읽었음)
    m = (re.search(r"(운양|제일)\s*(?:고)?\s*([1-3])(?![\d년]|\s*학기)", file_name)
         or re.search(r"고\s*([1-3])(?![\d년]|\s*학기)", file_name)
         or re.search(r"고\s*(\d)|(\d)\s*학년", str(grade)))
    g = next((x for x in (m.groups() if m else ()) if x and x.isdigit()), "")
    grade = f"고{g}" if g else str(grade).strip()
    def parse(t):
        t = str(t)
        y = re.search(r"(?:20)?(\d{2})\s*(?:년|학년도)", t)
        term = re.search(r"(\d)\s*학기", t)
        kind = "중간" if re.search(r"중간|1\s*차", t) else ("기말" if re.search(r"기말|2\s*차", t) else "")
        return f"{y.group(1)}년 {term.group(1)}학기 {kind}".strip() if (y and term and kind) else ""
    exam_name = parse(file_name) or parse(exam_name) or str(exam_name).strip()
    return school, grade, exam_name

def gichul_light(res: Dict, file_name: str) -> Dict:
    """gichul_map 문서에 남길 가벼운 목록 (문제 본문은 question_bank 표에)"""
    keep = ("no", "points", "q_type", "source_hint", "matched_label", "link_off")
    school, grade, exam_name = gichul_norm_meta(res.get("school", ""), res.get("grade", ""), res.get("exam_name", ""), file_name)
    return {"school": school, "grade": grade, "exam_name": exam_name, "file": file_name,
            "display": f"{school} {exam_name} {grade}".strip(),
            "questions": [{k: q.get(k, "") for k in keep} for q in res.get("questions", [])]}

def gichul_bank_rows(gclient, model: str, exam_id: str, res: Dict, mock_by_label: Dict[str, Dict], existing_keys: set, logs: List[str]) -> List[Dict]:
    """연결된 기출 문항을 문제 표에 넣을 행으로 (AI가 따로 한 번 더 풀어 정답이 다르면 '정답 확인 필요' 표시)"""
    ex = gichul_light(res, res.get("_file", ""))
    labels = set(mock_by_label)
    by_label = {}
    for q in res.get("questions", []):
        lbl = gichul_question_label(ex, q, labels)
        if not lbl or not str(q.get("passage", "")).strip() or q.get("q_type") == "기타" or (exam_id, str(q.get("no", ""))) in existing_keys:
            continue
        if find_degenerate_text({k: q.get(k) for k in ("stem", "passage", "box", "choices", "explanation")}):
            continue
        if q["q_type"] in QB_ESSAY_TYPES:
            q["choices"] = []
        else:
            q["answer"] = qb_norm_answer(q.get("answer"))
            q["choices"] = [""] * 5 if q["q_type"] in QB_MARK_TYPES else [str(c).strip() for c in (q.get("choices") or [])][:5]
            if len(q["choices"]) != 5:
                continue
        by_label.setdefault(lbl, []).append(q)
    rows = []
    for lbl, qs in by_label.items():
        checks = qb_verify(gclient, model, mock_by_label[lbl].get("body_text", ""), qs, logs)
        for q, ok in zip(qs, checks):
            content = {k: q.get(k, "") for k in ("stem", "given", "passage", "box", "choices", "answer", "explanation")}
            content.update({"focus": f"실제 기출 {ex['display']} {q.get('no', '')}번", "verified": bool(ok) and not q.pop("_unverified", False),
                            "source": {"exam_id": exam_id, "exam": ex["display"], "short": gichul_short(ex), "no": str(q.get("no", "")), "points": q.get("points", "")}})
            if not ok:
                content["answer_note"] = "AI가 다시 푼 답과 달라요 — 정답 확인 필요"
            p = mock_by_label[lbl]
            rows.append({"id": str(uuid.uuid4()), "passage_label": lbl, "exam_key": p.get("exam_key", ""), "number": str(p.get("number", "")),
                         "q_type": q["q_type"], "difficulty": QB_GICHUL_LEVEL, "school": ex["school"], "content": content})
    return rows

def qb_existing_gichul_keys(rows: List[Dict]) -> set:
    return {(r["content"]["source"]["exam_id"], r["content"]["source"]["no"]) for r in rows
            if r.get("difficulty") == QB_GICHUL_LEVEL and (r.get("content") or {}).get("source")}

# ---- 시험지 구성 ----
def qb_set_key(r: Dict):
    """실제 기출 중 같은 시험·같은 지문에 딸린 문제 묶음 표시 (변형문제는 None)"""
    if r.get("difficulty") != QB_GICHUL_LEVEL:
        return None
    c = r.get("content") or {}
    eid = (c.get("source") or {}).get("exam_id")
    p = re.sub(r"\s+", " ", str(c.get("passage", ""))).strip()
    return (eid, p) if eid and p else None

def _qb_src_no(r: Dict) -> int:
    m = re.match(r"\d+", str(((r.get("content") or {}).get("source") or {}).get("no", "")))
    return int(m.group(0)) if m else 999

def qb_keep_sets(picked: List[Dict], rows: List[Dict]) -> (List[Dict], List[str]):
    """실제 기출에서 한 지문에 여러 문제가 나온 경우, 하나가 뽑히면 같은 묶음 문제도 함께(시험 번호순으로 연달아) 넣음.
    rows는 이미 유형·난이도 조건으로 거른 것이라 유형을 골라 뽑을 때는 조건에 맞는 문제만 따라옴.
    늘어난 만큼 변형문제를 빼서 총 문항 수를 유지 (변형문제가 가장 많은 지문에서 뒤에서부터)"""
    sets = {}
    for r in rows:
        k = qb_set_key(r)
        if k:
            sets.setdefault(k, []).append(r)
    if not any(len(v) > 1 for v in sets.values()):
        return picked, []
    out, seen = [], set()
    for q in picked:
        if id(q) in seen:
            continue
        k = qb_set_key(q)
        for g in (sorted(sets[k], key=_qb_src_no) if k else [q]):
            if id(g) not in seen:
                out.append(g)
                seen.add(id(g))
    extra, removed = len(out) - len(picked), 0
    while removed < extra:
        var = [i for i, r in enumerate(out) if r.get("difficulty") != QB_GICHUL_LEVEL]
        if not var:
            break
        cnt = {}
        for i in var:
            cnt[out[i]["passage_label"]] = cnt.get(out[i]["passage_label"], 0) + 1
        top = max(cnt, key=cnt.get)
        out.pop(max(i for i in var if out[i]["passage_label"] == top))
        removed += 1
    if not extra:
        return out, []
    return out, [f"실제 기출에서 한 지문에 함께 나온 문제 {extra}개를 묶어서 넣었습니다"
                 + (f" (대신 변형문제 {removed}개를 빼서 문항 수 유지)." if removed == extra else f" (총 {len(out)}문항).")]

def qb_pick(rows: List[Dict], labels: List[str], total: int, seed: int) -> (List[Dict], List[str]):
    """고른 지문에서 골고루(같은 개수씩) 뽑고, 지문 안에서는 유형을 돌아가며 뽑음. 부족한 지문 몫은 남는 지문이 채움."""
    rnd = random.Random(seed)
    rows = qb_dedupe_rows(rows)
    pools = {}
    for lbl in labels:
        by_type = {}
        for r in rows:
            if r["passage_label"] == lbl:
                by_type.setdefault(r["q_type"], []).append(r)
        for lst in by_type.values():
            rnd.shuffle(lst)
        # 유형을 돌아가며 한 줄로 세움 (예: 주제·어법·순서·주제·어법·순서…)
        order = [t for t in QB_TYPE_NAMES if t in by_type]
        rnd.shuffle(order)
        seq = []
        while any(by_type[t] for t in order):
            for t in order:
                if by_type[t]:
                    seq.append(by_type[t].pop())
        pools[lbl] = seq
    active = [l for l in labels if pools.get(l)]
    empty = [l for l in labels if not pools.get(l)]
    notes = [f"조건에 맞는 문제가 없는 지문 {len(empty)}개: " + ", ".join(empty[:8]) + (" 외" if len(empty) > 8 else "")] if empty else []
    if not active:
        return [], notes
    # 나머지 몫(+1)은 무작위 지문에 → 문제 수가 지문 수보다 적어도 '다시 뽑기'마다 다른 지문이 들어감
    #   (예전엔 늘 목록 앞쪽 지문에만 줘서 뒤쪽 지문은 한 번도 안 뽑혔음)
    extra = set(rnd.sample(active, total % len(active)))
    quota = {l: total // len(active) + (1 if l in extra else 0) for l in active}
    taken = {l: pools[l][:quota[l]] for l in active}
    short = sum(quota[l] - len(taken[l]) for l in active)
    lacking = [l for l in active if len(taken[l]) < quota[l]]
    while short > 0:
        spare = [l for l in active if len(taken[l]) < len(pools[l])]
        if not spare:
            break
        for l in sorted(spare, key=lambda x: (len(taken[x]), rnd.random())):
            if short <= 0:
                break
            taken[l].append(pools[l][len(taken[l])])
            short -= 1
    if lacking:
        notes.append("쌓인 문제가 적은 지문: " + ", ".join(f"{l} ({len(pools[l])}문제)" for l in lacking)
                     + (" → 모자란 만큼 다른 지문에서 더 뽑았습니다." if short <= 0 else ""))
    if short > 0:
        notes.append(f"조건에 맞는 문제가 모자라 {total - short}문제로 만들었습니다 ({short}문제 부족). '문제 만들기'에서 더 생성해 주세요.")
    picked = [q for l in labels for q in taken.get(l, [])]
    # 어법 문제는 시험지 전체에서 개념이 고르게: 같은 지문·유형·난이도의 안 뽑힌 문제 중 이미 나온 개념과 덜 겹치는 것으로 바꿈
    ids = {id(q) for q in picked}
    seen: Dict[str, int] = {}
    for i, q in enumerate(picked):
        if q.get("q_type") not in QB_GRAMMAR_TYPES:
            continue
        cands = [q] + [r for r in pools.get(q["passage_label"], []) if id(r) not in ids and r.get("q_type") == q["q_type"] and r.get("difficulty") == q.get("difficulty")]
        best = min(cands, key=lambda r: (sum(seen.get(c, 0) for c in qb_grammar_concepts(r)), 0 if r is q else 1))
        if best is not q:
            ids.discard(id(q)); ids.add(id(best)); picked[i] = best
        for c in qb_grammar_concepts(best):
            seen[c] = seen.get(c, 0) + 1
    picked, set_note = qb_keep_sets(picked, rows)
    notes += set_note
    # 예전에 쌓인 순서 변형문제도(실제 기출 문항은 그대로) 시험지에는 수능식 고정 선택지로 (저장된 문제는 그대로, 문항마다 같은 결과가 나오게 id로 고정)
    for i, q in enumerate(picked):
        if q.get("q_type") == "순서" and q.get("difficulty") != QB_GICHUL_LEVEL and isinstance(q.get("content"), dict):
            c = copy.deepcopy(q["content"])
            if qb_fix_order(c, random.Random(str(q.get("id", ""))), only_if_needed=True):
                picked[i] = {**q, "content": c}
        # 예전에 <보기> 조합형과 섞여 나온 변형문제도 시험지에는 정리해서 (저장된 문제는 그대로)
        elif q.get("q_type") in QB_ONE_CHOICE_TYPES and q.get("difficulty") != QB_GICHUL_LEVEL and isinstance(q.get("content"), dict):
            c = copy.deepcopy(q["content"])
            fx = qb_fix_combo(c, q["q_type"])
            if fx:
                picked[i] = {**q, "q_type": fx, "content": c}
    return picked, notes

# 시험지 난이도 비율 (2026-11-09 선생님 요청: 최상·상·중 비중을 직접 정하기) — '비율 정하기'를 처음 켤 때 기본값(%)
QB_LEVEL_RATIO_DEFAULT = {"최상": 30, "상": 40, "중": 30, "하": 0, QB_GICHUL_LEVEL: 10}

def qb_level_name(lv: str) -> str:
    return "실제 기출" if lv == QB_GICHUL_LEVEL else lv

def qb_level_counts(total: int, weights: Dict[str, float]) -> Dict[str, int]:
    """난이도 비율 → 난이도별 문제 수 (합이 total, 반올림 나머지는 소수점이 큰 난이도부터). 비율 합이 100이 아니어도 비율로 나눔"""
    w = {k: max(0.0, float(v or 0)) for k, v in weights.items()}
    s = sum(w.values())
    if s <= 0 or total <= 0:
        return {k: 0 for k in w}
    raw = {k: total * v / s for k, v in w.items()}
    out = {k: int(raw[k]) for k in w}
    for k in sorted(w, key=lambda x: -(raw[x] - out[x])):
        if sum(out.values()) >= total:
            break
        out[k] += 1
    return out

def qb_pick_levels(rows: List[Dict], labels: List[str], total: int, seed: int, weights: Dict[str, float]) -> (List[Dict], List[str]):
    """난이도 비율대로 나눠 뽑음 (난이도마다 qb_pick으로 지문 골고루·유형 돌아가며).
    모자란 난이도 몫은 비율을 준 다른 난이도에서 채우고 알려 줌. 지문 순서로 합치고, 한 지문 안에서는 난이도를 번갈아 놓음"""
    rows = qb_dedupe_rows(rows)  # 난이도 이름만 다르게 두 번 저장된 문제가 두 난이도에서 함께 뽑히지 않게
    want = qb_level_counts(total, weights)
    keep_note = lambda nt: [x for x in nt if x.startswith("실제 기출에서 한 지문에")]
    picked, notes, lack = [], [], []
    for i, (lv, n) in enumerate(want.items()):
        if n <= 0:
            continue
        got, nt = qb_pick([r for r in rows if r.get("difficulty") == lv], labels, n, seed + i)
        picked += got
        notes += keep_note(nt)
        if len(got) < n:
            lack.append(f"{qb_level_name(lv)} {n - len(got)}문제")
    filled = 0
    if len(picked) < total:
        have = {r.get("id") for r in picked}
        rest = [r for r in rows if r.get("id") not in have and float(weights.get(r.get("difficulty"), 0) or 0) > 0]
        extra, nt = qb_pick(rest, labels, total - len(picked), seed + 97)
        picked += extra
        notes += keep_note(nt)
        filled = len(extra)
    while len(picked) > total:  # 실제 기출 묶음이 함께 들어와 넘친 만큼, 가장 많이 들어간 난이도의 변형문제를 뒤에서부터 뺌
        var = [i for i, r in enumerate(picked) if r.get("difficulty") != QB_GICHUL_LEVEL]
        if not var:
            break
        cnt = {}
        for i in var:
            cnt[picked[i]["difficulty"]] = cnt.get(picked[i]["difficulty"], 0) + 1
        top = max(cnt, key=lambda lv: cnt[lv] - want.get(lv, 0))
        picked.pop(max(i for i in var if picked[i]["difficulty"] == top))
    if lack:
        notes.append("난이도 비율대로 뽑기에 " + " · ".join(lack) + "가 모자랐어요"
                     + (f" → 비율을 준 다른 난이도에서 {filled}문제를 채웠어요." if filled else ".")
                     + (" '문제 만들기'에서 그 난이도를 더 만들어 주세요." if any(not x.startswith("실제 기출") for x in lack) else ""))
    if len(picked) < total:
        notes.append(f"조건에 맞는 문제가 모자라 {len(picked)}문제로 만들었습니다 ({total - len(picked)}문제 부족).")
    pos = {l: i for i, l in enumerate(labels)}
    lv_pos = {lv: i for i, lv in enumerate(want)}
    turn, keys = {}, []
    for r in picked:  # 한 지문 안 난이도 번갈아: 최상·상·중·최상… (실제 기출 묶음은 시험 번호순 그대로 붙여 둠)
        k = (r["passage_label"], r.get("difficulty"))
        turn[k] = turn.get(k, -1) + (0 if r.get("difficulty") == QB_GICHUL_LEVEL and k in turn else 1)
        keys.append((pos.get(r["passage_label"], 0), turn[k], lv_pos.get(r.get("difficulty"), 0)))
    order = sorted(range(len(picked)), key=lambda i: keys[i])
    return [picked[i] for i in order], notes

def qb_html(s) -> str:
    """AI가 쓴 글에서 <u>, <b>만 살리고 나머지는 글자로 표시"""
    s = html.escape(html.unescape(str(s or "")))
    s = re.sub(r"&lt;(/?)(u|b)&gt;", r"<\1\2>", s)
    return re.sub(r"<u>\s*</u>", QB_BLANK, s)  # 빈 밑줄(기출 빈칸)은 그대로 두면 안 보여서 빈칸 줄로

QB_BLANK = "<span style='display:inline-block;min-width:34mm;border-bottom:1px solid #111;'>&nbsp;</span>"
QB_SUM_BLANK = ("<span style='display:inline-block;min-width:24mm;border-bottom:1px solid #111;"
                "text-align:center;font-weight:700;text-indent:0;'>(\\1)</span>")

def qb_paras(s, order: bool = False, given: str = "", summary: bool = False) -> str:
    s = re.sub(r"(?i)<br\s*/?>", "\n", str(s or ""))  # AI가 줄바꿈을 <br>로 써 온 글 (그대로 두면 '<br>' 글자가 찍힘)
    if order:  # 순서: (A)(B)(C) 덩어리가 한 줄로 붙어 온 문항은 덩어리마다 새 줄, 주어진 글을 지문 앞에 또 쓴 문항은 앞부분 지움
        g = str(given or "").strip()
        if g and s.strip().startswith(g):
            s = s.strip()[len(g):]
        s = re.sub(r"\s+(?=\((?:A|B|C)\)\s)", "\n", s)
    out = "".join(f"<p>{qb_html(line)}</p>" for line in s.split("\n") if line.strip())
    if summary:  # 요약문 박스: (A)(B)를 실제 시험지처럼 단어가 들어갈 밑줄 칸으로 (이미 ___ 칸이 있으면 그대로)
        out = re.sub(r"_{2,}\s*\(([AB])\)\s*_{2,}|_{2,}\s*\(([AB])\)|\(([AB])\)\s*_{2,}|\(([AB])\)",
                     lambda m: QB_SUM_BLANK.replace("\\1", next(g for g in m.groups() if g)), out)
    return out

QB_EXAM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700;900&family=Noto+Serif+KR:wght@400;700&display=swap');
@page { size: A4; margin: 14mm 12mm 14mm 12mm; }
* { box-sizing: border-box; }
body { font-family: 'Noto Serif KR', serif; font-size: 10.3pt; line-height: 1.62; color: #111; margin: 0; background: #fff;
  -webkit-print-color-adjust: exact; print-color-adjust: exact; }  /* 인쇄 창의 '배경 그래픽'을 꺼도 표지 색·검은 배지가 나오게 */
.q .stem .tm { float: right; font-family: 'Noto Sans KR'; font-size: 7.8pt; font-weight: 700; color: #0047AB; border: 1px solid #0047AB; border-radius: 9px; padding: 0 6px; margin-left: 4px; }
.sheet { max-width: 186mm; margin: 0 auto; }
.head { border: 2px solid #111; margin-bottom: 8px; }
.head .top { display: flex; align-items: center; justify-content: space-between; padding: 6px 12px; border-bottom: 1px solid #111; }
.brand { font-family: 'Noto Sans KR'; font-weight: 900; font-size: 12pt; letter-spacing: .5px; }
.brand b { color: #fff; background: #111; padding: 1px 7px; margin-right: 6px; }
.kind { font-family: 'Noto Sans KR'; font-size: 9pt; }
.head h1 { font-family: 'Noto Sans KR'; font-weight: 900; font-size: 19pt; text-align: center; margin: 8px 0 2px; }
.head .sub { text-align: center; font-family: 'Noto Sans KR'; font-size: 10pt; margin-bottom: 8px; }
.info { display: flex; border-top: 1px solid #111; font-family: 'Noto Sans KR'; font-size: 9.5pt; }
.info div { flex: 1; padding: 5px 10px; border-right: 1px solid #111; }
.info div:last-child { border-right: 0; }
.cols { column-count: 2; column-gap: 9mm; column-rule: 1px solid #111; column-fill: auto; }  /* 왼쪽 단을 쪽 끝까지 채운 뒤 오른쪽 단으로 (문항이 적어도 반씩 나뉘지 않게) */
.q { margin: 0 0 14px; }  /* 긴 문항은 쪽·단을 넘겨 이어지게 (통째로 다음 쪽으로 밀리면 첫 쪽이 비어 버림) */
.q .stem, .q .tag, .q .given { break-after: avoid; }
.passage p { orphans: 3; widows: 3; }
.choices, .box, .write { break-inside: avoid; }
.box.long { break-inside: auto; }
.box.long p { orphans: 2; widows: 2; }
.q .stem { font-family: 'Noto Sans KR'; font-size: 10pt; margin-bottom: 3px; }
.q .stem .no { font-weight: 900; font-size: 11pt; margin-right: 2px; }
.q.setq { margin-bottom: 8px; }
.q .stem.sethead { font-weight: 700; break-after: avoid; }
.q .tag { display: block; font-family: 'Noto Sans KR'; font-size: 7.8pt; color: #444; margin: -1px 0 3px 16px; }
.q .tag .bd { display: inline-block; font-weight: 700; padding: 0 5px; margin-left: 3px; border: 1px solid #111; }
.q .tag .bd.real { background: #111; color: #fff; }
.q .tag .hist { display: inline-block; margin-top: 1px; }
.keynote { font-family: 'Noto Sans KR'; font-size: 8pt; color: #444; margin: -2px 0 8px; }
.passage { text-align: justify; margin: 0 0 6px; }
.passage p { margin: 0 0 4px; text-indent: 1em; }
.given { border: 1px solid #111; padding: 5px 8px; margin: 0 0 7px; }
.given p { margin: 0; }
.box { border: 1px solid #111; padding: 4px 8px; margin: 2px 0 4px 10px; font-size: 9.8pt; }
.box p { margin: 0; }
u { text-underline-offset: 2px; }
.choices { margin: 0 0 0 10px; }
.choices div { padding-left: 1.3em; text-indent: -1.3em; }
.choices.inline { display: flex; justify-content: space-between; padding-right: 12px; }
.choices.inline div { padding: 0; text-indent: 0; }
.write { border-bottom: 1px solid #999; height: 22px; margin: 0 0 4px 10px; }
.ans-page { break-before: page; page-break-before: always; clear: both; }  /* 정답·해설은 항상 새 쪽에서 시작 */
.anshead { border: 2px solid #111; text-align: center; font-family: 'Noto Sans KR'; font-weight: 900; font-size: 15pt; padding: 6px; margin-bottom: 8px; }
table.key { width: 100%; border-collapse: collapse; font-family: 'Noto Sans KR'; font-size: 9.5pt; margin-bottom: 6px; text-align: center; table-layout: fixed; }
table.key th, table.key td { border: 1px solid #111; padding: 3px 2px; }
table.key th { background: #e8e8e8; }
.exp { break-inside: avoid; margin-bottom: 7px; font-size: 9.2pt; line-height: 1.45; }
.exp h3 { font-family: 'Noto Sans KR'; font-size: 10pt; margin: 0 0 2px; border-bottom: 1px solid #999; padding-bottom: 1px; }
.exp h3 span { font-weight: 400; font-size: 8.2pt; color: #444; margin-left: 4px; }
.exp ul { margin: 0; padding-left: 14px; }
.foot { font-family: 'Noto Sans KR'; text-align: center; font-size: 8pt; color: #555; margin-top: 8px; }
.printbar { position: sticky; top: 0; text-align: right; padding: 6px; background: #fff; }
.printbar button { font-size: 13px; padding: 6px 14px; cursor: pointer; }
@media print { .printbar { display: none; } }
"""

# ---- ANATOMY VISION 단행본: 앞표지 · 사용법 · 제작 소개 2쪽 · (문제·해설) · 저작권 · 뒤표지 ----
#   양면 인쇄 기준: 앞쪽 4쪽 뒤라 문제는 5쪽(오른쪽 면)부터. 뒤표지가 맨 뒷면(짝수 쪽)이 되려면 전체 쪽수가 짝수여야 하는데
#   크롬은 '오른쪽 면에서 시작'(break-before: right)을 지원하지 않아서, 인쇄 막대의 '빈 쪽 넣기'로 맞춤
#   소개 페이지의 숫자(분석한 시험지 수·연도·어법 선택지 수 등)는 기출 분석 자료에서 그때그때 셈
QB_PASSAGE_MEASURE = [  # 실제 기출 지문 vs 원문: 원문 표현(4단어 묶음)이 남은 비율 (2026-11 측정)
    ("삽입", "93%", "62~94%"), ("순서", "88%", "12%까지"), ("빈칸", "91~95%", "55~75%"),
]
QB_BOOK_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800;900&display=swap');
@page vcover { size: A4; margin: 0; }
.vcover { width: 210mm; height: 297mm; margin: 0 auto 10mm; position: relative; overflow: hidden; background: #fff;
  box-shadow: 0 0 12px rgba(0,0,0,.12); font-family: 'Inter', 'Noto Sans KR', sans-serif; color: #0f172a; break-after: page; page-break-after: always; }
/* 프린터가 종이 가장자리 5~10mm를 못 찍어서, 표지·소개 쪽의 글자와 띠는 모두 종이 끝에서 25mm 이상 안쪽에 둠 */
.vcover .bar { position: absolute; top: 10mm; left: 14mm; right: 14mm; height: 5mm; border-radius: 3px; background: linear-gradient(90deg, #0f172a 0%, #0047AB 65%, #38bdf8 100%); }
.vcover .inner { position: absolute; inset: 30mm 28mm 26mm; display: flex; flex-direction: column; justify-content: space-between; }
.vcover .top { display: flex; justify-content: space-between; border-bottom: 1.5px solid #0f172a; padding-bottom: 12px; font-size: 10.5px; font-weight: 800; letter-spacing: 3px; color: #0047AB; }
.vcover .pill { display: inline-block; background: #b91c1c; color: #fff; font-size: 10.5px; font-weight: 800; letter-spacing: 2px; padding: 5px 12px; border-radius: 4px; margin-bottom: 18px; }
.vcover h1 { font-size: 50px; font-weight: 900; letter-spacing: -1.5px; line-height: 1.02; margin: 0 0 14px; white-space: nowrap; }
.vcover h1 span { color: #0047AB; }
.vcover .en { font-size: 15px; font-weight: 600; color: #475569; margin-bottom: 8px; }
.vcover .ko { font-size: 14px; font-weight: 500; color: #334155; }
.vcover .rule { width: 65px; height: 5px; background: #0047AB; margin: 26px 0 20px; }
.vcover .ttl { font-family: 'Noto Sans KR'; font-size: 17px; font-weight: 900; margin-bottom: 4px; }
.vcover .meta { font-family: 'Noto Sans KR'; font-size: 12px; color: #475569; }
.vcover .credit { font-size: 13px; font-weight: 700; letter-spacing: 1.2px; margin-top: 18px; }
.vcover .credit b { color: #0047AB; font-weight: 900; }
.vcover .bottom { display: flex; justify-content: space-between; align-items: flex-end; border-top: 1px solid #cbd5e1; padding-top: 16px; font-size: 11.5px; color: #475569; line-height: 1.6; }
.vcover .student { width: 210px; border: 1.5px solid #0f172a; border-radius: 6px; padding: 8px 14px; background: #f8fafc; color: #0f172a; font-weight: 700; }
.vcover .student div { display: flex; justify-content: space-between; border-bottom: 1px dashed #cbd5e1; padding: 5px 0; }
.vcover .student div:last-child { border-bottom: 0; }
.vcover.blank { box-shadow: 0 0 12px rgba(0,0,0,.06); }
.vcover.back .inner { justify-content: center; align-items: center; text-align: center; }
.vcover.back h1 { font-size: 46px; }
.vcover.back .line { font-family: 'Noto Sans KR'; font-size: 13px; color: #334155; margin: 10px 0 26px; }
.vcover.back .series { font-size: 12px; font-weight: 600; color: #64748b; letter-spacing: .5px; }
.vcover.back .series b { color: #0047AB; }
.vcover.back .foot2 { position: absolute; bottom: 26mm; left: 28mm; right: 28mm; border-top: 1px solid #cbd5e1; padding-top: 12px;
  font-family: 'Noto Sans KR'; font-size: 11px; color: #475569; display: flex; justify-content: space-between; }
.vcover.copy .inner { justify-content: flex-end; align-items: flex-end; }
.vcover.copy .cbox { text-align: right; max-width: 125mm; border-right: 4px solid #0047AB; padding-right: 14px; }
.vcover.copy .cbox .t { font-size: 12px; font-weight: 900; letter-spacing: 1.5px; margin-bottom: 6px; }
.vcover.copy .cbox .k { font-family: 'Noto Sans KR'; font-size: 10.5px; font-weight: 700; color: #334155; line-height: 1.65; word-break: keep-all; margin-bottom: 6px; }
.vcover.copy .cbox .e { font-size: 9px; color: #94a3b8; line-height: 1.4; }
@page vintro { size: A4; margin: 20mm 25mm; }
.vpage { width: 210mm; min-height: 297mm; box-sizing: border-box; padding: 20mm 25mm; margin: 0 auto 10mm; background: #fff; box-shadow: 0 0 12px rgba(0,0,0,.12);
  font-family: 'Noto Sans KR', sans-serif; color: #0f172a; break-after: page; page-break-after: always; line-height: 1.6; text-align: left; }
.vpage .kicker { font-family: 'Inter'; font-size: 10px; font-weight: 800; letter-spacing: 3px; color: #0047AB; }
.vpage h2 { font-size: 21pt; font-weight: 900; margin: 4px 0 4px; letter-spacing: -.5px; }
.vpage .lead { font-size: 12pt; font-weight: 700; color: #0047AB; margin: 0 0 10px; }
.vpage p { font-size: 10pt; margin: 0 0 8px; word-break: keep-all; }
.vpage .step { border-left: 4px solid #0047AB; padding: 2px 0 2px 12px; margin: 12px 0; }
.vpage .step h3 { font-size: 11.5pt; margin: 0 0 3px; }
.vpage .step h3 span { font-family: 'Inter'; color: #0047AB; font-weight: 900; margin-right: 6px; }
.vpage .step ul { margin: 2px 0 0; padding-left: 16px; font-size: 9.8pt; }
.vpage table { width: 100%; border-collapse: collapse; font-size: 9.5pt; margin: 5px 0 4px; }
.vpage th, .vpage td { border: 1px solid #94a3b8; padding: 4px 6px; text-align: center; }
.vpage th { background: #eef2ff; }
.vpage td:first-child { font-weight: 700; }
.vpage .two { display: flex; gap: 10mm; }
.vpage .two > div { flex: 1; border: 1.5px solid #0f172a; border-radius: 8px; padding: 10px 14px; }
.vpage .two h3 { font-size: 13pt; margin: 0 0 6px; }
.vpage .two h3.u { color: #0047AB; } .vpage .two h3.j { color: #b91c1c; }
.vpage .two ul { margin: 0; padding-left: 16px; font-size: 9.8pt; }
.vpage .two li { margin-bottom: 6px; }
.vpage .so { background: #f1f5f9; border-radius: 8px; padding: 10px 16px; margin-top: 12px; font-size: 10pt; }
.vpage .so ul { margin: 4px 0 0; padding-left: 16px; }
.vpage .quote { text-align: right; margin-top: 16px; font-size: 11pt; font-weight: 700; }
.vpage .quote span { display: block; font-size: 9.5pt; font-weight: 400; color: #475569; }
.vpage .howto li { margin-bottom: 12px; font-size: 10.5pt; }
.vpage .howto b { color: #0047AB; }
.vpage .series-tbl td.nw { white-space: nowrap; }
.vpage .series-tbl tr.me td { background: #eef2ff; font-weight: 700; }
.vcover.filler { display: none; }
body.addblank .vcover.filler { display: block; }
.printbar label { font-family: 'Noto Sans KR'; font-size: 12px; margin-right: 10px; color: #334155; }
@media print {
  .vcover { page: vcover; margin: 0 !important; box-shadow: none !important; height: 296mm; }
  .vpage { page: vintro; width: auto; min-height: 255mm; padding: 0; margin: 0 !important; box-shadow: none !important; background: #fff; }  /* Self-Note처럼 바탕이 회색인 교재에서도 소개 쪽은 흰 바탕 */
}
"""

def qb_book_stats(gmap: Dict, gidx: Dict) -> Dict:
    """소개 페이지에 넣을 숫자: 분석한 실제 기출 시험지 수·연도, 연결된 지문 수, 학교별 어법 선택지 수·많이 나온 개념"""
    exams = list((gmap.get("exams") or {}).values())
    years = sorted({int(m.group(1)) for ex in exams for m in [re.search(r"(\d{2})\s*년", str(ex.get("exam_name", "")))] if m})
    grammar = {}
    for school, text in QB_GRAMMAR_DNA.items():
        m = re.search(r"개념 빈도\((\d+)개 선택지\):\s*(.+)", text)
        if m:
            grammar[school] = (int(m.group(1)), [x.strip() for x in m.group(2).split(">")][:3])
    return {"exams": len(exams), "by_school": {sc: sum(1 for ex in exams if sc in str(ex.get("school", ""))) for sc in QB_SCHOOLS},
            "years": (2000 + years[0], 2000 + years[-1]) if years else None, "linked": len(gidx), "grammar": grammar}

def render_vision_book(meta: Dict, stats: Dict, n_total: int, n_real: int, total_min: int = 0) -> (str, str):
    """앞쪽 4쪽(앞표지·사용법·제작 소개 ①②)과 뒤쪽 2쪽(저작권·뒤표지) HTML"""
    e = html.escape
    acad = e(meta.get("academy") or "포피엠 입시 영어")
    yrs = stats.get("years")
    yr_txt = f"{yrs[0]}~{yrs[1]}년 " if yrs else ""
    n_ex = stats.get("exams", 0)
    by = stats.get("by_school", {})
    gram = stats.get("grammar", {})
    n_gram = sum(v[0] for v in gram.values())
    gram_rows = "".join(f"<tr><td>{e(sc)}</td><td>{v[0]}개</td><td>{e(' · '.join(v[1]))}</td></tr>" for sc, v in gram.items())
    pass_rows = "".join(f"<tr><td>{t}</td><td>{u}</td><td>{j}</td></tr>" for t, u, j in QB_PASSAGE_MEASURE)
    title = e(meta.get("title", ""))
    time_li = (f"<li><b>문항마다 적힌 ⏱ 권장 시간 안에 푸세요.</b><br>지문 단어 수와 유형(어법·조합형·순서·삽입 등)으로 계산한 시간이에요. "
               f"이 시험지 전체 권장 시간은 <b>약 {total_min}분</b>이에요. 시간 안에 못 푼 문제는 번호에 표시해 두었다가 해설로 다시 확인하세요.</li>"
               if total_min else "<li><b>실제 시험처럼 시간을 재고 푸세요.</b></li>")
    front = f"""
<div class='vcover'><div class='bar'></div><div class='inner'>
  <div class='top'><span>PQRS YOUNG ENGLISH CURRICULUM SERIES</span><span>ANATOMY SERIES</span></div>
  <div>
    <div class='pill'>REAL EXAM DNA EDITION &bull; 운양고 &middot; 제일고</div>
    <h1>ANATOMY <span>VISION</span></h1>
    <div class='en'>School-Specific Variant Exam System &mdash; Built from Real Exam DNA</div>
    <div class='ko'>운양고·제일고 실제 기출을 해부해, 출제자의 눈으로 다시 만든 변형문제</div>
    <div class='rule'></div>
    <div class='ttl'>{title}</div>
    <div class='meta'>총 {n_total}문항{f' · 실제 기출 {n_real}문항 포함' if n_real else ''}{f' · 권장 풀이 시간 약 {total_min}분' if total_min else ''}</div>
    <div class='credit'>Created &amp; Designed by <b>PQRS Young</b> &times; {acad}</div>
  </div>
  <div class='bottom'>
    <div><b style='color:#0f172a;font-size:13px'>PQRS Young &times; {acad}</b><br>High School English Exam &amp; CSAT Preparation</div>
    <div class='student'><div><span>SCHOOL</span><span style='font-weight:500;color:#64748b'>{e(meta.get('kind', ''))}</span></div><div><span>GRADE / CLASS</span><span></span></div><div><span>STUDENT NAME</span><span></span></div></div>
  </div>
</div></div>
<div class='vpage'>
  <div class='kicker'>HOW TO USE</div><h2>이 교재 사용법</h2>
  <ol class='howto'>
    {time_li}
    <li><b>문항 아래 작은 표시를 보세요.</b><br>실제 기출인지 변형문제인지, 난이도가 어떤지 알 수 있어요.</li>
    <li><b>'기출 이력'을 꼭 보세요.</b><br>같은 지문이 운양고·제일고 어느 시험 몇 번에 나왔는지 적혀 있어요. 굵게 표시된 것은 같은 유형으로 나온 문제예요.</li>
    <li><b>틀린 문제는 해설의 '근거'를 지문에서 직접 찾아 밑줄 치세요.</b><br>재진술 문제는 해설 끝 '원문 표현 → 바꾼 표현'을 외우는 것이 핵심이에요.</li>
  </ol>
</div>
<div class='vpage'>
  <div class='kicker'>HOW ANATOMY VISION WAS MADE</div><h2>ANATOMY VISION은 이렇게 만들었습니다</h2>
  <div class='lead'>"외운 지문으로는 풀 수 없게, 그러나 우리 학교 시험과 똑같이."</div>
  <p>시중 변형문제는 대부분 어느 학교 시험인지 상관없이 똑같이 만들어집니다. 그런데 운양고와 제일고는 같은 지문을 내도 묻는 방식이 전혀 다릅니다. ANATOMY VISION은 문제를 만들기 전에, 먼저 두 학교의 실제 시험지를 해부했습니다.</p>
  <div class='step'><h3><span>STEP 1</span>실제 기출 시험지 수집과 해부</h3><ul>
    <li>운양고·제일고 {yr_txt}중간·기말 시험지 <b>{n_ex}회분</b>(운양고 {by.get('운양고', 0)} · 제일고 {by.get('제일고', 0)})을 문항 하나하나 분석했습니다. 번호, 배점, 유형, 출처 지문을 모두 확인했습니다.</li>
    <li>출처 지문을 원문과 연결해, 지문 <b>{stats.get('linked', 0)}개</b>에 어느 시험 몇 번에 나왔는지 <b>기출 이력</b>을 만들었습니다.</li></ul></div>
  <div class='step'><h3><span>STEP 2</span>어법 선택지 {n_gram:,}개 전수 분석</h3>
    <table><tr><th>학교</th><th>선택지</th><th>많이 나온 개념 TOP 3</th></tr>{gram_rows}</table></div>
  <div class='step'><h3><span>STEP 3</span>지문을 얼마나 바꾸는지 측정</h3>
    <p style='font-size:9.5pt;margin:0'>실제 시험지 지문과 교과서·모의고사 원문을 네 단어씩 비교해, 원문 표현이 몇 % 남아 있는지 쟀습니다.</p>
    <table><tr><th>유형</th><th>운양고</th><th>제일고</th></tr>{pass_rows}</table>
    <ul><li><b>결론:</b> 운양고는 지문을 거의 그대로 두고 '주어진 문장'만 바꿔 씁니다. 제일고는 지문 본문까지 다시 써서, 외운 학생이 바로 찾지 못하게 합니다.</li></ul></div>
  <div class='step'><h3><span>STEP 4</span>학교별로 출제하고 정답을 한 번 더 검증</h3><ul>
    <li>위 분석 결과를 출제 기준으로 삼아 학교 스타일과 난이도(하·중·상·최상)별로 문제를 만들었습니다.</li>
    <li>정답은 따로 한 번 더 풀어 맞지 않는 문제는 버렸습니다.</li></ul></div>
</div>
<div class='vpage'>
  <div class='kicker'>THROUGH THE EXAMINER'S EYES</div><h2>운양고·제일고 출제자의 시선</h2>
  <div class='two'>
    <div><h3 class='u'>운양고 출제자는 이렇게 냅니다</h3><ul>
      <li><b>"원문을 잘 이해했는지 정확히 확인한다."</b><br>지문은 원문에 가깝게 두고, 주어진 문장과 선택지를 바꿔 씁니다.<br><i>at three degrees of warming</i> → <i>when the temperature rises by three or four more degrees</i></li>
      <li><b>"있는 대로 모두 고르게 한다."</b><br>어법은 ⓐ~ⓖ 중 틀린/옳은 것을 모두 고르는 조합형이 대부분이라, 하나만 알아서는 풀 수 없습니다.</li>
      <li><b>{e(gram.get('운양고', (0, ['수일치']))[1][0].split()[0])}를 가장 많이 묻습니다.</b><br>긴 수식어 뒤의 진짜 주어, 동명사구 주어 같은 함정입니다.</li></ul></div>
    <div><h3 class='j'>제일고 출제자는 이렇게 냅니다</h3><ul>
      <li><b>"외운 문장 모양으로는 못 찾게 한다."</b><br>지문 본문까지 동의어·구조 바꾸기로 재진술하고, 순서 문제는 도입부를 새로 쓰기도 합니다.</li>
      <li><b>"선택지가 길고 어휘가 어렵다."</b><br>주제·제목·요지 선택지를 학술적 상위 개념어로 바꿉니다.<br><i>help a child in distress</i> → <i>altruistic behavior</i></li>
      <li><b>{e(gram.get('제일고', (0, ['분사·분사구문']))[1][0].split()[0])}을 가장 많이 묻습니다.</b><br>감정 분사와 분사구문의 능동·수동을 특히 자주 내고, 도치 구문(So + 형용사, Not only)도 자주 활용합니다.</li></ul></div>
  </div>
  <div class='so'><b>그래서 ANATOMY VISION은</b><ul>
    <li>모든 변형문제에 <b>어느 학교 스타일로 만들었는지</b> 표시했습니다.</li>
    <li>실제 기출문제는 <b>'실제 기출'</b>로 따로 표시해 함께 실었습니다.</li>
    <li>해설마다 <b>근거 문장</b>과 <b>원문 표현 → 바꾼 표현</b>을 적었습니다.</li></ul></div>
  <div class='quote'>"우리 학교 시험지를 가장 많이 본 사람이 만든 변형문제."<span>— {acad} PQRS영T</span></div>
</div>"""
    back = f"""
<div class='vcover blank filler'></div>
<div class='vcover copy'><div class='inner'><div class='cbox'>
  <div class='t'>ANATOMY VISION &mdash; CREATED BY PQRS YOUNG</div>
  <div class='k'>본 교재의 모든 저작권은 제작자인 <b>PQRS영쌤</b>에게 있으며, 저작권자의 사전 서면 동의 없는 무단 복제, 배포, 전송, 스캔 및 도용 시 저작권법에 의거하여 민·형사상의 엄중한 법적 처벌을 받을 수 있습니다.</div>
  <div class='e'>Copyright &copy; PQRS Young ({acad}). All rights reserved.<br>Unauthorized reproduction, distribution, or plagiarism of this material is strictly prohibited and subject to legal prosecution.</div>
</div></div></div>
<div class='vcover back' style='break-after:auto;page-break-after:auto'><div class='bar'></div><div class='inner'>
  <h1>ANATOMY <span>VISION</span></h1>
  <div class='line'>실제 기출 {n_ex}회분 · 어법 선택지 {n_gram:,}개 분석 · 학교별 출제 DNA</div>
  <div class='series'>Anatomy One-Shot &middot; Self-Note &middot; Core &middot; Voca &middot; <b>Vision</b></div>
</div><div class='foot2'><span>장기동 {acad} · PQRS영T</span><span>031-991-1199</span></div></div>"""
    return front, back

# 문항별 권장 풀이 시간 (AI 없이 계산): 읽을 영어 단어 수(지문·주어진 글·박스·선택지) ÷ 분당 약 130단어 + 유형별로 따져 보는 시간
#   평균 지문이면 문항당 약 2분 → 24문항 약 50분(실제 내신 시험 시간과 비슷)
QB_TIME_EXTRA = {"어법 (단일)": 20, "어법 (조합형)": 40, "낱말 (문맥상)": 20, "<보기> 일치 조합형": 40, "순서": 25, "삽입": 25,
                 "무관한 문장": 15, "요약문 (A)(B)": 20, "빈칸": 20, "함축 의미": 15, "추론 (추론 가능·불가)": 30}
QB_TIME_LEVEL = {"최상": 1.15, "상": 1.05, "중": 1.0, "하": 0.9}

def qb_time_sec(r: Dict) -> int:
    c = r.get("content") or {}
    text = " ".join([str(c.get(k) or "") for k in ("given", "passage", "box")] + [str(x) for x in (c.get("choices") or [])])
    words = len(re.findall(r"[A-Za-z][A-Za-z'’-]*", re.sub(r"<[^>]+>", " ", text)))
    extra = 120 if r.get("q_type") in QB_ESSAY_TYPES else QB_TIME_EXTRA.get(r.get("q_type"), 10)
    sec = (20 + words * 60 / 130 + extra) * QB_TIME_LEVEL.get(r.get("difficulty"), 1.0)
    return int(min(300, max(60, round(sec / 10) * 10)))

def qb_time_text(sec: int) -> str:
    m, s_ = divmod(int(sec), 60)
    return f"{m}분" + (f" {s_}초" if s_ else "")

QB_VOCAB_CSS = """
.vocab-cols { display: flex; gap: 14px; align-items: flex-start; }
.vocab-page .vocab { width: 50%; border-collapse: collapse; font-size: 8.8pt; }
.vocab th, .vocab td { border-bottom: 1px solid #ccc; padding: 2.5px 5px; text-align: left; vertical-align: top; }
.vocab th { border-bottom: 2px solid #111; font-family: 'Noto Sans KR'; }
.vocab td.vw { font-weight: 700; width: 38%; }
.vocab td.vn { color: #666; white-space: nowrap; width: 12%; font-size: 8.5pt; }
.vocab tr { break-inside: avoid; }
"""
SCHEMA_QB_VOCAB = {"type": "object", "properties": {"words": {"type": "array", "items": {"type": "object", "properties": {
    "word": {"type": "string"}, "meaning": {"type": "string"}, "nos": {"type": "array", "items": {"type": "integer"}}},
    "required": ["word", "meaning", "nos"]}}}, "required": ["words"]}

def qb_choice_vocab(qs: List[Dict], model: str, logs: List[str]) -> List[Dict]:
    """시험지 문항의 선택지·<보기>에 나온 어휘를 뽑아 뜻과 나온 문항 번호로 정리 (가벼운 모델 1번 호출)"""
    blocks = []
    for n, r in enumerate(qs, 1):
        c = r.get("content") or {}
        parts = [str(x) for x in (c.get("choices") or []) if str(x).strip()] + ([str(c["box"])] if c.get("box") else [])
        if parts:
            blocks.append(f"[{n}번] " + " / ".join(re.sub(r"<[^>]+>", "", x) for x in parts))
    if not blocks:
        return []
    k = max(20, min(50, round(len(qs) * 2.5)))
    prompt = (f"아래는 고등학교 영어 시험지의 문항별 선택지와 <보기>다. 학생이 꼭 알아야 할 어휘를 {k}개 안팎으로 뽑아라.\n"
              "- 고1 수준 이상(쉬운 기초 단어 제외). 패러프레이즈에 쓰인 상위 개념어, 헷갈리게 만든 비슷한 철자 단어, 숙어·구동사·연어를 우선한다.\n"
              "- word는 사전형(동사 원형, 명사 단수)으로, 숙어는 'supply A with B'처럼. meaning은 그 문항에 쓰인 뜻의 짧은 한국어(필요하면 뜻 2개까지).\n"
              "- nos는 그 어휘가 나온 문항 번호들. 같은 어휘는 한 번만. 처음 나온 문항 번호 순서로 정렬.\n\n" + "\n".join(blocks))
    out = _qb_call(None, model, prompt, SCHEMA_QB_VOCAB, 0.2, logs, "선지 어휘 정리")
    words = [w for w in ((out or {}).get("words") or []) if isinstance(w, dict) and str(w.get("word", "")).strip()]
    words.sort(key=lambda w: min([int(x) for x in (w.get("nos") or []) if str(x).lstrip("-").isdigit()] or [999]))
    return words

# ---- 선지 해석본: 선지(<보기> 조합형은 <보기> 진술)를 한국어로 옮기고, 그 선지에 나온 단어 뜻을 바로 아래에 ----
#   한 번 해석한 문항은 문항 content['choice_ko']에 저장해서 다시 출력하면 AI 호출 없음 (선지가 바뀌면 다시 해석)
QB_TRANS_SKIP = {"순서", "어법 (조합형)"} | QB_MARK_TYPES | QB_ESSAY_TYPES  # 선지가 번호·조합뿐인 유형
QB_TRANS_BATCH = 6  # 한 번 호출에 넣는 문항 수 (답이 너무 길어져 잘리지 않게)
SCHEMA_QB_TRANS = {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {
    "index": {"type": "integer"},
    "lines": {"type": "array", "items": {"type": "object", "properties": {
        "ko": {"type": "string"},
        "words": {"type": "array", "items": {"type": "object", "properties": {"word": {"type": "string"}, "meaning": {"type": "string"}},
                                             "required": ["word", "meaning"]}}},
        "required": ["ko", "words"]}}},
    "required": ["index", "lines"]}}}, "required": ["items"]}

def qb_trans_parts(r: Dict) -> List[str]:
    """해석할 영어 줄 (없으면 빈 목록)"""
    c = r.get("content") or {}
    if r.get("q_type") in QB_TRANS_SKIP:
        return []
    if r.get("q_type") == "<보기> 일치 조합형":
        lines = [x for x in re.sub(r"(?i)<br\s*/?>", "\n", str(c.get("box") or "")).split("\n")]
    else:
        lines = [str(x) for x in (c.get("choices") or [])]
    lines = [re.sub(r"<[^>]+>", "", html.unescape(x)).strip() for x in lines]
    lines = [x for x in lines if x]
    return lines if sum(1 for x in lines if re.search(r"[A-Za-z]{3}", x)) >= 2 else []

def _qb_trans_hash(lines: List[str]) -> str:
    return hashlib.md5("\n".join(lines).encode("utf-8")).hexdigest()[:12]

def qb_trans_cached(r: Dict):
    parts = qb_trans_parts(r)
    ck = (r.get("content") or {}).get("choice_ko") or {}
    if parts and ck.get("h") == _qb_trans_hash(parts) and len(ck.get("lines") or []) == len(parts):
        return ck["lines"]
    return None

def qb_trans_estimate(qs: List[Dict], model: str) -> (int, float):
    """아직 해석 안 된 문항 수와 예상 비용(원)"""
    todo = [qb_trans_parts(r) for r in qs if qb_trans_parts(r) and qb_trans_cached(r) is None]
    n_lines = sum(len(p) for p in todo)
    p_in, _, p_out = ai_price(model)
    calls = -(-len(todo) // QB_TRANS_BATCH)
    return len(todo), (calls * 500 + n_lines * 45) * p_in / 1e6 * 1400 + n_lines * 120 * p_out / 1e6 * 1400

def qb_choice_trans(sb, qs: List[Dict], model: str, logs: List[str]) -> Dict[str, List[Dict]]:
    """문항 id → [{'en','ko','words'}] (저장된 해석은 그대로 쓰고, 없는 것만 가벼운 모델로 해석해 문항에 저장)"""
    out, todo = {}, []
    for r in qs:
        parts = qb_trans_parts(r)
        if not parts:
            continue
        got = qb_trans_cached(r)
        if got is not None:
            out[r["id"]] = [{"en": e, **g} for e, g in zip(parts, got)]
        else:
            todo.append((r, parts))

    def one(batch):
        blocks = [f"### 문항 {k}\n" + "\n".join(f"{j + 1}) {x}" for j, x in enumerate(parts)) for k, (_, parts) in enumerate(batch, 1)]
        prompt = ("아래는 고등학교 1학년 영어 시험 문항의 선지(또는 <보기> 진술)다. 선지가 길어서 어려워하는 학생을 위한 해석본을 만든다.\n"
                  "- 각 줄(1), 2), …)마다 ko = 자연스럽고 정확한 한국어 해석. 영어 문장 구조가 보이게 직역에 가깝게, 뜻을 빼거나 보태지 않는다. "
                  "'word1 …… word2' 같은 어휘 짝은 '(A) 뜻 …… (B) 뜻'처럼 옮긴다. 줄 앞의 ①·ⓐ 같은 기호는 ko에 다시 쓰지 않는다.\n"
                  "- words = 그 줄에 나온 단어·숙어 중 고1 학생이 모를 만한 것 2~5개(쉬운 기초 단어 제외, 짧은 줄은 0~2개). "
                  "word는 사전형(동사 원형, 명사 단수, 숙어는 'be influenced by'처럼), meaning은 그 줄에서 쓰인 뜻의 짧은 한국어.\n"
                  "- 문항마다 index(문항 번호)와, 받은 줄 수와 똑같은 개수의 lines를 순서대로 반환한다.\n\n" + "\n\n".join(blocks))
        res = _qb_call(None, model, prompt, SCHEMA_QB_TRANS, 0.2, logs, "선지 해석")
        got = {int(it.get("index", 0)): it.get("lines") or [] for it in ((res or {}).get("items") or []) if isinstance(it, dict)}
        done = []
        for k, (r, parts) in enumerate(batch, 1):
            lines = [{"ko": str(x.get("ko", "")).strip(),
                      "words": [{"word": str(w.get("word", "")).strip(), "meaning": str(w.get("meaning", "")).strip()}
                                for w in (x.get("words") or []) if isinstance(w, dict) and str(w.get("word", "")).strip()][:6]}
                     for x in got.get(k, []) if isinstance(x, dict)]
            if len(lines) == len(parts) and all(x["ko"] for x in lines):
                done.append((r, parts, lines))
        return done

    batches = [todo[i:i + QB_TRANS_BATCH] for i in range(0, len(todo), QB_TRANS_BATCH)]
    if batches:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(4, len(batches))) as pool:
            results = [x for b in pool.map(one, batches) for x in b]
        for r, parts, lines in results:
            out[r["id"]] = [{"en": e, **g} for e, g in zip(parts, lines)]
            c = dict(r.get("content") or {})
            c["choice_ko"] = {"h": _qb_trans_hash(parts), "lines": lines}
            r["content"] = c
            if sb is not None:
                try:
                    sb.table(QB_TABLE).update({"content": c}, **_UPSERT_MIN).eq("id", r["id"]).execute()
                except Exception as e:
                    logs.append(f"⚠️ 해석 저장 실패(다음에 다시 해석): {str(e)[:80]}")
        miss = len(todo) - len(results)
        if miss:
            logs.append(f"⚠️ {miss}문항은 해석을 받지 못했어요 (한 번 더 누르면 그 문항만 다시 해석해요)")
    return out

# 선지 해석도 변형문제처럼 서버 작업으로 (2026-11-10): 창을 닫거나 인터넷이 끊겨도 서버에서 계속하고,
#   문항 6개마다 1건이라 끊기거나 멈춰도 해석한 문항은 이미 문제에 저장돼 있음 → 다시 누르면 남은 문항만 이어서 해석
def qb_trans_queued(job: Dict, qids: List[str]) -> int:
    """작업 목록에서 아직 해석을 기다리는 (이 문항들 중) 문항 수"""
    want = set(qids)
    return len({q for it in job.get("items", []) if it.get("kind") == "trans" and it.get("state") == "pending" for q in it.get("qids", []) if q in want})

def enqueue_trans(qs: List[Dict], text_model: str, api_delay: int, academy_name: str, school_name: str, note: str) -> int:
    """아직 해석 안 된 문항을 6개씩 작업에 추가 (이미 기다리는 문항은 다시 넣지 않음) → 추가한 문항 수"""
    sb = supabase if IS_CLOUD_DB else None
    todo = [r["id"] for r in qs if qb_trans_parts(r) and qb_trans_cached(r) is None]
    added = {"n": 0}

    def add(job):
        if job.get("status") not in ("running", "stopping"):
            job["status"] = "running"
            job["items"] = [it for it in job.get("items", []) if it.get("state") in ("pending", "failed", "waiting")]
            job["created_at"] = time.time()
            job["worker"] = None
        queued = {q for it in job.get("items", []) if it.get("kind") == "trans" and it.get("state") == "pending" for q in it.get("qids", [])}
        ids = [i for i in todo if i not in queued]
        job["opts"] = {"model": text_model, "api_delay": api_delay, "academy_name": academy_name, "school_name": school_name}
        for k in range(0, len(ids), QB_TRANS_BATCH):
            job["items"].append({"id": str(uuid.uuid4()), "kind": "trans", "label": f"선지 해석 · {note} · {k // QB_TRANS_BATCH + 1}",
                                 "qids": ids[k:k + QB_TRANS_BATCH], "light": True, "state": "pending", "error": ""})
        added["n"] = len(ids)
        if ids:
            _job_log(job, f"➕ 선지 해석 {len(ids)}문항 추가 ({note})")
    _job_update(sb, add)
    start_gen_thread()
    return added["n"]

def _trans_item(sb_factory, item: Dict, opts: Dict, logs: List[str]) -> (bool, str):
    """작업 1건: 문항 몇 개의 선지를 해석해 문제에 저장 (이미 해석된 문항은 건너뜀)"""
    sb = sb_factory()
    rows = [r for r in qb_fetch_ids(sb, item.get("qids") or []) if qb_trans_parts(r) and qb_trans_cached(r) is None]
    if not rows:
        return True, ""
    qb_choice_trans(sb, rows, job_item_model(item, opts), logs)
    left = [r for r in rows if qb_trans_cached(r) is None]
    if left:
        return False, f"{len(left)}문항은 해석을 받지 못했어요"
    logs.append(f"✅ 선지 해석 {len(rows)}문항 저장")
    return True, ""

QB_TRANS_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700;900&family=Noto+Serif+KR:wght@400;700&display=swap');
@page { size: A4; margin: 13mm 12mm; }
* { box-sizing: border-box; }
body { font-family: 'Noto Sans KR', sans-serif; font-size: 9.6pt; line-height: 1.5; color: #111; margin: 0; background: #fff;
  -webkit-print-color-adjust: exact; print-color-adjust: exact; }
.sheet { max-width: 186mm; margin: 0 auto; }
.head { border: 2px solid #111; padding: 6px 12px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: baseline; }
.head h1 { font-size: 15pt; font-weight: 900; margin: 0; }
.head .sub { font-size: 9pt; color: #444; }
.cols { column-count: 2; column-gap: 8mm; column-rule: 1px solid #bbb; }
.tq { margin: 0 0 10px; }
.tq .no { font-weight: 900; font-size: 10.5pt; break-after: avoid; border-bottom: 1px solid #111; margin-bottom: 4px; }
.tq .no span { font-weight: 400; font-size: 8.3pt; color: #555; margin-left: 4px; }
.tl { break-inside: avoid; margin: 0 0 6px; }
.tl .en { font-family: 'Noto Serif KR', serif; font-size: 9.4pt; padding-left: 1.3em; text-indent: -1.3em; }
.tl .ko { color: #0b3d91; padding-left: 1.3em; }
.tl .wd { font-size: 8.4pt; color: #333; padding-left: 1.3em; }
.tl .wd b { font-family: 'Noto Serif KR', serif; color: #111; }
.foot { text-align: center; font-size: 8pt; color: #555; margin-top: 8px; }
.printbar { position: sticky; top: 0; text-align: right; padding: 6px; background: #fff; }
.printbar button { font-size: 13px; padding: 6px 14px; cursor: pointer; }
@media print { .printbar { display: none; } }
"""

def render_choice_trans(qs: List[Dict], trans: Dict[str, List[Dict]], meta: Dict) -> str:
    """선지 해석본 HTML (문항 번호는 시험지와 같음, 해석이 없는 유형은 건너뜀)"""
    o = []
    for n, r in enumerate(qs, 1):
        lines = trans.get(r.get("id"))
        if not lines:
            continue
        box = r.get("q_type") == "<보기> 일치 조합형"
        o.append(f"<div class='tq'><div class='no'>{n}. <span>{html.escape(r['q_type'])}</span></div>")
        for j, x in enumerate(lines):
            en = html.escape(x["en"])
            if not box and not re.match(r"^[①-⑤]", x["en"]):
                en = f"{CIRCLED[j]} {en}"
            wd = " · ".join(f"<b>{html.escape(w['word'])}</b> {html.escape(w['meaning'])}" for w in x.get("words") or [])
            o.append(f"<div class='tl'><div class='en'>{en}</div><div class='ko'>→ {html.escape(x['ko'])}</div>"
                     + (f"<div class='wd'>{wd}</div>" if wd else "") + "</div>")
        o.append("</div>")
    body = "".join(o) or "<p>해석할 선지가 있는 문항이 없어요 (순서·삽입·어법·낱말 등은 선지가 번호라서 빠져요).</p>"
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(meta.get('title', ''))} 선지 해석</title>"
            f"<style>{QB_TRANS_CSS}</style></head><body><div class='printbar'><button onclick='window.print()'>🖨️ 인쇄 / PDF로 저장</button></div>"
            f"<div class='sheet'><div class='head'><h1>{html.escape(meta.get('title', ''))} · 선지 해석</h1>"
            f"<div class='sub'>{html.escape(meta.get('academy', ''))} · 문항 번호는 시험지와 같아요</div></div>"
            f"<div class='cols'>{body}</div><div class='foot'>{html.escape(meta.get('academy', ''))}</div></div></body></html>")

def qb_trans_words(qs: List[Dict], trans: Dict[str, List[Dict]]) -> List[Dict]:
    """해석본에 나온 단어 → 기출 어휘 단어장 형식 (같은 단어는 처음 나온 문항에 한 번만)"""
    seen, out = set(), []
    for n, r in enumerate(qs, 1):
        for x in trans.get(r.get("id")) or []:
            for w in x.get("words") or []:
                k = w["word"].lower()
                if k in seen:
                    continue
                seen.add(k)
                out.append({"word": w["word"], "pos": "", "pronunciation": "", "meaning": w["meaning"],
                            "category": f"{n}번 [선지]", "synonyms": [], "antonyms": []})
    return out

@st.cache_data(ttl=60, show_spinner=False)
def qb_rows_for_voca(_sb, labels: tuple) -> List[Dict]:
    """단어장 지문들에 딸린 문제 (연결 지문 'A + B'로 만든 문제도 각 지문에 포함)"""
    want = set(labels)
    links = [l for l in qb_link_labels(_sb) if set(qb_link_parts(l)[1]) & want]
    return qb_fetch(_sb, list(labels) + links, "id,passage_label,q_type,difficulty,content")

def qb_voca_choice_words(rows: List[Dict], label: str, have: set) -> List[Dict]:
    """한 지문의 변형·기출 문제 선지에 나온 단어 (해석본·선지 단어 뽑기로 저장된 것) → 단어장 행. 단어장에 이미 있는 단어는 뺌"""
    out, seen = [], set(have)
    for r in rows:
        if label not in qb_link_parts(r.get("passage_label", ""))[1]:
            continue
        for x in qb_trans_cached(r) or []:
            for w in x.get("words") or []:
                k = str(w.get("word", "")).strip().lower()
                if not k or k in seen:
                    continue
                seen.add(k)
                out.append({"word": w["word"], "pos": "", "pronunciation": "", "meaning": w.get("meaning", ""),
                            "category": "기출 선지" if r.get("difficulty") == QB_GICHUL_LEVEL else "변형 선지",
                            "synonyms": [], "antonyms": []})
    return out

def render_voca_choice_option(prefix: str, sel_passages: List[Dict], text_model: str) -> List[Dict]:
    """Anatomy Voca 출력: 그 지문 변형문제 선지 단어를 넣을지 고르기 (넣으면 출력용 사본에만 붙이고 저장된 단어장은 그대로)"""
    sb = supabase if IS_CLOUD_DB else None
    labels = tuple(p.get("label", "") for p in sel_passages)
    try:
        rows = qb_rows_for_voca(sb, labels)
    except Exception:
        return sel_passages
    with st.container(border=True):
        on = st.checkbox("➕ 변형문제 선지에 나온 단어도 같이 넣기 (지문마다 '변형 선지' 표시로 맨 뒤에)", key=f"{prefix}_with_choice")
        added = {p.get("label", ""): qb_voca_choice_words(rows, p.get("label", ""),
                                                          {str(w.get("word", "")).strip().lower() for w in (p.get("data") or {}).get("voca_list", [])})
                 for p in sel_passages}
        todo = [r for r in rows if qb_trans_parts(r) and qb_trans_cached(r) is None]
        model = light_model_for(text_model)
        n_todo, won = qb_trans_estimate(todo, model)
        c1, c2 = st.columns([3, 2], vertical_alignment="center")
        c1.caption(f"선지 단어 {sum(len(v) for v in added.values())}개 (단어장에 이미 있는 단어는 뺌) · "
                   f"선지 단어를 아직 안 뽑은 문제 {n_todo}개" + (f" → 뽑으면 약 {max(1, round(won)):,}원" if n_todo else ""))
        n_wait = qb_trans_queued(load_gen_job(), [r["id"] for r in todo]) if n_todo else 0
        if n_wait:
            c2.caption(f"⏳ 서버에서 선지 단어를 뽑는 중 ({n_wait}문제 남음) · 창을 닫아도 계속돼요")
            if c2.button("🔄 새로 보기", use_container_width=True, key=f"{prefix}_pull_refresh"):
                qb_rows_for_voca.clear()
                st.rerun()
        elif n_todo and c2.button(f"🔎 남은 {n_todo}문제 선지 단어 뽑기", use_container_width=True, key=f"{prefix}_pull_choice",
                                  help="문제마다 선지 해석과 단어를 한 번 뽑아 저장해요 (시험지 '선지 해석본'과 같은 데이터, 다시 뽑지 않음). "
                                       "서버 작업이라 창을 닫아도 계속되고, 뽑은 문제는 바로 저장돼요."):
            n = enqueue_trans(todo, text_model, api_delay, academy_name, school_name, "Voca 선지 단어")
            qb_rows_for_voca.clear()
            st.session_state.gen_flash = f"✅ 선지 단어 뽑기 {n}문제를 서버 작업에 넣었어요. 다 되면 '새로 보기'를 누르세요."
            st.rerun()
    if not on:
        return sel_passages
    out = []
    for p in sel_passages:
        extra = added.get(p.get("label", ""), [])
        if extra:
            p = copy.deepcopy(p)
            p.setdefault("data", {})["voca_list"] = list(p["data"].get("voca_list", [])) + extra
        out.append(p)
    return out

def render_choice_trans_ui(sb, key: str, qids: List[str], title: str, text_model: str, academy_name: str, school_name: str):
    """시험지·클리닉 아래: 선지 해석본 만들기 + 단어장에 넣기
    해석은 서버 작업으로 (창을 닫아도 계속, 끊겨도 해석한 문항은 저장). 모두 해석되면 버튼을 누르는 즉시 HTML (AI 비용 없음)"""
    with st.container(border=True):
        job = load_gen_job()
        n_wait = qb_trans_queued(job, qids)
        cached = st.session_state.get(f"{key}_rows")
        # 아직 해석 안 된 문항이 있으면 매번 새로 읽음 (서버 작업이 그사이 저장했을 수 있어서 · 문항 수십 개라 가벼움)
        if (not cached or cached[0] != qids or n_wait or st.session_state.pop(f"{key}_refresh", False)
                or any(qb_trans_parts(r) and qb_trans_cached(r) is None for r in cached[1])):
            got_rows = {r["id"]: r for r in qb_fetch_ids(sb, qids)}
            st.session_state[f"{key}_rows"] = (qids, [got_rows[i] for i in qids if i in got_rows])
        qs = st.session_state[f"{key}_rows"][1]
        model = light_model_for(text_model)
        n_todo, won = qb_trans_estimate(qs, model)
        n_has = sum(1 for r in qs if qb_trans_parts(r))
        n_done = n_has - n_todo
        c1, c2 = st.columns([2, 3], vertical_alignment="center")
        if n_todo and not n_wait:
            go = c1.button(f"📝 선지 해석 시작 ({n_todo}문항)", use_container_width=True, key=f"{key}_btn", type="primary")
            c2.caption(f"선지가 문장인 문항 {n_has}개 중 해석된 {n_done}개 · 남은 {n_todo}문항 해석 예상 약 {max(1, round(won)):,}원 · "
                       "서버에서 해석해서 창을 닫거나 인터넷이 끊겨도 계속되고, 해석한 문항은 바로 저장돼요 (멈추면 남은 문항부터 이어서)")
            if go:
                n = enqueue_trans(qs, text_model, api_delay, academy_name, school_name, title)
                st.session_state.gen_flash = f"✅ 선지 해석 {n}문항을 서버 작업에 넣었어요. 위 작업 칸에서 진행을 볼 수 있고, 다 되면 여기서 '선지 해석본 만들기'를 누르세요."
                st.rerun()
        elif n_wait:
            c1.button("🔄 진행 상황 새로 보기", use_container_width=True, key=f"{key}_refresh_btn",
                      on_click=lambda: st.session_state.__setitem__(f"{key}_refresh", True))
            c2.markdown(f"<div class='qb-note' style='margin-top:0'>⏳ 서버에서 해석 중 · <b>{n_done}/{n_has}문항</b> 완료 · 기다리는 {n_wait}문항<br>"
                        "창을 닫아도 계속돼요. 다 되면 이 자리에 '선지 해석본 만들기'가 나와요.</div>", unsafe_allow_html=True)
        else:
            c2.caption(f"선지가 문장인 문항 {n_has}개 모두 해석돼 있어요 (순서·삽입·어법·낱말은 빠짐) · 만들기는 AI 비용 0원")
        if not n_wait and n_has and (not n_todo or n_done):
            mk = st.columns([2, 3])[0].button("📄 선지 해석본 만들기" + (f" (해석된 {n_done}문항만)" if n_todo else ""), use_container_width=True,
                                              key=f"{key}_make", disabled=not n_done)
            if mk:
                trans = qb_choice_trans(sb, [r for r in qs if qb_trans_cached(r) is not None], model, [])  # 저장된 해석만 씀 (AI 호출 없음)
                meta = {"title": title, "academy": academy_name}
                st.session_state[f"{key}_out"] = (qids, render_choice_trans(qs, trans, meta), qb_trans_words(qs, trans), [])
        got = st.session_state.get(f"{key}_out")
        if got and got[0] == qids:
            _, page, words, logs = got
            for m in logs:
                st.warning(m)
            safe = re.sub(r'[\\/:*?"<>|]', "", title).strip() or "시험지"
            d1, d2 = st.columns([3, 2])
            d1.download_button("📥 선지 해석본 HTML 다운로드 — 열어서 '인쇄 / PDF로 저장'", page, f"{safe}_선지해석.html", "text/html",
                               use_container_width=True, key=f"{key}_dl")
            d2.caption(f"📚 선지 단어 {len(words)}개는 문제마다 그 지문 이름으로 저장됐어요. "
                       "Anatomy Voca에서 그 지문 단어장을 출력할 때 '변형문제 선지 단어도 같이 넣기'를 켜면 함께 나와요.")
            with st.expander("해석본 미리 보기", expanded=False):
                components.html(page, height=700, scrolling=True)

def render_qbank_exam(qs: List[Dict], meta: Dict, gidx: Dict[str, List[Dict]]) -> str:
    def tag_of(r):
        """1줄: 지문 · 유형 + [실제 기출]/[변형 · ○○고 스타일 · 난이도]  2줄: 기출 이력 (같은 시험은 번호를 묶고, 같은 유형은 굵게)"""
        c = r["content"]
        src = c.get("source") if r["difficulty"] == QB_GICHUL_LEVEL else None
        if src:
            badge = (f"<span class='bd real'>실제 기출</span> {html.escape(src.get('exam', ''))} {html.escape(src.get('no', ''))}번"
                     + (f" · {html.escape(re.sub(r'\s*점\s*$', '', str(src['points'])))}점" if src.get("points") else ""))
        else:
            style = f"{html.escape(r.get('school') or '')} 스타일 · " if meta.get("show_style", True) else ""
            badge = f"<span class='bd var'>변형 · {style}{html.escape(r['difficulty'])}</span>"  # 🎣 표시는 학생이 알아채지 않게 해설 쪽에만
        t = f"▶ {html.escape(r['passage_label'])} · {html.escape(r['q_type'])} {badge}"
        if meta.get("show_gichul"):
            groups = {}
            for h in gidx.get(r["passage_label"], []):
                if src and h["exam_id"] == src.get("exam_id") and h["no"] == src.get("no"):
                    continue  # 이 문제 자신은 이력에서 뺌
                g = groups.setdefault(h["exam_id"], {"short": h["short"], "nos": [], "types": [], "same": False})
                g["nos"].append(h["no"])
                if h["q_type"] not in g["types"]:
                    g["types"].append(h["q_type"])
                g["same"] |= h["q_type"] == r["q_type"]
            if groups:
                items = []
                for g in groups.values():
                    one = f"{html.escape(g['short'])} {html.escape('·'.join(g['nos']))}번({html.escape('·'.join(g['types']))})"
                    items.append(f"<b>{one}</b>" if g["same"] else one)
                t += f"<br><span class='hist'>{'다른 ' if src else ''}기출 이력: {' · '.join(items)}</span>"
        return t

    def kind_short(r):
        if r["difficulty"] == QB_GICHUL_LEVEL:
            return "기출"
        return (f"변({(r.get('school') or '?')[:1]})" if meta.get("show_style", True) else "변형") + ("🎣" if qb_is_yt(r) else "")

    total_min = round(sum(qb_time_sec(r) for r in qs) / 60)
    if meta.get("show_time", True):
        meta = {**meta, "sub": " · ".join(x for x in [meta.get("sub", ""), f"권장 풀이 시간 약 {total_min}분"] if x)}
    book = meta.get("book_stats")
    front, back = render_vision_book(meta, book, len(qs), sum(1 for r in qs if r["difficulty"] == QB_GICHUL_LEVEL),
                                     total_min if meta.get("show_time", True) else 0) if book else ("", "")
    blank_toggle = ("<label title='양면 인쇄 때 뒤표지가 맨 뒷면에 오게 하려면 전체 쪽수가 짝수여야 해요'>"
                    "<input type='checkbox' onchange=\"document.body.classList.toggle('addblank', this.checked)\"> "
                    "인쇄 미리보기 쪽수가 홀수면 체크 → 뒤표지 앞에 빈 쪽 넣기</label>") if book else ""
    o = [f"<div class='printbar'>{blank_toggle}<button onclick='window.print()'>🖨️ 인쇄 / PDF로 저장</button></div>{front}<div class='sheet'>",
         f"""<div class='head'><div class='top'><div class='brand'><b>PQRS</b>영T {html.escape(meta.get('brand', '변형문제 TEST'))}</div><div class='kind'>{html.escape(meta.get('kind', ''))}</div></div>
<h1>{html.escape(meta.get('title', ''))}</h1><div class='sub'>{html.escape(meta.get('sub', ''))}</div>
<div class='info'><div>반 :</div><div>번호 :</div><div>이름 :</div><div>점수 :</div></div></div><div class='cols'>"""]
    keys = [qb_set_key(r) for r in qs]
    for n, r in enumerate(qs, 1):
        c = r["content"]
        k = keys[n - 1]
        in_set = k is not None and ((n >= 2 and keys[n - 2] == k) or (n < len(qs) and keys[n] == k))
        if in_set and (n < 2 or keys[n - 2] != k):
            # 실제 기출처럼 한 지문에 문제 여러 개: 지문을 한 번만 싣고 아래에 문제들을 이어 붙임
            last = n
            while last < len(qs) and keys[last] == k:
                last += 1
            o.append(f"<div class='q setq'><div class='stem sethead'>[{n}~{last}] 다음 글을 읽고 물음에 답하시오.</div>"
                     f"<div class='passage'>{qb_paras(c.get('passage'))}</div></div>")
        tm = f"<span class='tm'>⏱ {qb_time_text(qb_time_sec(r))}</span>" if meta.get("show_time", True) else ""
        o.append(f"<div class='q'><div class='stem'>{tm}<span class='no'>{n}.</span> {qb_html(c.get('stem'))}</div>")
        if meta.get("show_tag"):
            o.append(f"<span class='tag'>{tag_of(r)}</span>")
        if c.get("given"):
            o.append(f"<div class='given'>{qb_paras(c['given'])}</div>")
        if not in_set:
            o.append(f"<div class='passage'>{qb_paras(c.get('passage'), r['q_type'] == '순서', c.get('given'))}</div>")
        if c.get("box"):
            # 긴 <보기>(진술 7~8개)는 통째로 다음 단으로 밀리면 앞 단이 비어서, 단을 넘어 이어지게
            o.append(f"<div class='box{' long' if len(str(c['box'])) > 500 else ''}'>{qb_paras(c['box'], summary=r['q_type'] == '요약문 (A)(B)')}</div>")
        choices = c.get("choices") or []
        if r["q_type"] in QB_ESSAY_TYPES:
            o.append("<div class='write'></div><div class='write'></div>")
        elif all(not str(x).strip() for x in choices):
            o.append("<div class='choices inline'>" + "".join(f"<div>{ch}</div>" for ch in CIRCLED) + "</div>")
        else:
            o.append("<div class='choices'>" + "".join(f"<div>{CIRCLED[i]} {qb_html(re.sub(r'^\s*[①-⑤]\s*', '', str(x)))}</div>" for i, x in enumerate(choices[:5])) + "</div>")
        o.append("</div>")
    o.append("</div>")
    if meta.get("with_answers"):
        o.append("<div class='ans-page'><div class='anshead'>정답 및 해설</div>")
        for i in range(0, len(qs), 10):
            part = qs[i:i + 10]
            pad = "<td></td>" * (10 - len(part))  # 마지막 줄도 10칸 너비로 맞춤
            o.append("<table class='key'><tr><th>번호</th>" + "".join(f"<th>{i + k + 1}</th>" for k in range(len(part))) + pad.replace("td", "th") + "</tr><tr><th>정답</th>"
                     + "".join(f"<td><b>{'서술형' if r['q_type'] in QB_ESSAY_TYPES else html.escape(r['content'].get('answer', ''))}</b></td>" for r in part) + pad
                     + "</tr><tr><th>구분</th>" + "".join(f"<td>{html.escape(kind_short(r))}</td>" for r in part) + pad + "</tr></table>")
        o.append("<div class='keynote'>구분: 기출 = 실제 기출문제 · " + ("변(운) = 운양고 스타일 변형 · 변(제) = 제일고 스타일 변형" if meta.get("show_style", True) else "변형 = 변형문제")
                 + (" · 🎣 = 영쌤 출제 포인트 문항" if any(qb_is_yt(r) for r in qs) else "") + " · 굵은 기출 이력 = 같은 유형으로 출제됨</div>")
        o.append("<div class='cols'>")
        for n, r in enumerate(qs, 1):
            c = r["content"]
            ans = qb_html(c.get("answer", ""))
            src = c.get("source") if r["difficulty"] == QB_GICHUL_LEVEL else None
            kind = (f"실제 기출 {src.get('short', '')} {src.get('no', '')}번" if src else
                    (f"{r.get('school') or ''} 스타일 변형 · {r['difficulty']}" if meta.get("show_style", True) else f"변형 · {r['difficulty']}")
                    + (" · 🎣 영쌤 포인트" if qb_is_yt(r) else ""))
            note = f"<li><b>⚠️ {qb_html(c['answer_note'])}</b></li>" if c.get("answer_note") else ""
            o.append(f"<div class='exp'><h3>{n}번 정답 {ans}<span>({html.escape(r['passage_label'])} · {html.escape(r['q_type'])} · {html.escape(kind)})</span></h3><ul>"
                     + note + "".join(f"<li>{qb_html(e)}</li>" for e in (c.get("explanation") or [])) + "</ul></div>")
        o.append("</div></div>")
    if meta.get("vocab"):
        # 선지 어휘 정리 (주간 클리닉): 문항 순서대로, 2단 표
        vs = meta["vocab"]
        half = (len(vs) + 1) // 2
        tbl = lambda part: ("<table class='vocab'><tr><th>어휘</th><th>뜻</th><th>문항</th></tr>" + "".join(
            f"<tr><td class='vw'>{html.escape(str(v.get('word', '')))}</td><td>{html.escape(str(v.get('meaning', '')))}</td>"
            f"<td class='vn'>{html.escape(', '.join(str(x) for x in (v.get('nos') or [])))}</td></tr>" for v in part) + "</table>")
        o.append("<div class='ans-page vocab-page'><div class='anshead'>선지 어휘 정리</div>"
                 "<div class='keynote'>문제 선택지·&lt;보기&gt;에 나온 어휘 · 오른쪽 숫자 = 나온 문항 번호</div>"
                 f"<div class='vocab-cols'>{tbl(vs[:half])}{tbl(vs[half:])}</div></div>")
    o.append("<div class='foot'>PQRS 영T 자체제작 · 장기동 포피엠입시영어 031-991-1199</div></div>")
    o.append(back)
    css = QB_EXAM_CSS + (QB_BOOK_CSS if book else "") + (QB_VOCAB_CSS if meta.get("vocab") else "")
    return f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{html.escape(meta.get('title', '변형문제'))}</title><style>{css}</style></head><body>{''.join(o)}</body></html>"

# ---- 화면 ----
# 지문 고르기: 출처 → 학년 → 시험을 고르면 번호가 칩으로 펼쳐지고, 칩 옆 작은 숫자는 쌓인 문제 수.
# 여러 시험에서 고른 번호는 아래 '고른 지문'에 모여 함께 쓰임
QB_MOCK_ROWS = [(18, 22), (23, 34), (35, 45)]  # 모의고사 번호 줄 나누기 (대의 파악 · 세부·어법·빈칸 · 순서·삽입·장문)
QB_SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")

QB_PICKER_CSS = """<style>
[class*="st-key-qb_card_"] { border: 1px solid rgba(128,128,128,.16); border-radius: 20px; padding: 18px 22px 14px;
  background: rgba(128,128,128,.03); box-shadow: 0 1px 2px rgba(0,0,0,.03), 0 12px 32px rgba(0,0,0,.04); gap: .5rem; }
.qb-title { font-size: 15px; font-weight: 700; letter-spacing: -.01em; margin-bottom: 6px; }
.qb-title span { font-size: 12px; font-weight: 400; color: rgba(128,128,128,1); margin-left: 8px; }
.qb-label { font-size: 13px; font-weight: 600; color: rgba(128,128,128,1); padding-top: 6px; white-space: nowrap; font-variant-numeric: tabular-nums; }
.qb-note { font-size: 12px; color: rgba(128,128,128,1); line-height: 1.6; margin: -2px 0 6px; }
.qb-note b { font-weight: 600; color: inherit; }
.qb-hr { border: none; border-top: 1px solid rgba(128,128,128,.15); margin: 0; }
[class*="st-key-qbsep_"] { border-top: 1px solid rgba(128,128,128,.15); border-bottom: 1px solid rgba(128,128,128,.15); padding: 12px 0; gap: .45rem; }
.qb-basket { background: rgba(10,132,255,.07); border-radius: 12px; padding: 9px 14px; font-size: 13px; line-height: 1.65; }
.qb-basket .muted { color: rgba(128,128,128,1); }
.qb-basket b { font-weight: 600; }
[class*="st-key-qb_card_"] [data-testid="stBaseButton-pills"] { border-radius: 999px !important; background: transparent !important;
  border: 1px solid rgba(128,128,128,.28) !important; font-variant-numeric: tabular-nums; transition: background .15s ease, border-color .15s ease; }
[class*="st-key-qb_card_"] [data-testid="stBaseButton-pills"]:hover { border-color: #0A84FF !important; }
[class*="st-key-qb_card_"] [data-testid="stBaseButton-pillsActive"] { border-radius: 999px !important; font-variant-numeric: tabular-nums; }
[class*="st-key-qb_card_"] [data-testid="stBaseButton-pillsActive"],
[class*="st-key-qb_card_"] [data-testid="stBaseButton-segmented_controlActive"] { background: #0A84FF !important; border-color: #0A84FF !important; color: #fff !important; }
[class*="st-key-qb_card_"] [data-testid="stBaseButton-pillsActive"] *,
[class*="st-key-qb_card_"] [data-testid="stBaseButton-segmented_controlActive"] * { color: #fff !important; }
/* Streamlit 1.60대 이후(클라우드): 칩 줄이 가로 스크롤 한 줄이 되고 버튼 표시가 aria-pressed(칩)·aria-checked(막대 버튼)로 바뀜 → 줄바꿈 + 같은 색 */
[class*="st-key-qb_card_"] [data-testid="stButtonGroup"] > div { flex-wrap: wrap !important; overflow: visible !important; row-gap: 6px !important; }
[class*="st-key-qb_card_"] [data-testid="stButtonGroup"] button[aria-pressed] { background: transparent !important; border: 1px solid rgba(128,128,128,.28) !important;
  font-variant-numeric: tabular-nums; transition: background .15s ease, border-color .15s ease; }
[class*="st-key-qb_card_"] [data-testid="stButtonGroup"] button[aria-pressed]:hover { border-color: #0A84FF !important; }
[class*="st-key-qb_card_"] [data-testid="stButtonGroup"] button[aria-pressed="true"],
[class*="st-key-qb_card_"] [data-testid="stButtonGroup"] button[role="radio"][aria-checked="true"] { background: #0A84FF !important; border-color: #0A84FF !important; color: #fff !important; }
[class*="st-key-qb_card_"] [data-testid="stButtonGroup"] button[aria-pressed="true"] *,
[class*="st-key-qb_card_"] [data-testid="stButtonGroup"] button[role="radio"][aria-checked="true"] * { color: #fff !important; }
[class*="st-key-qb_card_"] [data-testid="stBaseButton-tertiary"] { color: #0A84FF !important; padding: 0 4px; min-height: 30px; white-space: nowrap; }
[class*="st-key-qb_card_"] [data-testid="stBaseButton-tertiary"] p { font-size: 13px; white-space: nowrap; }
[class*="st-key-qb_card_"] [data-testid="stBaseButton-tertiary"]:disabled { color: rgba(128,128,128,.6) !important; }
</style>"""

@st.cache_data(ttl=60, show_spinner=False)
def qb_bank_summary(_sb, labels: tuple) -> List[Dict]:
    """지문별·유형별 개수 표시용 (가벼운 열만, 1분마다 새로 읽음)"""
    return qb_fetch(_sb, list(labels), "id,passage_label,q_type,difficulty,school")

# 저장한 시험지 기록: 시험지마다 넣은 문항 id → 다음 시험지에서 '이미 낸 문항 빼기' (app_storage 문서 하나, 읽기는 1분 캐시)
QB_EXAM_LOG_FILE = os.path.join(BASE_DIR, "qb_exam_log.json")

@st.cache_data(ttl=60, show_spinner=False)
def qb_exam_log(_sb) -> List[Dict]:
    return (_store_get(_sb, QB_EXAM_LOG_FILE, {}) or {}).get("records", [])

def qb_exam_log_update(sb, change):
    """저장 직전에 최신 기록을 다시 읽어 바꿈 (다른 창에서 저장한 기록을 덮어쓰지 않게)"""
    records = (_store_get(sb, QB_EXAM_LOG_FILE, {}) or {}).get("records", [])
    _store_put(sb, QB_EXAM_LOG_FILE, {"records": change(records)})
    qb_exam_log.clear()

def _qb_is_mock(p: Dict) -> bool:
    return bool(p.get("year") and p.get("month"))

def _qb_group_name(ps: List[Dict]) -> str:
    p = ps[0]
    return f"{str(p['year'])[2:]}년 {p['month']}월" if _qb_is_mock(p) else (p.get("exam_key") or extract_group_prefix(p["label"])).strip("[] ")

def _qb_chip_counts(rows: List[Dict], field: str) -> Dict[str, int]:
    out = {}
    for r in rows:
        out[r[field]] = out.get(r[field], 0) + 1
    return out

def _qb_sup(n) -> str:
    return f" {str(n).translate(QB_SUP)}" if n else ""

def _qb_row(label: str):
    """설정 화면처럼 왼쪽 회색 이름 + 오른쪽 조작 칸 한 줄"""
    c1, c2 = st.columns([1.15, 8.85])
    c1.markdown(f"<div class='qb-label'>{label}</div>", unsafe_allow_html=True)
    return c2

def _qb_hr():
    st.markdown("<hr class='qb-hr'>", unsafe_allow_html=True)

def _qb_count_line(holder, counts: Dict[str, int], names: List[str]):
    """칩 아래에 고른 지문 기준 개수 (칩 글자를 바꾸면 Streamlit이 다른 칩으로 여겨 선택이 풀리므로 따로 표시)"""
    have = [f"{n} <b>{counts[n]}</b>" for n in names if counts.get(n)]
    holder.markdown(f"<div class='qb-note'>{'고른 지문에 있는 문제 &nbsp;' + ' · '.join(have) if have else '고른 지문에 해당하는 문제가 아직 없어요.'}</div>",
                    unsafe_allow_html=True)

# ---- 주간 클리닉: 그 주에 배운 지문만으로 클리닉 미니모의고사 (기본 20문항 · 한 학교 스타일 · 변형문제만) ----
#   app_storage 문서 weekly_clinic = {"weeks": [{id, name, school, labels, used: [문항 id], updated}]}
QB_CLINIC_FILE = os.path.join(BASE_DIR, "weekly_clinic.json")
QB_CLINIC_MIX = {"중": 4, "상": 8, "최상": 8}       # 20문항 기준 난이도 배분 (문항 수를 바꾸면 비율대로)
QB_CLINIC_TYPES = [t for t in QB_TYPE_NAMES if t not in QB_ESSAY_TYPES]
QB_CLINIC_EXTRA = 1.3                                 # 모자란 문제를 만들 때 정답 검증 탈락·난이도 낮춤을 생각해 더 만드는 비율

@st.cache_data(ttl=60, show_spinner=False)
def qb_clinic_weeks(_sb) -> List[Dict]:
    return (_store_get(_sb, QB_CLINIC_FILE, {}) or {}).get("weeks", [])

def qb_clinic_update(sb, change):
    """저장 직전에 최신 목록을 다시 읽어 바꿈"""
    weeks = (_store_get(sb, QB_CLINIC_FILE, {}) or {}).get("weeks", [])
    _store_put(sb, QB_CLINIC_FILE, {"weeks": change(weeks)})
    qb_clinic_weeks.clear()

def qb_clinic_mix(total: int, levels: List[str]) -> Dict[str, int]:
    """총 문항 수 → 난이도별 문항 수 (중4·상8·최상8 비율, 고른 난이도만)"""
    base = {lv: QB_CLINIC_MIX.get(lv, 4) for lv in levels} or {"상": 1}
    s = sum(base.values())
    mix = {lv: int(total * w / s) for lv, w in base.items()}
    for lv in sorted(base, key=lambda x: -(total * base[x] / s - mix[x])):
        if sum(mix.values()) >= total:
            break
        mix[lv] += 1
    return mix

def qb_clinic_pick(rows: List[Dict], labels: List[str], mix: Dict[str, int], seed: int) -> (List[Dict], Dict[str, int]):
    """난이도별로 지문 골고루 뽑아 합침 → (지문 순서대로 문항, 난이도별 모자란 수)"""
    rows = qb_dedupe_rows(rows)  # 난이도마다 따로 뽑으므로, 난이도 이름만 다른 같은 문제를 먼저 하나로
    picked, short = [], {}
    for lv, n in mix.items():
        if n <= 0:
            continue
        got, _ = qb_pick([r for r in rows if r["difficulty"] == lv], labels, n, seed)
        got = [r for r in got if r["difficulty"] == lv][:n]
        picked += got
        if len(got) < n:
            short[lv] = n - len(got)
    pos = {l: i for i, l in enumerate(labels)}
    picked.sort(key=lambda r: pos.get(r["passage_label"], 0))
    return picked, short

def qb_clinic_plan(summary: List[Dict], passages: List[Dict], short: Dict[str, int], school: str, types: List[str], seed: int) -> List[tuple]:
    """모자란 난이도만큼 새로 만들 (지문, 유형, 난이도, 문제 수): 그 난이도 문제가 적은 지문부터, 지문 안에서는 적게 쌓인 유형부터"""
    rnd = random.Random(seed)
    if not passages or not types:
        return []
    mine = [r for r in summary if r.get("school") == school and r["difficulty"] != QB_GICHUL_LEVEL]
    plan = {}
    for lv, k in short.items():
        have = {p["label"]: sum(1 for r in mine if r["passage_label"] == p["label"] and r["difficulty"] == lv) for p in passages}
        for _ in range(math.ceil(k * QB_CLINIC_EXTRA)):
            p = min(passages, key=lambda x: (have[x["label"]], rnd.random()))
            have[p["label"]] += 1
            # 이번에 만들 문제 전체에서 유형이 골고루 → 그다음 이 지문·난이도에 적게 쌓인 유형
            in_plan = {t: sum(n for (_, t2, _), n in plan.items() if t2 == t) for t in types}
            here = {t: sum(1 for r in mine if r["passage_label"] == p["label"] and r["q_type"] == t and r["difficulty"] == lv)
                       + sum(n for (l, t2, _), n in plan.items() if l == p["label"] and t2 == t) for t in types}
            t = min(types, key=lambda x: (in_plan[x], here[x], rnd.random()))
            key = (p["label"], t, lv)
            plan[key] = plan.get(key, 0) + 1
    by = {p["label"]: p for p in passages}
    return [(by[l], t, lv, n) for (l, t, lv), n in plan.items()]

def render_weekly_clinic(sb, mock_lib: List[Dict], by_label: Dict, summary: List[Dict], label_counts: Dict,
                         text_model: str, api_delay: int, academy_name: str, school_name: str):
    st.caption("그 주에 배운 지문을 주차로 저장해 두고, 그 지문으로 만든 변형문제만 모아 클리닉 미니모의고사를 만들어요 "
               "(실제 기출은 넣지 않고, 한 학교 스타일만). 문제가 모자라면 모자란 만큼만 골라서 새로 만들 수 있어요.")
    st.caption("🧭 이럴 때: 매주 배운 지문으로 같은 구성의 클리닉지를 만들 때 · 자유 구성은 '📄 시험지 만들기', 학교 실제 시험과 같은 구성·배점은 왼쪽 '🏁 영T FINAL 모의고사'")
    weeks = qb_clinic_weeks(sb)
    wmap = {w["id"]: w for w in weeks}
    ids = ["__new__"] + [w["id"] for w in weeks]
    if "qb_clinic_week_next" in st.session_state:  # 저장·삭제 뒤 고를 주차 (위젯을 만든 뒤에는 값을 바꿀 수 없어서 다음 실행 때 넣음)
        st.session_state["qb_clinic_week"] = st.session_state.pop("qb_clinic_week_next")
    if st.session_state.get("qb_clinic_week") not in ids:
        st.session_state["qb_clinic_week"] = "__new__"
    wid = st.selectbox("📅 주차", ids, format_func=lambda i: "➕ 새 주차 만들기" if i == "__new__" else
                       f"{wmap[i]['name']} · {wmap[i].get('school', '')} · 지문 {len(wmap[i].get('labels', []))}개", key="qb_clinic_week")
    w = wmap.get(wid, {})
    if st.session_state.get("qb_clinic_loaded") != wid:  # 주차를 바꾸면 그 주차 지문으로 채움
        st.session_state["qb_clinic_loaded"] = wid
        st.session_state["qb_clinic_labels"] = list(w.get("labels", []))
        st.session_state["qb_clinic_name"] = w.get("name", "")
        st.session_state["qb_clinic_school"] = w.get("school", QB_SCHOOLS[0])
        st.session_state.pop("qb_clinic_html", None)
    sel = _qb_passage_picker("qb_clinic_labels", mock_lib, label_counts)
    with st.container(key="qb_card_clinic_opts"):
        st.markdown("<div class='qb-title'>주차 정보</div>", unsafe_allow_html=True)
        name = _qb_row("주차 이름").text_input("주차 이름", key="qb_clinic_name", placeholder="예: 11월 2주차 운양고1 A반", label_visibility="collapsed")
        school = _qb_row("학교 스타일").segmented_control("학교 스타일", QB_SCHOOLS, key="qb_clinic_school", label_visibility="collapsed") or QB_SCHOOLS[0]
        c1, c2, _ = st.columns([1.3, 1, 3])
        if c1.button("💾 주차 저장" if wid == "__new__" else "💾 주차 고친 내용 저장", type="primary", use_container_width=True,
                     disabled=not sel or not name.strip(), key="qb_clinic_save"):
            from datetime import datetime, timedelta, timezone
            rec = {**w, "id": w.get("id") or uuid.uuid4().hex[:10], "name": name.strip(), "school": school, "labels": list(sel),
                   "used": w.get("used", []), "updated": datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d")}
            qb_clinic_update(sb, lambda ws: [rec] + [x for x in ws if x.get("id") != rec["id"]])
            st.session_state["qb_clinic_week_next"] = rec["id"]
            st.session_state["qb_clinic_loaded"] = rec["id"]
            st.rerun()
        if wid != "__new__":
            with c2.popover("주차 삭제", use_container_width=True):
                st.caption("주차 기록만 지워요. 문제은행의 문제는 그대로 남아요.")
                if st.button("삭제 확정", key="qb_clinic_del"):
                    qb_clinic_update(sb, lambda ws: [x for x in ws if x.get("id") != wid])
                    st.session_state["qb_clinic_week_next"] = "__new__"
                    st.rerun()
    if wid == "__new__":
        st.caption("지문을 고르고 이름을 붙여 저장하면 아래에서 클리닉을 만들 수 있어요.")
        return
    labels = sorted(w.get("labels", []), key=lambda l: natural_sort_key({"label": l}))
    labels = [l for l in labels if l in by_label]
    passages = [by_label[l] for l in labels]
    if not passages:
        # 저장된 주차 지문이 원문 라이브러리에서 모두 지워졌거나 이름이 바뀐 경우 (예전엔 여기서 오류가 나 아래 탭까지 멈췄음)
        st.warning("이 주차에 저장된 지문이 지금 원문 라이브러리에 하나도 없어요 (지워졌거나 이름이 바뀜). 위에서 지문을 다시 골라 저장해 주세요.")
        return
    # 이 주차 지문을 묶어 만든 문제(이어 붙이기·(A)(B))도 함께 · 모자란 문제는 짧은 교과서 단락을 자동으로 묶어 만듦
    links_w = [l for l in qb_link_labels(sb) if set(qb_link_parts(l)[1]) <= set(labels)]
    labels_all = sorted(labels + links_w, key=lambda l: natural_sort_key({"label": l}))
    gen_passages = qb_auto_join(passages, mock_lib)[0]
    if set(sel) != set(w.get("labels", [])) or name.strip() != w.get("name") or school != w.get("school"):
        st.info("고친 내용이 아직 저장되지 않았어요. 아래 클리닉은 저장된 주차 기준으로 만들어요.")
    school = w.get("school", QB_SCHOOLS[0])

    with st.container(key="qb_card_clinic_make"):
        st.markdown("<div class='qb-title'>클리닉 구성</div>", unsafe_allow_html=True)
        n1, n2 = _qb_row("문항 수").columns([1.3, 6], vertical_alignment="center")
        total = int(n1.number_input("문항 수", 5, 60, 20, key="qb_clinic_total", label_visibility="collapsed"))
        lvs = _qb_row("난이도").pills("난이도", ["최상", "상", "중"], selection_mode="multi", default=["최상", "상", "중"], key="qb_clinic_levels",
                                    label_visibility="collapsed") or ["상"]
        mix = qb_clinic_mix(total, [lv for lv in ["중", "상", "최상"] if lv in lvs])
        types = _qb_row("유형").pills("유형", QB_CLINIC_TYPES, selection_mode="multi", default=QB_CLINIC_TYPES, key="qb_clinic_types",
                                    label_visibility="collapsed") or QB_CLINIC_TYPES
        want_vocab = _qb_row("선지 어휘").checkbox(f"맨 뒤에 문제 선지·<보기>에 쓰인 어휘 정리 쪽 넣기 (가벼운 모델 {light_model_for(text_model).split(':')[-1]} · 약 10~30원)",
                                                 value=True, key="qb_clinic_vocab")
        c_yt = _qb_row("영쌤 포인트").segmented_control("영쌤 포인트", QB_YT_FILTER, default=QB_YT_FILTER[0], key="qb_clinic_ytf",
                                                     label_visibility="collapsed") or QB_YT_FILTER[0]
        skip_used = _qb_row("이미 낸 문항").checkbox(f"이 주차 클리닉에 이미 넣은 문항은 빼기 ({len(w.get('used', []))}개)", value=True, key="qb_clinic_skip")
        # 영쌤 포인트로 거를 때는 문제 내용이 있어야 알 수 있어서 이 주차 문제만 내용까지 읽음
        src_rows = summary if c_yt == QB_YT_FILTER[0] else [r for r in qb_fetch(sb, labels_all) if qb_yt_keep(r, c_yt)]
        mine = [r for r in src_rows if r["passage_label"] in set(labels_all) and r.get("school") == school and r["difficulty"] != QB_GICHUL_LEVEL
                and r["q_type"] in types and not (skip_used and r["id"] in set(w.get("used", [])))]
        have = {lv: sum(1 for r in mine if r["difficulty"] == lv) for lv in mix}
        n2.markdown("<div class='qb-note' style='margin-top:0'>" + " · ".join(f"{lv} <b>{n}</b>문항 (쌓인 문제 {have[lv]}개)" for lv, n in mix.items())
                    + f" · {school} 스타일 · 지문 {len(labels)}개</div>", unsafe_allow_html=True)
        short = {lv: n - have[lv] for lv, n in mix.items() if have[lv] < n}
        if short:
            plan = qb_clinic_plan(summary, gen_passages, short, school, types, sum(map(ord, wid)))
            n_q = sum(n for *_, n in plan)
            cost = sum(qb_job_cost_krw(text_model, n) for *_, n in plan)
            st.warning("문제가 모자라요: " + " · ".join(f"{lv} {k}문항" for lv, k in short.items())
                       + f" → 지문·유형을 골고루 나눠 {n_q}문제(검증 탈락 대비 조금 더)를 새로 만들 수 있어요. 예상 비용 약 {cost:,.0f}원")
            g1, g2 = st.columns([1.3, 3], vertical_alignment="center")
            yt = g2.toggle("🎣 영쌤 출제 포인트 위주로", value=True, key="qb_clinic_yt")
            if g1.button(f"🛠️ 모자란 {n_q}문제 만들기", use_container_width=True, key="qb_clinic_gen"):
                n = enqueue_qbank(passages, [], [], school, 1, True, text_model, api_delay, academy_name, school_name,
                                  plan=plan, yt_focus=yt, note=f"주간 클리닉 '{w['name']}' 모자란 문제 · {school}")
                st.session_state.gen_flash = f"✅ 주간 클리닉용 변형문제 {n}건을 생성 작업에 추가했습니다. 다 만들어지면 여기서 '클리닉 만들기'를 누르세요."
                st.rerun()
        title = _qb_row("제목").text_input("제목", value="", placeholder=f"{w['name']} 클리닉", key="qb_clinic_title", label_visibility="collapsed")
    b1, b2 = st.columns([2, 1])
    make = b1.button("📄 클리닉 만들기", type="primary", use_container_width=True, key="qb_clinic_btn", disabled=not labels)
    again = b2.button("🔀 다른 조합으로", use_container_width=True, key="qb_clinic_again", disabled="qb_clinic_html" not in st.session_state)
    if make or again:
        rows = [r for r in qb_fetch(sb, labels_all) if r.get("school") == school and r["difficulty"] in mix and r["q_type"] in types
                and not (skip_used and r["id"] in set(w.get("used", []))) and qb_yt_keep(r, c_yt)]
        picked, short2 = qb_clinic_pick(rows, labels_all, mix, random.randint(1, 10**9))
        if not picked:
            st.session_state.pop("qb_clinic_html", None)
            st.warning("이 주차 지문으로 만든 문제가 없어요. 위에서 '모자란 문제 만들기'를 먼저 눌러 주세요.")
        else:
            meta = {"brand": "클리닉 미니모의고사", "kind": school_name, "title": title or f"{w['name']} 클리닉",
                    "sub": f"{school} 스타일 · " + " · ".join(f"{lv} {sum(1 for r in picked if r['difficulty'] == lv)}" for lv in mix) + f" · 총 {len(picked)}문항",
                    "show_tag": True, "show_gichul": False, "with_answers": True, "academy": academy_name, "show_style": False, "show_time": True}
            if want_vocab:
                v_logs = []
                with st.spinner("문제 선지에 쓰인 어휘를 정리하는 중..."):
                    meta["vocab"] = qb_choice_vocab(picked, light_model_for(text_model), v_logs)
                flush_ai_usage()
                if not meta["vocab"]:
                    st.warning("선지 어휘 정리를 만들지 못했어요 (문제지는 그대로 만들었어요). " + " ".join(v_logs)[:200])
            st.session_state.qb_clinic_html = render_qbank_exam(picked, meta, gichul_index(load_gichul_map(), set(by_label)))
            st.session_state.qb_clinic_info = (len(picked), short2, meta["title"], [r["id"] for r in picked])
            st.session_state.qb_clinic_marked = False
    if st.session_state.get("qb_clinic_html"):
        n, short2, fname, qids = st.session_state.qb_clinic_info
        if short2:
            st.warning("문제가 모자라 " + " · ".join(f"{lv} {k}문항" for lv, k in short2.items()) + " 빠진 채로 만들었어요.")
        safe = re.sub(r'[\\/:*?"<>|]', "", fname).strip() or "클리닉"
        d1, d2 = st.columns([3, 1.3])
        d1.download_button(f"📥 클리닉 HTML 다운로드 ({n}문항) — 열어서 '인쇄 / PDF로 저장'", st.session_state.qb_clinic_html, f"{safe}.html", "text/html",
                           use_container_width=True, key="qb_clinic_dl")
        if st.session_state.get("qb_clinic_marked"):
            d2.button("✅ 낸 문항으로 표시함", disabled=True, use_container_width=True, key="qb_clinic_mark_done")
        elif d2.button("✔️ 이 문항들 낸 것으로 표시", use_container_width=True, key="qb_clinic_mark",
                       help="표시해 두면 이 주차로 다음 클리닉을 만들 때 이 문항은 빼요"):
            qb_clinic_update(sb, lambda ws: [{**x, "used": list(dict.fromkeys(x.get("used", []) + qids))} if x.get("id") == wid else x for x in ws])
            st.session_state.qb_clinic_marked = True
            st.rerun()
        render_choice_trans_ui(sb, "qb_clinic_tr", qids, fname, text_model, academy_name, school_name)
        components.html(st.session_state.qb_clinic_html, height=850, scrolling=True)

def _qb_passage_picker(key: str, mock_lib: List[Dict], counts: Dict[str, int]) -> List[str]:
    """출처·학년·시험을 고른 뒤 번호 칩을 눌러 지문을 고름 (여러 시험에 걸쳐 모아 담김)"""
    lib = sorted(mock_lib, key=natural_sort_key)
    groups, group_of = {}, {}
    for p in lib:
        g = p.get("exam_key") or extract_group_prefix(p["label"])
        groups.setdefault(g, []).append(p)
        group_of[p["label"]] = g
    order = {p["label"]: i for i, p in enumerate(lib)}
    by_lbl = {p["label"]: p for p in lib}
    st.session_state[key] = [l for l in st.session_state.get(key, []) if l in order]
    # 이어 붙인(연결) 지문 문제: 짧은 교과서 단락은 '4-3 + 4-4'처럼 묶은 지문으로만 만들어짐
    #   칩 숫자 = 그 지문이 들어간 문제 모두, 고른 지문 합계 = 시험지에 실제로 뽑힐 수 있는 문제 (묶인 지문을 모두 골랐을 때만)
    link_parts = {l: set(qb_link_parts(l)[1]) for l, n in counts.items() if n and qb_link_parts(l)[0]}
    chip_n = dict(counts)
    for l, parts in link_parts.items():
        for x in parts:
            chip_n[x] = chip_n.get(x, 0) + counts[l]

    def set_chosen(labels):
        st.session_state[key] = sorted(set(labels), key=order.get)

    def sync(pkey, opts):
        set_chosen([l for l in st.session_state[key] if l not in opts] + list(st.session_state.get(pkey) or []))

    with st.container(key=f"qb_card_{key}"):
        srcs = [s for s in ("모의고사", "교과서·부교재") if any(_qb_is_mock(ps[0]) == (s == "모의고사") for ps in groups.values())]
        st.markdown("<div class='qb-title'>지문 고르기<span>번호 옆 작은 숫자 = 쌓인 문제 수 (이어 붙인 지문 문제 포함)</span></div>", unsafe_allow_html=True)
        src = _qb_row("출처").segmented_control("출처", srcs, default=srcs[0], key=f"{key}_src", label_visibility="collapsed") or srcs[0]
        gks = [g for g, ps in groups.items() if _qb_is_mock(ps[0]) == (src == "모의고사")]
        grade = ""
        if src == "모의고사":
            grades = sorted({str(groups[g][0].get("grade", "")) for g in gks})
            grade = _qb_row("학년").segmented_control("학년", grades, default=grades[0], key=f"{key}_grade", label_visibility="collapsed") or grades[0]
            gks = sorted([g for g in gks if str(groups[g][0].get("grade", "")) == grade],
                         key=lambda g: (int(groups[g][0]["year"]), int(groups[g][0]["month"])), reverse=True)
        chosen = st.session_state[key]
        exam = _qb_row("시험" if src == "모의고사" else "단원").pills(
            "시험", gks, default=gks[0], key=f"{key}_exam_{src}_{grade}", label_visibility="collapsed",
            format_func=lambda g: _qb_group_name(groups[g])) or gks[0]

        ps = groups[exam]
        if _qb_is_mock(ps[0]):
            rows = [(f"{a}–{b}", [p for p in ps if a <= (extract_question_number(p["label"]) or 0) <= b]) for a, b in QB_MOCK_ROWS]
            used = {p["label"] for _, r in rows for p in r}
            rest = [p for p in ps if p["label"] not in used]
            rows = [r for r in rows if r[1]] + ([("기타", rest)] if rest else [])
        else:
            rows = [("번호", ps)]
        fmt = lambda l: str(by_lbl[l].get("number") or extract_question_number(l) or l) + _qb_sup(chip_n.get(l))
        nums_box = st.container(key=f"qbsep_{key}")
        for i, (row_name, row_ps) in enumerate(rows):
            opts = [p["label"] for p in row_ps]
            pkey = f"{key}_p_{exam}_{i}"
            st.session_state[pkey] = [l for l in opts if l in chosen]
            with nums_box:
                _qb_row(row_name).pills("번호", opts, selection_mode="multi", key=pkey, format_func=fmt, label_visibility="collapsed",
                                        on_change=sync, args=(pkey, opts))

        here = [p["label"] for p in ps]
        f1 = st.container()
        if chosen:
            lines = {}
            for l in chosen:
                lines.setdefault(group_of[l], []).append(str(by_lbl[l].get("number") or l))
            body = " &nbsp;·&nbsp; ".join(
                f"<b>{html.escape(_qb_group_name(groups[g]) + ((' ' + str(groups[g][0].get('grade'))) if _qb_is_mock(groups[g][0]) else ''))}</b> "
                f"{html.escape(', '.join(nums))}" for g, nums in lines.items())
            cs = set(chosen)
            n_link = sum(counts[l] for l, parts in link_parts.items() if parts <= cs)
            n_part = sum(counts[l] for l, parts in link_parts.items() if parts & cs and not parts <= cs)
            extra = (f" (이어 붙인 지문 문제 {n_link}개 포함)" if n_link else "") + (
                f" · 묶인 지문을 다 고르지 않아 빠지는 문제 {n_part}개" if n_part else "")
            f1.markdown(f"<div class='qb-basket'>고른 지문 <b>{len(chosen)}개</b> <span class='muted'>· 쌓인 문제 "
                        f"{sum(counts.get(l, 0) for l in chosen) + n_link}개{extra}</span>"
                        f"<br>{body}</div>", unsafe_allow_html=True)
        else:
            f1.markdown("<div class='qb-basket'><span class='muted'>번호를 눌러 지문을 고르세요. 다른 시험으로 바꿔도 고른 번호는 그대로 남아요.</span></div>",
                        unsafe_allow_html=True)
        f2, f3, f4, _ = st.columns([1, 1, 1, 4])
        f2.button("이 시험 전체", key=f"{key}_all", type="tertiary", on_click=lambda: set_chosen(st.session_state[key] + here))
        f3.button("이 시험 해제", key=f"{key}_none", type="tertiary",
                  on_click=lambda: set_chosen([l for l in st.session_state[key] if l not in here]))
        f4.button("모두 비우기", key=f"{key}_clear", type="tertiary", disabled=not chosen, on_click=lambda: set_chosen([]))
    return st.session_state[key]

def render_qbank_tab(text_model: str, api_delay: int, academy_name: str, school_name: str):
    sb = supabase if IS_CLOUD_DB else None
    ready, err = qb_table_status(sb)
    if not ready:
        st.error("변형문제를 저장할 표가 Supabase에 아직 없습니다. Supabase → SQL Editor에서 아래 문장을 한 번 실행한 뒤 새로고침해 주세요.")
        st.code(QB_SETUP_SQL, language="sql")
        st.caption(f"오류 내용: {err}")
        return
    mock_lib = [p for p in st.session_state.saved_mock if p.get("body_text")]
    if not mock_lib:
        st.info("원문 라이브러리에 지문이 없습니다. 왼쪽 메뉴 '📝 원문 라이브러리'에서 모의고사·교과서 원문을 먼저 저장해 주세요.")
        return
    by_label = {p["label"]: p for p in mock_lib}
    st.markdown(QB_PICKER_CSS, unsafe_allow_html=True)
    links = [l for l in qb_link_labels(sb) if all(x in by_label for x in qb_link_parts(l)[1])]
    summary = qb_bank_summary(sb, tuple(sorted(by_label) + links))
    label_counts = _qb_chip_counts(summary, "passage_label")
    # 탭 상태를 기억해서(on_change="rerun"), 안 보이는 '문제 보기·삭제' 탭은 문제를 DB에서 읽지 않음 (예전에는 화면을 누를 때마다 읽었음)
    t_gen, t_exam, t_clinic, t_manage, t_gichul = st.tabs(["🛠️ 문제 만들기 (누적)", "📄 시험지 만들기", "📅 주간 클리닉", "🗂️ 문제 보기·삭제", "🏫 기출 표시"],
                                                          key="qb_tabs", on_change="rerun")

    with t_gen:
        st.caption("고른 지문 × 유형 × 난이도마다 문제를 만들어 계속 쌓습니다. 같은 지문을 다시 만들면 이미 만든 문제와 다른 포인트로 출제합니다.")
        sel = _qb_passage_picker("qb_gen_labels", mock_lib, label_counts)
        with st.container(key="qb_card_gen_opts"):
            st.markdown("<div class='qb-title'>만들 문제</div>", unsafe_allow_html=True)
            q_types = _qb_row("유형").pills("유형", QB_TYPE_NAMES, selection_mode="multi", default=[t for t in QB_TYPE_NAMES if t not in QB_ESSAY_TYPES],
                                            key="qb_gen_types", label_visibility="collapsed") or []
            levels = _qb_row("난이도").pills("난이도", QB_LEVELS, selection_mode="multi", default=["상", "중"], key="qb_gen_levels", label_visibility="collapsed") or []
            school = _qb_row("학교 스타일").segmented_control("학교 스타일", QB_SCHOOLS, default=QB_SCHOOLS[0], key="qb_gen_school",
                                                          label_visibility="collapsed") or QB_SCHOOLS[0]
            # 학년: 선지 길이를 그 학교·학년 실제 기출에 맞춤. 기본은 지문 따라 (모의고사 학년·교과서 이름)
            g_found = {}
            for l in sel:
                g_ = qb_passage_grade(by_label[l]) or "모름"
                g_found[g_] = g_found.get(g_, 0) + 1
            g_cols = _qb_row("학년").columns([3.2, 4.8], vertical_alignment="center")
            gen_grade = g_cols[0].segmented_control("학년", ["지문 따라"] + QB_GRADES, default="지문 따라", key="qb_gen_grade",
                                                    label_visibility="collapsed",
                                                    help="선지 길이를 이 학교·학년의 24~26년 실제 기출에 맞춰요. '지문 따라'는 모의고사 학년, 교과서 이름(공통영어 = 고1, 영어Ⅰ·Ⅱ = 고2)으로 정해요.") or "지문 따라"
            g_cols[1].markdown("<div class='qb-note' style='margin-top:0'>" + (
                ("고른 지문: " + " · ".join(f"{g_} <b>{n_}개</b>" for g_, n_ in sorted(g_found.items())) + " → 선지 길이를 그 학년 실제 기출에 맞춰요"
                 + (" (모름은 학교 전체 평균)" if "모름" in g_found else "")) if gen_grade == "지문 따라"
                else f"모든 지문을 <b>{school} {gen_grade}</b> 실제 기출 선지 길이에 맞춰요") + "</div>", unsafe_allow_html=True)
            # 난이도마다 만들 문제 수 (지문 × 유형마다) = 난이도 비중 (예: 최상 3 · 상 2 · 중 1 → 50% · 33% · 17%)
            lv_n = {}
            levels = [lv for lv in QB_LEVELS if lv in levels]  # 누른 순서와 관계없이 최상·상·중·하 순서로
            if levels:
                n_cols = _qb_row("난이도별 개수").columns([1.1] * len(levels) + [max(1.0, 8 - 1.1 * len(levels))], vertical_alignment="bottom")
                for i, lv in enumerate(levels):
                    lv_n[lv] = int(n_cols[i].number_input(f"{lv} (문제)", 0, 5, 2, key=f"qb_gen_n_{lv}",
                                                          help="지문 하나 · 유형 하나마다 이 난이도로 만들 문제 수 (0이면 안 만듦)"))
                tot_n = sum(lv_n.values())
                n_cols[-1].markdown(f"<div class='qb-note' style='margin-top:0'>지문·유형마다 <b>{tot_n}문제</b>"
                                    + (" · 비중 " + " · ".join(f"{lv} <b>{round(100 * n / tot_n)}%</b>" for lv, n in lv_n.items() if n) if tot_n else "")
                                    + "</div>", unsafe_allow_html=True)
            verify = _qb_row("정답 검증").checkbox("AI가 한 번 더 풀어서 정답이 다르면 버림", value=True, key="qb_gen_verify")
            underline = "mix"
            if set(q_types) & (QB_GRAMMAR_TYPES - {"서술형 (조건 영작)"}):
                underline = _qb_row("어법 밑줄").segmented_control("어법 밑줄", list(QB_UNDERLINE_OPTS), format_func=QB_UNDERLINE_OPTS.get, default="mix",
                                                              key="qb_gen_underline", label_visibility="collapsed",
                                                              help="섞기: 학교 기출처럼 단어·구 밑줄과 문장 전체 밑줄을 문항마다 무작위로 섞어요") or "mix"
            light_easy = _qb_row("비용").checkbox(f"하·중은 가벼운 모델로 ({light_model_for(text_model).split(':')[-1]}) · 상·최상은 지금 모델({split_model_spec(text_model)[1]})",
                                                 value=True, key="qb_gen_light", disabled=light_model_for(text_model) == text_model)
            batch = _qb_row("반값 모드").toggle("💸 반값 모드 — 생성 비용 절반, 대신 결과가 보통 몇 분~몇 시간, 길면 24시간 뒤에 들어와요 (급할 때는 끄세요)",
                                              value=False, key="qb_gen_batch",
                                              help="각 AI 회사의 Batch API를 써요. 요청을 모아 보내 두면 회사가 한가할 때 처리하고 값을 50% 깎아 줘요. "
                                                   "정답 검증은 결과가 들어온 뒤 바로 해요(검증 값은 그대로). 창을 닫아도 서버가 2분마다 확인해서 들어온 것부터 저장해요.")
            n_yt = sum(1 for p in qb_yt_points() if p.get("on", True))
            yt_focus = _qb_row("영쌤 포인트").toggle(f"🎣 영쌤 출제 포인트 위주로 — 모든 문항에 포인트 하나씩 (켜 둔 포인트 {n_yt}개)",
                                                  value=False, key="qb_gen_yt", disabled=not n_yt,
                                                  help="끄면 문항 절반쯤에만 포인트를 넣어요. 포인트는 왼쪽 메뉴 출제 DNA 화면의 '🎣 영쌤 출제 포인트'에서 "
                                                       "추가·수정해요. 학교(운양고·제일고)와 관계없이 적용되고, 난이도 '하'에는 넣지 않아요.")
            # 지문 연결: 고른 지문 여러 개를 한 지문으로 묶어 출제 (실제 내신처럼)
            link = "따로"
            if len(sel) >= 2:
                link = _qb_row("지문 연결").segmented_control("지문 연결", list(QB_LINK_MODES), format_func=QB_LINK_MODES.get, default="따로",
                                                            key="qb_gen_link", label_visibility="collapsed",
                                                            help="이어 붙이기: 교과서 짧은 단락 2~4개를 실제 시험처럼 한 지문으로. "
                                                                 "(A)(B): 같은 범위의 비슷한 내용 두 지문을 함께 묻는 추론·일치·주제형") or "따로"
            gen_ps = [by_label[l] for l in sel]
            auto_joined = []
            if link == "따로" and any(by_label[l].get("source_type") == "textbook" and _qb_words(by_label[l]) < QB_SHORT_WORDS for l in sel):
                if _qb_row("짧은 단락").toggle(f"📚 짧은 교과서 단락({QB_SHORT_WORDS}단어 미만)은 앞뒤 단락과 자동으로 이어 붙여 출제", value=True, key="qb_gen_autojoin",
                                              help="실제 내신처럼 짧은 단락은 같은 단원의 바로 앞·뒤 단락(고른 것 중)과 2~3개를 묶어 한 지문으로 내요. "
                                                   "짧은 단락은 묶은 지문으로만, 함께 묶인 긴 단락은 따로도 만들어요. 모의고사 지문은 그대로예요."):
                    gen_ps, auto_joined = qb_auto_join(gen_ps, mock_lib)
                    if auto_joined:
                        st.markdown("<div class='qb-note'>자동으로 묶은 지문: " + " · ".join(
                            f"<b>{html.escape(l)}</b> ({len(by_label.get(l, {}).get('body_text', '').split()) or sum(_qb_words(by_label[x]) for x in qb_link_parts(l)[1])}단어)"
                            for l in auto_joined) + "</div>", unsafe_allow_html=True)
                    else:
                        st.caption("💡 짧은 단락 옆에 함께 고른 단락이 없어서 이번에는 묶지 않아요 (앞·뒤 단락도 함께 골라 주세요).")
            if link == "join":
                if len(sel) > 4:
                    st.warning("이어 붙이기는 지문 4개까지예요. 앞의 4개만 묶어요.")
                gen_ps = [qb_link_passage(gen_ps[:4], "join")]
            elif link == "pair":
                if len(sel) != 2:
                    st.warning("(A)(B)는 지문을 정확히 2개 골라 주세요.")
                    gen_ps = []
                else:
                    gen_ps = [qb_link_passage(gen_ps, "pair")]
                    dropped_t = [t for t in q_types if t not in QB_PAIR_TYPES]
                    q_types = [t for t in q_types if t in QB_PAIR_TYPES]
                    if dropped_t:
                        st.caption("💡 (A)(B) 두 지문은 " + " · ".join(QB_PAIR_TYPES) + " 유형만 만들어요 (나머지 유형은 이번에 빼요).")
            if link != "따로" and gen_ps:
                st.markdown(f"<div class='qb-note'>묶은 지문 <b>{html.escape(gen_ps[0]['label'])}</b> · "
                            f"{len(gen_ps[0]['body_text'].split())}단어 · 시험지를 만들 때 묶은 지문을 모두 고르면 함께 뽑혀요</div>", unsafe_allow_html=True)
        combos = list(dict.fromkeys(("어법 (조합형)" if (t == "어법 (단일)" and lv == "최상") else t, lv)
                                    for t in q_types for lv in levels if lv_n.get(lv, 0) > 0))
        n_jobs = len(gen_ps) * len(combos)
        if "어법 (단일)" in q_types and "최상" in levels:
            st.caption("💡 어법 최상은 실제 내신처럼 '옳은/틀린 것만을 있는 대로 고른 것'(ⓐ~ⓗ 조합형)으로 만듭니다.")
        n_q = len(gen_ps) * sum(lv_n[lv] for _, lv in combos)
        if n_jobs:
            cost = len(gen_ps) * sum(qb_job_cost_krw(light_model_for(text_model) if (light_easy and lv in ("하", "중")) else text_model, lv_n[lv], verify, batch)
                                  for _, lv in combos)
            when = "결과는 보통 몇 분~몇 시간, 길면 24시간 뒤" if batch else f"예상 시간 약 {max(1, n_jobs * 40 // 60)}분"
            st.caption(f"📦 작업 {n_jobs}건 · 문제 약 {n_q}개 · 예상 비용 약 {cost:,.0f}원 (문제당 약 {cost / max(1, n_q):,.0f}원){' · 💸 반값' if batch else ''} · "
                       f"{when} (창을 닫아도 서버에서 계속)")
            st.caption("💡 난이도별 개수를 늘리면 지문·규칙을 읽는 고정 비용을 함께 나눠 문제당 값이 내려가요 (1개 → 2개 → 3개로 갈수록 싸지고, 그 뒤로는 차이가 작아요).")
        if st.button("🚀 변형문제 생성 시작", type="primary", disabled=not n_jobs, key="qb_gen_btn"):
            used_lv = [lv for lv in levels if lv_n.get(lv, 0) > 0]
            n = enqueue_qbank(gen_ps, q_types, used_lv, school, max(lv_n.values()), verify, text_model, api_delay, academy_name, school_name, light_easy, underline,
                              batch, yt_focus, plan=[(p, t, lv, lv_n[lv]) for p in gen_ps for t, lv in combos],
                              note=f"{school}{'' if gen_grade == '지문 따라' else ' ' + gen_grade} · " + " · ".join(f"{lv} {lv_n[lv]}" for lv in used_lv) + "문제씩",
                              grade="" if gen_grade == "지문 따라" else gen_grade)
            st.session_state.gen_flash = f"✅ 변형문제 {n}건을 생성 작업에 추가했습니다. 위 작업 칸에서 진행 상황을 볼 수 있어요."
            st.rerun()

        if sel:
            rows = qb_fetch(sb, sel, "passage_label,q_type,difficulty")
            if rows:
                df = pd.DataFrame(rows)
                pv = df.pivot_table(index="passage_label", columns="q_type", values="difficulty", aggfunc="count", fill_value=0)
                pv = pv.reindex(columns=[t for t in QB_TYPE_NAMES if t in pv.columns])
                pv["합계"] = pv.sum(axis=1)
                st.markdown("##### 📊 고른 지문에 쌓인 문제 수")
                st.dataframe(pv, use_container_width=True)
            else:
                st.caption("고른 지문에는 아직 쌓인 문제가 없습니다.")

    with t_exam:
        st.caption("고른 번호에서 같은 개수씩 골고루 뽑고, 번호 안에서는 유형을 돌아가며 섞습니다.")
        st.caption("🧭 이럴 때: 지문·유형·문항 수를 마음대로 정해 시험지를 만들 때 · 매주 같은 구성은 '📅 주간 클리닉', 학교 실제 시험과 같은 구성·배점은 왼쪽 '🏁 영T FINAL 모의고사'")
        sel_e = _qb_passage_picker("qb_exam_labels", mock_lib, label_counts)
        with st.container(key="qb_card_exam_opts"):
            st.markdown("<div class='qb-title'>시험지 구성</div>", unsafe_allow_html=True)
            e_school = _qb_row("학교").segmented_control("학교", ["전체"] + QB_SCHOOLS, default="전체", key="qb_exam_school",
                                                       label_visibility="collapsed", help="변형문제는 그 학교 스타일, 실제 기출은 그 학교 시험만") or "전체"
            e_real = _qb_row("실제 기출").segmented_control("실제 기출", ["함께 섞기", "빼기", "기출만"], default="함께 섞기", key="qb_exam_real",
                                                         label_visibility="collapsed") or "함께 섞기"
            e_yt = _qb_row("영쌤 포인트").segmented_control("영쌤 포인트", QB_YT_FILTER, default=QB_YT_FILTER[0], key="qb_exam_yt",
                                                         label_visibility="collapsed",
                                                         help="🎣 영쌤 출제 포인트로 만든 변형문제만 뽑거나 빼요 (위 개수 표시는 이 조건을 빼고 센 수예요)") or QB_YT_FILTER[0]
            logs = qb_exam_log(sb)
            excl = set()
            if logs:
                names = {x["id"]: f"{x['name']} · {x['date']} · {x['n']}문항" for x in logs}
                st.session_state["qb_exam_excl"] = [i for i in st.session_state.get("qb_exam_excl", []) if i in names]
                x1, x2 = _qb_row("이미 낸 문항").columns([5, 1.3], vertical_alignment="center")
                chosen = x1.multiselect("이미 낸 문항 빼기", list(names), format_func=names.get, key="qb_exam_excl", label_visibility="collapsed",
                                        placeholder="저장한 시험지를 고르면 그 시험지에 들어간 문항은 빼고 뽑아요")
                for x in logs:
                    if x["id"] in chosen:
                        excl.update(x["qids"])
                with x2.popover("기록 관리", use_container_width=True):
                    st.caption("기록만 지워요. 문제은행의 문제는 그대로 남아요.")
                    for x in logs:
                        d1, d2 = st.columns([4, 1], vertical_alignment="center")
                        d1.markdown(f"{x['name']}  \n<span style='color:gray;font-size:12px'>{x['date']} · {x['n']}문항 · {x.get('title', '')}</span>", unsafe_allow_html=True)
                        if d2.button("삭제", key=f"qb_exam_log_del_{x['id']}"):
                            qb_exam_log_update(sb, lambda rs, i=x["id"]: [r for r in rs if r["id"] != i])
                            st.rerun()
                if excl:
                    x1.markdown(f"<div class='qb-note'>고른 기록 {len(chosen)}개 · 문항 <b>{len(excl)}개</b>를 빼고 뽑아요</div>", unsafe_allow_html=True)
            # 연결 지문 문제는 묶인 지문을 모두 골랐을 때 함께 뽑음
            sel_links = [l for l in links if set(qb_link_parts(l)[1]) <= set(sel_e)]
            sel_set = set(sel_e) | set(sel_links)
            pool = [r for r in summary if r["passage_label"] in sel_set and (e_school == "전체" or r.get("school") == e_school) and r.get("id") not in excl]
            lv_counts = _qb_chip_counts([r for r in pool if r["difficulty"] != QB_GICHUL_LEVEL], "difficulty")
            lv_box = _qb_row("난이도")
            e_levels = lv_box.pills("난이도", QB_LEVELS, selection_mode="multi", default=QB_LEVELS, key="qb_exam_levels", label_visibility="collapsed") or []
            _qb_count_line(lv_box, lv_counts, QB_LEVELS)
            pool = [r for r in pool if (r["difficulty"] == QB_GICHUL_LEVEL and e_real != "빼기") or (r["difficulty"] in e_levels and e_real != "기출만")]
            ty_counts = _qb_chip_counts(pool, "q_type")
            ty_box = _qb_row("유형")
            e_types = ty_box.pills("유형", QB_TYPE_NAMES, selection_mode="multi", default=QB_TYPE_NAMES, key="qb_exam_types", label_visibility="collapsed") or []
            _qb_count_line(ty_box, ty_counts, QB_TYPE_NAMES)
            u1, u2, _ = ty_box.columns([1, 1.2, 6])
            u1.button("전체 선택", key="qb_exam_types_all", type="tertiary", on_click=lambda: st.session_state.update(qb_exam_types=list(QB_TYPE_NAMES)))
            u2.button("있는 유형만", key="qb_exam_types_have", type="tertiary", disabled=not ty_counts,
                      on_click=lambda: st.session_state.update(qb_exam_types=[t for t in QB_TYPE_NAMES if ty_counts.get(t)]))
            order = _qb_row("순서").segmented_control("순서", ["지문 순서 (오름차순)", "지문 역순 (내림차순)"], default="지문 순서 (오름차순)",
                                                     key="qb_exam_order", label_visibility="collapsed") or "지문 순서 (오름차순)"
            avail = sum(ty_counts.get(t, 0) for t in e_types)
            mix = "자동으로 섞기"
            if e_school == "전체":
                mix = _qb_row("학교 비율").segmented_control("학교 비율", ["자동으로 섞기", "학교별 문제 수 정하기"], default="자동으로 섞기",
                                                         key="qb_exam_mix", label_visibility="collapsed",
                                                         help="예: 운양고 스타일 20문제 + 제일고 스타일 10문제 (실제 기출은 그 학교 시험 문제로 셈)") or "자동으로 섞기"
            split = {}
            if mix == "학교별 문제 수 정하기":
                s_avail = {sc: sum(1 for r in pool if r["q_type"] in e_types and r.get("school") == sc) for sc in QB_SCHOOLS}
                n_box = _qb_row("문제 수")
                n_cols = n_box.columns([1.3] * len(QB_SCHOOLS) + [6 - 1.3 * len(QB_SCHOOLS) + 1.3], vertical_alignment="bottom")
                for i, sc in enumerate(QB_SCHOOLS):
                    split[sc] = int(n_cols[i].number_input(f"{sc} 스타일", 0, 300, 15, key=f"qb_exam_n_{sc}"))
                total = sum(split.values())
                n_cols[-1].markdown("<div class='qb-note' style='margin-top:0'>총 <b>" + str(total) + "문제</b> · 뽑을 수 있는 문제 "
                                    + " · ".join(f"{sc} <b>{s_avail[sc]}개</b>" for sc in QB_SCHOOLS) + "</div>", unsafe_allow_html=True)
            else:
                n1, n2 = _qb_row("문제 수").columns([1.3, 6], vertical_alignment="center")
                total = n1.number_input("총 문제 수", 1, 300, 30, key="qb_exam_total", label_visibility="collapsed")
                n2.markdown(f"<div class='qb-note' style='margin-top:0'>지금 조건으로 뽑을 수 있는 문제 <b>{avail}개</b></div>", unsafe_allow_html=True)
            # 난이도 비율: 자동(쌓인 문제에서 섞어 뽑음) / 비율 정하기(최상·상·중 … % → 그 비율대로 뽑음)
            lv_mix = None
            has_real = any(r["difficulty"] == QB_GICHUL_LEVEL for r in pool)  # 고른 지문에 실제 기출이 있을 때만 '실제 기출' 비율 칸
            lv_keys = ([lv for lv in QB_LEVELS if lv in e_levels] if e_real != "기출만" else []) + ([QB_GICHUL_LEVEL] if e_real == "함께 섞기" and has_real else [])
            if e_real != "기출만" and lv_keys:
                lv_mode = _qb_row("난이도 비율").segmented_control("난이도 비율", ["자동으로 섞기", "비율 정하기"], default="자동으로 섞기", key="qb_exam_lvmode",
                                                               label_visibility="collapsed",
                                                               help="비율 정하기: 예) 최상 30 · 상 40 · 중 30 → 30문제면 최상 9 · 상 12 · 중 9문제. 합이 100이 아니어도 비율로 나눠요") or "자동으로 섞기"
                if lv_mode == "비율 정하기":
                    r_cols = _qb_row("비율 (%)").columns([1.1] * len(lv_keys) + [max(1.0, 8 - 1.1 * len(lv_keys))], vertical_alignment="bottom")
                    lv_mix = {lv: int(r_cols[i].number_input(qb_level_name(lv), 0, 100, QB_LEVEL_RATIO_DEFAULT.get(lv, 0), step=5, key=f"qb_exam_ratio_{lv}"))
                              for i, lv in enumerate(lv_keys)}
                    plan_n = qb_level_counts(int(total), lv_mix)
                    have_n = {lv: sum(1 for r in pool if r["difficulty"] == lv and r["q_type"] in e_types) for lv in lv_keys}
                    r_cols[-1].markdown("<div class='qb-note' style='margin-top:0'>" + (
                        f"{int(total)}문제 → " + " · ".join(
                            f"{qb_level_name(lv)} <b>{n}</b>" + (f" <span style='color:#d33'>(쌓인 {have_n[lv]}개뿐)</span>" if n > have_n[lv] else "")
                            for lv, n in plan_n.items() if n) if sum(lv_mix.values()) else "비율을 하나 이상 넣어 주세요") + "</div>", unsafe_allow_html=True)
            _qb_hr()
            default_title = (" · ".join(sel_e[:3]) + (" 외" if len(sel_e) > 3 else "")) if sel_e else ""
            title = _qb_row("제목").text_input("시험지 제목", value="", placeholder=default_title or "예: 고1 26년 6월 19·20·21번 변형문제", key="qb_exam_title",
                                                label_visibility="collapsed")
            kind = _qb_row("머리말 오른쪽").text_input("머리말 오른쪽", value=school_name, key="qb_exam_kind", label_visibility="collapsed")
            o1, o2, o3, o4 = _qb_row("표시").columns(4)
            o5, o6, _ = _qb_row("").columns([1, 1, 2])
            show_tag = o1.checkbox("문항 아래 출처·구분", value=True, key="qb_exam_tag", help="실제 기출/변형, 학교 스타일, 난이도")
            show_gichul = o2.checkbox("같은 지문 기출 이력", value=True, key="qb_exam_gichul", help="운양고·제일고에서 이 지문이 나온 시험·번호 전부")
            with_answers = o3.checkbox("마지막에 정답·해설", value=True, key="qb_exam_ans")
            show_style = o5.checkbox("변형문제에 학교 스타일 표시", value=True, key="qb_exam_style", help="끄면 '변형 · 상'처럼 난이도만 나와요")
            show_time = o6.checkbox("문항별 권장 풀이 시간", value=True, key="qb_exam_time", help="지문 단어 수·유형으로 계산 (AI 비용 없음)")
            as_book = o4.checkbox("ANATOMY VISION 책으로", value=True, key="qb_exam_book",
                                  help="앞표지·사용법·제작 소개 2쪽을 앞에, 저작권·뒤표지를 뒤에 붙여 양면 인쇄하면 한 권의 책처럼 나와요")
        b1, b2 = st.columns([2, 1])
        make = b1.button("📄 시험지 만들기", type="primary", disabled=not sel_e or not e_types or (not e_levels and e_real != "기출만") or total < 1
                         or (lv_mix is not None and not sum(lv_mix.values())), use_container_width=True, key="qb_exam_btn")
        reshuffle = b2.button("🔀 다른 조합으로 다시 뽑기", disabled="qb_exam_html" not in st.session_state, use_container_width=True, key="qb_exam_shuffle")
        if make or reshuffle:
            seed = random.randint(1, 10**9)
            labels_ordered = sorted(sel_e + sel_links, key=lambda l: natural_sort_key({"label": l}), reverse=order.startswith("지문 역순"))
            def wanted(r):
                is_real = r["difficulty"] == QB_GICHUL_LEVEL
                if (is_real and e_real == "빼기") or (not is_real and (e_real == "기출만" or r["difficulty"] not in e_levels)):
                    return False
                if (is_real and e_yt == QB_YT_FILTER[1]) or (not is_real and not qb_yt_keep(r, e_yt)):
                    return False
                return r["q_type"] in e_types and (e_school == "전체" or r.get("school") == e_school) and r["id"] not in excl
            rows = [r for r in qb_fetch(sb, labels_ordered) if wanted(r)]
            if split:
                # 학교마다 따로 골고루 뽑은 뒤 지문 순서대로 합침 (이미 불러온 문제 안에서만 계산 · DB·AI 호출 추가 없음)
                picked, notes = [], []
                for sc, n in split.items():
                    if n <= 0:
                        continue
                    sc_rows = [r for r in rows if r.get("school") == sc]
                    if lv_mix:
                        got, nt = qb_pick_levels(sc_rows, labels_ordered, n, seed, lv_mix)
                        notes += [f"{sc} 스타일 · {x}" for x in nt if not x.startswith("조건에 맞는 문제가 모자라")]
                    else:
                        got, _ = qb_pick(sc_rows, labels_ordered, n, seed)
                    picked += got
                    if len(got) < n:
                        notes.append(f"{sc} 스타일은 조건에 맞는 문제가 {len(got)}개뿐이라 {n}문제 중 {len(got)}문제만 넣었습니다. '문제 만들기'에서 더 생성해 주세요.")
                pos = {l: i for i, l in enumerate(labels_ordered)}
                picked.sort(key=lambda r: pos.get(r["passage_label"], 0))
            elif lv_mix:
                picked, notes = qb_pick_levels(rows, labels_ordered, int(total), seed, lv_mix)
            else:
                picked, notes = qb_pick(rows, labels_ordered, int(total), seed)
            if picked:
                labels_all = set(by_label)
                lv_got = _qb_chip_counts(picked, "difficulty")
                meta = {"brand": "변형문제 TEST", "kind": kind, "title": title or default_title,
                        "sub": " · ".join(x for x in [
                            f"실제 기출 {sum(1 for r in picked if r['difficulty'] == QB_GICHUL_LEVEL)}" if e_real != "빼기" else "",
                            f"변형 {sum(1 for r in picked if r['difficulty'] != QB_GICHUL_LEVEL)} ("
                            + (" · ".join(f"{lv} {lv_got[lv]}" for lv in QB_LEVELS if lv_got.get(lv)) if lv_mix else ", ".join(e_levels))
                            + ")" if e_real != "기출만" else "",
                            f"총 {len(picked)}문항"] if x), "show_tag": show_tag, "show_gichul": show_gichul, "with_answers": with_answers,
                        "academy": academy_name, "show_style": show_style, "show_time": show_time}
                gmap = load_gichul_map()
                gidx = gichul_index(gmap, labels_all)
                if as_book:
                    meta["book_stats"] = qb_book_stats(gmap, gidx)
                st.session_state.qb_exam_html = render_qbank_exam(picked, meta, gidx)
                st.session_state.qb_exam_info = (len(picked), notes, title or default_title or "변형문제")
                st.session_state.qb_exam_qids = [r["id"] for r in picked]
                st.session_state.qb_exam_logged = False
            else:
                st.session_state.pop("qb_exam_html", None)
                st.warning("조건에 맞는 문제가 없습니다. '문제 만들기'에서 먼저 생성해 주세요.")
        if st.session_state.get("qb_exam_html"):
            n, notes, fname = st.session_state.qb_exam_info
            for note in notes:
                st.warning(note)
            safe = re.sub(r'[\\/:*?"<>|]', "", fname).strip() or "변형문제"
            st.download_button(f"📥 시험지 HTML 다운로드 ({n}문항) — 열어서 '인쇄 / PDF로 저장'", st.session_state.qb_exam_html,
                               f"{safe}.html", "text/html", use_container_width=True, key="qb_exam_dl")
            s1, s2 = st.columns([3, 1.2], vertical_alignment="bottom")
            log_name = s1.text_input("이 시험지 기록 이름", placeholder="예: 김민수 1차 · 고1 A반 중간 대비", key="qb_exam_log_name",
                                     help="기록해 두면 다음 시험지를 만들 때 '이미 낸 문항'에서 골라 이 문항들을 뺄 수 있어요")
            if st.session_state.get("qb_exam_logged"):
                s2.button("✅ 기록했어요", disabled=True, use_container_width=True, key="qb_exam_log_done")
            elif s2.button("💾 문항 기록 저장", use_container_width=True, key="qb_exam_log_save", disabled=not log_name.strip()):
                from datetime import datetime, timedelta, timezone
                rec = {"id": uuid.uuid4().hex[:10], "name": log_name.strip(), "title": fname, "n": n,
                       "date": datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d"), "qids": st.session_state.get("qb_exam_qids", [])}
                qb_exam_log_update(sb, lambda rs: [rec] + rs)
                st.session_state.qb_exam_logged = True
                st.rerun()
            render_choice_trans_ui(sb, "qb_exam_tr", st.session_state.get("qb_exam_qids", []), fname, text_model, academy_name, school_name)
            components.html(st.session_state.qb_exam_html, height=850, scrolling=True)

    with t_clinic:
        render_weekly_clinic(sb, mock_lib, by_label, summary, label_counts, text_model, api_delay, academy_name, school_name)

    with t_manage:
        m_label = st.selectbox("📌 지문", sorted([p["label"] for p in mock_lib] + links, key=lambda l: natural_sort_key({"label": l})), key="qb_manage_label")
        rows = qb_fetch(sb, [m_label]) if (m_label and t_manage.open is not False) else []
        if not rows:
            st.caption("이 지문에는 아직 쌓인 문제가 없습니다.")
        else:
            table = pd.DataFrame([{"삭제": False, "유형": r["q_type"], "구분": "실제 기출" if r["difficulty"] == QB_GICHUL_LEVEL else f"변형 · {r['difficulty']}",
                                   "학교": r.get("school", "") + ("" if r["difficulty"] == QB_GICHUL_LEVEL else " 스타일"),
                                   "정답": (r["content"].get("answer", "") if r["q_type"] not in QB_ESSAY_TYPES else "서술형"),
                                   "출제 포인트": r["content"].get("focus", ""),
                                   "검증": "✅" if r["content"].get("verified") else ("⚠️ 정답 확인" if r["content"].get("answer_note") else ""),
                                   "id": r["id"]} for r in rows])
            edited = st.data_editor(table, hide_index=True, use_container_width=True, disabled=[c for c in table.columns if c != "삭제"],
                                    column_config={"id": None}, key=f"qb_manage_editor_{m_label}")
            to_del = edited.loc[edited["삭제"], "id"].tolist()
            v1, v2 = st.columns([1, 1])
            with v1.popover(f"🗑️ 고른 문제 삭제 ({len(to_del)})", use_container_width=True, disabled=not to_del):
                st.warning(f"{len(to_del)}문제를 영구 삭제합니다. 되돌릴 수 없습니다.")
                if st.button("삭제 확정", type="primary", key="qb_manage_del"):
                    qb_delete(sb, to_del)
                    qb_bank_summary.clear()
                    st.rerun()
            pick = v2.selectbox("👁️ 미리보기", range(len(rows)), format_func=lambda i: f"{i + 1}. {rows[i]['q_type']} · {rows[i]['difficulty']} · {rows[i]['content'].get('focus', '')[:40]}", key="qb_manage_preview")
            components.html(render_qbank_exam([rows[pick]], {"title": m_label, "show_tag": True, "with_answers": True, "show_gichul": True},
                                              gichul_index(load_gichul_map(), set(by_label))), height=600, scrolling=True)

    with t_gichul:
        gmap = load_gichul_map()
        exams = gmap.get("exams") or {}
        gidx = gichul_index(gmap, set(by_label))
        st.caption(f"실제 기출 시험지 {len(exams)}개를 분석해 두었고, 원문 라이브러리 지문 {len(gidx)}개가 기출로 연결되어 있습니다. "
                   "새 시험지 PDF를 올리면 문항별 출처를 찾아 '기출 이력'에 쓰고, 원문 라이브러리 지문으로 낸 문항은 **실제 기출문제**로 문제 은행에 수록합니다. "
                   "모든 문항의 발문·지문·선지·정답도 기출 분석용으로 따로 보관해요 "
                   "(시험지 1개에 약 2~3분, 약 200원 · 영T FINAL용 지문 단어 수 재기 약 50원 포함).")
        g_pdfs = st.file_uploader("📄 운양고·제일고 기출 시험지 PDF (여러 개 가능)", type=["pdf"], accept_multiple_files=True, key="gichul_pdf_uploader")
        if g_pdfs and st.button("🔎 기출 문항 출처 분석해서 추가", type="primary", key="gichul_btn"):
            cands = gichul_candidates_text(mock_lib)
            prog = st.progress(0.0)
            for i, f in enumerate(g_pdfs, 1):
                key_md5 = hashlib.md5(f.getvalue()).hexdigest()[:12]
                if key_md5 in exams:
                    continue
                with st.spinner(f"{f.name} 분석 중... ({i}/{len(g_pdfs)})"):
                    res = gichul_extract(client, text_model, pdf_to_part(f), f.name, cands)
                if res:
                    res["_file"] = f.name
                    n_off = gichul_check_links(res, by_label)
                    if n_off:
                        st.toast(f"⚠️ {f.name}: 원문과 내용이 맞지 않는 {n_off}문항은 연결하지 않았어요")
                    exams[key_md5] = gichul_light(res, f.name)
                    gichul_full_add(key_md5, gichul_full_record(res, f.name, text_model))  # 전 문항 보관 (분석용)
                    w_stat = final_measure_words(client, text_model, pdf_to_part(f), f.name)  # 영T FINAL 단어 수 기준 (Gemini Flash 기준 약 50원)
                    if w_stat:
                        final_save_words(sb, key_md5, w_stat)
                    linked = list({gichul_question_label(exams[key_md5], q, set(by_label)) for q in res.get("questions", [])} - {""})
                    bank_rows = gichul_bank_rows(client, text_model, key_md5, res, by_label, qb_existing_gichul_keys(qb_fetch(sb, linked)), [])
                    qb_insert(sb, bank_rows)
                    # 시험지 하나 끝날 때마다 바로 저장 (DB 최신 기록을 다시 읽어 이 시험지만 더함)
                    #   예전엔 모든 PDF가 끝난 뒤에만 저장해서, 중간에 끊기거나 다른 창에서 올리면 비용을 낸 분석이 기출 이력에서 빠졌음
                    gichul_add_exam(key_md5, exams[key_md5])
                    st.toast(f"✅ {exams[key_md5]['display']}: 실제 기출문제 {len(bank_rows)}개 수록")
                prog.progress(i / len(g_pdfs))
            dna_board_exam_stats.clear()  # 출제DNA 탭 현황판 숫자 갱신 (보기용 캐시만 비움)
            st.success("✅ 기출 출처를 저장했습니다.")
            st.rerun()
        if gidx:
            st.markdown("##### 🏫 기출로 연결된 지문")
            st.dataframe(pd.DataFrame([{"지문": l, "기출 이력": " / ".join(f"{h['short']} {h['no']}번({h['q_type']})" for h in hs)}
                                       for l, hs in sorted(gidx.items(), key=lambda x: natural_sort_key({"label": x[0]}))]),
                         hide_index=True, use_container_width=True)


# -----------------------------
# 5-B. 모의고사 PDF -> 정답 반영 원문 복원 & 년도/월/번호별 원문 라이브러리
# -----------------------------
SCHEMA_MOCK_RESTORE = {
    "type": "object",
    "properties": {
        "exam_year": {"type": "string", "description": "시행 연도 4자리 (예: 2026)"},
        "exam_month": {"type": "string", "description": "시행 월 숫자 (예: 3, 6, 9, 11)"},
        "grade": {"type": "string", "description": "학년 (예: 고1, 고2, 고3)"},
        "answer_key_found": {"type": "boolean", "description": "업로드 파일에서 공식 정답표를 찾았는지 여부"},
        "passages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "number": {"type": "string", "description": "문항 번호 (예: 18, 31, 41~42, 43~45)"},
                    "q_type": {"type": "string", "description": "문제 유형 (예: 글의 목적, 빈칸 추론, 글의 순서, 문장 삽입)"},
                    "answer": {"type": "string", "description": "정답 번호 (예: ③, 41~42는 '41:② 42:④')"},
                    "restored_text": {"type": "string", "description": "정답이 반영된 원래의 영어 원문"},
                    "needs_review": {"type": "boolean", "description": "정답표가 없거나 복원이 불확실해 사람 확인이 필요하면 true"},
                    "note": {"type": "string", "description": "복원 시 적용한 내용 요약 (예: 빈칸에 ③ 표현 삽입, (B)-(A)-(C) 순서로 배열)"}
                },
                "required": ["number", "q_type", "answer", "restored_text", "needs_review", "note"]
            }
        }
    },
    "required": ["exam_year", "exam_month", "grade", "answer_key_found", "passages"]
}

MOCK_RESTORE_PROMPT = """너는 대한민국 고등학교 영어 모의고사(학력평가·수능 모의평가) 원문 복원 전문가다.
업로드된 모의고사 시험지 PDF{answer_note}에서 **독해 문항(18번~45번)** 의 지문을 꺼내, 문제로 변형되기 전의 **정답이 모두 반영된 원래 영어 원문**으로 복원해 JSON으로 반환하라. 듣기 문항(1~17번)은 제외한다.

[정답 출처]
- 별도 정답표 파일이 있으면 그 정답을 최우선으로 사용하라.
- 정답표 파일이 없으면 시험지 PDF 안(보통 마지막 페이지)의 정답표를 찾아 사용하라.
- 어디에도 정답표가 없으면 직접 풀어서 복원하되, 해당 문항의 needs_review를 true로 하고 answer_key_found는 false로 하라.

[유형별 복원 규칙]
- 빈칸 추론(31~34번 등): 빈칸 자리에 정답 선지의 표현을 문법적으로 자연스럽게 넣는다.
- 무관한 문장(35번): 정답에 해당하는 흐름과 무관한 문장을 삭제한다.
- 글의 순서(36~37번): 주어진 글 뒤에 (A)(B)(C)를 정답 순서대로 이어 붙인다.
- 문장 삽입(38~39번): 주어진 문장을 정답 위치(①~⑤)에 넣는다.
- 요약문(40번): 본문만 원문으로 남기고 요약문 문장은 넣지 않는다. (A)(B) 정답은 note에 적는다.
- 어법(29번): 틀린 밑줄 표현을 올바른 형태로 고친다. 나머지 밑줄 표현은 그대로 둔다.
- 어휘(30번, 42번 등): 문맥상 부적절한 낱말을 문맥에 맞는 낱말로 바꾼다.
- 함축 의미(21번) 등 밑줄 문항: 밑줄 표시만 없애고 문장은 그대로 둔다.
- 장문 41~42번: 한 지문이므로 number를 "41~42"로 하나만 만든다.
- 장문 43~45번: (A) 다음에 (B)(C)(D)를 정답 순서대로 배열하고, number를 "43~45"로 하나만 만든다.
- 도표·안내문·내용 일치 문항(25~28번 등)도 지문 텍스트를 그대로 복원한다.

[공통 규칙]
- 발문(문제 지시문), 선택지, 문항 번호, 원문자(①~⑤), (A)(B)(C)/(a)~(e) 표시, 밑줄, 하단 어휘 각주(*표시)는 모두 제거한다.
- 원문 문장을 절대 요약·의역·생략하지 말고 철자와 구두점을 시험지 그대로 옮긴다.
- 문단 구분은 빈 줄로 표시한다.
- note에는 어떤 정답을 어떻게 반영했는지 한국어로 짧게 적는다.
- exam_year, exam_month, grade는 시험지 표지/머리말에서 파악한다."""

def normalize_mock_meta(year: str, month: str, grade: str) -> (str, str, str):
    y = "".join(re.findall(r'\d+', str(year)))
    if len(y) == 2:
        y = "20" + y
    m = "".join(re.findall(r'\d+', str(month))).lstrip("0")
    g_digits = re.findall(r'\d', str(grade))
    g = f"고{g_digits[0]}" if g_digits else str(grade).strip()
    return y, m, g

def mock_exam_key(year: str, month: str, grade: str) -> str:
    return f"[{year}년 {month}월 {grade}]"

def pdf_to_part(uploaded):
    """업로드한 PDF를 AI에 보낼 파일로 (큰 PDF는 Gemini일 때 자동으로 파일 업로드 방식으로 보냄)"""
    return AiFile(uploaded.getvalue(), "application/pdf", uploaded.name)

def extract_mock_exam_passages(pdf_file, answer_file, model_name: str) -> Dict:
    contents = [pdf_to_part(pdf_file)]
    answer_note = ""
    if answer_file is not None:
        ans_mime = answer_file.type if getattr(answer_file, "type", None) else "application/pdf"
        contents.append(AiFile(answer_file.getvalue(), ans_mime, answer_file.name))
        answer_note = f"(파일명: {pdf_file.name})과 별도 정답표 파일(파일명: {answer_file.name})"
    else:
        answer_note = f"(파일명: {pdf_file.name})"
    contents.append(MOCK_RESTORE_PROMPT.format(answer_note=answer_note))

    for attempt in range(4):
        try:
            return llm_json(model_name, contents, SCHEMA_MOCK_RESTORE, 0.1)
        except Exception as e:
            err_s = str(e).lower()
            if attempt < 3 and any(k in err_s for k in ["503", "429", "overloaded", "unavailable", "quota", "high demand"]):
                wait_t = 10 + attempt * 10
                st.toast(f"⏳ 서버 대기 감지 ({pdf_file.name}). {wait_t}초 후 재시도...")
                time.sleep(wait_t)
                continue
            st.error(f"❌ '{pdf_file.name}' 원문 복원 중 오류: {e}")
            return None
    return None

def mock_number_sort_key(num: str):
    """'18' → (18,), '41~42' → (41, 42), 교과서 '4-2' → (4, 2) 순서로 정렬"""
    nums = re.findall(r'\d+', str(num))
    return tuple(int(n) for n in nums) if nums else (999,)

# ---- 교과서 PDF → 단락(섹션)별 본문 원문 ----
SCHEMA_TEXTBOOK = {
    "type": "object",
    "properties": {
        "subject": {"type": "string", "description": "과목명 (예: 공통영어2, 영어I)"},
        "publisher": {"type": "string", "description": "출판사(저자) (예: 능률(오), 천재(조), YBM(박))"},
        "unit": {"type": "string", "description": "단원 (예: 4과)"},
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "number": {"type": "string", "description": "단원번호-순서 (예: 4-1, 4-2)"},
                    "title": {"type": "string", "description": "섹션 제목 또는 소제목 (없으면 본문 / Read More 등)"},
                    "text": {"type": "string", "description": "섹션의 영어 본문 원문"},
                    "needs_review": {"type": "boolean"},
                    "note": {"type": "string"}
                },
                "required": ["number", "title", "text", "needs_review", "note"]
            }
        }
    },
    "required": ["subject", "publisher", "unit", "sections"]
}

TEXTBOOK_PROMPT = """너는 대한민국 고등학교 영어 교과서 본문 정리 전문가다.
업로드된 교과서 PDF(파일명: {fname})에서 **본문 읽기 지문(Reading)** 을 찾아 원문 그대로 옮기고, 교과서가 나눈 단락·페이지 구획대로 섹션을 나눠 JSON으로 반환하라.
{hint}
[포함]
- 단원의 메인 본문(Reading)과 추가 읽기 지문(Read More, Further Reading 등). 추가 읽기 지문은 title에 그 이름을 적는다.
[제외]
- 듣기·말하기 대화문, 문제·활동(Check Up, Think & Talk 등), 어휘 목록, 한국어 해석, 각주(*), 그림 설명, 페이지 번호, 머리말·꼬리말.
[나누는 규칙]
- 교과서에 표시된 단락 번호·소제목·페이지 구획이 있으면 그대로 따른다. 없으면 내용 흐름에 따라 한 섹션이 5~10문장 정도가 되게 나눈다.
- number는 "단원번호-순서" 형식(예: 4과의 첫 섹션 → "4-1", 둘째 → "4-2")으로 쓴다.
- text는 원문 문장을 요약·의역·생략 없이 철자와 구두점 그대로 옮기고, 문단 구분은 빈 줄로 표시한다.
- 글자가 흐리거나 페이지가 잘려 확실하지 않은 섹션은 needs_review를 true로 하고 note에 이유를 적는다.
- subject, publisher, unit은 PDF 표지·머리말에서 파악한다."""

def textbook_exam_key(subject: str, publisher: str, unit: str) -> str:
    return "[" + re.sub(r'\s+', ' ', f"{subject} {publisher} {unit}").strip() + "]"

def extract_textbook_sections(pdf_file, model_name: str, subject: str, publisher: str, unit: str) -> Dict:
    known = ", ".join(f"{k}: {v}" for k, v in [("과목", subject), ("출판사", publisher), ("단원", unit)] if v)
    hint = f"(사용자가 알려준 정보 → {known}. 이 정보를 우선 사용하라.)" if known else ""
    contents = [pdf_to_part(pdf_file), TEXTBOOK_PROMPT.format(fname=pdf_file.name, hint=hint)]
    for attempt in range(4):
        try:
            return llm_json(model_name, contents, SCHEMA_TEXTBOOK, 0.1)
        except Exception as e:
            err_s = str(e).lower()
            if attempt < 3 and any(k in err_s for k in ["503", "429", "overloaded", "unavailable", "quota", "high demand"]):
                wait_t = 10 + attempt * 10
                st.toast(f"⏳ 서버 대기 감지 ({pdf_file.name}). {wait_t}초 후 재시도...")
                time.sleep(wait_t)
                continue
            st.error(f"❌ '{pdf_file.name}' 교과서 본문 추출 중 오류: {e}")
            return None
    return None

def build_mock_excel_bytes(records: List[Dict]) -> bytes:
    """기존 '엑셀 원클릭 통합 생성' 탭에 그대로 올릴 수 있는 형식(라벨/지문/학교명/이미지명)"""
    rows = [{"라벨": p["label"], "지문": p.get("body_text", ""), "학교명": "", "이미지명": ""} for p in records]
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame(rows, columns=["라벨", "지문", "학교명", "이미지명"]).to_excel(writer, sheet_name="모의고사_원문", index=False)
        try:
            ws = writer.book["모의고사_원문"]
            ws.column_dimensions["A"].width = 28
            ws.column_dimensions["B"].width = 120
        except Exception:
            pass
    return output.getvalue()

# ---- 5-C. 지문 이미지 생성 (기존 교재 이미지 스타일 참고) ----
# 3.1 flash: 장당 약 20초, 교재용으로 충분한 품질에 저렴 / pro: 더 정교하지만 비싸고 느림
IMAGE_MODELS = ["gemini-3.1-flash-image", "gemini-3-pro-image"]
IMAGE_PARALLEL = 4
BOOKLET_IMG_MAX_SIDE = 1280

# 지문마다 화풍을 바꿔 학생들이 지루하지 않게 함 (화면 이름, 프롬프트 지시)
IMAGE_STYLES = [
    ("영화 같은 실사 장면", "a cinematic photorealistic still shot on location (outdoors or in the real place where the story happens), dramatic natural light"),
    ("수채화 동화 일러스트", "a soft, colorful storybook watercolor illustration with hand-drawn charm"),
    ("3D 애니메이션 장면", "a bright, expressive 3D animated movie still, playful character acting and vivid colors"),
    ("클레이 / 미니어처", "a cute claymation / tilt-shift miniature diorama with tactile handmade textures"),
    ("포스터 일러스트", "a bold flat vector poster illustration with strong shapes and a limited vibrant palette"),
    ("초현실 은유 아트", "a surreal conceptual artwork that turns the main idea into one striking visual metaphor"),
    ("다큐멘터리 사진", "a documentary / National Geographic style photograph of the real subject (nature, science, history or society)"),
    ("그래픽 노블 한 컷", "a dynamic graphic-novel panel with ink lines, dramatic angle and cel shading"),
]
IMAGE_STYLE_AUTO = "자동 (지문마다 다양하게)"

MOCK_IMAGE_PROMPT = """Create one picture for a Korean high school English textbook page that makes the English passage below easy and fun to grasp at a glance.

[What to show]
- First decide the single most vivid, concrete image of this passage: the actual situation, place, event, creature, object or experiment it talks about, or one clear visual metaphor of its main idea.
- Show it in the place where it really happens (outdoors, nature, city, lab, stadium, history, space, ocean, etc.).
- Avoid generic office or study scenes: NO people sitting around a desk or table, NO meetings, NO laptops or paperwork, NO classroom or library, unless the passage is literally about that place.

[Style]
- Render it as {style}.
- 16:9 landscape, one clear focal point, readable at a small size on a printed page, cheerful or intriguing mood.
- Do not put any words, letters or numbers in the picture (no signs, labels or speech bubbles), so nothing can be misspelled.
- No watermark, no logo, no border, no caption outside the picture.{ref_note}

[Passage]
\"\"\"{passage}\"\"\""""

def pick_image_styles(n: int, choice: str) -> List[str]:
    """선택한 화풍을 n개 지문에 배정. '자동'이면 섞어서 돌아가며 배정해 이웃한 지문끼리 화풍이 겹치지 않게 함."""
    if choice != IMAGE_STYLE_AUTO:
        directive = next((d for name, d in IMAGE_STYLES if name == choice), IMAGE_STYLES[0][1])
        return [directive] * n
    order = [d for _, d in IMAGE_STYLES]
    random.shuffle(order)
    return [order[i % len(order)] for i in range(n)]

def mock_image_src(p: Dict) -> str:
    """st.image에 넘길 이미지 위치 (로컬 파일 우선, 없으면 Supabase 공개 URL)"""
    path = p.get("image_path", "")
    if not path:
        return ""
    fn = os.path.basename(str(path).replace("\\", "/"))
    local_p = os.path.join(IMAGE_DIR, fn)
    if os.path.exists(local_p):
        return local_p
    if IS_CLOUD_DB:
        return f"{SUPABASE_URL}/storage/v1/object/public/image_assets/{fn}"
    return ""

def shrink_image_bytes(raw: bytes, max_side: int = 1024) -> bytes:
    """스타일 참고용 이미지를 작게 줄여 요청 용량을 줄임 (기존 교재 이미지는 장당 7~9MB)"""
    from PIL import Image
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return buf.getvalue()

@st.cache_data(show_spinner=False, max_entries=10)
def load_style_reference_bytes(img_filename: str) -> bytes:
    local_p = os.path.join(IMAGE_DIR, img_filename)
    try:
        if os.path.exists(local_p):
            with open(local_p, "rb") as f:
                raw = f.read()
        elif IS_CLOUD_DB:
            import urllib.request
            raw = urllib.request.urlopen(f"{SUPABASE_URL}/storage/v1/object/public/image_assets/{img_filename}", timeout=60).read()
        else:
            return b""
        return shrink_image_bytes(raw)
    except Exception:
        return b""

def default_style_reference_files(limit: int = 2) -> List[str]:
    """한판승 교재에 쓰인 이미지 중 앞에서부터 참고 이미지로 사용"""
    files = []
    for p in st.session_state.saved_v:
        if p.get("image_path"):
            fn = os.path.basename(str(p["image_path"]).replace("\\", "/"))
            if fn not in files:
                files.append(fn)
        if len(files) >= limit:
            break
    return files

def image_cost_krw(model_name: str, n_refs: int = 0) -> float:
    """이미지 1장 예상 비용(원): 1K 이미지 출력 1120토큰 + 프롬프트(지문 포함 약 900토큰) + 참고 이미지 장당 560토큰"""
    p_in, _, p_out = ai_price(model_name)
    return (1120 * p_out + (900 + 560 * n_refs) * p_in) / 1_000_000 * 1400

def generate_passage_image(passage: str, style_refs: List[bytes], model_name: str, style: str = "") -> (bytes, str):
    """(이미지 bytes, 오류 메시지) 반환. 동시 생성 스레드에서 호출되므로 st.* 를 쓰지 않는다.
    참고 이미지는 직접 올린 경우에만 보냄 (입력이 줄어 비용도 적음)."""
    if client is None:
        return None, "AI 이미지 생성은 Gemini 키가 있어야 합니다 (Secrets에 GEMINI_API_KEY 등록)"
    contents = [types.Part.from_bytes(data=b, mime_type="image/jpeg") for b in style_refs if b]
    ref_note = "\n- Match the color, lighting and rendering feel of the attached reference images, but draw a completely new scene for this passage (do not copy their people, places or text)." if contents else ""
    contents.append(MOCK_IMAGE_PROMPT.format(passage=passage, style=style or IMAGE_STYLES[0][1], ref_note=ref_note))
    last_err = ""
    for attempt in range(3):
        try:
            res = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(aspect_ratio="16:9")
                )
            )
            u = getattr(res, "usage_metadata", None)
            if u:
                # 'AI 사용 금액'에 이미지 비용도 넣음 (예전엔 빠졌음). 생각 토큰은 글 출력 단가(이미지 단가의 1/20·1/10)라 그만큼 환산
                ratio = 0.1 if "pro" in model_name else 0.05
                _log_usage("gemini", model_name, u.prompt_token_count, u.cached_content_token_count,
                           round((u.candidates_token_count or 0) + (u.thoughts_token_count or 0) * ratio))
            for cand in (res.candidates or []):
                for part in (cand.content.parts if cand.content else []) or []:
                    if part.inline_data and part.inline_data.data:
                        return part.inline_data.data, ""
            return b"", "모델이 이미지를 반환하지 않았습니다."
        except Exception as e:
            last_err = str(e)
            if attempt < 2 and any(k in last_err.lower() for k in ["503", "429", "overloaded", "unavailable", "quota", "high demand"]):
                time.sleep(10 + attempt * 10)
                continue
            return b"", last_err
    return b"", last_err

def save_booklet_image(raw: bytes, name_prefix: str, sb_client=None, quiet: bool = False) -> str:
    """교재용으로 1280px JPEG로 줄여 로컬 + Supabase Storage에 저장 (장당 약 200~300KB)
    sb_client/quiet: 백그라운드 생성 작업에서 호출할 때 자기 DB 연결을 쓰고 st.* 를 쓰지 않기 위함"""
    from PIL import Image
    try:
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        im.thumbnail((BOOKLET_IMG_MAX_SIDE, BOOKLET_IMG_MAX_SIDE))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=82, optimize=True)
        jpg = buf.getvalue()
    except Exception:
        return ""
    # 다시 만들 때 캐시에 옛 이미지가 남지 않도록 매번 새 파일명 사용
    filename = f"{name_prefix}_{uuid.uuid4().hex[:6]}.jpg"
    img_path = os.path.join(IMAGE_DIR, filename)
    with open(img_path, "wb") as f:
        f.write(jpg)
    if IS_CLOUD_DB:
        try:
            (sb_client or supabase).storage.from_("image_assets").upload(
                path=filename, file=jpg,
                file_options={"content-type": "image/jpeg", "upsert": "true"}
            )
        except Exception as e:
            if quiet:
                return ""  # 클라우드에 못 올린 이미지는 배포 서버에서 사라지므로 연결하지 않음
            st.warning(f"⚠️ 이미지 클라우드 업로드 실패 ({filename}): {e}")
    return img_path

def set_mock_image(p: Dict, img_path: str):
    """모의고사 지문에 이미지를 연결하고, 같은 라벨의 한판승·Self-Note 교재에도 반영
    (교재 이미지가 비어 있거나 이 지문의 이전 이미지였던 경우에만 교체)"""
    old_fn = os.path.basename(str(p.get("image_path", "")).replace("\\", "/"))
    p["image_path"] = img_path
    for saved_key, fpath in [("saved_v", VICTORY_FILE), ("saved_a", ANATOMY_FILE)]:
        hit = False
        for bp in st.session_state[saved_key]:
            if bp.get("label") != p["label"]:
                continue
            bp_fn = os.path.basename(str(bp.get("image_path", "")).replace("\\", "/"))
            if not bp_fn or (old_fn and bp_fn == old_fn):
                bp["image_path"] = img_path
                hit = True
        if hit:
            save_to_file(fpath, st.session_state[saved_key])

def render_mock_image_section(selected: List[Dict]):
    with st.container(border=True):
        st.markdown("##### 🎨 선택 지문 이미지 생성")
        st.caption("💡 지문 내용을 가장 잘 보여주는 장면을 지문마다 다른 화풍(실사·수채화·3D 애니메이션·클레이·포스터 등)으로 그립니다. 만든 이미지는 이 지문으로 생성하는 한판승·Self-Note 교재에 자동으로 들어갑니다.")
        i_col1, i_col2 = st.columns([2, 1])
        img_style = i_col1.selectbox("🖌️ 화풍", [IMAGE_STYLE_AUTO] + [name for name, _ in IMAGE_STYLES], key="mock_img_style")
        with i_col1.expander("🖼️ 원하는 느낌의 참고 이미지 올리기 (선택)", expanded=False):
            ref_uploads = st.file_uploader("참고 이미지 (여러 장 가능)", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="mock_style_refs")
            use_book_style = st.checkbox("기존 한판승 교재 이미지 느낌 따라하기 (참고 이미지 2장 자동 첨부)", value=False, key="mock_use_book_style")
        img_model = i_col2.selectbox("🎨 이미지 모델", IMAGE_MODELS, key="mock_img_model", help="gemini-3.1-flash-image: 빠르고 저렴(교재용 권장) / gemini-3-pro-image: 더 정교하지만 비싸고 느림")
        redo = i_col2.checkbox("이미 있는 이미지도 다시 만들기", value=False, key="mock_img_redo")

        # 참고 이미지는 직접 고른 경우에만 첨부 (없으면 입력이 줄어 비용이 적고 화풍도 자유로움)
        if ref_uploads:
            style_refs = [shrink_image_bytes(u.getvalue()) for u in ref_uploads]
        elif use_book_style:
            style_refs = [load_style_reference_bytes(fn) for fn in default_style_reference_files()]
        else:
            style_refs = []
        style_refs = [b for b in style_refs if b]
        if style_refs:
            st.image(style_refs, width=120, caption=[f"참고 {i+1}" for i in range(len(style_refs))])

        targets = [p for p in selected if redo or not p.get("image_path")]
        styles_for = dict(zip([p["id"] for p in targets], pick_image_styles(len(targets), img_style)))
        if targets:
            per = image_cost_krw(img_model, len(style_refs))
            st.caption(f"🧾 예상 비용 약 {per * len(targets):,.0f}원 (이미지 {len(targets)}장 · 장당 약 {per:,.0f}원 · {img_model} · 대략)")
        if st.button(f"🎨 이미지 생성 ({len(targets)}개, {IMAGE_PARALLEL}장씩 동시 생성)", type="primary", use_container_width=True, disabled=not targets, key="mock_img_gen_btn"):
            from concurrent.futures import ThreadPoolExecutor, as_completed
            prog = st.progress(0)
            status = st.empty()
            status.info(f"⏳ 이미지 {len(targets)}개 생성 중... ({IMAGE_PARALLEL}장씩 동시에 진행, 장당 약 20초)")
            done, errors = 0, []
            with ThreadPoolExecutor(max_workers=IMAGE_PARALLEL) as pool:
                futures = {pool.submit(generate_passage_image, p.get("body_text", ""), style_refs, img_model, styles_for[p["id"]]): p for p in targets}
                for n, fut in enumerate(as_completed(futures), start=1):
                    p = futures[fut]
                    img_bytes, err = fut.result()
                    img_path = save_booklet_image(img_bytes, f"mock_{p['id'][:8]}") if img_bytes else ""
                    if img_path:
                        set_mock_image(p, img_path)
                        done += 1
                        # 한 장 만들 때마다 바로 저장 (예전엔 다 끝난 뒤에만 저장해서 중간에 끊기면 비용을 낸 그림 연결이 사라졌음)
                        save_to_file(MOCK_FILE, st.session_state.saved_mock)
                    else:
                        errors.append(f"{p['label']}: {err or '이미지 저장 실패'}")
                    prog.progress(n / len(targets))
                    status.info(f"⏳ {n}/{len(targets)} 완료: {p['label']}")
            prog.empty()
            status.success(f"✅ 이미지 {done}개를 만들었습니다.")
            for e in errors:
                st.error(f"❌ {e}")

        st.markdown(f"**🖼️ 선택 지문 이미지 ({sum(1 for p in selected if mock_image_src(p))}/{len(selected)}개)** · 마음에 안 드는 이미지는 '📤 내 이미지로 교체'로 직접 올릴 수 있습니다.")
        g_cols = st.columns(3)
        for i, p in enumerate(selected):
            with g_cols[i % 3]:
                src = mock_image_src(p)
                if src:
                    st.image(src, caption=p["label"], width="stretch")
                else:
                    st.caption(f"🖼️ {p['label']} · 이미지 없음")
                with st.popover("📤 내 이미지로 교체", use_container_width=True):
                    up = st.file_uploader("이미지 선택", type=["png", "jpg", "jpeg", "webp"], key=f"mock_own_img_{p['id']}")
                    if up is not None and st.button("이 이미지로 저장", type="primary", key=f"mock_own_save_{p['id']}"):
                        img_path = save_booklet_image(up.getvalue(), f"mock_{p['id'][:8]}")
                        if img_path:
                            set_mock_image(p, img_path)
                            save_to_file(MOCK_FILE, st.session_state.saved_mock)
                            st.rerun()
                        else:
                            st.error("❌ 이미지를 읽지 못했습니다.")

def render_mock_exam_tab(text_model: str, api_delay: int, academy_name: str, school_name: str):
    kind_label = st.segmented_control("자료 종류", ["모의고사", "교과서"], default="모의고사", key="src_kind") or "모의고사"
    is_textbook = kind_label == "교과서"
    if is_textbook:
        st.info(
            "💡 교과서 PDF를 올리면 AI가 **본문(Reading)을 단락별로 나눠** `공통영어2 능률(오) 4과 4-1, 4-2…`처럼 원문을 저장합니다. "
            "단원 하나만 담긴 PDF가 가장 정확하고, 교과서 전체 PDF도 올릴 수 있습니다(단원을 적어 주세요)."
        )
    else:
        st.info(
            "💡 모의고사 시험지 PDF와 정답표를 올리면 AI가 **18~45번 지문을 정답이 반영된 원래 원문**으로 복원합니다. "
            "(빈칸 채움, 순서 재배열, 삽입 문장 반영, 무관한 문장 삭제, 어법·어휘 수정) "
            "복원 결과를 표에서 확인·수정한 뒤 저장하면, 아래에서 **년도·월·학년별로 번호를 클릭해 원하는 지문만 교재로 생성**할 수 있습니다."
        )

    # ---- 1) 업로드 & 원문 추출 ----
    if is_textbook:
        with st.container(border=True):
            st.markdown("##### 1️⃣ 교과서 PDF 업로드 & 단락별 본문 추출")
            tb_pdf = st.file_uploader("📘 교과서 PDF (단원 단위 권장)", type=["pdf"], key="tb_pdf_uploader")
            t_col1, t_col2, t_col3, t_col4 = st.columns([1.2, 1.2, 1, 2])
            tb_subject = t_col1.text_input("과목 (비우면 자동 인식)", placeholder="예: 공통영어2", key="tb_in_subject")
            tb_publisher = t_col2.text_input("출판사(저자)", placeholder="예: 능률(오)", key="tb_in_publisher")
            tb_unit = t_col3.text_input("단원", placeholder="예: 4과", key="tb_in_unit")
            if tb_pdf and t_col4.button("📘 단락별 본문 추출 시작", type="primary", use_container_width=True, key="tb_extract_btn"):
                with st.spinner(f"'{tb_pdf.name}'에서 본문을 단락별로 추출 중입니다... (1~3분 소요)"):
                    parsed = extract_textbook_sections(tb_pdf, text_model, tb_subject.strip(), tb_publisher.strip(), tb_unit.strip())
                if parsed:
                    draft_rows = []
                    for sec in sorted(parsed.get("sections", []), key=lambda x: mock_number_sort_key(x.get("number", ""))):
                        num = str(sec.get("number", "")).strip()
                        if not num:
                            continue
                        draft_rows.append({
                            "번호": num, "유형": sec.get("title", ""), "정답": "",
                            # 교과서는 AI가 이유 없이 모두 '확인 필요'로 표시하는 일이 많아, 이유(메모)를 적은 단락만 표시
                            "확인필요": bool(sec.get("needs_review", False)) and bool(str(sec.get("note", "")).strip()),
                            "복원 원문": sec.get("text", "").strip(), "복원 메모": sec.get("note", ""),
                        })
                    for k in ["mock_draft_editor", "tb_draft_subject", "tb_draft_publisher", "tb_draft_unit"]:
                        st.session_state.pop(k, None)
                    st.session_state.mock_draft = {
                        "kind": "textbook",
                        "subject": tb_subject.strip() or parsed.get("subject", ""),
                        "publisher": tb_publisher.strip() or parsed.get("publisher", ""),
                        "unit": tb_unit.strip() or parsed.get("unit", ""),
                        "source": tb_pdf.name,
                        "rows": draft_rows,
                    }
                    st.rerun()

    if not is_textbook:
        with st.container(border=True):
            st.markdown("##### 1️⃣ 모의고사 PDF 업로드 & 원문 복원")
            u_col1, u_col2 = st.columns(2)
            mock_pdf = u_col1.file_uploader("📄 모의고사 시험지 PDF", type=["pdf"], key="mock_pdf_uploader")
            mock_ans = u_col2.file_uploader(
                "✅ 정답표 (PDF 또는 이미지, 시험지 PDF 안에 정답표가 있으면 생략 가능)",
                type=["pdf", "png", "jpg", "jpeg"], key="mock_ans_uploader"
            )
            m_col1, m_col2, m_col3, m_col4 = st.columns([1, 1, 1, 2])
            in_year = m_col1.text_input("연도 (비우면 자동 인식)", placeholder="예: 2026", key="mock_in_year")
            in_month = m_col2.text_input("월 (비우면 자동 인식)", placeholder="예: 3", key="mock_in_month")
            in_grade = m_col3.selectbox("학년", ["자동 인식", "고1", "고2", "고3"], key="mock_in_grade")

            if mock_pdf and m_col4.button("🧠 정답 반영 원문 복원 시작", type="primary", use_container_width=True, key="mock_extract_btn"):
                with st.spinner(f"'{mock_pdf.name}'의 18~45번 지문을 정답 반영 원문으로 복원 중입니다... (1~2분 소요)"):
                    parsed = extract_mock_exam_passages(mock_pdf, mock_ans, text_model)
                if parsed:
                    y, m, g = normalize_mock_meta(
                        in_year.strip() or parsed.get("exam_year", ""),
                        in_month.strip() or parsed.get("exam_month", ""),
                        in_grade if in_grade != "자동 인식" else parsed.get("grade", "")
                    )
                    draft_rows = []
                    for ps in sorted(parsed.get("passages", []), key=lambda x: mock_number_sort_key(x.get("number", ""))):
                        num = re.sub(r'\s*번\s*$', '', str(ps.get("number", "")).strip()).replace("-", "~")
                        if not num:
                            continue
                        draft_rows.append({
                            "번호": num,
                            "유형": ps.get("q_type", ""),
                            "정답": ps.get("answer", ""),
                            "확인필요": bool(ps.get("needs_review", False)) or not parsed.get("answer_key_found", False),
                            "복원 원문": ps.get("restored_text", "").strip(),
                            "복원 메모": ps.get("note", ""),
                        })
                    for k in ["mock_draft_editor", "mock_draft_year", "mock_draft_month", "mock_draft_grade"]:
                        st.session_state.pop(k, None)
                    st.session_state.mock_draft = {
                        "kind": "mock",
                        "year": y, "month": m, "grade": g,
                        "answer_key_found": parsed.get("answer_key_found", False),
                        "source": mock_pdf.name,
                        "rows": draft_rows
                    }
                    st.rerun()

    # ---- 2) 검수 & 저장 ----
    draft = st.session_state.get("mock_draft")
    if draft:
        draft_is_tb = draft.get("kind") == "textbook"
        with st.container(border=True):
            st.markdown("##### 2️⃣ " + ("교과서 본문 검수" if draft_is_tb else "복원 결과 검수") + " (셀을 더블클릭해 수정 → 저장)")
            d_col1, d_col2, d_col3 = st.columns(3)
            if draft_is_tb:
                draft["subject"] = d_col1.text_input("과목", value=draft["subject"], key="tb_draft_subject")
                draft["publisher"] = d_col2.text_input("출판사(저자)", value=draft["publisher"], key="tb_draft_publisher")
                draft["unit"] = d_col3.text_input("단원", value=draft["unit"], key="tb_draft_unit")
                meta_ok = bool(draft["subject"].strip() and draft["publisher"].strip() and draft["unit"].strip())
                ekey = textbook_exam_key(draft["subject"], draft["publisher"], draft["unit"])
            else:
                draft["year"] = d_col1.text_input("연도", value=draft["year"], key="mock_draft_year")
                draft["month"] = d_col2.text_input("월", value=draft["month"], key="mock_draft_month")
                draft["grade"] = d_col3.text_input("학년", value=draft["grade"], key="mock_draft_grade")
                y, m, g = normalize_mock_meta(draft["year"], draft["month"], draft["grade"])
                meta_ok = bool(y and m and g)
                ekey = mock_exam_key(y, m, g)
                if not draft.get("answer_key_found"):
                    st.warning("⚠️ 정답표를 찾지 못해 AI가 직접 푼 정답으로 복원했습니다. **확인필요** 표시된 문항을 꼭 검토해주세요.")

            review_cnt = sum(1 for r in draft["rows"] if r.get("확인필요"))
            st.caption(f"📌 저장될 이름: **{md_safe(ekey)}** · 지문 {len(draft['rows'])}개 · 확인필요 {review_cnt}개 · 원본 파일: {draft.get('source', '')}")

            if any(p.get("exam_key") == ekey for p in st.session_state.saved_mock):
                st.info(f"ℹ️ {md_safe(ekey)} 자료가 이미 라이브러리에 있습니다. 저장하면 같은 번호의 지문은 새 내용으로 교체됩니다.")

            editor_cols = ["번호", "유형", "확인필요", "복원 원문", "복원 메모"] if draft_is_tb else ["번호", "유형", "정답", "확인필요", "복원 원문", "복원 메모"]
            edited = st.data_editor(
                pd.DataFrame(draft["rows"], columns=editor_cols),
                num_rows="dynamic",
                use_container_width=True,
                height=420,
                column_config={
                    "유형": st.column_config.TextColumn("단락 제목" if draft_is_tb else "유형"),
                    "복원 원문": st.column_config.TextColumn("본문 원문" if draft_is_tb else "복원 원문", width="large"),
                    "확인필요": st.column_config.CheckboxColumn(),
                },
                key="mock_draft_editor"
            )

            with st.expander("🔍 지문 하나씩 크게 보기 (표에서 수정한 내용 반영)", expanded=False):
                nums = [str(n) for n in edited["번호"].tolist() if str(n).strip() and str(n) != "nan"]
                if nums:
                    v_num = st.selectbox("번호", nums, key="mock_draft_view_num")
                    v_row = edited[edited["번호"].astype(str) == v_num].iloc[0]
                    meta_bits = [f"{'단락 제목' if draft_is_tb else '유형'}: {v_row.get('유형', '')}"]
                    if not draft_is_tb:
                        meta_bits.append(f"정답: {v_row.get('정답', '')}")
                    meta_bits.append(f"메모: {v_row.get('복원 메모', '')}")
                    st.caption(" · ".join(meta_bits))
                    with st.container(border=True):
                        st.markdown(
                            f'<div style="white-space:pre-wrap; line-height:1.7; font-size:14px;">{esc(v_row.get("복원 원문", ""))}</div>',
                            unsafe_allow_html=True
                        )

            s_col1, s_col2 = st.columns([3, 1])
            if s_col1.button(f"💾 {md_safe(ekey)} 원문 라이브러리에 저장", type="primary", use_container_width=True, key="mock_draft_save"):
                if not meta_ok:
                    st.error("과목·출판사·단원을 모두 입력해주세요." if draft_is_tb else "연도·월·학년을 모두 입력해주세요.")
                else:
                    new_records = []
                    for _, r in edited.iterrows():
                        num = str(r.get("번호", "")).strip()
                        body = str(r.get("복원 원문", "")).strip()
                        if not num or num == "nan" or not body or body == "nan":
                            continue
                        rec = {
                            "id": str(uuid.uuid4()),
                            "exam_key": ekey,
                            "number": num,
                            "q_type": str(r.get("유형", "") or ""),
                            "needs_review": bool(r.get("확인필요", False)),
                            "note": str(r.get("복원 메모", "") or ""),
                            "body_text": body,
                        }
                        if draft_is_tb:
                            # 교과서: [공통영어2 능률(오) 4과] 4-1
                            rec.update({"label": f"{ekey} {num}", "source_type": "textbook",
                                        "subject": draft["subject"].strip(), "publisher": draft["publisher"].strip(), "unit": draft["unit"].strip()})
                        else:
                            rec.update({"label": f"{ekey} {num}번", "source_type": "mock",
                                        "year": y, "month": m, "grade": g, "answer": str(r.get("정답", "") or "")})
                        new_records.append(rec)
                    # 같은 지문을 다시 저장할 때: 예전 기록의 id와 만들어 둔 그림(image_*)은 그대로 이어받음
                    #   (예전엔 새 기록으로 통째로 바꿔서 비용을 내고 만든 지문 그림 연결이 빠졌음)
                    old_by_label = {p.get("label"): p for p in st.session_state.saved_mock if isinstance(p, dict)}
                    for rec in new_records:
                        old = old_by_label.get(rec["label"])
                        if old:
                            rec["id"] = old.get("id") or rec["id"]
                            for k, v in old.items():
                                if k.startswith("image_") and v and not rec.get(k):
                                    rec[k] = v
                    new_labels = {p["label"] for p in new_records}
                    st.session_state.saved_mock = [p for p in st.session_state.saved_mock if p.get("label") not in new_labels] + new_records
                    save_to_file(MOCK_FILE, st.session_state.saved_mock)
                    del st.session_state["mock_draft"]
                    st.session_state.gen_flash = f"✅ {ekey} 지문 {len(new_records)}개를 원문 라이브러리에 저장했습니다!"
                    st.rerun()
            if s_col2.button("✖️ 검수 취소", use_container_width=True, key="mock_draft_cancel"):
                del st.session_state["mock_draft"]
                st.rerun()

    if not draft:
        st.caption("2️⃣ 검수 — 위에서 PDF를 올려 본문을 뽑으면 여기에서 확인·수정한 뒤 저장해요.")

    # ---- 3) 모의고사(연도/월/학년) 또는 교과서(과목/출판사/단원) → 번호 클릭 선택 ----
    saved_mock: List[Dict] = st.session_state.saved_mock
    kind_items = [p for p in saved_mock if (p.get("source_type") == "textbook") == is_textbook]
    st.markdown(f"##### 3️⃣ {kind_label} 선택 → 번호 클릭 → 교재 생성")
    if not kind_items:
        st.info(f"저장된 {kind_label} 원문이 없습니다. 위에서 {kind_label} PDF를 업로드해 저장해주세요.")

    exams: Dict[str, List[Dict]] = {}
    f_col1, f_col2, f_col3 = st.columns(3)
    if is_textbook:
        subjects = sorted({p.get("subject", "") for p in kind_items if p.get("subject")})
        sel_a = f_col1.selectbox("📘 과목", ["전체"] + subjects, key="tb_f_subject")
        publishers = sorted({p.get("publisher", "") for p in kind_items if p.get("publisher") and sel_a in ("전체", p.get("subject"))})
        sel_b = f_col2.selectbox("🏢 출판사", ["전체"] + publishers, key="tb_f_publisher")
        units = sorted({p.get("unit", "") for p in kind_items if p.get("unit")}, key=natural_sort_key)
        sel_c = f_col3.selectbox("📑 단원", ["전체"] + units, key="tb_f_unit")
        fields = ("subject", "publisher", "unit")
    else:
        years = sorted({p.get("year", "") for p in kind_items if p.get("year")}, reverse=True)
        sel_a = f_col1.selectbox("📅 연도", ["전체"] + years, key="mock_f_year")
        months = sorted({p.get("month", "") for p in kind_items if p.get("month") and sel_a in ("전체", p.get("year"))}, key=lambda x: int(x) if x.isdigit() else 99)
        sel_b = f_col2.selectbox("🗓️ 월", ["전체"] + months, key="mock_f_month")
        grades = sorted({p.get("grade", "") for p in kind_items if p.get("grade")})
        sel_c = f_col3.selectbox("🎓 학년", ["전체"] + grades, key="mock_f_grade")
        fields = ("year", "month", "grade")
    for p in kind_items:
        if any(sel != "전체" and p.get(fld) != sel for sel, fld in zip((sel_a, sel_b, sel_c), fields)):
            continue
        exams.setdefault(p.get("exam_key", extract_group_prefix(p["label"])), []).append(p)

    def exam_sort_key(k: str):
        first = exams[k][0]
        if is_textbook:
            return (first.get("subject", ""), first.get("publisher", ""), natural_sort_key(first.get("unit", "")))
        return (-int(first.get("year", 0) or 0), -int(first.get("month", 0) or 0), first.get("grade", ""))

    # 위젯 상태는 화면에 안 그려지면 사라지므로, 시험별 선택 번호를 별도로 보관 (필터를 바꿔도 선택 유지)
    pick_store: Dict[str, List[str]] = st.session_state.setdefault("mock_pick_store", {})

    with st.container(height=420, border=True):
        if not exams:
            st.caption(f"조건에 맞는 {kind_label}가 없습니다.")
        for ekey in sorted(exams.keys(), key=exam_sort_key):
            items = sorted(exams[ekey], key=lambda x: mock_number_sort_key(x.get("number", "")))
            num_to_item = {p["number"]: p for p in items}
            pill_key = f"mock_pick_{ekey}"
            if pill_key not in st.session_state:
                st.session_state[pill_key] = [n for n in pick_store.get(ekey, []) if n in num_to_item]
            h_col1, h_col2, h_col3 = st.columns([4, 1, 1])
            h_col1.markdown(f"**{md_safe(ekey)}** · 지문 {len(items)}개")
            if h_col2.button("전체 선택", key=f"mock_all_{ekey}", use_container_width=True):
                st.session_state[pill_key] = list(num_to_item.keys())
            if h_col3.button("선택 해제", key=f"mock_none_{ekey}", use_container_width=True):
                st.session_state[pill_key] = []
            picked = st.pills(
                ekey,
                options=list(num_to_item.keys()),
                selection_mode="multi",
                format_func=lambda n, m=num_to_item, tb=is_textbook: (n if tb else f"{n}번") + (" ⚠️" if m[n].get("needs_review") else ""),
                key=pill_key,
                label_visibility="collapsed"
            )
            pick_store[ekey] = list(picked or [])

    # 모든 시험·교과서의 선택 번호 모으기 (필터·종류를 바꿔도 선택 유지, 모의고사와 교과서를 섞어 선택 가능)
    by_label = {p["label"]: p for p in saved_mock}
    by_key_num = {(p.get("exam_key"), p.get("number")): p for p in saved_mock}
    selected: List[Dict] = []
    for ekey, nums in pick_store.items():
        for num in nums:
            item = by_key_num.get((ekey, num))
            if item:
                selected.append(item)
    selected.sort(key=natural_sort_key)

    if not selected:
        st.caption("👆 번호를 클릭해 교재로 만들 지문을 선택하세요. (⚠️ = 정답 확인 필요)")
        return

    st.markdown(f"🛒 **선택된 지문 {len(selected)}개:** " + ", ".join(md_safe(p["label"]) for p in selected[:20]) + (f" ... 외 {len(selected) - 20}개" if len(selected) > 20 else ""))

    def _drop_pill_state():
        # 번호 칩 상태를 지워 다음 화면에서 pick_store 기준으로 다시 그림 (그려진 칩의 상태는 콜백에서만 바꿀 수 있음)
        for k in [k for k in st.session_state.keys() if str(k).startswith("mock_pick_") and k != "mock_pick_store"]:
            del st.session_state[k]

    def _clear_all_picks():
        st.session_state["mock_pick_store"] = {}
        _drop_pill_state()

    def _mark_checked(ids):
        for p in st.session_state.saved_mock:
            if p["id"] in ids:
                p["needs_review"] = False
        save_to_file(MOCK_FILE, st.session_state.saved_mock)
        _drop_pill_state()   # ⚠️가 빠지면 칩 글자가 바뀌므로 선택이 풀리지 않게 다시 그림

    flagged = [p for p in selected if p.get("needs_review")]
    c_col1, c_col2, _ = st.columns([1, 1.6, 2.4])
    c_col1.button("🧹 선택 모두 해제", use_container_width=True, key="mock_clear_all", on_click=_clear_all_picks,
                  help="필터에 안 보이는 시험·교과서에서 고른 번호까지 모두 해제해요")
    if flagged:
        c_col2.button(f"✅ 확인 완료 — 선택 지문 ⚠️ {len(flagged)}개 지우기", use_container_width=True, key="mock_mark_checked",
                      on_click=_mark_checked, args=({p["id"] for p in flagged},),
                      help="⚠️는 AI가 정답표를 못 찾았거나 복원이 확실하지 않을 때 붙여요. 원문을 확인했다면 지워도 돼요")

    with st.expander("🔎 선택 지문 원문 확인·수정", expanded=False):
        pick_lbl = st.selectbox("지문 선택", [p["label"] for p in selected], key="mock_view_select")
        target = by_label[pick_lbl]
        st.caption(f"유형: {target.get('q_type', '')} · 정답: {target.get('answer', '')} · 메모: {target.get('note', '')}")
        new_body = st.text_area("원문", value=target.get("body_text", ""), height=260, key=f"mock_body_{target['id']}")
        new_review = st.checkbox("⚠️ 확인필요", value=bool(target.get("needs_review")), key=f"mock_rev_{target['id']}")
        if st.button("💾 이 지문 수정 저장", key=f"mock_save_one_{target['id']}"):
            target["body_text"] = new_body.strip()
            target["needs_review"] = new_review
            save_to_file(MOCK_FILE, st.session_state.saved_mock)
            st.success(f"✅ {pick_lbl} 원문을 저장했습니다.")

    render_mock_image_section(selected)

    a_col1, a_col2 = st.columns(2)
    a_col1.download_button(
        "📊 선택 지문 엑셀 다운로드 (엑셀 원클릭 생성 메뉴 형식)",
        data=build_mock_excel_bytes(selected),
        file_name="모의고사_원문.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        key="mock_excel_dl"
    )
    with a_col2.popover("🗑️ 선택 지문 삭제", use_container_width=True):
        st.warning(f"선택한 지문 {len(selected)}개를 원문 라이브러리에서 영구 삭제합니다. (이미 만든 교재는 지워지지 않습니다)")
        if st.button("삭제 확정", type="primary", key="mock_del_confirm"):
            del_ids = {p["id"] for p in selected}
            st.session_state.saved_mock = [p for p in saved_mock if p["id"] not in del_ids]
            for k in [k for k in st.session_state.keys() if str(k).startswith("mock_pick_")]:
                del st.session_state[k]
            st.session_state["mock_pick_store"] = {}
            save_to_file(MOCK_FILE, st.session_state.saved_mock)
            st.rerun()

    with st.container(border=True):
        st.markdown("##### 🎯 선택 지문으로 교재 생성")
        g_col1, g_col2, g_col3, g_col4 = st.columns(4)
        mdo_v = g_col1.checkbox("🏆 Anatomy One-Shot (한판승)", value=True, key="mock_do_v")
        mdo_a = g_col2.checkbox("🦅 Anatomy Self-Note", value=True, key="mock_do_a")
        mdo_c = g_col3.checkbox("💎 Anatomy Core", value=True, key="mock_do_c")
        mdo_w = g_col4.checkbox("📖 Anatomy Voca", value=True, key="mock_do_w")
        if any(p.get("needs_review") for p in selected):
            st.warning("⚠️ 선택한 지문 중 정답 확인이 필요한 지문이 있습니다. 위 '원문 확인·수정'에서 먼저 검토하는 것을 권장합니다.")
        m_types = [t for t, on in [("victory", mdo_v), ("anatomy", mdo_a), ("core", mdo_c), ("voca", mdo_w)] if on]
        if selected and m_types:
            st.caption(book_cost_line(text_model, book_new_counts([p["label"] for p in selected], m_types)))
        if st.button(f"🚀 선택한 {len(selected)}개 지문 교재 생성 시작", type="primary", use_container_width=True, key="mock_gen_btn"):
            if not (mdo_v or mdo_a or mdo_c or mdo_w):
                st.warning("생성할 교재 종류를 최소 1개 이상 체크해주세요.")
            else:
                rows = [{"label": p["label"], "text": p["body_text"], "school": "", "img_path": p.get("image_path", "")} for p in selected]
                types_list = [t for t, on in [("victory", mdo_v), ("anatomy", mdo_a), ("core", mdo_c), ("voca", mdo_w)] if on]
                n_added = enqueue_generation(rows, types_list, text_model, api_delay, academy_name, school_name)
                st.session_state.gen_flash = f"✅ {n_added}개 지문을 생성 작업에 추가했습니다. 창을 닫아도 서버에서 계속 진행됩니다."
                st.rerun()
        st.caption("👉 생성된 교재는 왼쪽 메뉴 '교재 출력'(한판승·Self-Note·Core·Voca)에서 같은 이름으로 확인·렌더링할 수 있습니다.")


# -----------------------------
# 6. 메인 UI 화면 및 사이드바 구성
# -----------------------------

# ---- 사용 설명서 (처음 쓰는 사람용 · 글만 보여 주므로 DB·AI 호출 없음) ----
APP_HELP = {
"overview": """
##### 🧭 이 시스템이 하는 일
지문(원문)을 모아 두고 → AI로 교재·변형문제를 만들어 **클라우드 DB(Supabase)에 쌓은 뒤** → 원하는 것만 골라 **HTML 단행본으로 출력**하는 앱이에요.
만든 결과는 모두 DB에 저장되므로 창을 닫거나 앱이 다시 시작돼도 사라지지 않아요.

##### 🔁 기본 흐름 (왼쪽 메뉴)
| 순서 | 어디서 | 무엇을 |
|---|---|---|
| ① 원문 모으기 | 📝 원문 라이브러리 | 모의고사·교과서 PDF를 올리면 AI가 원문을 복원 → 표에서 확인·수정 → 저장 |
| ② 교재 만들기 | 📝 원문 라이브러리 아래쪽 / 🚀 엑셀 원클릭 | 지문을 골라 One-Shot·Self-Note·Core·Voca 생성 (서버에서 백그라운드로 진행) |
| ③ 출력 | 🖨️ 교재 출력 (🏆 🦅 💎 📖) | 지문을 골라 '단행본 렌더링' → HTML 다운로드 → 열어서 **인쇄 / PDF로 저장** |
| ④ 변형문제 | 🧩 변형문제 은행·시험지 | 문제를 계속 쌓아 두고, 조건에 맞게 뽑아 시험지·책으로 출력 |
| ⑤ 학교 스타일 학습 | 🎯 출제 DNA · 📚 기출 어휘 | 운양고·제일고 기출 PDF로 출제 규칙과 어휘를 쌓음 → ②·④ 생성에 반영 |

##### ⚙️ 왼쪽 메뉴 아래 (모든 화면에 공통으로 적용)
- **🤖 AI 모델**: 모든 AI 작업(교재·변형문제·PDF 분석)에 이 모델을 써요. 모델마다 **값과 품질이 크게 달라요** → 바로 아래 '📊 모델 고르기 참고표'를 보고 고르세요.
- **💵 AI 사용 금액**: 이번 달 모델별 금액이에요. 실제로 쓴 토큰으로 계산한 금액이에요 (제공사 청구서와 조금 다를 수 있어요).

##### 🛠️ ⚙️ 설정 메뉴
- **로고·학원명·학교명**: 교재 머리말·꼬리말·표지에 들어가요. 바꾸면 자동 저장돼요.
- **지문 간 API 대기 시간**: 지문 하나를 끝내고 다음 지문까지 쉬는 시간이에요. 너무 줄이면 제공사가 요청을 막을 수 있어요.
- **저장소 상태**: 🟢 클라우드 DB 연결됨이어야 정상이에요. 🟡 로컬 모드면 저장이 이 PC에만 돼요.

##### 📦 위쪽 '작업' 칸 (생성 중일 때만 보임)
생성 버튼을 누르면 작업이 **서버 대기열**에 들어가 하나씩 처리돼요. 창을 닫아도 계속 진행돼요.
- **⏸️ 생성 중단**: 지금 하던 지문만 끝내고 멈춰요. **▶️ 다시 시작**: 남은 것부터 이어서 해요 (사이드바에서 고른 모델로).
- **🔁 실패한 N개 다시 시도**: 오류 난 지문만 다시 해요. **🧹 작업 기록 지우기**: 끝난 작업 목록만 지워요 (만든 교재는 그대로).
- **남은 작업을 사이드바 모델로**: 작업 중에 모델을 바꿨다면, 남은 지문부터 새 모델로 만들어요.

##### 💰 비용이 드는 버튼 / 안 드는 버튼
- **AI 비용이 드는 것**: 생성 시작·자동 생성·원문 복원·본문 추출·PDF 분석·DNA 추출·이미지 생성 버튼.
- **무료인 것**: 고르기·검색·정렬·렌더링·다운로드·인쇄·시험지 만들기(이미 쌓인 문제에서 뽑기만 함)·삭제.
- **되돌릴 수 없는 것**: 🗑️ 삭제 (한 번 더 '삭제 확정'을 눌러야 지워져요)와 '기본값으로 초기화'.
""",
"excel": """
##### 🚀 엑셀 원클릭 통합 생성
엑셀 한 장으로 여러 지문의 교재 4종을 한 번에 만들어요. **원문 라이브러리에 없는 지문**(학교 프린트·외부 지문 등)에 쓰면 편해요.
1. **엑셀 업로드**: 열 이름은 `label`(라벨)·`passage`(지문)·`school`(학교명, 선택)·`image_name`(이미지명, 선택).
2. **이미지 대량 업로드 (선택)**: 엑셀의 이미지명과 파일명이 같으면 그 지문에 붙어요.
3. **AI 이미지 생성 체크**: 이미지가 없는 지문만 AI가 지문 내용으로 그려요 (장당 약 10초, 비용 발생). 화풍·이미지 모델을 고를 수 있어요.
4. **만들 교재 체크** → **🚀 올인원 통합 생성 시작**: 작업 칸에 올라가 서버에서 진행돼요.
   - Core에서 쓴 패러프레이징 어휘는 Voca 단어장에 자동으로 들어가요.
5. 다 되면 왼쪽 메뉴 '교재 출력'(🏆 🦅 💎 📖)에서 골라 출력하세요.
""",
"mock": """
##### 📝 원문 라이브러리 — 모든 교재·변형문제의 '원본 창고'
**① 올리기 → ② 확인·저장 → ③ 골라서 교재 만들기** 순서예요. 맨 위에서 **모의고사 / 교과서**를 먼저 고르세요.

**모의고사**
- 시험지 PDF + 정답표를 올리고 **🧠 정답 반영 원문 복원 시작**: AI가 18~45번 지문을 정답이 반영된 원래 글로 되돌려요
  (빈칸 채우기, 순서 재배열, 삽입 문장 넣기, 무관한 문장 빼기, 어법·어휘 고치기).
- 결과 표에서 고칠 곳을 바로 고치고, **🔍 지문 하나씩 크게 보기**로 확인 → **💾 원문 라이브러리에 저장**. 마음에 안 들면 **✖️ 검수 취소**.

**교과서**
- 과목·출판사·단원을 적고(비우면 자동 인식) **📘 단락별 본문 추출 시작** → 단락마다 `4-1, 4-2…`처럼 나뉘어 저장돼요. 단원 하나씩 올리면 가장 정확해요.

**저장된 원문 쓰기 (아래쪽)**
- 년도·월·학년 필터로 좁힌 뒤 번호를 눌러 지문을 골라요 (**전체 선택 / 선택 해제**).
- 고른 번호는 필터를 바꿔도 유지돼요. 한 번에 다 풀려면 '선택된 지문' 아래 **🧹 선택 모두 해제**를 누르세요.
- **⚠️ 표시**: AI가 정답표를 못 찾았거나 복원이 확실하지 않은 지문이에요. 원문을 확인했다면 **✅ 확인 완료**로 지울 수 있어요 (지워도 기능에는 영향 없음).
- **🔎 선택 지문 원문 확인·수정**: 저장된 원문을 고치고 **💾 이 지문 수정 저장**. ⚠️ 표시는 정답 확인이 필요한 지문이에요.
- **🎨 이미지 생성 / 📤 내 이미지로 교체**: 한판승·Self-Note에 들어갈 지문 그림이에요.
- 만들 교재를 체크하고 **🚀 선택한 N개 지문 교재 생성 시작** → 작업 칸에서 진행돼요.
- **🗑️ 선택 지문 삭제**: 원문만 지워요. 이미 만든 교재는 남지만, 그 지문의 변형문제는 지문 목록에서 고를 수 없게 돼요.
""",
"final": """
**🏁 영T FINAL 모의고사 — 시험지 만들기는 AI 비용 없음**

학생들이 시험 직전에 **우리 학교와 똑같은 시험**을 회차별로 볼 수 있게 만드는 화면이에요.

1. **시험 정보**: 학교(운양고/제일고)·학년·시험 이름(예: 26년 2학기 기말)을 정해요. 같은 시험 이름으로 만들면 1회차, 2회차 … 5회차로 이어져요.
2. **시험 범위**: 시험에 나오는 지문을 골라요. 다음 회차를 만들 때는 지난 회차의 범위가 그대로 골라져 있어요.
3. **시험 구성**: 그 학교·학년 **실제 기출 한 회**를 고르면 유형·배점·문항 수(선택형·서술형)를 그대로 따라요. 기본은 가장 최근 시험이에요.
   - **목표 평균**: 난이도별 예상 정답률(하 90%·중 75%·상 55%·최상 40%)로 평균을 계산해서, 목표에 맞게 난이도를 섞어요.
   - **단어 수**: 그 학교·학년 실제 기출의 지문 단어 수 평균에 가깝게 골라요. 원문을 그대로 쓰는 주제·일치형 같은 문제는 실제 시험처럼 **한 지문에 여러 문제**([1~3] 다음 글을 읽고…)로 묶여요.
   - **이미 낸 문제는 안 나와요**: 같은 학교·학년 FINAL에 저장된 문제는 모두 빼고 뽑아요.
4. **📄 만들기** → 미리보기·학생용/교사용(정답·해설·난이도) HTML. 마음에 들면 **💾 확정 저장**해야 회차로 기록되고 다음 회차에서 빠져요.
   - 모자란 유형이 있으면 **🛠️ 부족한 유형 문제 만들기 작업 추가** 버튼으로 바로 만들 수 있어요 (이때만 AI 비용).
5. **회차 기록**: 시험을 본 뒤 표에 **실제 평균·응시 인원**을 적고 저장하세요. 실제 평균과 예상 평균의 차이를 최근 회차일수록 크게 쳐서 다음 회차 예상에 보정해요. 기록이 쌓일수록 목표 평균에 더 정확히 맞춰져요.

시험지 모양: 운양고는 '○학년 ○학기 지필평가 문제지' 머리말·과목명·실시 학급 칸, 제일고는 '학년 | 과목 | 시행일시' 표와 네모 칸 지문으로 실제 시험지와 같은 배치예요. 학교 로고·학교 이름 대신 PQRS 영T가 들어가요.
""",
"qbank": """
##### 🧩 변형문제 은행 — '쌓기'와 '뽑기'가 나뉘어 있어요
문제를 만들 때마다 **문제 은행(DB)에 계속 쌓이고**, 시험지는 쌓인 문제에서 **뽑기만** 해요. 그래서 시험지를 몇 번 만들어도 AI 비용이 들지 않아요.

**🛠️ 문제 만들기 (누적) — 비용 발생**
- 위 칩에서 **시험·번호**를 골라 지문을 정해요.
- **유형 × 난이도**마다 **난이도별 개수**만큼 만들어요. 난이도마다 개수를 따로 정해 비중을 맞춰요 (예: 최상 3 · 상 2 · 중 1 → 지문·유형마다 6문제, 비중 50% · 33% · 17%, 0이면 그 난이도는 안 만듦). 지문 3개 × 유형 2개 × (최상 3 + 상 2 + 중 1)이면 36문제예요. 버튼 위에 예상 개수·비용·시간이 나와요.
- **학교 스타일**: 운양고 / 제일고 중 어느 학교 출제 방식으로 만들지 정해요 (🎯 출제DNA가 반영돼요).
- **정답 검증**: AI가 만든 문제를 한 번 더 풀어 보고, 정답이 다르면 버려요 (비용 약 10~20% 추가, 품질↑).
  상·최상은 같은 검증에서 **난이도도 판정**해요. 학생이 원문을 외우고 온다고 보고, 원문 기억만으로 풀리거나 지문을 안 읽고 지울 수 있는 오답이 있으면 **상·중으로 낮춰 저장**해요(최상은 상위권도 실제로 고민할 구조일 때만). 그래서 '최상'으로 요청해도 상·중으로 쌓일 수 있어요.
- **순서·삽입·빈칸 상·최상**(두 학교 모두): 외운 학생도 다시 읽어야 풀리게, 순서는 덩어리를 다시 쓰고 순서를 바꾸며, 삽입은 주어진 문장과 주변 문장을 바꿔 쓰고, 빈칸은 원문 표현을 그대로 쓴 선지를 오답으로 둬요.
- **하·중은 가벼운 모델로**: 쉬운 문제는 싼 모델로, 상·최상은 지금 모델로 만들어 비용을 줄여요.
- **💸 반값 모드**: 켜면 생성 비용이 절반이에요. 대신 결과가 바로 안 나오고 보통 몇 분~몇 시간, 길면 24시간 뒤에 들어와요. 급할 때는 끄고, 시간이 있을 때(예: 밤에 넣어 두고 다음 날) 켜세요. 창을 닫아도 서버가 2분마다 확인해서 들어온 것부터 저장해요.
- 같은 지문을 다시 만들면 이미 쌓인 문제와 **다른 포인트**로 출제해요.
- **🔗 지문 연결**(지문을 2개 이상 고르면 나와요): **이어 붙이기**는 교과서의 짧은 단락 2~4개를 실제 시험처럼 한 지문으로 묶어 출제해요 (예: 4-1 + 4-2). **(A)(B) 두 지문 함께**는 같은 범위의 비슷한 내용 두 지문을 (A)(B)로 싣고 '(A)와 (B)를 읽고 추론할 수 없는 것'처럼 두 글을 함께 묻는 문제를 만들어요 (추론·내용 일치·<보기>·주제·요지·제목 유형만). 묶은 지문 문제는 '4-1 + 4-2'처럼 따로 쌓이고, 시험지를 만들 때 묶인 지문을 모두 고르면 함께 뽑혀요.
- **🎣 영쌤 출제 포인트**: 왼쪽 메뉴 출제 DNA 화면 맨 아래 목록에 선생님의 '낚시' 출제 방식을 한 줄씩 적어 두면, 학교(운양고·제일고)와 관계없이 맞는 유형 문항에 하나씩 넣어요 (평소 문항 절반쯤, '🎣 영쌤 출제 포인트 위주로'를 켜면 모든 문항 · 난이도 '하'는 빼요). 새 포인트는 맨 아래 빈 줄에 계속 추가하면 돼요. **✨ 정리해서 저장하기**를 누르면 새로 쓰거나 고친 포인트를 AI가 출제 지시문으로 정리해 보여 주고(한 칸에 섞어 쓴 포인트는 나눠 줌), 고칠 곳을 고친 뒤 **✅ 확정해서 저장**해야 올라가요 (몇 개에 약 1~5원). '교과서만'을 켠 포인트는 교과서 지문에만 쓰고 같은 단원 다른 단락 원문을 함께 넘겨요.
- **📚 짧은 교과서 단락 자동 묶기**: 교과서 단락을 고르면 120단어보다 짧은 단락은 같은 단원의 바로 앞·뒤 단락(함께 고른 것)과 2~3개를 묶어 한 지문으로 출제해요 (예: 4-1 + 4-2). 짧은 단락은 묶은 지문으로만, 함께 묶인 긴 단락은 따로도 만들어요. 모의고사 (A)(B) 묶기는 지금처럼 직접 골라요. 영쌤 포인트로 만든 문항은 학생이 알아채지 않게 문제지에는 표시하지 않고 정답표('변(운)🎣')·해설에만 표시되며, 시험지·클리닉에서 '영쌤 포인트 문제만/빼기'로 고를 수 있어요.
- **📅 주간 클리닉**: 그 주에 배운 지문을 주차로 저장해 두면, 그 지문으로 만든 변형문제만 모아 클리닉 미니모의고사(기본 20문항 · 중4/상8/최상8 · 한 학교 스타일 · 실제 기출 제외)를 만들어요. 모자라면 '모자란 문제 만들기'가 지문·유형을 골고루 나눠 필요한 만큼만 생성하고, '낸 것으로 표시'하면 다음 클리닉에서 그 문항은 빠져요.
- **순서 유형 선택지**는 수능처럼 ① (A)-(C)-(B) ② (B)-(A)-(C) ③ (B)-(C)-(A) ④ (C)-(A)-(B) ⑤ (C)-(B)-(A)로 고정돼요 ((A)로 시작하는 선지는 ① 하나). 예전에 만든 순서 문제도 시험지에서는 이렇게 바뀌어 나와요.
- **어법 밑줄**: 섞기(학교 기출 비율로 단어·구 밑줄과 문장 전체 밑줄을 무작위로) / 단어·구 / 문장 전체.
- **문항마다 출제 장치를 무작위로 정해요** (AI 비용 추가 없음): 어법은 실제 기출 개념 약 20가지 중 2~3개를 '꼭 묻기'로 뽑아요. 학교마다 지금까지 만든 문제를 세서 덜 나온 개념부터 채우기 때문에, 분사처럼 많이 나온 개념만 반복되지 않고 실제 출제된 개념(5형식·관계부사·대명사 지칭·복합관계사 등)이 모두 고르게 나와요. 시험지를 뽑을 때도 어법 문제는 이미 나온 개념과 덜 겹치는 것을 골라요.
  영어 선택지·<보기>·요약문은 두 학교 모두 늘 재진술하고, 주제를 뒤집은 오답(제일고 1~2개, 운양고 가끔)을 넣어요.
  정답 번호도 프로그램이 ①~⑤에 고르게 정해요. 상·최상은 실제 내신처럼 지문 단어를 그대로 쓴 '낚는 선지'를 ①·②에, 완전히 다른 말로 바꾼 정답을 ③·④(가끔 ⑤)에 둬요.
  제일고 상·최상은 순서 문제의 덩어리를 새로 써서 원문과 다른 순서로 만들거나, 지문 전체를 다시 쓰기도 해요.
  **헷갈림 장치**(중 25%·상 60%·최상 90%): 100점 학생도 '어렵다'고 표시한 실제 기출 방식을 유형마다 하나씩 넣어요 — 주제·제목은 글 일부의 구체어를 묶은 오답, 요약문은 한 칸만으로 안 갈리는 쌍,
  내용 일치는 맞는 선지를 낯설게 바꿔 쓰고 틀린 선지는 지문 단어 그대로 한 단어만 뒤집기, 삽입은 단어만 겹치는 가짜 자리, 어법은 같은 함정 두 번 + 긴 주어. 해설 끝에 '헷갈리는 포인트'가 한 줄 붙어요.

**📄 시험지 만들기 — 무료**
- 지문을 고르고 **학교·실제 기출 포함 여부·난이도·유형·순서·문제 수**를 정해요. 칩 아래에 '지금 뽑을 수 있는 문제 수'가 나와요.
- **학교 비율 → 학교별 문제 수 정하기**: 예) 운양고 스타일 20 + 제일고 스타일 10. 한쪽이 모자라면 다른 학교로 채우지 않고 알려 줘요.
- **난이도 비율 → 비율 정하기**: 최상·상·중·하(고른 지문에 실제 기출이 있으면 실제 기출도) 비율(%)을 넣으면 그 비율대로 뽑아요. 예) 최상 30 · 상 40 · 중 30, 30문제 → 최상 9 · 상 12 · 중 9. 합이 100이 아니어도 비율로 나눠요. 쌓인 문제가 모자란 난이도는 빨간 글씨로 알려 주고, 모자란 만큼은 비율을 준 다른 난이도에서 채워요. 학교별 문제 수와 함께 쓰면 학교마다 이 비율로 뽑아요.
- **이미 낸 문항**: 전에 '문항 기록 저장'한 시험지를 고르면 **그 문항을 빼고** 뽑아요 (추가로 풀 학생에게 새 문제만 줄 때).
- 뽑는 원리: 고른 지문에서 **같은 개수씩 골고루**, 지문 안에서는 **유형을 돌아가며** 뽑아요. 모자란 지문 몫은 문제가 많은 지문이 채워요.
- **실제 기출 묶음**: 실제 시험에서 한 지문에 여러 문제가 나온 경우(예: 5·6번), 하나가 뽑히면 같은 지문 문제도 함께 넣고 '[5~6] 다음 글을 읽고 물음에 답하시오.' 아래에 지문을 한 번만 실어요. 유형을 골라 뽑으면 고른 유형만 따라오고, 늘어난 만큼 변형문제를 빼서 문항 수는 그대로예요.
- **📄 시험지 만들기**: 미리보기·HTML 다운로드. **🔀 다른 조합으로 다시 뽑기**: 같은 조건에서 다른 문제로 다시 뽑아요.
- **💾 문항 기록 저장**: 이름(예: 김민수 1차)을 붙여 이 시험지 문항을 기록해요. 다음 시험지에서 빼는 데 써요. 지우기는 '기록 관리'에서 해요 (문제는 그대로).
- **표시 옵션**: 출처·구분, 같은 지문 기출 이력, 정답·해설, 학교 스타일 표시, 문항별 권장 풀이 시간(단어 수·유형으로 계산), **ANATOMY VISION 책으로**(앞뒤 표지·사용법·제작 소개·저작권 쪽 추가).

**🗂️ 문제 보기·삭제**: 지문 하나를 골라 쌓인 문제를 표로 보고, 체크해서 지울 수 있어요. ⚠️ 정답 확인 = 검증에서 답이 엇갈린 문제예요.

**🏫 기출 표시 — 비용 발생 (시험지 1개 약 200원)**: 운양고·제일고 실제 시험지 PDF를 올리면 문항마다 **어느 모의고사 몇 번 지문인지** 찾아요.
그 결과로 시험지에 '같은 지문 기출 이력'이 붙고, 원문 라이브러리 지문으로 낸 문항은 **실제 기출문제**로 문제 은행에 들어가요. 모든 문항의 발문·지문·선지·정답은 기출 분석용으로 따로 보관해요. 이미 올린 PDF는 다시 분석하지 않아요.
AI가 짝지은 원문과 내용 단어가 20%도 안 겹치면 잘못 연결된 것으로 보고 연결하지 않아요.
""",
"library": """
##### 📚 교재 출력 (🏆 One-Shot · 🦅 Self-Note · 💎 Core · 📖 Voca) — 만든 교재를 골라 출력하는 곳
- **🔄 누락된 N개 지문 자동 생성** (Core·Voca만, 비용 발생): 다른 교재는 있는데 이 교재만 없는 지문을 찾아 만들어요. 엑셀을 다시 올릴 필요가 없어요.
- **🔗 CORE 어휘를 VOCA에 병합** (Voca만, 무료): Core에 쓰인 패러프레이징 어휘를 단어장에 넣어요.
- **고르기**: 그룹 필터·검색 → **✅ 검색 결과 전체 선택**, 또는 번호를 적고 **⚡ 입력한 번호만 선택 추가**. 카드를 눌러도 선택/해제돼요.
  - **누른 순서대로 출력**돼요. 번호순으로 하려면 **🔢 이름/번호순 정렬**을 누르세요.
- **⬜ 전체 선택 해제** / **🗑️ 선택 항목 삭제**: 삭제는 이 교재 데이터만 지워요 (원문은 남아요).
- **👁️ 단행본 렌더링** → **HTML 다운로드** → 받은 파일을 크롬으로 열어 **인쇄 / PDF로 저장**. Voca는 엑셀·CSV로도 받을 수 있어요.
""",
"dna": """
##### 🎯 출제 DNA — 학교 출제 스타일을 AI에게 가르치는 곳
여기 저장된 **출제 DNA 규칙**은 교재·변형문제를 만들 때 AI에게 함께 전달돼요. 그래서 이 규칙이 좋을수록 '그 학교다운' 문제가 나와요.
- 맨 위 **📊 현황판**: 학교별로 참고하는 기출 시험지 수, 실제 시험에 많이 나온 유형·어법 개념, 지문을 얼마나 바꾸는지, 그래서 문제를 어떻게 만드는지를 보여 줘요. 시험지를 더 올리면 바로 바뀌어요 (보기만 하는 화면이라 비용 없음).
  - 시험지 수·유형 통계는 **🧩 변형문제 은행·시험지 → 🏫 기출 표시**에 올린 시험지에서, 학습 기록은 이 화면의 PDF 업로드에서 쌓여요.
- **기출 PDF 업로드 → 🧠 출제DNA 추출 및 누적 적용** (비용 발생): AI가 주제·요약 패러프레이징 방식, 빈칸 어휘 수준, 어휘 함정 패턴을 분석해 규칙에 **덧붙여요**.
- **규칙 글을 직접 고치고 💾 수동 저장**할 수도 있어요. 서술형 규칙은 따로 **💾 서술형 출제 DNA 저장**.
- **🔄 운양고·제일고 기본값으로 초기화**: 지금까지 쌓은 규칙이 처음 상태로 돌아가요. **되돌릴 수 없어요.**
""",
"pastvoca": """
##### 📚 기출 어휘 독립 라이브러리 — 학교 시험지별 단어장
- 기출 시험지 PDF를 여러 개 올리고 **🚀 어휘 일괄 추출 및 저장** (비용 발생): 시험지마다 단어를 뽑아 **출처(몇 년도·학년·몇 번, 본문/선지)**와 함께 저장해요.
  - 학교명을 비우면 파일명·시험지에서 자동으로 알아내요. 틀리게 잡히면 '학교명 강제 지정'에 적으세요.
- 아래 목록에서 **운양고 / 제일고 / 전체 일괄 선택** 또는 카드로 고른 뒤 렌더링해서 출력해요.
- 한 시험지 단어는 이어서 나오고, **다음 시험지부터 새 쪽**에서 시작해요.
""",
}

# -----------------------------
# 출제DNA 탭 현황판: 지금 참고하는 기출 시험지와 실제 출제 방식을 한눈에 (읽기 전용 · AI 호출 없음 · 프롬프트·캐시와 무관)
#   숫자는 저장된 기출 시험지(gichul_map)·출제 DNA 기록과 출제 코드의 상수(QB_*)에서 그때그때 읽으므로 시험지를 더 올리면 바로 바뀜
# -----------------------------
DNA_BOARD_CSS = """<style>
.dnab-card { border: 1px solid #cbd5e1; border-radius: 10px; padding: 14px 16px; background: #ffffff; color: #0f172a; margin-bottom: 10px; }
.dnab-card h4 { margin: 0 0 8px; font-size: 16px; }
.dnab-card h4.u { color: #0047AB; } .dnab-card h4.j { color: #b91c1c; }
.dnab-sub { font-size: 12.5px; color: #475569; margin-bottom: 10px; line-height: 1.55; }
.dnab-sec { font-size: 13px; font-weight: 800; margin: 12px 0 6px; color: #0f172a; }
.dnab-bar { display: flex; align-items: center; gap: 8px; font-size: 12.5px; margin: 3px 0; }
.dnab-bar .lb { width: 128px; flex-shrink: 0; color: #334155; }
.dnab-bar .tr { flex: 1; background: #f1f5f9; border-radius: 4px; height: 12px; overflow: hidden; }
.dnab-bar .fl { height: 100%; border-radius: 4px; }
.dnab-bar .vl { width: 64px; text-align: right; color: #475569; flex-shrink: 0; }
.dnab-li { font-size: 12.8px; line-height: 1.6; margin: 0; padding-left: 18px; color: #1e293b; }
.dnab-li li { margin-bottom: 4px; }
.dnab-tbl { width: 100%; border-collapse: collapse; font-size: 12.5px; color: #0f172a; background: #ffffff; }
.dnab-tbl th, .dnab-tbl td { border: 1px solid #cbd5e1; padding: 6px 8px; text-align: left; vertical-align: top; }
.dnab-tbl th { background: #eef2ff; white-space: nowrap; }
</style>"""


@st.cache_data(ttl=600, show_spinner=False)
def dna_board_exam_stats() -> Dict:
    """저장된 실제 기출 시험지(변형문제 탭 '🏫 기출 표시'에 올린 것)를 학교별로 셈. 시험지를 더 올리면 그 자리에서 캐시를 비움"""
    try:
        exams = list((load_gichul_map().get("exams") or {}).values())
    except Exception:
        exams = []
    out = {}
    for sc in QB_SCHOOLS:
        ex = [e for e in exams if e.get("school") == sc]
        qs = [q for e in ex for q in e.get("questions", []) or []]
        pts = [float(m.group()) for q in qs for m in [re.search(r"\d+(?:\.\d+)?", str(q.get("points", "")))] if m]
        yrs = sorted({int(m.group(1)) for e in ex for m in [re.search(r"(\d{2})\s*년", str(e.get("exam_name", "")))] if m})
        types = {}
        for q in qs:
            t = q.get("q_type") or "기타"
            types[t] = types.get(t, 0) + 1
        grades = {}
        for e in ex:
            grades[e.get("grade") or "?"] = grades.get(e.get("grade") or "?", 0) + 1
        out[sc] = {"exams": len(ex), "questions": len(qs), "avg_pts": round(sum(pts) / len(pts), 2) if pts else 0,
                   "years": (2000 + yrs[0], 2000 + yrs[-1]) if yrs else None, "types": sorted(types.items(), key=lambda x: -x[1]),
                   "grades": dict(sorted(grades.items())),
                   "list": sorted([(e.get("grade", ""), e.get("exam_name", ""), len(e.get("questions", []) or [])) for e in ex],
                                  key=lambda x: (x[1], x[0]), reverse=True)}
    return out


def _dnab_bars(items, total, color, n=8):
    rows = []
    top = max([v for _, v in items[:n]] or [1])
    for name, v in items[:n]:
        pct = v / total * 100 if total else 0
        rows.append(f"<div class='dnab-bar'><span class='lb'>{esc(name)}</span><span class='tr'><span class='fl' style='display:block;width:{v / top * 100:.0f}%;background:{color}'></span></span>"
                    f"<span class='vl'>{pct:.0f}% ({v})</span></div>")
    return "".join(rows)


def render_dna_board():
    stats = dna_board_exam_stats()
    dna = st.session_state.get("exam_dna") or {}
    hist = dna.get("history", []) or []
    n_pdf = sum(str(h.get("title", "")).lower().count(".pdf") for h in hist)
    n_base = 8 if any("기본 탑재" in str(h.get("title", "")) for h in hist) else 0
    st.markdown(DNA_BOARD_CSS, unsafe_allow_html=True)
    st.markdown("#### 📊 지금 참고하는 기출 시험지와 실제 출제 방식")
    st.caption("저장된 기출 시험지와 출제 규칙을 그대로 읽어 보여 줍니다. 시험지를 더 올리면 이 숫자가 바로 바뀝니다. (보기만 하는 화면이라 AI 비용·토큰과 상관없어요)")

    m = st.columns(4)
    for col, sc in zip(m[:2], QB_SCHOOLS):
        s_ = stats.get(sc, {})
        yr = f"{s_['years'][0]}~{s_['years'][1]}년" if s_.get("years") else "-"
        col.metric(f"🏫 {sc} 기출 시험지", f"{s_.get('exams', 0)}회분")
        col.caption(f"{s_.get('questions', 0)}문항 · {yr}")
    m[2].metric("🧠 출제 규칙 학습에 쓴 PDF", f"{n_base + n_pdf}개")
    m[2].caption(f"기본 탑재 {n_base} + 업로드 {n_pdf}")
    m[3].metric("🕒 출제 규칙 최근 업데이트", str(dna.get("updated_at", "-"))[:16])
    m[3].caption(f"학습 기록 {len(hist)}건")

    cols = st.columns(2)
    for col, sc, cls, color in zip(cols, QB_SCHOOLS, ["u", "j"], ["#0047AB", "#b91c1c"]):
        s_ = stats.get(sc, {})
        grades = " · ".join(f"{g} {n}회" for g, n in (s_.get("grades") or {}).items())
        gfreq = sorted(QB_GRAMMAR_FREQ.get(sc, {}).items(), key=lambda x: -x[1])
        m_tot = re.search(r"(\d+)개 선택지", QB_GRAMMAR_DNA.get(sc, ""))  # 전수 정리한 선택지 수 (개념 목록 합보다 큼)
        g_total = int(m_tot.group(1)) if m_tot else sum(v for _, v in gfreq)
        rate = QB_SENTENCE_RATE.get(sc, {})
        if sc == "운양고":
            how = [
                "지문 본문은 원문 표현을 약 90% 이상 그대로 두고, 난이도는 선택지와 문항 설계로 냅니다. 선택지·<보기>는 전부 긴 영어 재진술이고 5개 모두 지문 소재를 다룹니다.",
                "단, 상·최상 순서·삽입은 학생이 원문을 외우고 오므로 덩어리·주어진 문장을 다시 쓰고 배치를 바꿉니다(아래 '암기 무력화').",
                "주제·제목·요지·빈칸 같은 선택지 문제는 절반 정도에 '지문 핵심어를 그대로 쓰면서 결론을 뒤집은 오답' 1개를 넣습니다(하 난이도 제외).",
                f"어법 최상(조합형)은 문장 전체 밑줄 ⓐ~ⓔ(ⓕ) {QB_HARD_SENT_RATE[('운양고', '최상', '어법 (조합형)')]:.0%} + '옳은 것만을 있는 대로' {QB_RIGHT_ONLY_RATE[('운양고', '최상')]:.0%}, "
                f"상은 문장 밑줄 {QB_HARD_SENT_RATE[('운양고', '상', '어법 (조합형)')]:.0%}·옳은 것만 {QB_RIGHT_ONLY_RATE[('운양고', '상')]:.0%}. 원문 그대로인데 틀려 보이는 문장을 '옳은 것'에 섞습니다. 하·중 문장 밑줄 — 단일 {rate.get('어법 (단일)', 0):.0%} · 조합형 {rate.get('어법 (조합형)', 0):.0%}.",
                "빈칸은 원문을 그대로 두고 비울 어구만 정해, 프로그램이 원문에 빈칸을 뚫습니다. 추론형(추론 가능·불가, 필자 관점으로 상황 설명)도 냅니다.",
            ]
        else:
            how = [
                "지문 본문도 원문 표현을 약 60~85%만 남기고 다시 씁니다. 상·최상은 문항의 약 25%를 지문 전체 재진술로 만듭니다.",
                "상·최상 순서·삽입은 덩어리·주어진 문장을 다시 쓰고 배치를 바꿉니다(아래 '암기 무력화').",
                "선택지 문제는 '결론을 뒤집은 오답'을 상·최상에서 1~2개 넣고, 선택지는 길고 학술적인 상위 개념어로 씁니다.",
                f"어법은 ①~⑤ 단일형과 문장 단위 밑줄이 많습니다. 문장 단위 비율 — 단일 {rate.get('어법 (단일)', 0):.0%} · 조합형 {rate.get('어법 (조합형)', 0):.0%}. "
                f"최상 조합형은 문장 전체 밑줄 + '옳은 것만' {QB_RIGHT_ONLY_RATE[('제일고', '최상')]:.0%}.",
            ]
        measure = "".join(f"<tr><td>{t}</td><td>{(u if sc == '운양고' else j)}</td></tr>" for t, u, j in QB_PASSAGE_MEASURE)
        col.markdown(f"""<div class='dnab-card'>
<h4 class='{cls}'>{sc}</h4>
<div class='dnab-sub'>{esc(grades) or '아직 올린 시험지가 없어요'} · 평균 배점 {s_.get('avg_pts', 0)}점<br>{esc(QB_SCHOOL_STYLE.get(sc, ''))}</div>
<div class='dnab-sec'>실제 시험에 많이 나온 유형 (전체 {s_.get('questions', 0)}문항 중)</div>
{_dnab_bars([t for t in s_.get('types', []) if t[0] != '기타'], s_.get('questions', 0), color)}
<div class='dnab-sec'>어법 선택지에서 많이 물은 개념 (선택지 {g_total}개 중)</div>
{_dnab_bars(gfreq, g_total, color, 6)}
<div class='dnab-sec'>지문을 원문에서 얼마나 바꾸나 (원문 표현이 남은 비율)</div>
<table class='dnab-tbl'><tr><th>유형</th><th>원문 표현 유지</th></tr>{measure}</table>
<div class='dnab-sec'>그래서 {sc} 스타일 문제는 이렇게 만듭니다</div>
<ul class='dnab-li'>{''.join(f'<li>{esc(x)}</li>' for x in how)}</ul>
</div>""", unsafe_allow_html=True)

    lv_rows = "".join(
        f"<tr><td><b>{lv}</b></td><td>{QB_TRAP_RATE.get(lv, 0):.0%}</td><td>{esc(QB_LEVEL_RULES.get(lv, ''))}</td></tr>" for lv in QB_LEVELS)
    pos_txt = " · ".join(f"{CIRCLED[p]} {w:.0%}" for p, w in zip(QB_TRAP_ANS_POS, QB_TRAP_ANS_W))
    st.markdown(f"""<div class='dnab-card'>
<h4>두 학교 공통 — 난이도별로 이렇게 만듭니다</h4>
<table class='dnab-tbl'><tr><th>난이도</th><th>함정형 문항 비율<br><span style='font-weight:400'>(선택지 유형)</span></th><th>난이도 기준</th></tr>{lv_rows}</table>
<ul class='dnab-li' style='margin-top:10px'>
<li><b>함정형 문항</b>: 지문 단어를 그대로 많이 쓴 '낚는 선지'를 앞 번호(①·②)에 두고, 정답은 지문 단어를 거의 쓰지 않은 다른 말로 뒤에 둡니다. 정답 번호 — {pos_txt}.</li>
<li><b>헷갈림 장치</b>(중 {QB_FEEL_RATE['중']:.0%} · 상 {QB_FEEL_RATE['상']:.0%} · 최상 {QB_FEEL_RATE['최상']:.0%}): 100점 학생도 '어렵다'고 표시한 실제 기출 방식을 유형마다 하나 넣고(구체어 오답·한 칸으로 안 갈리는 요약문·맞는 선지 낯설게·가짜 삽입 자리·같은 어법 함정 두 번), 해설 끝에 '헷갈리는 포인트'를 붙입니다.</li>
<li><b>나머지 문항</b>은 정답을 ①·②·⑤ 쪽으로 조금 더 보내서, 시험지 전체로 보면 정답 번호가 고르게 나오게 합니다.</li>
<li><b>선택지·&lt;보기&gt;·요약문 문장</b>은 난이도와 상관없이 원문 문장을 그대로 옮기지 않고 모두 다른 말로 씁니다(실제 기출에서 원문 표현이 0~6%만 남음).</li>
<li>문제를 만든 뒤 <b>정답을 한 번 더 따로 풀어 보고</b>, 맞지 않으면 버립니다.</li>
</ul></div>""", unsafe_allow_html=True)
    recipes = [("순서", QB_ORDER_RECIPE), ("삽입", QB_INSERT_RECIPE), ("빈칸", QB_BLANK_RECIPE), ("어법 (문장 밑줄·옳은 것만)", QB_SENT_GRAMMAR_RECIPE)]
    judge_lines = [l[2:] for l in QB_LEVEL_JUDGE.splitlines() if l.startswith("- ")]
    st.markdown(f"""<div class='dnab-card'>
<h4>난이도 표시 원칙 — 학생은 시험 범위 지문을 외우고 온다</h4>
<div class='dnab-sub'>'최상'은 상위권 학생도 실제로 고민하거나 실수할 구조일 때만 붙입니다. 상·최상을 만들면 정답 검증 때 아래 기준으로 다시 판정해서, 못 미치면 <b>상·중으로 낮춰 저장</b>합니다(요청 난이도와 이유는 문항에 남김).</div>
<ul class='dnab-li'>{''.join(f'<li>{esc(x)}</li>' for x in judge_lines)}</ul>
<div class='dnab-sec'>상·최상 '암기 무력화' 레시피 (두 학교 공통)</div>
<table class='dnab-tbl'><tr><th>유형</th><th>이렇게 만듭니다</th></tr>{''.join(f'<tr><td>{esc(t)}</td><td>{esc(r)}</td></tr>' for t, r in recipes)}</table>
</div>""", unsafe_allow_html=True)

    with st.expander(f"📚 누적된 기출 시험지 목록 (운양고 {stats.get('운양고', {}).get('exams', 0)} · 제일고 {stats.get('제일고', {}).get('exams', 0)}회분) · 출제 규칙 학습 기록 {len(hist)}건"):
        lc = st.columns(2)
        for col, sc in zip(lc, QB_SCHOOLS):
            rows = "".join(f"<tr><td>{esc(g)}</td><td>{esc(nm)}</td><td>{n}문항</td></tr>" for g, nm, n in stats.get(sc, {}).get("list", []))
            col.markdown(f"<div class='dnab-sec'>{sc}</div><table class='dnab-tbl'><tr><th>학년</th><th>시험</th><th>문항</th></tr>{rows}</table>", unsafe_allow_html=True)
        st.markdown("<div class='dnab-sec'>출제 규칙 학습 기록 (최근 것 먼저)</div>", unsafe_allow_html=True)
        for h_idx, h_item in enumerate(reversed(hist), start=1):
            with st.container(border=True):
                st.markdown(f"**{h_idx}. {h_item.get('title', '기출 분석')}**")
                st.caption(h_item.get("summary", ""))
        st.caption("시험지 목록·유형 통계는 '🧩 변형문제 은행·시험지' 메뉴 → '🏫 기출 표시' 탭에 올린 시험지에서, 출제 규칙 학습 기록은 아래 '기출 PDF 업로드'에서 쌓입니다.")


# -----------------------------
# 6-B. 화면 구성: 왼쪽 메뉴(그룹별) → 고른 화면 하나만 그림
#   - 예전에는 탭 9개를 매번 전부 그려서 버튼 하나만 눌러도 모든 교재 목록·미리보기를 다시 만들었음
#   - 지금은 고른 메뉴만 그리므로 같은 동작이 더 빠르고, 주소(…/qbank)로 바로 들어갈 수 있음
# -----------------------------
# 다른 메뉴로 갔다 와도 필터·칩·입력값이 그대로 남도록 위젯 값을 붙잡아 둠 (Streamlit 공식 방법)
#   버튼·파일 업로드·표 편집처럼 값을 직접 넣을 수 없는 위젯은 건너뜀
_KEEP_VALUE_TYPES = {"bool_value", "string_value", "int_value", "double_value",
                     "double_array_value", "int_array_value", "string_array_value"}

def keep_widget_state():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        inner = get_script_run_ctx().session_state._state
        mapper, meta = inner._key_id_mapper, inner._new_widget_state.widget_metadata
    except Exception:
        return  # Streamlit 내부가 바뀌면 그냥 건너뜀 (값이 초기화될 뿐 기능은 그대로)
    for k in list(st.session_state.keys()):
        wid = mapper.get_id_from_key(k, None)
        m = meta.get(wid) if wid else None
        if m is None or m.value_type not in _KEEP_VALUE_TYPES:
            continue
        try:
            v = st.session_state[k]
            if v is None or isinstance(v, dict):
                continue
            st.session_state[k] = v
        except Exception:
            pass

keep_widget_state()

# 사이드바 밖(⚙️ 설정 화면)에서 고치는 값은 저장된 설정에서 바로 읽음
app_settings = st.session_state.app_settings
academy_name = app_settings.get("academy_name", "포피엠 입시 영어")
school_name = app_settings.get("school_name", "영어는 영쌤!")
api_delay = int(app_settings.get("api_delay", 3))
logo_urls = {s: app_settings.get(f"logo_{s}") or DEFAULT_LOGO_URL for s in "hfe"}
h_url, f_url, e_url = logo_urls["h"], logo_urls["f"], logo_urls["e"]
CSS_VICTORY = build_css_victory(h_url, f_url, e_url)
CSS_ANATOMY = build_css_anatomy(h_url, f_url, e_url)
CSS_CORE = build_css_core(h_url, f_url, e_url)
CSS_VOCA = build_css_voca(h_url, f_url)

# 화면 여백만 살짝 다듬음 (외부 글꼴·이미지 없이 몇 줄짜리 CSS라 속도 영향 없음)
st.markdown("""<style>
.block-container { padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1480px; }
.pg-crumb { font-size: 0.8rem; color: #64748b; margin-bottom: -0.2rem; }
.pg-desc { font-size: 0.9rem; color: #475569; margin-top: -0.4rem; line-height: 1.55; }
.home-tile-desc { font-size: 0.82rem; color: #64748b; margin: -0.3rem 0 0.2rem 0.2rem; line-height: 1.45; }
.home-count { font-size: 0.78rem; color: #0047AB; font-weight: 700; margin-left: 0.2rem; }
[data-testid="stSidebarNav"] { padding-top: 0.4rem; }
</style>""", unsafe_allow_html=True)


# =============================
# 영T FINAL 모의고사: 학교·학년 실제 시험과 같은 구성·배점·디자인의 회차별 동형 모의고사
#   - 구성(유형·배점·문항 수): 그 학교·학년 실제 기출(gichul_map) 중 고른 시험을 그대로 따름
#   - 문제: 변형문제 은행에서 뽑음 (AI 비용 없음). 같은 학교·학년 FINAL에서 이미 낸 문제는 다시 안 냄
#   - 난이도: 난이도별 예상 정답률로 평균을 예측 → 회차마다 기록한 실제 평균으로 예측을 보정 (누적)
#   - 단어 수: 실제 기출 지문 단어 수(시험지마다 한 번 재서 저장)에 가깝게 고름
# =============================
FINAL_FILE = os.path.join(BASE_DIR, "final_exams.json")          # 회차 기록 (문항 id·배점·예상/실제 평균)
FINAL_WORDS_FILE = os.path.join(BASE_DIR, "final_word_stats.json")  # 실제 기출 시험지별 지문 단어 수
FINAL_TITLE = "PQRS영T FINAL"
FINAL_P = {"하": 0.9, "중": 0.75, "상": 0.55, "최상": 0.4, "기출": 0.6}  # 처음 쓰는 난이도별 예상 정답률 (실제 평균 기록으로 보정)
FINAL_DECAY = 0.6  # 보정: 최근 회차일수록 크게 (한 회차 전 0.6, 두 회차 전 0.36 …)
_W_RE = re.compile(r"[A-Za-z][A-Za-z'’-]*")

SCHEMA_FINAL_WORDS = {
    "type": "object",
    "properties": {"questions": {"type": "array", "items": {"type": "object", "properties": {
        "no": {"type": "string", "description": "문항 번호 (예: 1, 서술형 2)"},
        "passage": {"type": "string", "description": "이 문항의 영어 지문 그대로 (주어진 글·<보기>·요약문 영어 포함). 앞 문항과 같은 지문이면 빈 문자열"},
        "same_as": {"type": "string", "description": "앞 문항과 지문을 같이 쓰면 그 첫 문항 번호, 아니면 빈 문자열"},
        "choices_text": {"type": "string", "description": "영어 선택지 전부를 한 줄로 (한국어 선택지·지문 속 ①~⑤ 유형은 빈 문자열)"},
    }, "required": ["no", "passage", "same_as", "choices_text"]}}},
    "required": ["questions"],
}

def _final_words(t) -> int:
    return len(_W_RE.findall(re.sub(r"<[^>]+>", " ", str(t or ""))))

def final_measure_words(gclient, model: str, pdf_part, file_name: str) -> Dict:
    """기출 시험지 PDF의 문항별 지문을 옮겨 적게 해서 단어 수를 셈 (시험지 1개에 한 번, Gemini Flash 기준 약 50원)"""
    prompt = (f"이 PDF는 고등학교 영어 지필평가 시험지다 (파일명: {file_name}). 객관식·서술형 문항마다 영어 지문을 시험지 그대로 옮겨라. "
              "여러 문항이 지문 하나를 같이 쓰면([1~3] 다음 글을 읽고…) 첫 문항에만 passage를 쓰고, 나머지 문항은 passage를 빈 문자열로, same_as에 첫 문항 번호를 적는다. "
              "주어진 글·<보기>·요약문의 영어도 passage 끝에 붙인다. choices_text에는 영어 선택지를 전부 한 줄로 적는다. 학생 필기·표시는 옮기지 않는다.")
    out = _qb_call(gclient, model, [pdf_part, prompt], SCHEMA_FINAL_WORDS, 0.0, [], f"{file_name} 단어 수")
    if not isinstance(out, dict) or not out.get("questions"):
        return {}
    per = [{"no": str(q.get("no", "")), "words": _final_words(q.get("passage")), "same_as": str(q.get("same_as") or ""),
            "choice_words": _final_words(q.get("choices_text"))} for q in out["questions"]]
    ps = [p["words"] for p in per if p["words"] >= 20]
    return {"n_q": len(per), "n_passages": len(ps), "passage_words": sum(ps), "choice_words": sum(p["choice_words"] for p in per),
            "per_q": per, "model": model, "at": time.strftime("%Y-%m-%d")}

def final_load_words(sb) -> Dict:
    return (_store_get(sb, FINAL_WORDS_FILE, {}) or {}).get("exams", {})

def final_save_words(sb, exam_id: str, stat: Dict):
    doc = _store_get(sb, FINAL_WORDS_FILE, {}) or {}
    doc.setdefault("exams", {})[exam_id] = stat
    _store_put(sb, FINAL_WORDS_FILE, doc)

def final_word_target(words: Dict, gmap: Dict, school: str, grade: str) -> Dict:
    """그 학교·학년 실제 기출의 평균 지문 단어 수 (잰 시험지만)"""
    ss = [w for eid, w in words.items() if (gmap.get("exams") or {}).get(eid, {}).get("school") == school
          and (gmap.get("exams") or {}).get(eid, {}).get("grade") == grade and w.get("n_passages")]
    if not ss:
        return {}
    n = len(ss)
    return {"n_exams": n, "passage_words": round(sum(w["passage_words"] for w in ss) / n),
            "n_passages": round(sum(w["n_passages"] for w in ss) / n, 1),
            "per_passage": round(sum(w["passage_words"] for w in ss) / max(1, sum(w["n_passages"] for w in ss))),
            "choice_words": round(sum(w["choice_words"] for w in ss) / n)}

def final_points(s) -> float:
    m = re.search(r"\d+(?:\.\d+)?", str(s or ""))
    return float(m.group(0)) if m else 0.0

def _final_exam_key(ex: Dict):
    m = re.search(r"(\d{2})\s*년.*?(\d)\s*학기.*?(중간|기말)", str(ex.get("exam_name", "")))
    return (int(m.group(1)), int(m.group(2)), 1 if m.group(3) == "기말" else 0) if m else (0, 0, 0)

def final_templates(gmap: Dict, school: str, grade: str) -> List:
    """그 학교·학년 실제 기출 (최신 시험 먼저, 문항이 너무 적게 분석된 시험은 뺌)"""
    exs = [(eid, ex) for eid, ex in (gmap.get("exams") or {}).items()
           if ex.get("school") == school and ex.get("grade") == grade and len(ex.get("questions", [])) >= 15]
    return sorted(exs, key=lambda x: _final_exam_key(x[1]), reverse=True)

def final_slots(ex: Dict) -> List[Dict]:
    """실제 기출 한 시험 → 문항 칸 목록 [{no, q_type, points}] (유형을 모르는 '기타'는 빼고 배점 합을 원래대로 맞춤)"""
    qs = ex.get("questions", [])
    total = sum(final_points(q.get("points")) for q in qs)
    slots = []
    for q in qs:
        t, no = q.get("q_type", ""), str(q.get("no", ""))
        if t == "기타" and "서술" in no:
            t = "서술형 (조건 영작)"
        if t in QB_TYPES:
            slots.append({"no": no, "q_type": t, "points": final_points(q.get("points")) or 4.0})
    kept = sum(s["points"] for s in slots)
    if slots and total and kept and abs(kept - total) > 0.05:
        k = total / kept
        for s in slots:
            s["points"] = round(s["points"] * k, 1)
        slots[-1]["points"] = round(slots[-1]["points"] + total - sum(s["points"] for s in slots), 1)  # 반올림 차이는 마지막 칸에
    return slots

def final_load(sb) -> List[Dict]:
    return (_store_get(sb, FINAL_FILE, {}) or {}).get("records", [])

def final_update(sb, change):
    """저장 직전에 최신 기록을 다시 읽어 바꿈 (다른 창에서 저장한 기록을 덮어쓰지 않게)"""
    recs = (_store_get(sb, FINAL_FILE, {}) or {}).get("records", [])
    _store_put(sb, FINAL_FILE, {"records": change(recs)})

def final_delta(records: List[Dict], school: str, grade: str) -> (float, int):
    """실제 평균 - 예측 평균(보정 전)의 가중 평균 → 다음 회차 예측에 더함. (보정값, 쓴 회차 수)"""
    done = [r for r in records if r.get("school") == school and r.get("grade") == grade and r.get("actual") not in (None, "")]
    done.sort(key=lambda r: r.get("date", ""), reverse=True)
    if not done:
        return 0.0, 0
    ws = [FINAL_DECAY ** i for i in range(len(done))]
    d = sum(w * (float(r["actual"]) - float(r.get("predicted_raw", 0))) for w, r in zip(ws, done)) / sum(ws)
    return round(d, 1), len(done)

def final_row_words(r: Dict) -> int:
    c = r.get("content") or {}
    return _final_words(" ".join(str(c.get(k) or "") for k in ("given", "passage", "box")))

def _final_pkey(r: Dict) -> str:
    """같은 글을 그대로 쓰는 문제끼리 묶는 열쇠 (원문 지문을 그대로 쓰는 주제·일치형 등은 한 지문에 여러 문제로 묶임)"""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", str((r.get("content") or {}).get("passage", "")))).strip()

def final_pick(rows: List[Dict], slots: List[Dict], labels: List[str], target_raw: float, word_target: int, seed: int) -> (List[Dict], List[str]):
    """문항 칸마다 같은 유형 문제를 골라 넣음 (AI 호출 없음). 목표: 예상 평균 ≈ 목표, 지문을 골고루, 지문 단어 수 ≈ 실제 기출.
    → 칸 순서가 아닌 지문 순서로 정렬한 문제 목록 (각 문제에 'points' 붙음), 알림"""
    rnd = random.Random(seed)
    rows = qb_dedupe_rows(rows)
    pools = {}
    for r in rows:
        pools.setdefault(r["q_type"], []).append(r)
    notes, need = [], {}
    for s in slots:
        need[s["q_type"]] = need.get(s["q_type"], 0) + 1
    lacking = {t: n - len(pools.get(t, [])) for t, n in need.items() if len(pools.get(t, [])) < n}
    if lacking:
        notes.append("문제가 모자란 유형: " + ", ".join(f"{t} {n}개" for t, n in lacking.items()) + " → 그 칸은 비워 두었어요. 아래에서 부족한 문제를 만들 수 있어요.")
    base = {l: i for i, l in enumerate(labels)}
    parts = lambda r: [p for p in qb_link_parts(r["passage_label"])[1] if p in base] or [r["passage_label"]]
    words_of = {id(r): final_row_words(r) for r in rows}

    def score(assign):
        chosen = [(s, r) for s, r in zip(slots, assign) if r is not None]
        # 실제로 채운 칸의 배점 기준 100점 환산 (칸이 비면 그만큼 만점이 줄어듦)
        pred = sum(s["points"] * FINAL_P.get(r["difficulty"], 0.6) for s, r in chosen) * 100 / (sum(s["points"] for s, _ in chosen) or 1)
        e = ((pred - target_raw) / 1.5) ** 2
        cnt = {}
        for _, r in chosen:
            for p in parts(r):
                cnt[p] = cnt.get(p, 0) + 1
        if cnt:
            m = sum(cnt.values()) / max(1, len(labels))
            e += sum((cnt.get(l, 0) - m) ** 2 for l in labels) * 0.8
        if word_target:
            uniq = {}
            for _, r in chosen:
                uniq.setdefault(_final_pkey(r), words_of[id(r)])
            e += ((sum(uniq.values()) - word_target) / (word_target * 0.08)) ** 2
        return e, pred

    used, assign = set(), []
    for s in slots:
        cands = [r for r in pools.get(s["q_type"], []) if id(r) not in used]
        r = rnd.choice(cands) if cands else None
        if r is not None:
            used.add(id(r))
        assign.append(r)
    best, _ = score(assign)
    idx = [i for i, r in enumerate(assign) if r is not None]
    for _ in range(4000 if idx else 0):
        i = rnd.choice(idx)
        cands = pools.get(slots[i]["q_type"], [])
        r = rnd.choice(cands)
        if id(r) in used:
            # 같은 유형 두 칸끼리 맞바꾸기는 점수가 같으니 건너뜀
            continue
        old = assign[i]
        assign[i] = r
        e, _ = score(assign)
        if e <= best:
            best = e
            used.discard(id(old))
            used.add(id(r))
        else:
            assign[i] = old
    _, pred = score(assign)
    # 지문 순서대로, 같은 글을 쓰는 문제는 붙여서, 한 지문 안에서는 원래 시험의 유형 순서대로 (서술형은 맨 뒤)
    t_order = {}
    for i, s in enumerate(slots):
        t_order.setdefault(s["q_type"], i)
    first_seen = {}
    picked = []
    for s, r in zip(slots, assign):
        if r is not None:
            picked.append({**r, "points": s["points"]})
    picked.sort(key=lambda r: (r["q_type"] in QB_ESSAY_TYPES, min(base.get(p, 999) for p in parts(r)), t_order.get(r["q_type"], 99)))
    for i, r in enumerate(picked):
        first_seen.setdefault(_final_pkey(r), i)
    picked.sort(key=lambda r: (r["q_type"] in QB_ESSAY_TYPES, first_seen[_final_pkey(r)], t_order.get(r["q_type"], 99)))
    for i, r in enumerate(picked):  # 순서·<보기> 형식 정리 (시험지에만)
        c = copy.deepcopy(r.get("content") or {})
        if r["q_type"] == "순서" and r.get("difficulty") != QB_GICHUL_LEVEL:
            qb_fix_order(c, random.Random(str(r.get("id", ""))), only_if_needed=True)
        elif r["q_type"] in QB_ONE_CHOICE_TYPES:
            fx = qb_fix_combo(c, r["q_type"])
            if fx:
                r = {**r, "q_type": fx}
        picked[i] = {**r, "content": c}
    return picked, notes, round(pred, 1)

FINAL_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700;900&family=Noto+Serif+KR:wght@400;700&display=swap');
@page { size: A4; margin: 13mm 12mm 16mm 12mm; }
* { box-sizing: border-box; }
body { font-family: 'Noto Serif KR', 'Times New Roman', serif; font-size: 10.4pt; line-height: 1.66; color: #111; margin: 0; background: #fff;
  -webkit-print-color-adjust: exact; print-color-adjust: exact; }
.sheet { max-width: 186mm; margin: 0 auto; }
.printbar { position: sticky; top: 0; text-align: right; padding: 6px; background: #fff; z-index: 5; }
.printbar button { font-size: 13px; padding: 6px 14px; cursor: pointer; }
@media print { .printbar { display: none; } .pfoot { position: fixed; bottom: -9mm; left: 0; right: 0; } }
.pfoot { font-family: 'Noto Sans KR'; font-size: 8.5pt; display: flex; justify-content: space-between; padding: 0 2mm; color: #222; }
.ttl { font-family: 'Noto Sans KR'; font-weight: 900; text-align: center; }
/* 운양고형 머리말: 위 칸 시험 이름, 가운데 큰 과목명, 오른쪽 실시 학급 */
.u-head { border: 2px solid #111; margin-bottom: 6px; font-family: 'Noto Sans KR'; }
.u-head .r1 { border-bottom: 1px solid #111; text-align: center; font-weight: 900; font-size: 14pt; padding: 6px; }
.u-head .r2 { display: flex; }
.u-head .logo { width: 30mm; display: flex; align-items: center; justify-content: center; border-right: 1px solid #111; }
.u-head .logo b { background: #111; color: #fff; padding: 2px 8px; font-size: 11pt; letter-spacing: 1px; }
.u-head .mid { flex: 1; text-align: center; padding: 4px 0; }
.u-head .mid .big { font-size: 25pt; font-weight: 900; letter-spacing: 3px; line-height: 1.25; }
.u-head .mid .date { font-size: 11pt; font-weight: 700; }
.u-head .cls { width: 48mm; border-left: 1px solid #111; display: flex; flex-direction: column; }
.u-head .cls div { flex: 1; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 11pt; }
.u-head .cls div:first-child { border-bottom: 1px solid #111; }
/* 제일고형 머리말: 위 제목 줄 + 학년 | 과목 | 시행일시 표 */
.j-title { font-family: 'Noto Sans KR'; font-weight: 900; font-size: 15pt; text-align: center; margin: 2px 0 6px; }
table.j-head { width: 100%; border-collapse: collapse; border: 2px solid #111; font-family: 'Noto Sans KR'; margin-bottom: 8px; }
table.j-head td { border: 1px solid #111; text-align: center; padding: 8px 4px; font-weight: 700; font-size: 11.5pt; }
table.j-head td.big { font-size: 20pt; font-weight: 900; }
.notice { border: 1px solid #111; padding: 5px 8px; font-family: 'Noto Sans KR'; font-size: 8.8pt; line-height: 1.5; margin-bottom: 8px; break-inside: avoid; }
.notice b { font-weight: 900; }
.cols { column-count: 2; column-gap: 8mm; column-rule: 1px solid #111; column-fill: auto; }
.q { margin: 0 0 15px; }
.q .stem, .sethead, .q .given { break-after: avoid; }
.q .stem { font-family: 'Noto Sans KR'; font-size: 10pt; margin-bottom: 4px; }
.q .stem .no { font-weight: 900; font-size: 11pt; margin-right: 3px; }
.q .stem .pt { white-space: nowrap; }
.sethead { font-family: 'Noto Sans KR'; font-weight: 700; font-size: 10.5pt; margin: 0 0 4px; }
.passage { text-align: justify; margin: 0 0 6px; }
.passage p { margin: 0 0 3px; text-indent: 1em; orphans: 3; widows: 3; }
.j .passage { border: 1px solid #111; padding: 6px 9px; }      /* 제일고는 지문을 네모 칸 안에 */
.given { border: 1px solid #111; padding: 5px 8px; margin: 0 0 7px; }
.given p, .box p { margin: 0; }
.box { border: 1px solid #111; padding: 4px 8px; margin: 2px 0 5px 8px; font-size: 9.8pt; break-inside: avoid; }
.box.long { break-inside: auto; }
.box .bx { text-align: center; font-family: 'Noto Sans KR'; font-size: 9pt; margin-bottom: 1px; }
u { text-underline-offset: 2px; }
.choices { margin: 0 0 0 8px; break-inside: avoid; }
.choices div { padding-left: 1.3em; text-indent: -1.3em; }
.choices.inline { display: flex; justify-content: space-between; padding-right: 12px; }
.choices.inline div { padding: 0; text-indent: 0; }
.essay-h { column-span: all; font-family: 'Noto Sans KR'; font-weight: 900; font-size: 12pt; border-top: 2px solid #111; border-bottom: 1px solid #111; padding: 4px 0; margin: 4px 0 10px; text-align: center; }
.write { border: 1px solid #111; height: 54px; margin: 2px 0 6px 8px; }
.ans-page { break-before: page; page-break-before: always; }
.anshead { border: 2px solid #111; text-align: center; font-family: 'Noto Sans KR'; font-weight: 900; font-size: 15pt; padding: 6px; margin-bottom: 8px; }
table.key { width: 100%; border-collapse: collapse; font-family: 'Noto Sans KR'; font-size: 9pt; margin-bottom: 6px; text-align: center; table-layout: fixed; }
table.key th, table.key td { border: 1px solid #111; padding: 3px 1px; }
table.key th { background: #e8e8e8; }
.keynote { font-family: 'Noto Sans KR'; font-size: 8.5pt; color: #333; margin: 0 0 8px; }
.exp { break-inside: avoid; margin-bottom: 7px; font-size: 9.2pt; line-height: 1.45; }
.exp h3 { font-family: 'Noto Sans KR'; font-size: 10pt; margin: 0 0 2px; border-bottom: 1px solid #999; padding-bottom: 1px; }
.exp h3 span { font-weight: 400; font-size: 8.2pt; color: #444; margin-left: 4px; }
.exp ul { margin: 0; padding-left: 14px; }
"""

def render_final_exam(qs: List[Dict], meta: Dict) -> str:
    """FINAL 모의고사 시험지 HTML. meta: school, grade, round, subject, term, date, with_answers, predicted"""
    school, grade = meta.get("school", "운양고"), str(meta.get("grade", "고1"))
    g = re.sub(r"\D", "", grade) or "1"
    subj = meta.get("subject") or "영어"
    rnd_no = meta.get("round", 1)
    title = f"{FINAL_TITLE} {rnd_no}회차"
    n_mc = sum(1 for r in qs if r["q_type"] not in QB_ESSAY_TYPES)
    n_es = len(qs) - n_mc
    total = round(sum(r.get("points", 0) for r in qs), 1)
    j = school == "제일고"
    o = ["<div class='printbar'><button onclick='window.print()'>🖨️ 인쇄 / PDF로 저장</button></div>", f"<div class='sheet{' j' if j else ''}'>"]
    if j:
        o.append(f"<div class='j-title'>{html.escape(title)} 모의고사 문제지 <span style='font-size:10.5pt;font-weight:700'>· {html.escape(meta.get('term', ''))} 제일고 대비</span></div>"
                 f"<table class='j-head'><tr><td>학 년</td><td class='big'>{g}</td><td class='big'>{html.escape(subj)}</td>"
                 f"<td>시 행 일 시</td><td>{html.escape(meta.get('date', ''))}</td></tr></table>")
        o.append(f"<div class='cols'><div class='notice'>※ 다음 내용을 반드시 확인하시오.<br>▣ 문항 수는 선택형 ( <b>{n_mc}</b> )문항"
                 + (f", 서술형 ( <b>{n_es}</b> )문항" if n_es else "") + f"입니다. (총 {total:g}점)<br>▣ 답안지에 인적사항(학년, 반, 번호, 성명)을 정확히 쓰시오.<br>"
                 "▣ 각 문항의 배점은 문항 끝에 표시되어 있습니다.</div>")
    else:
        o.append(f"<div class='u-head'><div class='r1'>{html.escape(title)} · ( {g} )학년 {html.escape(meta.get('term', ''))} 대비 모의고사 문제지</div>"
                 f"<div class='r2'><div class='logo'><b>PQRS 영T</b></div><div class='mid'><div class='big'>고{g} {html.escape(subj)}</div>"
                 f"<div class='date'>{html.escape(meta.get('date', ''))}</div></div><div class='cls'><div>대비 학교</div><div>운양고 {g}학년</div></div></div></div>")
        o.append(f"<div class='cols'><div class='notice'>※ 답안지에 필요한 인적 사항(반, 번호, 이름)을 정확히 표기한 후 답안을 작성하시오.<br>"
                 f"<b>선택형</b> 정답은 OMR카드 선택형 답란에 표기하시오.<br><b>배 점</b> 선택형 {n_mc}문항" + (f" · 서술형 {n_es}문항" if n_es else "")
                 + f", {total:g}점 (각 문항별 배점은 별도 표시)</div>")
    keys = [None if r["q_type"] in QB_ESSAY_TYPES else _final_pkey(r) for r in qs]
    mc_no, es_no, essay_started = 0, 0, False
    for i, r in enumerate(qs):
        c = r["content"]
        essay = r["q_type"] in QB_ESSAY_TYPES
        if essay:
            es_no += 1
            label = f"서술형 {es_no}"
            if not essay_started:
                essay_started = True
                o.append("<div class='essay-h'>서 술 형</div>")
        else:
            mc_no += 1
            label = f"{mc_no}."
        k = keys[i]
        in_set = k is not None and ((i > 0 and keys[i - 1] == k) or (i + 1 < len(qs) and keys[i + 1] == k))
        if in_set and (i == 0 or keys[i - 1] != k):
            last = i
            while last + 1 < len(qs) and keys[last + 1] == k:
                last += 1
            o.append(f"<div class='q'><div class='sethead'>[{mc_no}~{mc_no + last - i}] 다음 글을 읽고, 물음에 답하시오.</div>"
                     f"<div class='passage'>{qb_paras(c.get('passage'))}</div></div>")
        pt = r.get("points", 0)
        o.append(f"<div class='q'><div class='stem'><span class='no'>{label}</span> {qb_html(c.get('stem'))} <span class='pt'>[{pt:g}점]</span></div>")
        if c.get("given"):
            o.append(f"<div class='given'>{qb_paras(c['given'])}</div>")
        if not in_set:
            o.append(f"<div class='passage'>{qb_paras(c.get('passage'), r['q_type'] == '순서', c.get('given'))}</div>")
        if c.get("box"):
            bx = "<div class='bx'>&lt;보 기&gt;</div>" if re.search(r"[ⓐ-ⓩ]", str(c["box"])) else ""
            o.append(f"<div class='box{' long' if len(str(c['box'])) > 500 else ''}'>{bx}{qb_paras(c['box'], summary=r['q_type'] == '요약문 (A)(B)')}</div>")
        ch = c.get("choices") or []
        if essay:
            o.append("<div class='write'></div>")
        elif all(not str(x).strip() for x in ch):
            o.append("<div class='choices inline'>" + "".join(f"<div>{x}</div>" for x in CIRCLED) + "</div>")
        else:
            o.append("<div class='choices'>" + "".join(f"<div>{CIRCLED[n]} {qb_html(re.sub(r'^\s*[①-⑤]\s*', '', str(x)))}</div>" for n, x in enumerate(ch[:5])) + "</div>")
        o.append("</div>")
    o.append("</div>")
    if meta.get("with_answers"):
        o.append(f"<div class='ans-page'><div class='anshead'>{html.escape(title)} 정답 및 해설</div>")
        if meta.get("predicted") is not None:
            o.append(f"<div class='keynote'>교사용 · 예상 평균 {meta['predicted']:g}점 / {total:g}점 · 난이도 구성: "
                     + " · ".join(f"{lv} {sum(1 for r in qs if r['difficulty'] == lv)}" for lv in QB_LEVELS + [QB_GICHUL_LEVEL] if any(r['difficulty'] == lv for r in qs)) + "</div>")
        for s in range(0, len(qs), 12):
            part = qs[s:s + 12]
            nos = []
            for k2, r in enumerate(part, s + 1):
                nos.append(f"서{sum(1 for x in qs[:k2] if x['q_type'] in QB_ESSAY_TYPES)}" if r["q_type"] in QB_ESSAY_TYPES else str(sum(1 for x in qs[:k2] if x['q_type'] not in QB_ESSAY_TYPES)))
            pad = 12 - len(part)
            o.append("<table class='key'><tr><th>번호</th>" + "".join(f"<th>{n}</th>" for n in nos) + "<th></th>" * pad + "</tr>"
                     "<tr><th>정답</th>" + "".join(f"<td><b>{'서술형' if r['q_type'] in QB_ESSAY_TYPES else html.escape(r['content'].get('answer', ''))}</b></td>" for r in part) + "<td></td>" * pad + "</tr>"
                     "<tr><th>배점</th>" + "".join(f"<td>{r.get('points', 0):g}</td>" for r in part) + "<td></td>" * pad + "</tr>"
                     "<tr><th>난이도</th>" + "".join(f"<td>{html.escape(r['difficulty'])}</td>" for r in part) + "<td></td>" * pad + "</tr></table>")
        o.append("<div class='cols'>")
        mc_no = es_no = 0
        for r in qs:
            c = r["content"]
            if r["q_type"] in QB_ESSAY_TYPES:
                es_no += 1
                lab = f"서술형 {es_no}"
            else:
                mc_no += 1
                lab = f"{mc_no}번"
            o.append(f"<div class='exp'><h3>{lab} 정답 {qb_html(c.get('answer', ''))}<span>({html.escape(r['passage_label'])} · {html.escape(r['q_type'])} · {html.escape(r['difficulty'])})</span></h3><ul>"
                     + "".join(f"<li>{qb_html(e)}</li>" for e in (c.get("explanation") or [])) + "</ul></div>")
        o.append("</div></div>")
    o.append(f"<div class='pfoot'><span>{g}학년 ({html.escape(subj)})</span><span>{html.escape(title)} · {html.escape(meta.get('academy', '') or 'PQRS 영T')}</span></div>")
    o.append("</div>")
    return f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>{FINAL_CSS}</style></head><body>{''.join(o)}</body></html>"

def qb_fetch_ids(sb, ids: List[str]) -> List[Dict]:
    """문제 id로 불러옴 (저장한 회차 시험지를 다시 만들 때)"""
    ids = [i for i in dict.fromkeys(ids) if i]
    if sb is None:
        keep = set(ids)
        return [r for r in _qb_local_load() if r.get("id") in keep]
    out = []
    for i in range(0, len(ids), 100):
        out += sb.table(QB_TABLE).select("*").in_("id", ids[i:i + 100]).execute().data or []
    return out

def final_rebuild(sb, rec: Dict) -> List[Dict]:
    """저장한 회차 기록 → 그때와 같은 순서·배점의 문제 목록"""
    by_id = {r["id"]: r for r in qb_fetch_ids(sb, [x["id"] for x in rec.get("items", [])])}
    out = []
    for x in rec.get("items", []):
        r = by_id.get(x["id"])
        if not r:
            continue
        c = copy.deepcopy(r.get("content") or {})
        qt = r["q_type"]
        if qt == "순서" and r.get("difficulty") != QB_GICHUL_LEVEL:
            qb_fix_order(c, random.Random(str(r.get("id", ""))), only_if_needed=True)
        elif qt in QB_ONE_CHOICE_TYPES:
            qt = qb_fix_combo(c, qt) or qt
        out.append({**r, "q_type": qt, "content": c, "points": x.get("points", 4.0)})
    return out

def render_final_tab(text_model: str, api_delay: int, academy_name: str, school_name: str):
    sb = supabase if IS_CLOUD_DB else None
    st.caption("🧭 이럴 때: 학교 실제 시험과 같은 문항 수·배점·디자인의 회차별 모의고사가 필요할 때 · 자유 구성 시험지나 주간 클리닉은 '🧩 변형문제 은행·시험지'")
    ready, err = qb_table_status(sb)
    if not ready:
        st.error("변형문제 표가 아직 없습니다. '변형문제 은행·시험지' 화면의 안내대로 먼저 만들어 주세요.")
        return
    mock_lib = [p for p in st.session_state.saved_mock if p.get("body_text")]
    if not mock_lib:
        st.info("원문 라이브러리에 지문이 없습니다. '📝 원문 라이브러리'에서 원문을 먼저 저장해 주세요.")
        return
    by_label = {p["label"]: p for p in mock_lib}
    st.markdown(QB_PICKER_CSS, unsafe_allow_html=True)
    gmap = load_gichul_map()
    words = final_load_words(sb)
    recs = final_load(sb)

    # 1) 학교·학년·시험
    with st.container(key="qb_card_final_info"):
        st.markdown("<div class='qb-title'>시험 정보<span>같은 시험 이름으로 1회차, 2회차 … 이어서 만들어요</span></div>", unsafe_allow_html=True)
        school = _qb_row("학교").segmented_control("학교", QB_SCHOOLS, default=QB_SCHOOLS[0], key="final_school", label_visibility="collapsed") or QB_SCHOOLS[0]
        grade = _qb_row("학년").segmented_control("학년", ["고1", "고2", "고3"], default="고1", key="final_grade", label_visibility="collapsed") or "고1"
        mine = [r for r in recs if r.get("school") == school and r.get("grade") == grade]
        series_names = list(dict.fromkeys(r["series"] for r in sorted(mine, key=lambda r: r.get("date", ""), reverse=True)))
        NEW = "＋ 새 시험"
        s1, s2 = _qb_row("시험").columns([2, 3], vertical_alignment="center")
        pick = s1.selectbox("시험", series_names + [NEW], key=f"final_series_{school}_{grade}", label_visibility="collapsed")
        series = pick if pick != NEW else s2.text_input("새 시험 이름", placeholder="예: 26년 2학기 기말", key="final_series_new",
                                                         label_visibility="collapsed").strip()
        rounds = sorted([r for r in mine if r["series"] == series], key=lambda r: r.get("round", 0))
        next_round = (max((r.get("round", 0) for r in rounds), default=0) + 1) if series else 1
        last = rounds[-1] if rounds else {}
        c1, c2 = _qb_row("과목·날짜").columns([1, 1])
        subject = c1.text_input("과목", value=last.get("subject", "공통영어2" if grade == "고1" else "영어"), key=f"final_subject_{school}_{grade}_{series}",
                                placeholder="과목 (예: 공통영어2, 영어Ⅱ)", label_visibility="collapsed")
        from datetime import datetime, timedelta, timezone
        today = datetime.now(timezone(timedelta(hours=9)))
        date = c2.text_input("시행일", value=f"{today.year}년 {today.month}월 {today.day}일", key="final_date", label_visibility="collapsed")

    delta, n_cal = final_delta(recs, school, grade)
    if rounds:
        st.markdown(f"##### 📋 {html.escape(series)} 회차 기록")
        df = pd.DataFrame([{"회차": f"{r['round']}회차", "날짜": r.get("date", ""), "문항": len(r.get("items", [])),
                            "예상 평균": r.get("predicted"), "실제 평균": r.get("actual"), "응시 인원": r.get("n_students"),
                            "메모": r.get("note", ""), "id": r["id"]} for r in rounds])
        ed = st.data_editor(df, hide_index=True, use_container_width=True, key=f"final_rec_{school}_{grade}_{series}",
                            disabled=["회차", "날짜", "문항", "예상 평균"], column_config={
                                "id": None,
                                "실제 평균": st.column_config.NumberColumn("실제 평균 ✏️", min_value=0.0, max_value=100.0, step=0.1, format="%.1f"),
                                "응시 인원": st.column_config.NumberColumn("응시 인원 ✏️", min_value=0, step=1),
                                "메모": st.column_config.TextColumn("메모 ✏️")})
        r1, r2, r3 = st.columns([1.2, 1.6, 2])
        if r1.button("💾 점수 기록 저장", key=f"final_rec_save_{school}_{grade}_{series}", use_container_width=True):
            vals = {row["id"]: row for row in ed.to_dict("records")}

            def put(rs):
                for r in rs:
                    v = vals.get(r.get("id"))
                    if v:
                        a = v.get("실제 평균")
                        r["actual"] = None if a is None or (isinstance(a, float) and math.isnan(a)) else round(float(a), 1)
                        n = v.get("응시 인원")
                        r["n_students"] = None if n is None or (isinstance(n, float) and math.isnan(n)) else int(n)
                        r["note"] = str(v.get("메모") or "")
                return rs
            final_update(sb, put)
            st.success("✅ 점수를 저장했어요. 다음 회차 예상 평균에 바로 반영돼요.")
            st.rerun()
        which = r2.selectbox("다시 받을 회차", [r["id"] for r in rounds], format_func=lambda i: next(f"{r['round']}회차 · {r.get('date', '')}" for r in rounds if r["id"] == i),
                             key=f"final_re_{school}_{grade}_{series}", label_visibility="collapsed")
        if r3.button("📥 이 회차 시험지 다시 만들기", key=f"final_re_btn_{school}_{grade}_{series}", use_container_width=True):
            rec = next(r for r in rounds if r["id"] == which)
            qs = final_rebuild(sb, rec)
            if len(qs) < len(rec.get("items", [])):
                st.warning(f"저장 뒤에 지워진 문제 {len(rec['items']) - len(qs)}개는 빼고 만들었어요.")
            meta = {"school": school, "grade": grade, "round": rec["round"], "subject": rec.get("subject", subject), "term": series,
                    "date": rec.get("exam_date", date), "academy": academy_name, "predicted": rec.get("predicted")}
            st.session_state.final_re = (rec["round"], render_final_exam(qs, {**meta, "with_answers": False}), render_final_exam(qs, {**meta, "with_answers": True}))
        if st.session_state.get("final_re"):
            rn, stu, tea = st.session_state.final_re
            d1, d2 = st.columns(2)
            d1.download_button(f"📥 {rn}회차 학생용", stu, f"{FINAL_TITLE} {rn}회차 학생용.html", "text/html", use_container_width=True, key="final_re_dl1")
            d2.download_button(f"📥 {rn}회차 교사용 (정답·해설)", tea, f"{FINAL_TITLE} {rn}회차 교사용.html", "text/html", use_container_width=True, key="final_re_dl2")
    cal_txt = (f"지금까지 실제 평균을 기록한 {school} {grade} FINAL {n_cal}회를 보면, 실제 평균이 계산값보다 평균 <b>{delta:+.1f}점</b>이었어요. 다음 회차 예상 평균에 이만큼 보정해요."
               if n_cal else f"{school} {grade} FINAL의 실제 평균 기록이 아직 없어요. 시험을 본 뒤 위 표에 평균을 적어 주면, 다음 회차부터 예상 평균을 그만큼 보정해요.")
    st.markdown(f"<div class='qb-note'>🎯 {cal_txt}</div>", unsafe_allow_html=True)

    # 2) 시험 범위
    if series and rounds and f"final_labels_set_{series}" not in st.session_state:
        st.session_state["final_labels"] = [l for l in rounds[-1].get("labels", []) if l in by_label]
        st.session_state[f"final_labels_set_{series}"] = True
    links = [l for l in qb_link_labels(sb) if all(x in by_label for x in qb_link_parts(l)[1])]
    summary = qb_bank_summary(sb, tuple(sorted(by_label) + links))
    sel = _qb_passage_picker("final_labels", mock_lib, _qb_chip_counts(summary, "passage_label"))

    # 3) 구성
    tmpls = final_templates(gmap, school, grade)
    with st.container(key="qb_card_final_opts"):
        st.markdown("<div class='qb-title'>시험 구성<span>실제 기출 한 회의 유형·배점·문항 수를 그대로 따라요</span></div>", unsafe_allow_html=True)
        if not tmpls:
            st.warning(f"{school} {grade} 실제 기출이 아직 없어요. '변형문제 은행·시험지 → 🏫 기출 표시'에서 기출 시험지를 먼저 올려 주세요.")
            return
        t_id = _qb_row("참고 기출").selectbox("참고 기출", [e for e, _ in tmpls], key=f"final_tmpl_{school}_{grade}", label_visibility="collapsed",
                                             format_func=lambda e: (lambda x: f"{x['display']} · {len(x.get('questions', []))}문항")(gmap["exams"][e]),
                                             index=next((i for i, (e, _) in enumerate(tmpls) if e == last.get("template")), 0))
        slots = final_slots(gmap["exams"][t_id])
        tcount = {}
        for s in slots:
            tcount[s["q_type"]] = tcount.get(s["q_type"], 0) + 1
        st.markdown(f"<div class='qb-note'>선택형 <b>{sum(1 for s in slots if s['q_type'] not in QB_ESSAY_TYPES)}</b> · 서술형 <b>{sum(1 for s in slots if s['q_type'] in QB_ESSAY_TYPES)}</b> · "
                    f"총 <b>{sum(s['points'] for s in slots):g}점</b> &nbsp;|&nbsp; " + " · ".join(f"{t} {n}" for t, n in tcount.items()) + "</div>", unsafe_allow_html=True)
        tgt_words = final_word_target(words, gmap, school, grade)
        if tgt_words:
            st.markdown(f"<div class='qb-note'>📏 {school} {grade} 실제 기출 {tgt_words['n_exams']}회 평균: 지문 약 <b>{tgt_words['n_passages']:g}개</b>, "
                        f"지문 단어 합계 약 <b>{tgt_words['passage_words']:,}단어</b> (지문 1개 약 {tgt_words['per_passage']}단어). 이 길이에 가깝게 골라요.</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='qb-note'>📏 {school} {grade} 실제 기출의 단어 수를 아직 재지 않았어요. 단어 수는 맞추지 않고 골라요.</div>", unsafe_allow_html=True)
        a1, a2 = _qb_row("목표 평균").columns([1.3, 6], vertical_alignment="center")
        target = a1.number_input("목표 평균", 30.0, 95.0, float(last.get("target", 70.0)), 1.0, key=f"final_target_{school}_{grade}", label_visibility="collapsed")
        a2.markdown("<div class='qb-note' style='margin-top:0'>점 (100점 기준) · 난이도별 예상 정답률로 평균을 계산해 이 점수에 맞게 난이도를 섞어요</div>", unsafe_allow_html=True)
        o1, o2 = _qb_row("문제").columns(2)
        use_real = o1.checkbox("실제 기출 문제도 넣기", value=False, key="final_real", help="끄면 변형문제만 써요 (학생이 이미 본 실제 기출은 빼는 게 보통이에요)")
        other_school = o2.checkbox("다른 학교 스타일 변형도 쓰기", value=False, key="final_other", help="끄면 이 학교 스타일로 만든 변형문제만 써요")
    used_ids = {x["id"] for r in mine for x in r.get("items", [])}
    st.caption(f"🔁 {school} {grade} FINAL에서 이미 낸 문제 {len(used_ids)}개는 빼고 뽑아요.")

    b1, b2 = st.columns([2, 1])
    make = b1.button(f"📄 {FINAL_TITLE} {next_round}회차 만들기", type="primary", disabled=not sel or not series, use_container_width=True, key="final_make")
    again = b2.button("🔀 다른 조합으로 다시 뽑기", disabled="final_built" not in st.session_state, use_container_width=True, key="final_again")
    if not series:
        st.caption("시험 이름을 먼저 정해 주세요 (예: 26년 2학기 기말).")
    if make or again:
        sel_links = [l for l in links if set(qb_link_parts(l)[1]) <= set(sel)]
        lab_order = sorted(sel, key=lambda l: natural_sort_key({"label": l}))
        rows = [r for r in qb_fetch(sb, lab_order + sel_links)
                if r["id"] not in used_ids and (use_real or r["difficulty"] != QB_GICHUL_LEVEL)
                and (other_school or r.get("school") == school or r["difficulty"] == QB_GICHUL_LEVEL)]
        picked, notes, pred_raw = final_pick(rows, slots, lab_order, target - delta, (tgt_words or {}).get("passage_words", 0), random.randint(1, 10**9))
        st.session_state.final_built = {"school": school, "grade": grade, "series": series, "round": next_round, "picked": picked, "notes": notes,
                                        "pred_raw": pred_raw, "delta": delta, "target": target, "labels": sel, "template": t_id, "subject": subject, "date": date,
                                        "lacking": [t for t, n in tcount.items() if sum(1 for r in rows if r["q_type"] == t) < n]}
    fb = st.session_state.get("final_built")
    if fb and (fb["school"], fb["grade"], fb["series"]) == (school, grade, series):
        qs = fb["picked"]
        pred = round(fb["pred_raw"] + fb["delta"], 1)
        uniq = {}
        for r in qs:
            uniq.setdefault(_final_pkey(r), final_row_words(r))
        lv = {}
        for r in qs:
            lv[r["difficulty"]] = lv.get(r["difficulty"], 0) + 1
        st.markdown(f"<div class='qb-basket'><b>{FINAL_TITLE} {fb['round']}회차</b> · {len(qs)}문항 / 칸 {len(slots)}개 · "
                    f"예상 평균 <b>{pred:g}점</b> (목표 {fb['target']:g}점" + (f", 기록 보정 {fb['delta']:+.1f}점 포함" if fb["delta"] else "") + ")<br>"
                    f"난이도: " + " · ".join(f"{k} {lv[k]}" for k in QB_LEVELS + [QB_GICHUL_LEVEL] if lv.get(k)) +
                    f" &nbsp;|&nbsp; 지문 {len(uniq)}개 · 단어 합계 약 {sum(uniq.values()):,}" + (f" (실제 기출 평균 {tgt_words['passage_words']:,})" if tgt_words else "") + "</div>",
                    unsafe_allow_html=True)
        for n in fb["notes"]:
            st.warning(n)
        off = pred - fb["target"]
        if abs(off) > 5:
            st.warning(f"예상 평균이 목표보다 {abs(off):.0f}점 {'낮아요' if off < 0 else '높아요'}. 고른 범위에 "
                       f"{'쉬운(중·하)' if off < 0 else '어려운(상·최상)'} 문제가 부족해서예요. 아래 버튼으로 그 난이도 문제를 더 만들면 맞출 수 있어요.")
        if fb["lacking"] or abs(off) > 5:
            lk = list(dict.fromkeys(fb["lacking"] or [t for t in tcount if t not in QB_ESSAY_TYPES]))
            levels = ["중", "하"] if off < -5 else (["최상", "상"] if off > 5 else (["상", "중"] if fb["target"] >= 65 else ["최상", "상"]))
            cost = len(fb["labels"]) * sum(qb_job_cost_krw(text_model, 2, True) for _ in lk for _ in levels)
            if st.button(f"🛠️ 부족한 문제 만들기 작업 추가 ({', '.join(lk)} · {'·'.join(levels)}) · 고른 지문 {len(fb['labels'])}개 × 2문제 · 약 {cost:,.0f}원",
                         key="final_fill"):
                n = enqueue_qbank([by_label[l] for l in fb["labels"] if l in by_label], lk, levels, school, 2, True, text_model, api_delay,
                                  academy_name, school_name, True, "mix", False, grade=grade if grade in QB_GRADES else "")
                st.session_state.gen_flash = f"✅ 문제 만들기 작업 {n}건을 추가했어요. 다 만들어지면 다시 'FINAL 만들기'를 눌러 주세요."
                st.rerun()
        meta = {"school": school, "grade": grade, "round": fb["round"], "subject": fb["subject"], "term": series, "date": fb["date"],
                "academy": academy_name, "predicted": pred}
        stu = render_final_exam(qs, {**meta, "with_answers": False})
        tea = render_final_exam(qs, {**meta, "with_answers": True})
        d1, d2, d3 = st.columns([1, 1, 1.3])
        d1.download_button("📥 학생용 HTML", stu, f"{FINAL_TITLE} {fb['round']}회차 학생용.html", "text/html", use_container_width=True, key="final_dl1")
        d2.download_button("📥 교사용 HTML (정답·해설)", tea, f"{FINAL_TITLE} {fb['round']}회차 교사용.html", "text/html", use_container_width=True, key="final_dl2")
        if d3.button(f"💾 {fb['round']}회차로 확정 저장 (다음 회차에서 이 문제들은 빠짐)", type="primary", use_container_width=True, key="final_save", disabled=not qs):
            rec = {"id": uuid.uuid4().hex[:10], "school": school, "grade": grade, "series": series, "round": fb["round"],
                   "date": today.strftime("%Y-%m-%d"), "exam_date": fb["date"], "subject": fb["subject"], "labels": fb["labels"], "template": fb["template"],
                   "items": [{"id": r["id"], "points": r["points"], "level": r["difficulty"], "q_type": r["q_type"]} for r in qs],
                   "target": fb["target"], "predicted_raw": fb["pred_raw"], "delta": fb["delta"], "predicted": pred,
                   "actual": None, "n_students": None, "note": ""}
            final_update(sb, lambda rs: rs + [rec])
            st.session_state.pop("final_built", None)
            st.success(f"✅ {series} {fb['round']}회차를 저장했어요. 시험을 본 뒤 위 '회차 기록' 표에 실제 평균을 적어 주세요.")
            st.rerun()
        components.html(tea, height=900, scrolling=True)


def page_home():
    st.markdown("#### 무엇을 하시겠어요?")
    st.markdown("<div class='pg-desc'>자주 쓰는 일을 목적별로 모았어요. 왼쪽 메뉴에서도 언제든 바로 갈 수 있어요.</div>", unsafe_allow_html=True)
    st.write("")
    n_src = len(st.session_state.saved_mock)
    counts = {
        "source": f"원문 {n_src}개", "oneshot": f"{len(st.session_state.saved_v)}개", "selfnote": f"{len(st.session_state.saved_a)}개",
        "core": f"{len(st.session_state.saved_c)}개", "voca": f"{len(st.session_state.saved_w)}개",
        "past-voca": f"시험지 {len(st.session_state.saved_past_voca)}개",
    }
    for group, keys in HOME_GROUPS:
        st.markdown(f"##### {group}")
        cols = st.columns(len(keys))
        for i, key in enumerate(keys):
            info = PAGE_INFO[key]
            with cols[i].container(border=True):
                st.page_link(PAGES[key], label=f"**{info['title']}**", icon=info["icon"], use_container_width=True)
                cnt = f"<span class='home-count'>· 저장 {counts[key]}</span>" if key in counts else ""
                st.markdown(f"<div class='home-tile-desc'>{info['short']}{cnt}</div>", unsafe_allow_html=True)
        st.write("")
    with st.expander("📖 처음 쓰는 분을 위한 사용 설명서 — 시스템 원리·버튼 역할", expanded=False):
        st.markdown(APP_HELP["overview"])


def excel_template_bytes() -> bytes:
    """엑셀 원클릭 빈 양식 (열 이름 + 예시 한 줄)"""
    buf = io.BytesIO()
    pd.DataFrame([{"라벨": "[2025년 6월 고1] 20번", "지문": "Many people believe that ... (영어 원문을 그대로 붙여 넣기)",
                   "학교명": "운양고", "이미지명": "(선택) 함께 올릴 이미지 파일 이름, 예: q20.png"}]).to_excel(buf, index=False)
    return buf.getvalue()

def page_excel():
    e1, e2 = st.columns([3, 1.2], vertical_alignment="bottom")
    xlsx = e1.file_uploader("📂 엑셀 업로드", type=["xlsx"])
    e2.download_button("📄 빈 엑셀 양식 받기", excel_template_bytes(), "4pm_엑셀원클릭_양식.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True,
                       help="열 이름: 라벨 · 지문 · 학교명 · 이미지명 (영어 열 이름 label · passage · school · image_name도 돼요). 지문만 꼭 채우면 돼요.")
    imgs = st.file_uploader("🖼️ 이미지 대량 업로드 (선택)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    # 올린 엑셀을 미리 읽어 지문 수·예상 비용을 보여 줌 (생성 버튼을 눌렀을 때 읽는 것과 같은 규칙)
    xl_preview = []
    if xlsx:
        try:
            _df = pd.read_excel(io.BytesIO(xlsx.getvalue()))
            for i, r in _df.iterrows():
                raw_text = r.get('passage', r.get('지문', ''))
                if pd.isna(raw_text) or not str(raw_text).strip():
                    continue
                xl_preview.append({"label": clean_label(r.get('label', r.get('라벨', f'지문_{i+1}'))),
                                   "img": str(r.get('image_name', r.get('이미지명', ''))).strip()})
            if not xl_preview:
                st.warning("⚠️ 엑셀에서 지문을 찾지 못했어요. '지문'(또는 passage) 열에 영어 원문이 있는지 확인해 주세요. 위 '빈 엑셀 양식 받기'의 열 이름과 같아야 해요.")
        except Exception as e:
            st.warning(f"⚠️ 엑셀을 읽지 못했어요: {str(e)[:150]}")

    with st.container(border=True):
        x_col1, x_col2, x_col3 = st.columns([2.2, 1.4, 1.4])
        xl_gen_image = x_col1.checkbox(
            "🎨 이미지가 없는 지문은 AI가 지문 내용으로 이미지 생성 (한판승·Self-Note에 사용)", value=True, key="xl_gen_image",
            help="엑셀의 이미지명에 맞는 이미지를 올린 지문은 그 이미지를 쓰고, 나머지 지문만 새로 만듭니다. 장당 약 10초, 생성 작업 안에서 함께 진행됩니다."
        )
        xl_img_style = x_col2.selectbox("🖌️ 화풍", [IMAGE_STYLE_AUTO] + [name for name, _ in IMAGE_STYLES], key="xl_img_style", disabled=not xl_gen_image)
        xl_img_model = x_col3.selectbox("🎨 이미지 모델", IMAGE_MODELS, key="xl_img_model", disabled=not xl_gen_image)

    st.markdown("##### 🎯 생성할 교재 종류 선택 (원하는 항목만 체크 가능)")
    g_col1, g_col2, g_col3, g_col4 = st.columns(4)
    do_v = g_col1.checkbox("🏆 Anatomy One-Shot (한판승)", value=True)
    do_a = g_col2.checkbox("🦅 Anatomy Self-Note", value=True)
    do_c = g_col3.checkbox("💎 Anatomy Core", value=True)
    do_w = g_col4.checkbox("📖 Anatomy Voca", value=True)
    x_types = [t for t, on in [("victory", do_v), ("anatomy", do_a), ("core", do_c), ("voca", do_w)] if on]
    if xl_preview and x_types:
        up_names = {i.name for i in (imgs or [])}
        n_pic = sum(1 for r in xl_preview if r["img"] not in up_names) if xl_gen_image and set(x_types) & {"victory", "anatomy"} else 0
        st.caption(f"📄 엑셀 지문 {len(xl_preview)}개 · " + book_cost_line(text_model, book_new_counts([r["label"] for r in xl_preview], x_types),
                                                                       f" + AI 이미지 최대 {n_pic}장 약 {image_cost_krw(xl_img_model) * n_pic:,.0f}원" if n_pic else ""))

    if xlsx and st.button("🚀 올인원 통합 생성 시작", type="primary"):
        if not (do_v or do_a or do_c or do_w):
            st.warning("생성할 교재 종류를 최소 1개 이상 체크해주세요.")
        else:
            df = pd.read_excel(xlsx)
            img_dict = {
                i.name: f"data:{i.type if hasattr(i, 'type') and i.type else 'image/png'};base64,{base64.b64encode(i.getvalue()).decode()}"
                for i in imgs
            } if imgs else {}

            rows = []
            for i, r in df.iterrows():
                raw_text = r.get('passage', r.get('지문', ''))
                if pd.isna(raw_text) or not str(raw_text).strip():
                    continue
                excel_school = r.get('school', r.get('학교명', ''))
                img_name = str(r.get('image_name', r.get('이미지명', ''))).strip()
                img_path = ""
                if img_dict.get(img_name):
                    # 업로드한 이미지는 교재용 크기(1280px JPEG)로 줄여 저장
                    raw_img = base64.b64decode(img_dict[img_name].split(",", 1)[1])
                    img_path = save_booklet_image(raw_img, f"xl_{hashlib.md5(raw_img).hexdigest()[:10]}")
                rows.append({
                    "label": clean_label(r.get('label', r.get('라벨', f'지문_{i+1}'))),
                    "text": str(raw_text).strip(),
                    "school": str(excel_school).strip() if pd.notna(excel_school) and str(excel_school).strip() not in ["", "nan"] else "",
                    "img_path": img_path,
                    "gen_image": xl_gen_image and not img_path,
                    "image_model": xl_img_model,
                })
            # 화풍은 지문마다 돌아가며 배정 (자동이면 이웃한 지문끼리 겹치지 않게)
            for r, style in zip(rows, pick_image_styles(len(rows), xl_img_style)):
                r["image_style"] = style

            types_list = [t for t, on in [("victory", do_v), ("anatomy", do_a), ("core", do_c), ("voca", do_w)] if on]
            n_added = enqueue_generation(rows, types_list, text_model, api_delay, academy_name, school_name)
            st.session_state.gen_flash = f"✅ {n_added}개 지문을 생성 작업에 추가했습니다. 창을 닫아도 서버에서 계속 진행됩니다."
            st.rerun()


def page_source():
    render_mock_exam_tab(text_model, api_delay, academy_name, school_name)


def page_qbank():
    render_qbank_tab(text_model, api_delay, academy_name, school_name)


def page_final():
    render_final_tab(text_model, api_delay, academy_name, school_name)


def page_oneshot():
    render_library_manager("v", "Anatomy One-Shot", VICTORY_FILE, CSS_VICTORY, "victory", text_model, api_delay, academy_name, school_name, h_url, f_url)


def page_selfnote():
    render_library_manager("a", "Anatomy Self-Note", ANATOMY_FILE, CSS_ANATOMY, "anatomy", text_model, api_delay, academy_name, school_name, h_url, f_url)


def page_core():
    render_library_manager("c", "Anatomy Core", CORE_FILE, CSS_CORE, "core", text_model, api_delay, academy_name, school_name, h_url, f_url)


def page_voca():
    render_library_manager("w", "Anatomy Voca", VOCA_FILE, CSS_VOCA, "voca", text_model, api_delay, academy_name, school_name, h_url, f_url)


def page_dna():
    render_dna_board()
    st.divider()
    st.markdown("#### ➕ 기출 시험지 PDF로 출제 규칙 더 학습시키기")
    st.info(
        "💡 새 시험지 PDF를 올리면 AI가 **주제·요약문 패러프레이징 방식, 빈칸 타겟 어휘 수준, 어휘 추론 함정 패턴**을 분석해 출제 규칙에 덧붙이고 클라우드 DB에 영구 누적합니다. "
        "문항 유형·배점 통계와 '기출 이력'은 '🧩 변형문제 은행·시험지' 메뉴 → '🏫 기출 표시' 탭에 시험지를 올려야 쌓여요."
    )

    col_dna1, col_dna2 = st.columns([1.2, 1.8])
    with col_dna1:
        uploaded_pdfs = st.file_uploader(
            "📄 분석할 학교 기출 시험지 PDF 업로드 (여러 개 동시 가능)",
            type=["pdf"],
            accept_multiple_files=True,
            key="dna_pdf_uploader"
        )
        school_tag = st.text_input("🏫 시험지 출처 태그 (예: 26년 2학기 기말 제일고1)", placeholder="예: 26년 2학기 기말 운양고2")

        if uploaded_pdfs and st.button("🧠 업로드한 기출 PDF 출제DNA 추출 및 누적 적용", type="primary", use_container_width=True):
            with st.spinner("기출 시험지 PDF의 패러프레이징 패턴과 어휘 함정을 정밀 분석 중입니다..."):
                try:
                    contents_payload = [AiFile(up_pdf.getvalue(), "application/pdf", up_pdf.name) for up_pdf in uploaded_pdfs]
                    dna_prompt = f"""너는 대한민국 고등학교 영어 내신 기출 분석 전문가다.
업로드된 영어 시험지 PDF({', '.join([u.name for u in uploaded_pdfs])})를 정밀 분석하여 다음 4가지를 추출하라.
1. 주제/제목/요지/주장 선지에서 원문 표현을 어떤 수준의 상위 개념어(Meta-vocabulary)와 구문으로 패러프레이징하는지 (실제 시험지 속 구체적 영어 예시 포함)
2. 요약문 빈칸 (A), (B) 문항에서 어떤 품사와 성격의 단어를 빈칸 타겟으로 삼는지 (실제 시험지 속 영어 예시 포함)
3. '문맥상 낱말의 쓰임' 및 어휘 변형 문제에서 어떤 반의어/혼동어 함정을 사용하는지 (실제 영어 예시 포함)
4. <보기> 영문 일치/불일치 문항에서 오답과 정답 선지를 어떻게 변형하는지
5. 서술형·논술형 문항이 본문의 어떤 문장(주제문·결론·예시 등)을 고르는지, 어떤 문법(관계사·분사구문·가주어 등)을 묻는지, 어떤 형식(조건 영작·배열·어법 수정·요약문 완성 등)인지

기존 출제 DNA 규칙:
{st.session_state.exam_dna.get('rules_text', '')}

기존 서술형 출제 DNA:
{st.session_state.exam_dna.get('essay_rules_text', '') or '(아직 없음)'}

위 기존 규칙을 유지하면서 새로 발견된 출제 패턴과 구체적 어휘 예시를 통합하여, 앞으로 'ANATOMY CORE'와 'ANATOMY VOCA' 생성 프롬프트에 바로 넣을 수 있는 완성된 [출제 DNA 핵심 규칙] 텍스트(updated_rules_text),
5번 분석을 기존 서술형 DNA에 통합한 [서술형 출제 DNA] 텍스트(updated_essay_rules_text, 한판승 '서술형 출제 가능' 표시 기준으로 쓰이며 2500자 이내),
이번 시험지 핵심 요약(analysis_summary)을 JSON으로 반환하라."""

                    dna_schema = {
                        "type": "object",
                        "properties": {"updated_rules_text": {"type": "string"}, "updated_essay_rules_text": {"type": "string"}, "analysis_summary": {"type": "string"}},
                        "required": ["updated_rules_text", "updated_essay_rules_text", "analysis_summary"]
                    }
                    contents_payload.append(dna_prompt)
                    parsed_dna = llm_json(text_model, contents_payload, dna_schema, 0.2)
                    dna_title = school_tag.strip() if school_tag.strip() else ", ".join([u.name for u in uploaded_pdfs])

                    def _apply_dna(d, parsed=parsed_dna, title=dna_title):
                        d["rules_text"] = parsed["updated_rules_text"]
                        if parsed.get("updated_essay_rules_text", "").strip():
                            d["essay_rules_text"] = parsed["updated_essay_rules_text"].strip()
                        d["updated_at"] = time.strftime("%Y-%m-%d %H:%M")
                        d.setdefault("history", []).append({"title": title, "summary": parsed["analysis_summary"]})
                    st.session_state.pop("essay_dna_editor", None)  # 서술형 편집 칸이 예전 글을 붙잡고 있지 않게
                    if exam_dna_update(_apply_dna):
                        st.success("✅ 기출 PDF 출제DNA 분석 및 클라우드 DB 반영 완료!")
                        st.rerun()
                    else:
                        # 비용을 낸 분석 결과는 이 창에 남겨 둠 → 아래 '수동 저장'·'서술형 저장'으로 다시 저장 가능
                        _apply_dna(st.session_state.exam_dna)
                        st.warning("분석 결과는 이 창에 보관했어요. 잠시 뒤 아래 '💾 출제 DNA 규칙 수동 저장'과 '💾 서술형 출제 DNA 저장'을 눌러 저장해 주세요.")
                except Exception as e:
                    st.error(f"❌ PDF 분석 중 오류 발생: {e}")

        st.caption("학습 기록은 위 '📚 누적된 기출 시험지 목록'을 펼치면 볼 수 있어요.")

    with col_dna2:
        st.markdown(f"#### 🧬 현재 활성화된 출제 DNA 프롬프트 규칙 (최근 업데이트: `{st.session_state.exam_dna.get('updated_at', '')}`)")
        st.caption("이 규칙이 `Anatomy Core`와 `Anatomy Voca`를 생성할 때 AI에게 자동으로 주입됩니다. 직접 텍스트를 다듬어 저장하실 수도 있습니다.")
        edited_rules = st.text_area(
            "출제 DNA 프롬프트 규칙 편집",
            value=st.session_state.exam_dna.get("rules_text", ""),
            height=420,
            label_visibility="collapsed"
        )
        dc1, dc2 = st.columns(2)
        if dc1.button("💾 출제 DNA 규칙 수동 저장", type="primary", use_container_width=True):
            if exam_dna_update(lambda d, t=edited_rules: d.__setitem__("rules_text", t)):
                st.success("✅ 출제 DNA 규칙이 저장되었습니다!")
        with dc2.popover("🔄 운양고·제일고 기본값으로 초기화", use_container_width=True):
            # 한 번 잘못 눌러도 사라지지 않게 확인을 한 번 더 받음. 바뀌는 것은 위 '출제 DNA 규칙' 글뿐
            st.caption("위 '출제 DNA 규칙' 글만 처음 기본값으로 되돌려요. 서술형 출제 DNA · 🎣 영쌤 출제 포인트 · 학습 기록은 그대로 둬요. 되돌린 뒤에는 지금 규칙 글을 다시 볼 수 없어요.")
            if st.button("기본값으로 되돌리기 확정", type="primary", use_container_width=True, key="dna_reset_ok"):
                def _reset_rules(d):
                    d["rules_text"] = DEFAULT_EXAM_DNA["rules_text"]
                    d["updated_at"] = DEFAULT_EXAM_DNA["updated_at"]
                if exam_dna_update(_reset_rules):
                    st.rerun()

        st.markdown("#### ✍️ 서술형 출제 DNA (한판승 '서술형 출제 가능' 표시 기준)")
        st.caption("운양고·제일고 기출 서술형·논술형 문항을 분석한 기준입니다. 한판승을 만들 때 이 기준에 맞는 문장만 지문당 1개(긴 지문 2개) 표시합니다.")
        edited_essay = st.text_area(
            "서술형 출제 DNA 편집",
            value=st.session_state.exam_dna.get("essay_rules_text", "") or ESSAY_DNA_FALLBACK,
            height=300,
            label_visibility="collapsed",
            key="essay_dna_editor"
        )
        if st.button("💾 서술형 출제 DNA 저장", use_container_width=True, key="essay_dna_save"):
            if exam_dna_update(lambda d, t=edited_essay.strip(): d.__setitem__("essay_rules_text", t)):
                st.success("✅ 서술형 출제 DNA가 저장되었습니다!")

    render_yt_points_editor()


def render_yt_points_editor():
    """🎣 영쌤 출제 포인트 목록 편집 (출제 DNA 문서의 yt_points에 저장)"""
    st.markdown("#### 🎣 영쌤 출제 포인트 (학교와 관계없이 변형문제 만들 때 반영)")
    st.caption("선생님이 쓰는 '낚시' 출제 방식을 한 줄에 하나씩 적어 두면, 변형문제를 만들 때 맞는 유형의 문항에 하나씩 넣어요 "
               "(평소엔 문항 절반쯤, '문제 만들기'에서 '🎣 영쌤 출제 포인트 위주로'를 켜면 모든 문항). 맨 아래 빈 줄에 새 포인트를 계속 추가할 수 있어요. "
               "'교과서만'을 켜면 교과서 지문에만 쓰고, 같은 단원 다른 단락 원문을 함께 넘겨요.")
    pts = qb_yt_points()
    df = pd.DataFrame([{"사용": bool(p.get("on", True)), "출제 포인트": str(p.get("text", "")),
                        "적용 유형": p.get("scope") if p.get("scope") in QB_YT_SCOPES else "전체 유형", "교과서만": bool(p.get("tb"))} for p in pts],
                      columns=["사용", "출제 포인트", "적용 유형", "교과서만"])
    edited = st.data_editor(
        df, num_rows="dynamic", use_container_width=True, hide_index=True, key="yt_points_editor",
        column_config={
            "사용": st.column_config.CheckboxColumn("사용", default=True, width="small"),
            "출제 포인트": st.column_config.TextColumn("출제 포인트", width="large", required=True),
            "적용 유형": st.column_config.SelectboxColumn("적용 유형", options=list(QB_YT_SCOPES), default="전체 유형", required=True),
            "교과서만": st.column_config.CheckboxColumn("교과서만", default=False, width="small"),
        })
    y1, y2 = st.columns(2)
    if y1.button("✨ 정리해서 저장하기", type="primary", use_container_width=True, key="yt_points_save",
                 help="새로 쓰거나 고친 포인트를 AI가 출제 지시문으로 깔끔하게 정리해 보여 줘요. 확인·수정한 뒤 '확정해서 저장'을 눌러야 올라가요 (포인트 몇 개에 약 1~5원)."):
        rows = []
        for _, r in edited.iterrows():
            t = str(r.get("출제 포인트") or "").strip()
            if t and t.lower() != "nan":
                rows.append({"on": bool(r.get("사용")) if pd.notna(r.get("사용")) else True, "text": t,
                             "scope": r.get("적용 유형") if r.get("적용 유형") in QB_YT_SCOPES else "전체 유형",
                             "tb": bool(r.get("교과서만")) if pd.notna(r.get("교과서만")) else False})
        have = {str(p.get("text", "")).strip() for p in pts}
        changed = [i for i, r in enumerate(rows) if r["text"] not in have]
        if not changed:
            # 글은 그대로이고 사용·유형·교과서만만 바꾼 경우는 정리할 것이 없으니 바로 저장
            if exam_dna_update(lambda d, r=rows: d.__setitem__("yt_points", r)):
                st.success(f"✅ 영쌤 출제 포인트 {len(rows)}개를 저장했어요. 다음에 시작하는 변형문제 작업부터 반영돼요.")
        else:
            with st.spinner(f"새로 쓰거나 고친 포인트 {len(changed)}개를 정리하는 중..."):
                tidy = yt_tidy_points([rows[i] for i in changed], text_model)
            if tidy is None:
                st.error("정리에 실패했어요. 잠시 뒤 다시 눌러 주세요. (아무것도 저장하지 않았어요)")
            else:
                for t in tidy:
                    t["src"] = changed[t["src"]]  # 정리한 목록 번호 → 전체 목록 번호
                st.session_state.yt_draft = {"rows": rows, "tidy": tidy}
                st.session_state.pop("yt_draft_editor", None)
                st.rerun()
    if y2.button("↩️ 기본 포인트 다시 넣기", use_container_width=True, key="yt_points_reset",
                 help="지금 목록 뒤에 처음 넣어 둔 기본 포인트 중 빠진 것만 다시 붙여요 (직접 쓴 포인트는 그대로)"):
        def _add_defaults(d):
            cur = qb_yt_points(d)  # DB 최신 목록 뒤에 빠진 기본 포인트만 붙임
            have = {str(p.get("text", "")).strip() for p in cur}
            d["yt_points"] = cur + [dict(p) for p in DEFAULT_YT_POINTS if p["text"].strip() not in have]
        if exam_dna_update(_add_defaults):
            st.session_state.pop("yt_points_editor", None)
            st.rerun()

    draft = st.session_state.get("yt_draft")
    if draft:
        with st.container(border=True):
            st.markdown("##### ✨ 정리한 버전 — 확인하고 고칠 곳을 고친 뒤 확정해 주세요")
            st.caption("새로 쓰거나 고친 포인트만 정리했어요. '정리한 포인트' 칸은 직접 고칠 수 있고, 마음에 안 들면 '내가 쓴 그대로'를 켜면 원래 쓴 말로 저장해요. "
                       "한 칸에 여러 포인트를 섞어 쓴 경우는 여러 줄로 나눠 정리했어요. '확정해서 저장'을 누르기 전에는 아무것도 바뀌지 않아요.")
            ddf = pd.DataFrame([{"원래 쓴 말": t["raw"], "정리한 포인트": t["text"], "적용 유형": t["scope"] if t["scope"] in QB_YT_SCOPES else "전체 유형",
                                 "교과서만": bool(t.get("tb")), "내가 쓴 그대로": False, "무엇을 고쳤나": t.get("note", "")} for t in draft["tidy"]])
            ded = st.data_editor(ddf, hide_index=True, use_container_width=True, key="yt_draft_editor", disabled=["원래 쓴 말", "무엇을 고쳤나"],
                                 column_config={
                                     "원래 쓴 말": st.column_config.TextColumn("원래 쓴 말", width="medium"),
                                     "정리한 포인트": st.column_config.TextColumn("정리한 포인트 ✏️", width="large"),
                                     "적용 유형": st.column_config.SelectboxColumn("적용 유형 ✏️", options=list(QB_YT_SCOPES), required=True),
                                     "교과서만": st.column_config.CheckboxColumn("교과서만", width="small"),
                                     "내가 쓴 그대로": st.column_config.CheckboxColumn("내가 쓴 그대로", width="small"),
                                     "무엇을 고쳤나": st.column_config.TextColumn("무엇을 고쳤나", width="medium")})
            k1, k2 = st.columns(2)
            if k1.button("✅ 확정해서 저장", type="primary", use_container_width=True, key="yt_draft_ok"):
                out = yt_merge_draft(draft["rows"], draft["tidy"], ded.to_dict("records"))
                if exam_dna_update(lambda d, o=out: d.__setitem__("yt_points", o)):
                    for k in ("yt_draft", "yt_points_editor", "yt_draft_editor"):
                        st.session_state.pop(k, None)
                    st.session_state.yt_saved_msg = f"✅ 영쌤 출제 포인트 {len(out)}개를 저장했어요. 다음에 시작하는 변형문제 작업부터 반영돼요."
                    st.rerun()
            if k2.button("✖️ 취소 (저장 안 함)", use_container_width=True, key="yt_draft_cancel"):
                st.session_state.pop("yt_draft", None)
                st.session_state.pop("yt_draft_editor", None)
                st.rerun()
    msg = st.session_state.pop("yt_saved_msg", None)
    if msg:
        st.success(msg)


SCHEMA_YT_TIDY = {
    "type": "object",
    "properties": {"points": {"type": "array", "items": {"type": "object", "properties": {
        "src": {"type": "integer", "description": "원래 쓴 말의 번호 (입력 목록의 번호)"},
        "text": {"type": "string", "description": "정리한 출제 포인트 (한국어 지시문)"},
        "scope": {"type": "string", "enum": list(QB_YT_SCOPES)},
        "tb": {"type": "boolean", "description": "교과서 지문에만 쓰는 포인트인지"},
        "note": {"type": "string", "description": "무엇을 고쳤는지 25자 이내"},
    }, "required": ["src", "text", "scope", "tb", "note"]}}},
    "required": ["points"],
}

def yt_tidy_points(rows: List[Dict], model: str):
    """선생님이 급히 쓴 출제 포인트 → 변형문제 출제 지시문으로 정리 (뜻·예시는 그대로, AI 1회).
    → [{src(rows 번호), raw, text, scope, tb, note}] / 실패하면 None"""
    examples = "\n".join(f"- [{p['scope']}] {p['text']}" for p in DEFAULT_YT_POINTS[:3])
    items = "\n".join(f"{i}. [지금 고른 유형: {r['scope']}{' · 교과서만' if r['tb'] else ''}] {r['text']}" for i, r in enumerate(rows))
    prompt = f"""너는 고등 영어 내신 출제 팀장이다. 아래는 선생님이 급하게 적은 '낚시' 출제 방식 메모다. 각 메모를 변형문제를 만드는 AI가 그대로 따라 할 수 있는 깔끔한 출제 지시문으로 정리하라.
규칙:
- 선생님의 뜻·의도·예시·단어를 빠뜨리거나 바꾸지 않는다. 메모에 없는 새 장치를 지어내지 않는다. 오타·띄어쓰기·뒤죽박죽인 순서만 바로잡고, 무엇을(어디에) 어떻게 바꾸는지, 학생이 왜 낚이는지가 드러나게 쓴다.
- 영어 예시 표현은 그대로 살려 괄호 안에 '(예: …)'로 둔다.
- 한 메모에 서로 다른 포인트가 여러 개 섞여 있으면 포인트마다 나눠 같은 src 번호로 여러 개 낸다. 하나뿐이면 하나만.
- scope는 이 포인트가 맞는 문제 유형 묶음을 고른다(지금 고른 유형이 맞으면 그대로). tb는 교과서 지문에서만 쓸 수 있는 포인트(같은 단원 다른 단락 내용 등)면 true.
- '~한다.'체로 1~3문장.
- note에는 무엇을 고쳤는지 짧게(예: 오타 수정, 두 포인트로 나눔, 예시 정리).

[이미 저장된 포인트 문체 예시]
{examples}

[정리할 메모]
{items}"""
    out = _qb_call(client, model, prompt, SCHEMA_YT_TIDY, 0.2, [], "영쌤 출제 포인트 정리")
    if not isinstance(out, dict) or not out.get("points"):
        return None
    tidy = []
    for t in out["points"]:
        i = t.get("src")
        if isinstance(i, int) and 0 <= i < len(rows) and str(t.get("text", "")).strip():
            tidy.append({"src": i, "raw": rows[i]["text"], "text": str(t["text"]).strip(), "scope": t.get("scope") or rows[i]["scope"],
                         "tb": bool(t.get("tb")) or rows[i]["tb"], "note": str(t.get("note", ""))[:40]})
    for i, r in enumerate(rows):  # AI가 빠뜨린 메모는 원래 쓴 말 그대로 보여 줌
        if not any(t["src"] == i for t in tidy):
            tidy.append({"src": i, "raw": r["text"], "text": r["text"], "scope": r["scope"], "tb": r["tb"], "note": "정리 못 함 — 원래 쓴 말"})
    return sorted(tidy, key=lambda t: t["src"])

def yt_merge_draft(rows: List[Dict], tidy: List[Dict], edited: List[Dict]) -> List[Dict]:
    """전체 목록(rows) 중 정리한 줄만 확인 표의 내용으로 바꿈 (목록 순서 유지, '내가 쓴 그대로'면 원래 줄)"""
    by_src = {}
    for t, e in zip(tidy, edited):
        by_src.setdefault(t["src"], []).append(e)
    out = []
    for i, r in enumerate(rows):
        if i not in by_src:
            out.append(r)
            continue
        es = by_src[i]
        if any(e.get("내가 쓴 그대로") for e in es):
            out.append(r)  # 여러 줄로 나눈 것 중 하나라도 '그대로'면 원래 한 줄로
            continue
        for e in es:
            txt = str(e.get("정리한 포인트") or "").strip()
            if txt and txt.lower() != "nan":
                tb = e.get("교과서만")
                out.append({"on": r["on"], "text": txt,
                            "scope": e.get("적용 유형") if e.get("적용 유형") in QB_YT_SCOPES else r["scope"],
                            "tb": bool(tb) if tb is not None and not (isinstance(tb, float) and tb != tb) else r["tb"]})
    return out


def page_past_voca():

    with st.container(border=True):
        up_past_pdfs = st.file_uploader(
            "📂 어휘를 추출할 기출 시험지 PDF 여러 개 동시 업로드 (수십 개 또는 분할 PDF 동시 선택 가능)",
            type=["pdf"],
            accept_multiple_files=True,
            key="past_pdf_multi_uploader"
        )
        p_col1, p_col2 = st.columns([2, 4])
        override_school = p_col1.text_input("🏫 학교명 강제 지정 (비워두면 파일명/시험지에서 자동 인식)", placeholder="예: 운양고 또는 제일고", key="past_override_school")

        if up_past_pdfs and p_col2.button(f"🚀 업로드된 {len(up_past_pdfs)}개 기출 시험지 어휘 일괄 추출 및 시험지별 통합 저장", type="primary", use_container_width=True):
            prog_bar = st.progress(0)
            status_box = st.empty()
            total_added_words = 0

            for idx, pdf_file in enumerate(up_past_pdfs):
                status_box.info(f"⏳ ({idx + 1}/{len(up_past_pdfs)}) '{pdf_file.name}' 시험지 본문 및 선지 어휘 정밀 추출 중...")
                if idx > 0:
                    time.sleep(api_delay)

                prompt_past = f"""너는 대한민국 고등학교 영어 내신 기출 시험지 분석 전문가다.
업로드된 기출 시험지 PDF(파일명: {pdf_file.name})의 모든 문항(1번~끝번, 서술형/논술형 포함)을 전수 조사하여 다음을 JSON으로 반환하라.

1. 출처 메타데이터 자동 파악:
   - school_name: 파일명이나 시험지 상단/하단에서 학교명을 파악해 '운양고', '제일고' 등으로 기재 (파일명에 '[운양1]'이 있으면 '운양고', '[제일2]'가 있으면 '제일고').
   - exam_year_term: 연도 및 학기/시험 종류 (예: '26년 1학기 중간', '26년 1학기 기말', '26년 2학기 중간').
   - grade: 학년 (예: '고1', '고2', '고3').
2. 문항별(questions) 중고등 핵심 어휘 및 숙어 추출:
   - question_number: 문항 번호 (예: '1번', '2번', ..., '24번', '논술형1').
   - words: 해당 문항의 **지문 본문**과 **객관식 선지(1~5번 및 <보기>)**에 등장한 중고등 필수 어휘·숙어·패러프레이징 어휘를 빠짐없이 추출하라.
   - 초등 기초 단어(be동사, a, the, beautiful, happy, good 등)는 절대 제외하라.
   - 각 단어의 source_part에는 본문에서 나왔으면 '기출본문', 문제 선지나 <보기>/요약문에서 나왔으면 '기출선지'라고 정확히 구분해 적어라.
   - pronunciation은 반드시 자연스러운 '한글 발음'(IPA 금지)으로 표기하고, 문맥에 맞는 유의어(synonyms 최대 2개)와 반의어(antonyms 1~2개)도 각각 한글발음과 뜻을 포함해 작성하라. {SYN_RULE} 맞는 유의어가 없으면 빈 배열."""

                for attempt in range(4):
                    try:
                        parsed_pdf = llm_json(text_model, [pdf_to_part(pdf_file), prompt_past], SCHEMA_PAST_PDF_VOCA, 0.2)
                        sch_n = override_school.strip() if override_school.strip() else parsed_pdf.get("school_name", "기출").strip()
                        yr_term = parsed_pdf.get("exam_year_term", "").strip()
                        grd = parsed_pdf.get("grade", "").strip()

                        # 시험지 단위로 통합 라벨 생성 (예: [운양고 26년 1학기 중간 고1])
                        exam_title_clean = re.sub(r'\s+', ' ', f"{sch_n} {yr_term} {grd}").strip()
                        exam_full_label = clean_label(f"[{exam_title_clean}]")

                        exam_words = []
                        for q_obj in parsed_pdf.get("questions", []):
                            q_num = str(q_obj.get("question_number", "")).strip()
                            if not q_num:
                                continue
                            for w in q_obj.get("words", []):
                                src_part = str(w.get("source_part", "기출본문")).strip()
                                exam_words.append({
                                    "word": str(w.get("word", "")).strip(),
                                    "pos": str(w.get("pos", "")).strip(),
                                    "pronunciation": str(w.get("pronunciation", "")).strip(),
                                    "meaning": str(w.get("meaning", "")).strip(),
                                    "category": f"{exam_title_clean} {q_num} [{src_part}]",
                                    "synonyms": w.get("synonyms", []),
                                    "antonyms": w.get("antonyms", [])
                                })

                        if exam_words:
                            st.session_state.saved_past_voca.append({
                                "id": str(uuid.uuid4()),
                                "label": exam_full_label,
                                "data": {"voca_list": exam_words},
                                "body_text": f"{exam_full_label} 기출 본문 및 선지 통합 어휘",
                                "academy_name": academy_name,
                                "school_name": sch_n
                            })
                            total_added_words += len(exam_words)
                            # 동일 시험지(예: 전반부/후반부 분할 PDF)가 있으면 자동으로 하나의 시험지로 합본 및 문항순 정렬
                            st.session_state.saved_past_voca, _ = consolidate_past_voca_records(st.session_state.saved_past_voca)
                            save_to_file(PAST_VOCA_FILE, st.session_state.saved_past_voca)
                        break
                    except Exception as e:
                        err_s = str(e).lower()
                        if attempt < 3 and any(k in err_s for k in ["503", "429", "overloaded", "unavailable", "quota"]):
                            wait_t = 10 + attempt * 10
                            st.toast(f"⏳ 서버 대기 감지 ({pdf_file.name}). {wait_t}초 후 재시도...")
                            time.sleep(wait_t)
                            continue
                        st.error(f"❌ '{pdf_file.name}' 처리 중 오류: {e}")
                        break

                prog_bar.progress((idx + 1) / len(up_past_pdfs))

            status_box.success(f"✅ 총 {len(up_past_pdfs)}개 파일에서 {total_added_words}개의 기출 어휘가 시험지 단위로 통합 저장되었습니다!")
            prog_bar.empty()
            st.rerun()

    st.markdown("---")
    render_library_manager("past_v", "기출 어휘 단어장", PAST_VOCA_FILE, CSS_VOCA, "past_voca", text_model, api_delay, academy_name, school_name, h_url, f_url)



# ---- 전체 백업 · 되살리기 (설정 화면) ----
LIB_SESSION_KEYS = {VICTORY_FILE: "saved_v", ANATOMY_FILE: "saved_a", CORE_FILE: "saved_c", VOCA_FILE: "saved_w",
                    MOCK_FILE: "saved_mock", PAST_VOCA_FILE: "saved_past_voca"}
LIB_NAMES = {VICTORY_FILE: "One-Shot(한판승)", ANATOMY_FILE: "Self-Note", CORE_FILE: "Core", VOCA_FILE: "Voca",
             MOCK_FILE: "원문 라이브러리", PAST_VOCA_FILE: "기출 어휘"}
BACKUP_SKIP_DOCS = {"gen_job"}  # 진행 중 작업 상태는 되살리지 않음 (백업에는 넣음)

def build_full_backup(with_images: bool = False) -> (bytes, str, str):
    """DB의 모든 문서 + 변형문제 은행 (+ 이미지) → ZIP (파일 내용, 파일 이름, 요약)"""
    buf = io.BytesIO()
    counts = []
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if IS_CLOUD_DB:
            for k in sorted(_doc_versions()):
                res = supabase.table("app_storage").select("content").eq("doc_key", k).execute()
                content = res.data[0].get("content") if res.data else None
                zf.writestr(f"{k}.json", json.dumps(content, ensure_ascii=False))
                if isinstance(content, list):
                    counts.append(f"{k} {len(content)}개")
            rows, start = [], 0
            while True:
                r = supabase.table(QB_TABLE).select("*").order("id").range(start, start + 999).execute()
                rows += r.data or []
                if len(r.data or []) < 1000:
                    break
                start += 1000
        else:
            for fp in glob_local_docs():
                with open(fp, "r", encoding="utf-8") as f:
                    zf.writestr(os.path.basename(fp), f.read())
            rows = _qb_local_load()
        zf.writestr("question_bank.json", json.dumps(rows, ensure_ascii=False))
        counts.append(f"변형문제 {len(rows):,}개")
        n_img = 0
        if with_images:
            names = []
            if IS_CLOUD_DB:
                off = 0
                while True:
                    lst = supabase.storage.from_("image_assets").list("", {"limit": 1000, "offset": off}) or []
                    names += [x["name"] for x in lst if x.get("name") and x.get("id")]
                    if len(lst) < 1000:
                        break
                    off += 1000
            elif os.path.isdir(IMAGE_DIR):
                names = os.listdir(IMAGE_DIR)
            prog = st.progress(0.0, text="이미지 받는 중")
            for i, nm in enumerate(names):
                try:
                    if IS_CLOUD_DB:
                        raw = supabase.storage.from_("image_assets").download(nm)
                    else:
                        with open(os.path.join(IMAGE_DIR, nm), "rb") as f:
                            raw = f.read()
                    zf.writestr(f"image_assets/{nm}", raw)
                    n_img += 1
                except Exception:
                    pass
                prog.progress((i + 1) / max(1, len(names)), text=f"이미지 받는 중 {i + 1}/{len(names)}")
            prog.empty()
            counts.append(f"이미지 {n_img}개")
    stamp = time.strftime("%Y%m%d_%H%M", time.gmtime(time.time() + 9 * 3600))
    return buf.getvalue(), f"4pm_backup_{stamp}.zip", " · ".join(counts)

def glob_local_docs() -> List[str]:
    """로컬 PC 모드에서 백업할 JSON 파일들"""
    skip = {"question_bank.json"}
    return [os.path.join(BASE_DIR, f) for f in sorted(os.listdir(BASE_DIR)) if f.endswith(".json") and f not in skip
            and not f.startswith(".")]

def _mig_target(name: str, exact: bool):
    """파일 이름 → 교재 파일 (백업 ZIP 안은 문서 이름 그대로만, 따로 올린 JSON은 예전처럼 이름에 든 글자로)"""
    base = os.path.splitext(os.path.basename(name))[0].lower()
    by_key = {filepath_to_doc_key(fp): fp for fp in LIB_SESSION_KEYS}
    if base in by_key:
        return by_key[base]
    if exact:
        return None
    for word, fp in (("victory", VICTORY_FILE), ("anatomy", ANATOMY_FILE), ("core", CORE_FILE), ("mock", MOCK_FILE),
                     ("past", PAST_VOCA_FILE), ("voca", VOCA_FILE)):
        if word in base:
            return fp
    return None

def migration_read_uploads(files) -> (Dict, List[Dict], Dict, List[str]):
    """올린 JSON·백업 ZIP → ({교재 파일: {"name", "data"}}, 변형문제 행, 그 밖의 문서 {doc_key: 내용}, 문제 메시지)"""
    plan, qb_rows, others, bad = {}, [], {}, []

    def take(name, raw, exact):
        try:
            data = json.loads(raw)
        except Exception:
            bad.append(f"{name}: JSON으로 읽을 수 없어 건너뜀")
            return
        base = os.path.splitext(os.path.basename(name))[0]
        if base == "question_bank":
            if isinstance(data, list):
                qb_rows.extend(r for r in data if isinstance(r, dict) and r.get("id"))
            return
        fp = _mig_target(name, exact)
        if fp:
            if isinstance(data, list):
                plan[fp] = {"name": os.path.basename(name), "data": data}
            else:
                bad.append(f"{name}: 지문 목록 형식이 아니라 건너뜀")
        elif exact and base not in BACKUP_SKIP_DOCS and data is not None:
            others[base] = data
        elif not exact:
            bad.append(f"{name}: 어느 교재 파일인지 몰라 건너뜀 (파일 이름에 victory·anatomy·core·voca·past·mock 중 하나가 있어야 해요)")

    for uf in files:
        if uf.name.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(uf.getvalue())) as zf:
                    for info in zf.infolist():
                        if not info.is_dir() and info.filename.lower().endswith(".json") and "/" not in info.filename.strip("/"):
                            take(info.filename, zf.read(info.filename), True)
            except Exception as e:
                bad.append(f"{uf.name}: ZIP을 열 수 없음 ({str(e)[:80]})")
        else:
            take(uf.name, uf.getvalue(), False)
    return plan, qb_rows, others, bad

def migration_new_items(cur: List[Dict], incoming: List[Dict]) -> List[Dict]:
    """지금 목록에 없는 지문 (id도 지문 이름도 겹치지 않는 것)"""
    ids = {p.get("id") for p in cur if isinstance(p, dict)}
    labels = {p.get("label") for p in cur if isinstance(p, dict)}
    out = []
    for p in incoming:
        if isinstance(p, dict) and p.get("id") not in ids and p.get("label") not in labels:
            out.append(p)
            labels.add(p.get("label"))
    return out

def migration_restore_qbank(rows: List[Dict]) -> int:
    """백업의 변형문제 중 지금 은행에 없는 것만 다시 넣음 → 넣은 수"""
    sb = supabase if IS_CLOUD_DB else None
    if sb is None:
        have = {r.get("id") for r in _qb_local_load()}
    else:
        have, start = set(), 0
        while True:
            r = sb.table(QB_TABLE).select("id").order("id").range(start, start + 999).execute()
            have |= {x["id"] for x in (r.data or [])}
            if len(r.data or []) < 1000:
                break
            start += 1000
    cols = ("id", "created_at", "passage_label", "exam_key", "number", "q_type", "difficulty", "school", "content")
    todo = [{k: r[k] for k in cols if k in r} for r in rows if r.get("id") not in have]
    for i in range(0, len(todo), 200):
        chunk = todo[i:i + 200]
        if sb is None:
            qb_insert(None, chunk)
        else:
            sb.table(QB_TABLE).insert(chunk, **_UPSERT_MIN).execute()
    return len(todo)

def migration_restore_docs(docs: Dict) -> int:
    """설정·출제 DNA 등 그 밖의 문서: DB에 없는 것만 되살림 → 되살린 수"""
    have = set(_doc_versions()) if IS_CLOUD_DB else {os.path.splitext(f)[0] for f in os.listdir(BASE_DIR)}
    n = 0
    for k, content in docs.items():
        if k in have:
            continue
        if IS_CLOUD_DB:
            _doc_upsert(supabase, k, content)
        with open(os.path.join(BASE_DIR, f"{k}.json"), "w", encoding="utf-8") as f:
            json.dump(content, f, ensure_ascii=False, indent=2)
        n += 1
    return n


def page_settings():
    settings_changed = False
    with st.container(border=True):
        st.markdown("##### 🏫 교재에 들어가는 정보")
        st.caption("표지·머리말·꼬리말에 들어가요. 바꾸면 바로 저장되고, 다시 바꾸기 전까지 유지돼요.")
        s1, s2 = st.columns(2)
        new_academy = s1.text_input("학원명", academy_name, key="set_academy_name")
        new_school = s2.text_input("학교명 (표지 및 ANATOMY용)", school_name, key="set_school_name")
        for k, v in [("academy_name", new_academy), ("school_name", new_school)]:
            if app_settings.get(k) != v:
                app_settings[k] = v
                settings_changed = True

    with st.container(border=True):
        st.markdown("##### 🖼️ 로고")
        st.caption("한 번 올린 로고는 자동 저장돼요. 올리지 않은 자리는 기본 로고가 들어가요.")
        lcols = st.columns(3)
        for col, (slot, label) in zip(lcols, [("h", "머리말 로고"), ("f", "꼬리말 로고"), ("e", "영쌤 로고")]):
            with col:
                up = st.file_uploader(label, type=["png", "jpg", "jpeg"], key=slot)
                if up is not None:
                    # 같은 파일이 화면을 다시 그릴 때마다 저장되지 않도록 내용 해시로 비교
                    up_hash = hashlib.md5(up.getvalue()).hexdigest()
                    if app_settings.get(f"logo_{slot}_hash") != up_hash:
                        app_settings[f"logo_{slot}"] = logo_upload_to_data_url(up)
                        app_settings[f"logo_{slot}_hash"] = up_hash
                        settings_changed = True
                elif app_settings.get(f"logo_{slot}"):
                    st.image(app_settings[f"logo_{slot}"], width=56, caption=f"저장된 {label}")
        if any(app_settings.get(f"logo_{s}") for s in "hfe"):
            if st.button("🔄 저장된 로고 모두 기본값으로", key="reset_logos"):
                for s in "hfe":
                    app_settings.pop(f"logo_{s}", None)
                    app_settings.pop(f"logo_{s}_hash", None)
                save_app_settings(app_settings)
                st.rerun()

    with st.container(border=True):
        st.markdown("##### ⏱️ 생성 속도")
        new_delay = st.slider(
            "지문 간 API 대기 시간(초)", min_value=1, max_value=10, value=api_delay, key="set_api_delay",
            help="유료 Tier 1 이상이면 1~2초로 줄여 생성 속도를 높일 수 있습니다. 너무 줄이면 제공사가 요청을 막을 수 있어요."
        )
        if app_settings.get("api_delay") != new_delay:
            app_settings["api_delay"] = new_delay
            settings_changed = True
        if not AI_KEYS.get("gemini"):
            st.caption("ℹ️ AI 이미지 생성은 Gemini 키가 있어야 쓸 수 있습니다.")

    if settings_changed:
        save_app_settings(app_settings)

    with st.container(border=True):
        st.markdown(f"##### ☁️ 저장소 · {'클라우드 DB 연결됨 🟢' if IS_CLOUD_DB else '로컬 PC 모드 🟡'}")
        st.markdown("**💾 전체 백업 받기** (AI 비용 없음)")
        st.caption("교재 4종·원문 라이브러리·기출 어휘·설정·출제 DNA·기출 지도·클리닉 기록과 변형문제 은행 전체를 ZIP 파일 하나로 받아요. "
                   "한 달에 한 번쯤, 그리고 큰 작업(데이터 이식·대량 삭제) 전에 받아 두면 아래 '되살리기'로 돌릴 수 있어요.")
        bk1, bk2 = st.columns([1.3, 2])
        bk_img = bk2.checkbox("🖼️ 이미지 파일도 넣기 (시간이 더 걸리고 파일이 커져요)", value=False, key="bk_with_images")
        if bk1.button("📦 백업 파일 만들기", use_container_width=True, key="bk_build"):
            with st.spinner("DB에서 모든 데이터를 모으는 중이에요..."):
                try:
                    st.session_state["bk_zip"] = build_full_backup(bk_img)
                except Exception as e:
                    st.error(f"⚠️ 백업을 만들지 못했어요: {str(e)[:200]}")
        if st.session_state.get("bk_zip"):
            data, fname, summary = st.session_state["bk_zip"]
            st.download_button(f"📥 {fname} 받기 ({len(data) / 1_000_000:.1f}MB)", data, fname, "application/zip",
                               type="primary", use_container_width=True, key="bk_dl")
            st.caption(summary)

        st.divider()
        st.markdown("**📥 백업·JSON 되살리기 (데이터 이식)**")
        st.caption("💡 이미지는 Supabase의 [Storage -> image_assets 버킷]에 직접 드래그해 올리거나, 구글 드라이브에서 받은 ZIP 파일 1개로 한 번에 올릴 수 있습니다. "
                   "위에서 받은 백업 ZIP은 1)에 올리면 교재·원문·변형문제를, 2)에 올리면 이미지를 되살려요.")
        up_jsons = st.file_uploader(
            "1) JSON 파일 또는 전체 백업 ZIP 업로드 (victory, anatomy, core, voca, past_voca, mock)",
            type=["json", "zip"],
            accept_multiple_files=True,
            key="cloud_mig_json"
        )
        up_zip = st.file_uploader(
            "2) [선택] image_assets.zip 압축 파일 1개로 통째 업로드",
            type=["zip"],
            key="cloud_mig_zip"
        )
        plan, qb_rows, other_docs, bad = migration_read_uploads(up_jsons or [])
        for b in bad:
            st.warning(f"⚠️ {b}")
        mig_mode = "add"
        confirm_ok = True
        if plan:
            mig_mode = st.radio("되살리는 방식", ["add", "replace"], horizontal=True, key="mig_mode",
                                format_func=lambda m: {"add": "➕ 없는 지문만 추가 (지금 데이터는 그대로)",
                                                       "replace": "♻️ 파일 내용으로 전체 바꾸기"}[m])
            rows_view = []
            for fp, info in plan.items():
                cur = st.session_state.get(LIB_SESSION_KEYS[fp]) or []
                new_n = len(migration_new_items(cur, info["data"]))
                rows_view.append({"교재": LIB_NAMES[fp], "파일": info["name"], "지금": len(cur), "파일 속": len(info["data"]),
                                  "없는 지문만 추가": f"+{new_n}개",
                                  "전체 바꾸기": f"{len(cur)}개 → {len(info['data'])}개"})
            st.dataframe(pd.DataFrame(rows_view), hide_index=True, use_container_width=True)
            if mig_mode == "replace":
                shrink = [r for r in rows_view if r["파일 속"] < r["지금"]]
                st.warning("♻️ 전체 바꾸기는 위 교재의 지금 목록을 파일 내용으로 통째로 바꿔요. 파일에 없는 지문은 지워져요."
                           + (f" 특히 {', '.join(r['교재'] for r in shrink)}은(는) 지금보다 줄어들어요." if shrink else ""))
                confirm_ok = st.checkbox("위 표의 '전체 바꾸기' 개수대로 바뀌는 것을 확인했어요", key="mig_confirm")
        qb_restore = bool(qb_rows) and st.checkbox(f"🧩 변형문제 은행: 백업의 {len(qb_rows):,}문제 중 지금 없는 문제만 다시 넣기", value=True, key="mig_qb")
        docs_restore = bool(other_docs) and st.checkbox(
            f"⚙️ 그 밖의 문서 {len(other_docs)}개(설정·출제 DNA·기출 지도 등) 중 DB에 없는 것만 되살리기 (있는 문서는 그대로)", value=True, key="mig_docs")
        if st.button("🚀 클라우드 DB로 데이터 이식 실행", type="primary", use_container_width=True,
                     disabled=not (up_zip is not None or plan or qb_restore or docs_restore) or not confirm_ok):
            with st.spinner("데이터를 클라우드 DB에 동기화 중입니다..."):
                if up_zip is not None:
                    with zipfile.ZipFile(up_zip) as zf:
                        img_entries = [info for info in zf.infolist() if not info.is_dir() and info.filename.lower().endswith((".png", ".jpg", ".jpeg"))]
                        prog = st.progress(0)
                        for idx, info in enumerate(img_entries):
                            fname = os.path.basename(info.filename)
                            if not fname or fname.startswith("."):
                                continue
                            raw_bytes = zf.read(info.filename)
                            with open(os.path.join(IMAGE_DIR, fname), "wb") as f:
                                f.write(raw_bytes)
                            if IS_CLOUD_DB:
                                try:
                                    supabase.storage.from_("image_assets").upload(
                                        path=fname,
                                        file=raw_bytes,
                                        file_options={"content-type": "image/png", "upsert": "true"}
                                    )
                                except Exception:
                                    pass
                            prog.progress((idx + 1) / max(1, len(img_entries)))
                        prog.empty()

                done_msgs = []
                for fp, info in plan.items():
                    key = LIB_SESSION_KEYS[fp]
                    j_data, _ = normalize_passage_records(copy.deepcopy(info["data"]))
                    if fp == PAST_VOCA_FILE:
                        j_data, _ = consolidate_past_voca_records(j_data)
                    if mig_mode == "replace":
                        st.session_state[key] = j_data
                        save_to_file(fp, st.session_state[key], replace=True)
                        done_msgs.append(f"{LIB_NAMES[fp]} {len(j_data)}개로 바꿈")
                    else:
                        cur = st.session_state.get(key) or []
                        new_items = migration_new_items(cur, j_data)
                        if new_items:
                            cur.extend(new_items)
                            st.session_state[key] = cur
                            save_to_file(fp, st.session_state[key])
                        done_msgs.append(f"{LIB_NAMES[fp]} +{len(new_items)}개")
                if qb_restore:
                    n_qb = migration_restore_qbank(qb_rows)
                    done_msgs.append(f"변형문제 +{n_qb:,}개")
                if docs_restore:
                    n_doc = migration_restore_docs(other_docs)
                    done_msgs.append(f"문서 {n_doc}개 되살림")
                st.cache_data.clear()
                st.session_state["mig_done"] = "✅ 데이터 이식을 마쳤어요" + (f" · {' · '.join(done_msgs)}" if done_msgs else "")
                st.rerun()
        if st.session_state.get("mig_done"):
            st.success(st.session_state.pop("mig_done"))


# =============================
# ✅ 배포 전 검수 (2026-11-10): 조교샘이 교재·PPT를 내보내기 전에 지문마다 유의어와 정답만 훑어보는 화면 (AI 비용 없음)
#   - 유의어: Voca 단어장 · 한판승 키워드 · Self-Note 어휘 · Core 핵심 어휘 · Core 요약문 (A)(B) → PPT·교재에 나오는 유의어가 모두 여기서 옴
#     같은 지문·같은 단어·같은 유의어는 한 줄로 묶고, '틀림'으로 지우면 묶인 교재 모두에서 빠짐 (다음 출력·PPT부터 반영)
#   - 정답: Core 요약문 (A)(B) 정답, Core <보기>형 진위 O/X, Self-Note OX 퀴즈 → 고친 정답은 저장
#   - 한국어 뜻이 한 글자도 안 겹치는 유의어(예: enzyme 효소 ≈ protein 단백질)는 '먼저 볼 것'으로 표시
# =============================
REVIEW_SRC = {"w": "Voca", "v": "한판승", "a": "Self-Note", "c": "Core 어휘", "cs": "Core 요약문"}
REVIEW_LIBS = {"w": ("saved_w", VOCA_FILE), "v": ("saved_v", VICTORY_FILE), "a": ("saved_a", ANATOMY_FILE),
               "c": ("saved_c", CORE_FILE), "cs": ("saved_c", CORE_FILE)}
REVIEW_STOP = set("하다는은을를의에이가적인한되게고있것수기함성로으와과도된할들적")
REVIEW_FROM_SHARED = "다른 교재 화면에서 고른 지문"

def review_split_rel(s) -> (List[str], str):
    """'a(뜻), b(뜻) / [함정]c(뜻)' → (['a(뜻)', 'b(뜻)'], ' / [함정]c(뜻)')"""
    s = str(s or "").strip()
    m = re.search(r"\s*/\s*(?=\[함정\])", s)
    head, tail = (s[:m.start()], s[m.start():]) if m else (s, "")
    return [t.strip() for t in re.split(r",\s*(?![^()\[\]]*[)\]])", head) if t.strip()], tail

def review_join_rel(toks: List[str], tail: str) -> str:
    if not toks:
        return re.sub(r"^\s*/\s*", "", tail)  # 유의어를 다 지운 요약문은 '[함정]오답(뜻)'만
    return ", ".join(toks) + tail

def review_tok(t: str) -> (str, str):
    """'gather(모이다)' · 'gather[개더](모이다)' → ('gather', '모이다')"""
    m = re.match(r"\s*([^(\[]*?)\s*(?:\[[^\]]*\])?\s*(?:\((.*)\))?\s*$", str(t))
    return ((m.group(1) or "").strip(), (m.group(2) or "").strip()) if m else (str(t).strip(), "")

def review_suspect(m1, m2) -> bool:
    """두 한국어 뜻이 한 글자도 안 겹치면 '먼저 볼 것' (조사·'하다' 같은 흔한 글자는 빼고 셈, 뜻이 비면 판단 안 함)"""
    a = {ch for ch in str(m1) if "가" <= ch <= "힣"} - REVIEW_STOP
    b = {ch for ch in str(m2) if "가" <= ch <= "힣"} - REVIEW_STOP
    return bool(a and b and not (a & b))

def review_group_of(label: str) -> str:
    m = re.match(r"\s*\[([^\]]+)\]", str(label))
    return m.group(1).strip() if m else extract_group_prefix(label)

def review_short(label: str) -> str:
    return re.sub(r"^\s*\[[^\]]*\]\s*", "", str(label)) or str(label)

def review_collect(labels: List[str]) -> (List[Dict], List[Dict]):
    """고른 지문들의 유의어 행(한 교재 한 칸씩)과 정답 행 — 행마다 고칠 위치(src, i, tag) 포함"""
    by = {k: {p.get("label"): p for p in st.session_state[sk]} for k, (sk, _) in REVIEW_LIBS.items() if k != "cs"}
    syn, ans = [], []
    def add(lbl, src, i, word, meaning, sw, sm, tag=""):
        if str(sw).strip() and str(word).strip():
            syn.append({"label": lbl, "src": src, "i": i, "tag": tag, "word": str(word).strip(), "meaning": str(meaning or "").strip(),
                        "syn": str(sw).strip(), "syn_meaning": str(sm or "").strip()})
    for lbl in labels:
        d = (by["w"].get(lbl) or {}).get("data") or {}
        for i, it in enumerate(d.get("voca_list") or []):
            for x in it.get("synonyms") or []:
                if isinstance(x, dict):
                    add(lbl, "w", i, it.get("word"), it.get("meaning"), x.get("word"), x.get("meaning"))
        d = (by["v"].get(lbl) or {}).get("data") or {}
        for i, kw in enumerate(d.get("keywords") or []):
            for t in review_split_rel(kw.get("synonyms"))[0]:
                add(lbl, "v", i, kw.get("word"), kw.get("meaning"), *review_tok(t))
        d = (by["a"].get(lbl) or {}).get("data") or {}
        for i, v in enumerate(d.get("vocab") or []):
            for t in review_split_rel(v.get("synonyms_text"))[0]:
                add(lbl, "a", i, v.get("word"), v.get("meaning"), *review_tok(t))
        for i, q in enumerate(d.get("quizzes") or []):
            ans.append({"label": lbl, "src": "a", "i": i, "field": "answer", "kind": "Self-Note OX",
                        "text": q.get("english_statement", ""), "answer": q.get("answer", ""), "note": q.get("korean_statement", "")})
        d = (by["c"].get(lbl) or {}).get("data") or {}
        for i, cv in enumerate(d.get("core_vocab") or []):
            for x in cv.get("synonyms") or []:
                if isinstance(x, dict):
                    add(lbl, "c", i, cv.get("word"), cv.get("meaning"), x.get("word"), x.get("meaning"))
        for i, sm in enumerate(d.get("summaries") or []):
            for tag in ("a", "b"):
                for t in review_split_rel(sm.get(f"synonyms_{tag}"))[0]:
                    add(lbl, "cs", i, sm.get(f"blank_{tag}"), "", *review_tok(t), tag=tag)
                ans.append({"label": lbl, "src": "c", "i": i, "field": f"blank_{tag}", "kind": f"Core 요약문 ({tag.upper()})",
                            "text": sm.get("eng", ""), "answer": sm.get(f"blank_{tag}", ""), "note": "함정: " + str(sm.get(f"distractor_{tag}", ""))})
        for i, tf in enumerate(d.get("paraphrased_tf") or []):
            ans.append({"label": lbl, "src": "c", "i": i, "field": "tf", "kind": "Core 진위 O/X",
                        "text": tf.get("eng", ""), "answer": tf.get("answer", ""), "note": tf.get("point", "")})
    return syn, ans

def review_group_syn(syn: List[Dict]) -> List[Dict]:
    """같은 지문·같은 단어·같은 유의어는 한 줄로 (어느 교재에 있는지 함께 표시)"""
    out, seen = [], {}
    for r in syn:
        k = (r["label"], r["word"].lower(), r["syn"].lower())
        if k not in seen:
            seen[k] = dict(r, srcs=[REVIEW_SRC[r["src"]]])
            out.append(seen[k])
        else:
            g = seen[k]
            if REVIEW_SRC[r["src"]] not in g["srcs"]:
                g["srcs"].append(REVIEW_SRC[r["src"]])
            g["meaning"] = g["meaning"] or r["meaning"]
            g["syn_meaning"] = g["syn_meaning"] or r["syn_meaning"]
    for g in out:
        g["suspect"] = review_suspect(g["meaning"], g["syn_meaning"])
    return out

def review_delete_syns(picked: List[Dict], fixes: Dict = None) -> int:
    """'틀림'으로 고른 (지문, 단어, 유의어)를 그 지문의 모든 교재에서 지움 → 바뀐 교재만 저장. 고친 칸 수
    fixes: {(지문, 단어, 틀린 유의어): (바꿀 유의어, 한글 발음, 뜻)} → 지우는 대신 그 자리에 바꿔 넣음 (이미 있는 말이면 지우기만)"""
    keys = {(r["label"], r["word"].lower(), r["syn"].lower()) for r in picked}
    fixes = fixes or {}
    by = {k: {p.get("label"): p for p in st.session_state[sk]} for k, (sk, _) in REVIEW_LIBS.items()}
    fresh, _ = review_collect(sorted({r["label"] for r in picked}))
    n, dirty = 0, set()
    for r in fresh:
        if (r["label"], r["word"].lower(), r["syn"].lower()) not in keys:
            continue
        d = (by[r["src"]].get(r["label"]) or {}).get("data") or {}
        sw = r["syn"].lower()
        fx = fixes.get((r["label"], r["word"].lower(), sw))
        if r["src"] in ("w", "c"):
            it = (d.get("voca_list" if r["src"] == "w" else "core_vocab") or [])[r["i"]]
            old = [x for x in it.get("synonyms") or [] if isinstance(x, dict)]
            have = {str(x.get("word", "")).strip().lower() for x in old} | {r["word"].lower()}
            new = []
            for x in old:
                if str(x.get("word", "")).strip().lower() != sw:
                    new.append(x)
                elif fx and fx[0].strip() and fx[0].strip().lower() not in have:
                    new.append({"word": fx[0].strip(), "pronunciation": fx[1].strip(), "meaning": fx[2].strip()})
                    have.add(fx[0].strip().lower())
            it["synonyms"] = new
        else:
            obj, field = {"v": (lambda: (d["keywords"][r["i"]], "synonyms")), "a": (lambda: (d["vocab"][r["i"]], "synonyms_text")),
                          "cs": (lambda: (d["summaries"][r["i"]], f"synonyms_{r['tag']}"))}[r["src"]]()
            toks, tail = review_split_rel(obj.get(field))
            have = {review_tok(t)[0].lower() for t in toks} | {r["word"].lower()}
            new = []
            for t in toks:
                if review_tok(t)[0].lower() != sw:
                    new.append(t)
                elif fx and fx[0].strip() and fx[0].strip().lower() not in have:
                    new.append(f"{fx[0].strip()}({fx[2].strip()})")
                    have.add(fx[0].strip().lower())
            obj[field] = review_join_rel(new, tail)
        n += 1
        dirty.add(r["src"])
    for k in {REVIEW_LIBS[x] for x in dirty}:
        save_to_file(k[1], st.session_state[k[0]])
    return n

def review_save_answers(changes: List[tuple]) -> int:
    """고친 정답 저장. 요약문 (A)(B) 정답을 바꾸면 요약문 문장 속 '(A) 정답'도 같이 바꿈"""
    by = {"a": {p.get("label"): p for p in st.session_state.saved_a}, "c": {p.get("label"): p for p in st.session_state.saved_c}}
    dirty, n = set(), 0
    for a, new in changes:
        d = (by[a["src"]].get(a["label"]) or {}).get("data") or {}
        if a["kind"] == "Self-Note OX":
            obj, field = (d.get("quizzes") or [])[a["i"]], "answer"
        elif a["field"] == "tf":
            obj, field = (d.get("paraphrased_tf") or [])[a["i"]], "answer"
        else:
            obj, field = (d.get("summaries") or [])[a["i"]], a["field"]
            tag = field[-1].upper()
            old = str(obj.get(field, "")).strip()
            eng = str(obj.get("eng", ""))
            mk = re.search(rf"\({tag}\)", eng)
            if old and mk:  # '(A) **influence**' · '(A) culturally fabricated'처럼 (A) 뒤에서 처음 나오는 옛 정답을 바꿈
                obj["eng"] = eng[:mk.end()] + re.sub(rf"(?<![A-Za-z]){re.escape(old)}(?![A-Za-z])", lambda _m: new, eng[mk.end():], count=1)
        obj[field] = new.upper() if field == "answer" else new
        dirty.add(a["src"])
        n += 1
    for src in dirty:
        sk, fp = REVIEW_LIBS[src]
        save_to_file(fp, st.session_state[sk])
    return n

def review_print_html(labels: List[str], groups: List[Dict], ans: List[Dict]) -> str:
    """인쇄용 체크리스트 (지문마다 유의어 □ · 정답 □ · 확인자 칸)"""
    css = """body{font-family:'Malgun Gothic','Apple SD Gothic Neo',sans-serif;font-size:11px;color:#111;margin:14mm}
h1{font-size:17px;margin:0 0 4px}.sub{color:#555;margin-bottom:10px}
.ps{page-break-inside:avoid;border:1px solid #bbb;border-radius:6px;padding:8px 10px;margin:0 0 10px}
.ps h2{font-size:13px;margin:0 0 6px}table{border-collapse:collapse;width:100%;margin:2px 0 6px}
td,th{border-bottom:1px solid #e3e3e3;padding:2px 4px;text-align:left;vertical-align:top}th{background:#f4f6f8;font-size:10px}
.ck{width:14px;font-size:13px}.sus{color:#b45309;font-weight:700}.src{color:#777;font-size:9.5px}.sign{margin-top:4px;color:#555}
@page{size:A4;margin:10mm}"""
    out = [f"<html><head><meta charset='utf-8'><title>배포 전 검수 체크리스트</title><style>{css}</style></head><body>",
           f"<h1>✅ 배포 전 검수 체크리스트</h1><div class='sub'>{esc(time.strftime('%Y-%m-%d'))} · 지문 {len(labels)}개 · "
           f"유의어 {len(groups)}개 · 정답 {len(ans)}개 · 시험 선지에서 바꿔 써도 뜻이 같은 말인지만 보세요. 틀리면 ✗, ⚠ 표시는 먼저 보세요.</div>"]
    for lbl in labels:
        gs = [g for g in groups if g["label"] == lbl]
        aa = [a for a in ans if a["label"] == lbl]
        if not gs and not aa:
            continue
        out.append(f"<div class='ps'><h2>{esc(lbl)}</h2>")
        if gs:
            out.append("<table><tr><th class='ck'>□</th><th>단어</th><th>유의어</th><th>교재</th></tr>")
            for g in sorted(gs, key=lambda g: not g["suspect"]):
                out.append(f"<tr><td class='ck'>□</td><td>{esc(g['word'])} <span class='src'>{esc(g['meaning'])}</span></td>"
                           f"<td>{'<span class=sus>⚠ </span>' if g['suspect'] else ''}{esc(g['syn'])} <span class='src'>{esc(g['syn_meaning'])}</span></td>"
                           f"<td class='src'>{esc('·'.join(g['srcs']))}</td></tr>")
            out.append("</table>")
        if aa:
            out.append("<table><tr><th class='ck'>□</th><th>종류</th><th>문장</th><th>정답</th></tr>")
            for a in aa:
                out.append(f"<tr><td class='ck'>□</td><td class='src'>{esc(a['kind'])}</td><td>{esc(a['text'])}</td><td><b>{esc(a['answer'])}</b> <span class='src'>{esc(a['note'])}</span></td></tr>")
            out.append("</table>")
        out.append("<div class='sign'>확인자 ________ · 고친 것 ____개</div></div>")
    out.append("</body></html>")
    return "".join(out)

# ---- 🤖 AI 검수 (2026-11-10): 조교샘이 검수할 수 없어서 AI가 대신 봄 ----
#   1차: 지금 모델이 지문마다 유의어 쌍을 판정하고, 정답 문항은 저장된 답을 보지 않고 직접 풂
#   2차: 1차에서 '틀림'인 유의어와 저장된 답과 다른 정답만 더 강한 다른 모델(REVIEW_AI_SECOND)이 따로 다시 봄
#   적용: 두 번 다 틀림인 유의어 → 2차가 준 확실한 유의어로 바꾸고, 없으면 지움 / O·X 정답은 두 모델이 같은 다른 답이면 고침 /
#         요약문 (A)(B)는 두 모델 모두 함정 단어를 고르면 '확인 필요'로만 표시 (요약문 문장을 다시 써야 할 수 있어서)
#   결과는 syn_review 문서에 남겨, 다음에는 새로 생긴 유의어·정답만 봄 (다시 볼 때 비용이 안 듦)
REVIEW_FILE = os.path.join(BASE_DIR, "syn_review.json")
REVIEW_AI_SECOND = ("openai:gpt-6.1-sol", "anthropic:claude-sonnet-5-5")
REVIEW_ALL = "전체 지문"
SCHEMA_REVIEW1 = {  # 1차: 틀린 유의어 번호만 (출력을 줄여 비용을 낮춤)
    "type": "object",
    "properties": {
        "bad_ids": {"type": "array", "items": {"type": "integer"}},
        "answers": {"type": "array", "items": {"type": "object", "properties": {
            "id": {"type": "integer"}, "answer": {"type": "string"}}, "required": ["id", "answer"]}},
    },
    "required": ["bad_ids", "answers"],
}
SCHEMA_REVIEW = {  # 2차: 틀린 유의어의 이유와 바꿔 넣을 말
    "type": "object",
    "properties": {
        "bad_syns": {"type": "array", "items": {"type": "object", "properties": {
            "id": {"type": "integer"}, "why": {"type": "string"},
            "fix_word": {"type": "string"}, "fix_pron": {"type": "string"}, "fix_meaning": {"type": "string"}},
            "required": ["id", "why", "fix_word", "fix_pron", "fix_meaning"]}},
        "answers": {"type": "array", "items": {"type": "object", "properties": {
            "id": {"type": "integer"}, "answer": {"type": "string"}}, "required": ["id", "answer"]}},
    },
    "required": ["bad_syns", "answers"],
}
REVIEW_AI_RULES = """너는 고등학교 영어 내신 교재 검수자다. 아래 지문을 기준으로 [유의어]와 [정답]을 검수하고 JSON만 반환하라.
[유의어 판정] 줄마다 'id. 단어(뜻) ≈ 유의어(뜻)'이다. 학생이 이 둘을 '같은 뜻'으로 외워도 되는지 본다. 지문 속 그 단어 자리에 유의어를 넣었을 때 핵심 뜻이 그대로면 맞다.
- 맞음: 뜻이 같은 말, 시험 선택지 재진술에 흔히 쓰는 조금 넓거나 약한 말(toxic → harmful, gradually → slowly), 숙어·구로 바꾼 말.
- 틀림: 핵심 뜻이 다른 말(enzyme '효소' → protein '단백질', efficient '효율적인' → effective '효과적인'), 철자·소리가 비슷한 다른 단어와 헷갈린 말(perishable '상하기 쉬운' → degradable '분해되는'), 지문 속 뜻이 아닌 다른 뜻의 동의어, 반대·무관한 말, 품사가 달라 그 자리에 쓸 수 없는 말, 유의어의 한국어 뜻 풀이가 틀린 경우.
- 단어장은 원형으로 적으므로 시제·인칭·단복수 같은 형태 차이는 따지지 않는다. 확실히 틀린 것만 틀림으로 본다.
[정답 풀기] 문항마다 지문만 보고 직접 푼다. (O/X) 문항은 지문 내용과 맞으면 'O', 틀리면 'X'. (고르기) 문항은 _____에 들어갈 보기 단어 하나를 그대로 쓴다.
[답하는 방법] 맨 아래 [이번에 할 일]을 따른다. answers에는 문항마다 id와 answer를 넣는다."""

def review_syn_key(g: Dict) -> str:
    return f"{g['label']}|{g['word'].lower()}|{g['syn'].lower()}"

def review_ans_key(a: Dict) -> str:
    return f"{a['label']}|{a['kind']}|{a['i']}|{re.sub(r'[^a-z]', '', str(a['text']).lower())[:50]}"

def review_store_get() -> Dict:
    d = _store_get(supabase if IS_CLOUD_DB else None, REVIEW_FILE, {}) or {}
    d.setdefault("syn", {})
    d.setdefault("ans", {})
    return d

def review_second_model(model: str) -> str:
    for m in REVIEW_AI_SECOND:
        if m != model and AI_KEYS.get(split_model_spec(m)[0]):
            return m
    return model

def review_passage_text(label: str) -> str:
    for sk in ("saved_w", "saved_v", "saved_a", "saved_c", "saved_mock"):
        for p in st.session_state.get(sk) or []:
            if p.get("label") == label and str(p.get("body_text", "")).strip():
                return str(p["body_text"])
    return ""

def _rv_norm(v) -> str:
    return re.sub(r"[^a-z]", "", str(v).lower())

def _review_blank(a: Dict) -> (str, List[str]):
    """요약문 (A)(B) 문항: (A) 자리 정답을 _____로 가리고 보기 두 개(정답·함정, 순서는 섞음)"""
    tag = a["field"][-1].upper()
    eng = str(a["text"]).replace("**", "")
    ans_w = str(a["answer"]).strip()
    mk = re.search(rf"\({tag}\)", eng)
    if mk and ans_w:
        eng = eng[:mk.end()] + re.sub(rf"(?<![A-Za-z]){re.escape(ans_w)}(?![A-Za-z])", lambda _m: "_____", eng[mk.end():], count=1)
    opts = [ans_w, re.sub(r"^함정:\s*", "", str(a["note"])).strip()]
    random.Random(eng).shuffle(opts)
    return eng, opts

def review_ai_block(label: str, text: str, srows: List[Dict], arows: List[Dict], second: bool = False) -> str:
    out = [f"지문 정보: {label}", f'지문: """{text}"""']
    if srows:
        out.append("\n[유의어]")
        out += [f"{k}. {g['word']}({g['meaning'] or '-'}) ≈ {g['syn']}({g['syn_meaning'] or '-'})" for k, g in enumerate(srows, 1)]
    if arows:
        out.append("\n[정답]")
        for k, a in enumerate(arows, 1):
            if a["field"].startswith("blank_"):
                eng, opts = _review_blank(a)
                out.append(f"{k}. (고르기: {' / '.join(opts)}) {eng}")
            else:
                out.append(f"{k}. (O/X) {a['text']}")
    out.append("\n[이번에 할 일] " + ("틀린 유의어 줄마다 bad_syns에 id, why(한국어 20자 이내), 바꿔 넣을 확실한 유의어(fix_word·fix_pron 한글 발음·fix_meaning 한국어 뜻, "
                                      "원래 단어·이미 있는 유의어와 다른 말 — 확실한 것이 없으면 셋 다 빈 문자열)를 넣는다. 맞는 줄은 쓰지 않는다."
                                      if second else "틀린 유의어 줄의 id만 bad_ids에 넣는다. 맞는 줄은 쓰지 않는다."))
    return "\n".join(out)

def _review_differs(a: Dict, v: str) -> bool:
    v = str(v or "").strip()
    if not v:
        return False
    if a["field"].startswith("blank_"):
        return _rv_norm(v) != _rv_norm(a["answer"])
    return v[:1].upper() in ("O", "X") and v[:1].upper() != str(a["answer"]).strip()[:1].upper()

def review_ai_cost_krw(model: str, n_ps: int, n_syn: int, n_ans: int, n_words: int) -> float:
    """대략: 지문당 규칙 약 700토큰 + 지문 + 유의어 줄·정답 줄, 출력은 틀림 30% 가정(이유·바꿀 말) + 생각 토큰. 2차는 1차의 약 40%를 더 강한 모델로"""
    def usd(m, tin, tout):
        p_in, _, p_out = ai_price(m)
        return (tin * p_in + tout * p_out) / 1_000_000
    tin = n_ps * 700 + n_words * 1.4 + n_syn * 22 + n_ans * 45
    tout = (n_ps * 80 + n_syn * 0.3 * 35 + n_ans * 12) * 2
    m2 = review_second_model(model)
    return (usd(model, tin, tout) + usd(m2, tin * 0.4, tout * 0.4)) * 1400

def review_ai_run(labels: List[str], model: str, progress=None) -> Dict:
    """고른 지문 중 아직 AI가 안 본 유의어·정답을 검수하고 바로 고침 → 결과 요약 (바뀐 것 목록 포함)"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    store = review_store_get()
    syn, ans = review_collect(labels)
    gsyn = review_group_syn(syn)
    texts = {l: review_passage_text(l) for l in labels}
    tasks = []
    for l in labels:
        sr = [g for g in gsyn if g["label"] == l and review_syn_key(g) not in store["syn"]]
        ar = [a for a in ans if a["label"] == l and review_ans_key(a) not in store["ans"] and str(a["answer"]).strip()]
        if (sr or ar) and texts.get(l):
            tasks.append((l, sr, ar))
    logs: List[str] = []

    def judge(m, l, sr, ar, second):
        out = _qb_call(client, m, [REVIEW_AI_RULES, review_ai_block(l, texts[l], sr, ar, second)], SCHEMA_REVIEW if second else SCHEMA_REVIEW1,
                       0.0, logs, f"{l} AI 검수", cache_first=True)
        if not isinstance(out, dict):
            return False, {}, {}
        if second:
            bad = {int(x.get("id") or 0): x for x in out.get("bad_syns") or [] if isinstance(x, dict)}
        else:
            bad = {int(x): {} for x in out.get("bad_ids") or [] if str(x).lstrip("-").isdigit()}
        got = {int(x.get("id") or 0): str(x.get("answer", "")).strip() for x in out.get("answers") or [] if isinstance(x, dict)}
        return True, bad, got

    def run_all(m, jobs, lo, hi, what, second=False):
        res = {}
        if not jobs:
            return res
        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = {ex.submit(judge, m, l, sr, ar, second): l for l, sr, ar in jobs}
            for n, f in enumerate(as_completed(futs), 1):
                try:
                    res[futs[f]] = f.result()
                except Exception as e:
                    logs.append(f"❌ {futs[f]}: {str(e)[:120]}")
                    res[futs[f]] = (False, {}, {})
                if progress:
                    progress(lo + (hi - lo) * n / len(jobs), f"{what} {n}/{len(jobs)}")
        return res

    first = run_all(model, tasks, 0.0, 0.7, "1차 검수")
    m2 = review_second_model(model)
    jobs2 = []
    for l, sr, ar in tasks:
        ok, bad, got = first.get(l, (False, {}, {}))
        if not ok:
            continue
        s2 = [sr[i - 1] for i in sorted(bad) if 1 <= i <= len(sr)]
        a2 = [ar[i - 1] for i in sorted(got) if 1 <= i <= len(ar) and _review_differs(ar[i - 1], got[i])]
        if s2 or a2:
            jobs2.append((l, s2, a2))
    second = run_all(m2, jobs2, 0.7, 1.0, f"2차 다시 보기 ({split_model_spec(m2)[1]})", second=True)
    sent2 = {l: (s2, a2) for l, s2, a2 in jobs2}

    bad_rows, fixes, ans_fix, ans_check, n_syn_ok, n_ans_ok = [], {}, [], [], 0, 0
    for l, sr, ar in tasks:
        ok, bad, got = first.get(l, (False, {}, {}))
        if not ok:
            continue
        s2, a2 = sent2.get(l, ([], []))
        ok2, bad2, got2 = second.get(l, (False, {}, {}))
        if s2 and not ok2:
            s2_keys_pending = {review_syn_key(g) for g in s2}  # 2차가 실패한 지문의 '틀림' 후보는 기록하지 않음 → 다음에 다시 봄
        else:
            s2_keys_pending = set()
        bad2_by = {review_syn_key(s2[i - 1]): v for i, v in bad2.items() if 1 <= i <= len(s2)} if ok2 else {}
        for g in sr:
            k = review_syn_key(g)
            if k in s2_keys_pending:
                continue
            if k in bad2_by:
                v = bad2_by[k]
                fw, fp, fm = (str(v.get(x, "") or "").strip() for x in ("fix_word", "fix_pron", "fix_meaning"))
                bad_rows.append(dict(g, why=str(v.get("why", "")).strip(), fix=(f"{fw}({fm})" if fw else "")))
                if fw:
                    fixes[(g["label"], g["word"].lower(), g["syn"].lower())] = (fw, fp, fm)
                    store["syn"][f"{g['label']}|{g['word'].lower()}|{fw.lower()}"] = 1  # 2차 모델이 준 말은 다시 안 봄
                store["syn"][k] = [str(v.get("why", "")).strip(), fw]
            else:
                store["syn"][k] = 1
                n_syn_ok += 1
        a2_keys = {review_ans_key(a): a for a in a2}
        got2_by = {review_ans_key(a2[i - 1]): v for i, v in got2.items() if 1 <= i <= len(a2)} if ok2 else {}
        for i, a in enumerate(ar, 1):
            k = review_ans_key(a)
            if i not in got:
                continue  # AI가 답을 안 냈음 → 다음에 다시
            if k not in a2_keys:
                store["ans"][k] = "ok"
                n_ans_ok += 1
                continue
            if not ok2 or k not in got2_by:
                continue
            v1, v2 = got[i], got2_by[k]
            if not _review_differs(a, v2):
                store["ans"][k] = "ok"
                n_ans_ok += 1
            elif a["field"].startswith("blank_") or (v1[:1].upper() != v2[:1].upper()):
                ans_check.append(dict(a, ai1=v1, ai2=v2))
                store["ans"][k] = ["check", v1, v2]
            else:
                ans_fix.append((a, v2[:1].upper()))
                store["ans"][k] = ["fixed", str(a["answer"]), v2[:1].upper()]
    n_changed = review_delete_syns(bad_rows, fixes) if bad_rows else 0
    if ans_fix:
        review_save_answers(ans_fix)
    store["updated"] = time.strftime("%Y-%m-%d %H:%M")
    _store_put(supabase if IS_CLOUD_DB else None, REVIEW_FILE, store)
    return {"tasks": len(tasks), "failed": [l for l in first if not first[l][0]], "syn_ok": n_syn_ok, "bad": bad_rows,
            "n_fixed": sum(1 for b in bad_rows if b["fix"]), "n_changed": n_changed, "ans_ok": n_ans_ok,
            "ans_fix": [dict(a, new=v) for a, v in ans_fix], "ans_check": ans_check, "logs": logs[-10:], "model": model, "model2": m2}


def page_review():
    all_labels = sorted({p.get("label") for sk in ("saved_w", "saved_v", "saved_a", "saved_c") for p in st.session_state[sk] if p.get("label")},
                        key=natural_sort_key)
    if not all_labels:
        st.info("검수할 교재가 아직 없어요. 먼저 원문 라이브러리에서 교재를 만들어 주세요.")
        return
    if st.session_state.get("rv_flash"):
        st.success(st.session_state.pop("rv_flash"))
    label_set = set(all_labels)
    shared = [l for l in st.session_state.shared_sel_labels if l in label_set]
    groups = sorted({review_group_of(l) for l in all_labels}, key=natural_sort_key)
    opts = [REVIEW_ALL] + ([REVIEW_FROM_SHARED] if shared else []) + groups
    c1, c2 = st.columns([1.4, 2.6])
    grp = c1.selectbox("📁 검수할 범위", opts, key="rv_group",
                       format_func=lambda g: f"{g} ({len(shared)}개)" if g == REVIEW_FROM_SHARED else (f"{g} ({len(all_labels)}개)" if g == REVIEW_ALL else g))
    if grp == REVIEW_ALL:
        labels = all_labels
        c2.markdown(f"<div class='qb-note' style='margin-top:1.9rem'>교재가 있는 모든 지문 <b>{len(all_labels)}개</b></div>", unsafe_allow_html=True)
    else:
        pool = shared if grp == REVIEW_FROM_SHARED else [l for l in all_labels if review_group_of(l) == grp]
        labels = c2.multiselect("지문", pool, default=pool, key=f"rv_labels_{grp}", format_func=review_short)
    syn, ans = review_collect(labels)
    gsyn = review_group_syn(syn)
    n_sus = sum(g["suspect"] for g in gsyn)
    mins = max(1, round((len(gsyn) * 2 + len(ans) * 5) / 60)) if (gsyn or ans) else 0
    st.markdown(f"<div class='qb-note'>지문 <b>{len(labels)}개</b> · 유의어 <b>{len(gsyn)}개</b> (뜻이 한 글자도 안 겹쳐 먼저 볼 것 <b>{n_sus}개</b>) · "
                f"정답 <b>{len(ans)}개</b> → 훑어보는 데 약 <b>{mins}분</b> (유의어 하나 2초, 정답 하나 5초 기준)<br>"
                "기준: <b>시험 선지에서 그 단어 자리에 바꿔 넣어도 뜻이 같은가?</b> 상위어(enzyme → protein)·연상어(perishable → degradable)는 틀림이에요.</div>",
                unsafe_allow_html=True)
    ver = st.session_state.setdefault("rv_ver", 0)
    # 🤖 AI 검수: 아직 AI가 안 본 것만 (조교샘 대신)
    store = review_store_get()
    todo_s = [g for g in gsyn if review_syn_key(g) not in store["syn"]]
    todo_a = [a for a in ans if review_ans_key(a) not in store["ans"] and str(a["answer"]).strip()]
    todo_l = sorted({x["label"] for x in todo_s + todo_a}, key=natural_sort_key)
    with st.container(border=True):
        a1, a2 = st.columns([3.2, 1.6], vertical_alignment="center")
        if todo_l:
            n_words = sum(len(review_passage_text(l).split()) for l in todo_l)
            cost = review_ai_cost_krw(text_model, len(todo_l), len(todo_s), len(todo_a), n_words)
            a1.markdown(f"**🤖 AI 검수** — 아직 AI가 안 본 유의어 <b>{len(todo_s):,}개</b> · 정답 <b>{len(todo_a):,}개</b> (지문 {len(todo_l)}개) · "
                        f"예상 약 <b>{cost:,.0f}원</b><br><span style='color:#64748b;font-size:0.85rem'>"
                        f"{split_model_spec(text_model)[1]}가 먼저 보고, 틀림으로 본 것만 {split_model_spec(review_second_model(text_model))[1]}가 다시 봐요. "
                        "둘 다 틀리다고 한 유의어는 맞는 말로 바꾸거나 지우고, 둘 다 같은 다른 답을 낸 O/X 정답은 고쳐요.</span>", unsafe_allow_html=True)
            go = a2.button("🤖 AI로 검수하고 고치기", type="primary", use_container_width=True, key="rv_ai_go")
        else:
            a1.markdown("**🤖 AI 검수** — 고른 지문의 유의어·정답을 AI가 모두 봤어요 ✅" + (f" <span style='color:#64748b;font-size:0.85rem'>(마지막 {esc(store.get('updated', ''))})</span>" if store.get("updated") else ""),
                        unsafe_allow_html=True)
            go = False
        if go:
            bar = st.progress(0.0, text="AI 검수를 시작해요…")
            r = review_ai_run(todo_l, text_model, lambda f, t: bar.progress(min(1.0, f), text=t))
            flush_ai_usage()  # 사이드바 'AI 사용 금액'에 바로 반영
            st.session_state.rv_ai_last = r
            st.session_state.rv_ver = ver + 1
            st.session_state.rv_flash = (f"🤖 AI 검수를 마쳤어요 (지문 {r['tasks']}개). 유의어 {len(r['bad'])}개를 고쳤어요 "
                                         f"(맞는 말로 바꿈 {r['n_fixed']}개 · 지움 {len(r['bad']) - r['n_fixed']}개) · O/X 정답 {len(r['ans_fix'])}개 고침"
                                         + (f" · 확인 필요 {len(r['ans_check'])}개" if r["ans_check"] else "")
                                         + (f" · 실패한 지문 {len(r['failed'])}개 (다시 누르면 그것만 봐요)" if r["failed"] else ""))
            st.rerun()
        last = st.session_state.get("rv_ai_last")
        if last and (last["bad"] or last["ans_fix"] or last["ans_check"]):
            with st.expander(f"방금 AI가 고친 것 보기 (유의어 {len(last['bad'])} · 정답 {len(last['ans_fix'])} · 확인 필요 {len(last['ans_check'])})"):
                if last["bad"]:
                    st.dataframe(pd.DataFrame([{"지문": review_short(b["label"]), "단어": f"{b['word']} ({b['meaning']})", "틀린 유의어": f"{b['syn']} ({b['syn_meaning']})",
                                                "바꾼 말": b["fix"] or "(지움)", "이유": b["why"]} for b in last["bad"]]), hide_index=True, use_container_width=True)
                if last["ans_fix"]:
                    st.dataframe(pd.DataFrame([{"지문": review_short(a["label"]), "종류": a["kind"], "문장": a["text"], "고친 정답": f"{a['answer']} → {a['new']}"}
                                               for a in last["ans_fix"]]), hide_index=True, use_container_width=True)
                if last["ans_check"]:
                    st.caption("⚠️ 확인 필요: 두 AI가 저장된 정답과 다르게 풀었지만, 서로 답이 다르거나 요약문이라 자동으로 고치지 않았어요.")
                    st.dataframe(pd.DataFrame([{"지문": review_short(a["label"]), "종류": a["kind"], "문장": a["text"], "저장된 정답": a["answer"],
                                                "AI 1차": a["ai1"], "AI 2차": a["ai2"]} for a in last["ans_check"]]), hide_index=True, use_container_width=True)
    t1, t2, t3 = st.tabs([f"🔤 유의어 {len(gsyn)}", f"✅ 정답 {len(ans)}", "🖨️ 인쇄용 체크리스트"])
    with t1:
        if not gsyn:
            st.caption("고른 지문에 유의어가 없어요.")
        else:
            only = st.toggle(f"⚠️ 먼저 볼 것만 보기 ({n_sus}개)", key="rv_only_sus")
            rows = [g for g in gsyn if g["suspect"]] if only else gsyn
            df = pd.DataFrame([{"틀림": False, "지문": review_short(g["label"]), "단어": g["word"], "뜻": g["meaning"],
                                "유의어": ("⚠️ " if g["suspect"] else "") + g["syn"], "유의어 뜻": g["syn_meaning"], "교재": "·".join(g["srcs"]),
                                "AI": "✅" if store["syn"].get(review_syn_key(g)) == 1 else ""}
                               for g in rows])
            ed = st.data_editor(df, hide_index=True, use_container_width=True, height=min(620, 38 + 35 * len(rows)),
                                disabled=[c for c in df.columns if c != "틀림"], key=f"rv_syn_{ver}_{grp}_{only}",
                                column_config={"틀림": st.column_config.CheckboxColumn("❌ 틀림", width="small")})
            picked = [rows[i] for i, v in enumerate(ed["틀림"].tolist()) if bool(v)]
            if st.button(f"🗑️ 틀림 표시한 유의어 {len(picked)}개 지우기", type="primary", disabled=not picked, key="rv_del"):
                n = review_delete_syns(picked)
                st.session_state.rv_ver = ver + 1
                st.session_state.rv_flash = f"✅ 유의어 {len(picked)}개를 지웠어요 (교재 {n}곳). 다음에 출력하는 교재·PPT부터 빠져요."
                st.rerun()
            st.caption("지우면 그 지문의 Voca·한판승·Self-Note·Core에서 같은 단어의 같은 유의어가 모두 빠져요. 되돌리려면 그 교재 화면의 편집에서 다시 적어 주세요.")
    with t2:
        if not ans:
            st.caption("고른 지문에 Core 요약문·진위, Self-Note OX 정답이 없어요.")
        else:
            def ai_mark(a):
                v = store["ans"].get(review_ans_key(a))
                return "✅" if v == "ok" else ("🔧 고침" if isinstance(v, list) and v[0] == "fixed" else
                                              (f"⚠️ AI: {v[1]}/{v[2]}" if isinstance(v, list) and v[0] == "check" else ""))
            df2 = pd.DataFrame([{"지문": review_short(a["label"]), "종류": a["kind"], "문장": a["text"], "정답": str(a["answer"]), "참고": a["note"], "AI": ai_mark(a)}
                                for a in ans])
            ed2 = st.data_editor(df2, hide_index=True, use_container_width=True, height=min(620, 38 + 35 * len(ans)),
                                 disabled=["지문", "종류", "문장", "참고", "AI"], key=f"rv_ans_{ver}_{grp}",
                                 column_config={"정답": st.column_config.TextColumn("정답 ✏️", width="small"), "문장": st.column_config.TextColumn("문장", width="large")})
            changes = [(a, str(v).strip()) for a, v in zip(ans, ed2["정답"].tolist()) if str(v).strip() != str(a["answer"]).strip()]
            bad = [a for a, v in changes if a["field"] in ("answer", "tf") and v.upper() not in ("O", "X")] + [a for a, v in changes if not v]
            if bad:
                st.warning("O/X 문제의 정답은 O 또는 X로, 요약문 정답은 영어 단어로 적어 주세요.")
            if st.button(f"💾 고친 정답 {len(changes)}개 저장", type="primary", disabled=not changes or bool(bad), key="rv_save_ans"):
                n = review_save_answers(changes)
                st.session_state.rv_ver = ver + 1
                st.session_state.rv_flash = f"✅ 정답 {n}개를 고쳐 저장했어요. 요약문 정답은 요약문 문장 속 단어도 함께 바꿨어요."
                st.rerun()
    with t3:
        st.caption("종이로 보고 싶을 때 받아서 인쇄하세요 (지문마다 □ 칸 · 먼저 볼 유의어가 위에). 고친 내용은 이 화면에서 저장해야 교재에 반영돼요.")
        st.download_button("🖨️ 체크리스트 HTML 받기", data=review_print_html(labels, gsyn, ans).encode("utf-8"),
                           file_name=f"배포전검수_{time.strftime('%Y%m%d')}.html", mime="text/html", disabled=not labels, key="rv_print")


# ---- 메뉴 정의 (title: 메뉴 이름, short: 홈 화면 한 줄 설명, desc: 화면 위 설명, help: 사용법 키) ----
PAGE_INFO = {
    "home": {"title": "홈", "icon": "🏠", "group": "", "func": page_home, "short": "", "desc": "", "help": None},
    "source": {"title": "원문 라이브러리", "icon": "📝", "group": "만들기", "func": page_source,
               "short": "모의고사·교과서 PDF → 원문 저장 → 번호 골라 교재 생성",
               "desc": "모의고사·교과서 PDF를 올려 원문을 저장하고, 번호를 골라 교재를 만들어요. 모든 교재·변형문제의 원본 창고예요.", "help": "mock"},
    "qbank": {"title": "변형문제 은행·시험지", "icon": "🧩", "group": "만들기", "func": page_qbank,
              "short": "유형·난이도별 문제를 쌓고, 조건대로 뽑아 시험지 출력",
              "desc": "문제를 만들 때마다 은행에 쌓이고, 시험지는 쌓인 문제에서 뽑기만 해요 (시험지 만들기는 AI 비용 없음).", "help": "qbank"},
    "final": {"title": "영T FINAL 모의고사", "icon": "🏁", "group": "만들기", "func": page_final,
              "short": "학교·학년 실제 시험과 같은 구성·배점·디자인의 회차별 동형 모의고사 · 평균 기록으로 난이도 보정",
              "desc": "운양고·제일고 실제 기출과 같은 유형·배점·문항 수로 PQRS영T FINAL 1회차, 2회차 …를 만들어요. 이미 낸 문제는 다시 안 나오고, 회차마다 실제 평균을 적으면 다음 회차 난이도를 거기에 맞춰요 (AI 비용 없음).",
              "help": "final"},
    "excel": {"title": "엑셀 원클릭 생성", "icon": "🚀", "group": "만들기", "func": page_excel,
              "short": "엑셀 한 장으로 교재 4종을 한 번에",
              "desc": "엑셀과 이미지를 올리면 One-Shot(한판승)·Self-Note·Core·Voca를 함께 만들고, Core에 쓰인 패러프레이징 어휘는 Voca 단어장에 자동으로 들어가요.", "help": "excel"},
    "oneshot": {"title": "One-Shot (한판승)", "icon": "🏆", "group": "교재 출력", "func": page_oneshot,
                "short": "골라서 단행본 출력 · 수업용 PPT", "desc": "만든 Anatomy One-Shot(한판승)을 골라 단행본으로 출력하거나 수업용 PPT로 받아요.", "help": "library"},
    "selfnote": {"title": "Self-Note", "icon": "🦅", "group": "교재 출력", "func": page_selfnote,
                 "short": "골라서 단행본 출력", "desc": "만든 Anatomy Self-Note를 골라 단행본으로 출력해요.", "help": "library"},
    "core": {"title": "Core", "icon": "💎", "group": "교재 출력", "func": page_core,
             "short": "골라서 단행본 출력 · 빠진 지문 자동 생성", "desc": "만든 Anatomy Core를 골라 단행본으로 출력해요. 다른 교재만 있는 지문은 여기서 바로 만들 수 있어요.", "help": "library"},
    "voca": {"title": "Voca", "icon": "📖", "group": "교재 출력", "func": page_voca,
             "short": "단어장 출력 · 엑셀/CSV 받기", "desc": "만든 Anatomy Voca 단어장을 골라 출력하거나 엑셀·CSV로 받아요.", "help": "library"},
    "review": {"title": "배포 전 검수", "icon": "✅", "group": "교재 출력", "func": page_review,
               "short": "AI가 유의어·정답을 검수해 틀린 것 고치기 · 직접 확인 · 인쇄용 체크리스트",
               "desc": "교재·PPT를 학생에게 내보내기 전에 유의어와 정답을 확인해요. 🤖 AI 검수를 누르면 AI 두 개가 보고 틀린 유의어·O/X 정답을 바로 고쳐요. 직접 표시해서 지우거나 고칠 수도 있어요.",
               "help": None},
    "dna": {"title": "출제 DNA", "icon": "🎯", "group": "기출 분석", "func": page_dna,
            "short": "학교별 출제 방식 현황판 · 기출 PDF로 규칙 학습",
            "desc": "운양고·제일고 기출 시험지로 출제 규칙을 쌓아요. 여기 규칙이 교재·변형문제 생성에 함께 쓰여요.", "help": "dna"},
    "past-voca": {"title": "기출 어휘", "icon": "📚", "group": "기출 분석", "func": page_past_voca,
                  "short": "기출 시험지 PDF → 시험지별 단어장",
                  "desc": "기출 시험지 PDF에서 단어를 뽑아 출처(학년·번호·본문/선지)와 함께 시험지별 단어장으로 저장해요. 한 시험지 단어는 이어서, 다음 시험지는 새 쪽부터 출력돼요.", "help": "pastvoca"},
    "settings": {"title": "설정", "icon": "⚙️", "group": "설정", "func": page_settings,
                 "short": "로고·학원명·생성 속도·데이터 이전", "desc": "교재에 들어가는 로고·학원명, 생성 속도, 데이터 이전·백업을 관리해요. AI 모델은 왼쪽 메뉴 아래에서 골라요.", "help": None},
}
HOME_GROUPS = [("✍️ 만들기", ["source", "qbank", "final", "excel"]),
               ("🖨️ 교재 출력", ["oneshot", "selfnote", "core", "voca", "review"]),
               ("🏫 기출 분석", ["dna", "past-voca"])]
PAGES = {k: st.Page(v["func"], title=v["title"], icon=v["icon"], url_path=k, default=(k == "home")) for k, v in PAGE_INFO.items()}
nav = {}
for k, v in PAGE_INFO.items():
    nav.setdefault(v["group"], []).append(PAGES[k])
current = st.navigation(nav, position="sidebar")

with st.sidebar:
    # AI 제공사·모델: 모든 화면에서 쓰므로 메뉴 바로 아래에 둠. Secrets에 키가 있는 제공사만, 모델 목록은 그 키로 자동 조회
    st.markdown("##### 🤖 AI 모델")
    providers = [p for p in AI_PROVIDERS if AI_KEYS.get(p)]
    saved_provider, saved_name = split_model_spec(app_settings.get("text_model", "gemini:gemini-2.5-flash"))
    provider = st.radio("AI 제공사 (글 생성)", providers, format_func=lambda p: AI_PROVIDERS[p], horizontal=True,
                        index=providers.index(saved_provider) if saved_provider in providers else 0, key="ai_provider",
                        label_visibility="collapsed")
    model_options = list_ai_models(provider)
    if saved_provider == provider and saved_name and saved_name not in model_options:
        model_options = [saved_name] + model_options
    default_name = saved_name if saved_provider == provider else AI_DEFAULT_MODELS[provider][0]
    model_name = st.selectbox("모델", model_options, index=model_options.index(default_name) if default_name in model_options else 0,
                              key=f"ai_model_{provider}", label_visibility="collapsed",
                              help="Secrets의 API 키로 쓸 수 있는 모델을 자동으로 불러옵니다 (1시간마다 새로 확인).")
    text_model = f"{provider}:{model_name}"
    if app_settings.get("text_model") != text_model:
        app_settings["text_model"] = text_model
        save_app_settings(app_settings)
    render_ai_spend()
    render_model_guide()
    st.caption(f"{'🟢 클라우드 DB 연결됨' if IS_CLOUD_DB else '🟡 로컬 PC 모드 (이 PC에만 저장)'} · {academy_name}")

# ---- 화면 머리: 위치 표시 · 제목 · 한 줄 설명 · 사용법 ----
cur_key = next((k for k, v in PAGE_INFO.items() if v["title"] == current.title), "home")
cur = PAGE_INFO.get(cur_key, PAGE_INFO["home"])
h1, h2 = st.columns([6, 1], vertical_alignment="bottom")
with h1:
    crumb = "4pm 올인원 교재 시스템" + (f" › {cur['group']}" if cur["group"] and cur["group"] != cur["title"] else "")
    st.markdown(f"<div class='pg-crumb'>{crumb}</div>", unsafe_allow_html=True)
    st.markdown(f"## {cur['icon']} {'4pm 올인원 교재 시스템' if cur_key == 'home' else cur['title']}")
    if cur["desc"]:
        st.markdown(f"<div class='pg-desc'>{cur['desc']}</div>", unsafe_allow_html=True)
if cur["help"]:
    with h2.popover("❓ 사용법", use_container_width=True):
        st.markdown(APP_HELP[cur["help"]])

# 진행 중이거나 남아 있는 생성 작업은 어느 화면에서든 위에 보임
render_gen_job_panel()

current.run()
