from __future__ import annotations
import os
import re
import json
import logging
import hashlib
import importlib
from typing import Any, Dict, List, Optional
from datetime import datetime
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from bs4 import BeautifulSoup
import requests
import re as _re2
from django.shortcuts import render
from django.utils import timezone
from django.conf import settings
from django.http import JsonResponse, HttpRequest
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from pydantic import validator
from pydantic_settings import BaseSettings

AUTO_INGEST_AFTER_GEMINI = getattr(
    settings, "AUTO_INGEST_AFTER_GEMINI",
    os.environ.get("AUTO_INGEST_AFTER_GEMINI", "1").lower() not in ("0", "false", "no")
)

try:
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
except Exception:
    SentenceTransformerEmbeddingFunction = None

# 로깅
log = logging.getLogger(__name__)

# 안전한 기본 설정
if not hasattr(settings, "CHROMA_DB_DIR"):
    setattr(settings, "CHROMA_DB_DIR", os.environ.get("CHROMA_DB_DIR", str(Path.cwd() / "chroma_db_new")))
if not hasattr(settings, "CHROMA_COLLECTION"):
    setattr(settings, "CHROMA_COLLECTION", os.environ.get("CHROMA_COLLECTION", "my_notes"))
if not hasattr(settings, "GEMINI_EMBED_MODELS"):
    env_models = os.environ.get("GEMINI_EMBED_MODELS", "text-embedding-004")
    setattr(settings, "GEMINI_EMBED_MODELS", [m.strip() for m in env_models.split(",") if m.strip()])
if not hasattr(settings, "GEMINI_TEXT_MODEL"):
    setattr(settings, "GEMINI_TEXT_MODEL", os.environ.get("GEMINI_MODEL_DIRECT", "gemini-2.0-flash"))

# 공용 응답 헬퍼
def _ok(d: Dict[str, Any]) -> JsonResponse:
    d.setdefault("ok", True)
    return JsonResponse(d, status=200)

def _fail(message: str, extra: Dict[str, Any] | None = None) -> JsonResponse:
    payload = {"ok": False, "error": message}
    if extra:
        payload.update(extra)
    return JsonResponse(payload, status=200)

# ===== 언어 감지(간단 휴리스틱) & 프롬프트 유틸 =====
_LANG_RE_KO = re.compile(r'[\uac00-\ud7a3]')
_LANG_RE_ES = re.compile(r'[¿¡áéíóúüñÁÉÍÓÚÜÑ]')

def _detect_lang(text: str) -> str:
    s = text or ""
    if _LANG_RE_KO.search(s):
        return "ko"
    if _LANG_RE_ES.search(s):
        return "es"
    return "en"

def _mk_news_prompt(question: str, lang: str) -> str:
    if lang == "ko":
        return (
            "한국어로 간결하고 최신성 있게 답하세요.\n"
            "가능하면 참고할만한 기사/자료의 URL을 3~5개 본문 하단에 적어 주세요.\n\n"
            f"[질문]\n{question}\n\n[답변]\n"
        )
    if lang == "es":
        return (
            "Responde de forma concisa en español y teniendo en cuenta la actualidad.\n"
            "Si es posible, añade 3–5 URL de referencia al final del texto.\n\n"
            f"[Pregunta]\n{question}\n\n[Respuesta]\n"
        )
    # en
    return (
        "Answer concisely in English and keep the information up-to-date.\n"
        "When helpful, list 3–5 reference URLs at the end.\n\n"
        f"[Question]\n{question}\n\n[Answer]\n"
    )

def _make_rag_prompt(question: str, context: str, lang: str) -> str:
    if lang == "ko":
        return (
            "아래 제공된 자료만 근거로 한국어로 핵심을 정리해 답하세요.\n"
            "- 자료에서 확인되는 사실을 묶어서 요약하세요.\n"
            "- 확실한 근거가 보이면 항목화하고 문장 끝에 [1], [2]처럼 근거 블록 번호를 붙이세요.\n"
            "- 직접적 근거가 부족하면 한 줄로 '자료 내 직접 근거 부족'이라고 밝힌 뒤, "
            "자료에서 추론 가능한 범위 내 핵심 포인트를 요약하세요.\n"
            "- '본문에 없음' 같은 표현은 사용하지 마세요.\n\n"
            f"[질문]\n{question}\n\n[자료]\n{context}\n\n[답변]\n"
        )
    if lang == "es":
        return (
            "Responde en español usando únicamente el material proporcionado.\n"
            "- Agrupa y resume los hechos comprobables.\n"
            "- Cuando haya evidencia clara, usa viñetas y añade [1], [2]… al final indicando el bloque fuente.\n"
            "- Si falta evidencia directa, escribe una línea: 'Evidencia directa insuficiente en el material', "
            "y después resume puntos clave inferibles del material.\n"
            "- No uses la frase 'no aparece en el texto'.\n\n"
            f"[Pregunta]\n{question}\n\n[Material]\n{context}\n\n[Respuesta]\n"
        )
    # en
    return (
        "Answer in English using only the provided context.\n"
        "- Group and summarize the verifiable facts.\n"
        "- When evidence is clear, use bullet points and append [1], [2], etc., to cite context blocks.\n"
        "- If direct evidence is lacking, write one line: 'Insufficient direct evidence in the provided material', "
        "then summarize key points that are reasonable inferences from the material.\n"
        "- Do not use the phrase 'not found in the text'.\n\n"
        f"[Question]\n{question}\n\n[Context]\n{context}\n\n[Answer]\n"
    )

# Gemini 클라이언트/호출
try:
    from google import genai
    try:
        from google.genai.types import HttpOptions
    except Exception:
        HttpOptions = None
except Exception:
    genai = None
    HttpOptions = None

def _gemini_client():
    if genai is None:
        raise RuntimeError("google-genai 미설치: pip install google-generativeai google-genai")
    api_key = getattr(settings, "GEMINI_API_KEY", None) or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다.")
    api_version = getattr(settings, "GEMINI_API_VERSION", os.environ.get("GEMINI_API_VERSION", "v1"))
    try:
        if HttpOptions is not None:
            return genai.Client(api_key=api_key, http_options=HttpOptions(api_version=api_version))
        return genai.Client(api_key=api_key)
    except TypeError:
        return genai.Client(api_key=api_key)

def _gemini_model() -> str:
    return getattr(settings, "GEMINI_TEXT_MODEL", "gemini-2.0-flash")

def _ask_gemini(prompt: str, model: Optional[str] = None) -> str:
    try:
        c = _gemini_client()
        r = c.models.generate_content(model=model or _gemini_model(), contents=prompt)
        txt = getattr(r, "text", None)
        if not txt and getattr(r, "candidates", None):
            try:
                txt = r.candidates[0].content.parts[0].text
            except Exception:
                pass
        return (txt or "").strip() or "[빈 응답]"
    except Exception as e:
        log.warning("Gemini 응답 실패: %s", e)
        return f"[모델 응답 실패: {e}]"

# 임베딩 (Google GenAI)
LAST_EMBED_META = {"param": None, "model": None, "dim": None}

def _embed_texts(texts: List[str]) -> List[List[float]]:
    import inspect
    if not texts:
        return []
    c = _gemini_client()
    try:
        sig = inspect.signature(c.models.embed_content)
        if "contents" in sig.parameters:
            p = "contents"
        elif "content" in sig.parameters:
            p = "content"
        elif "input" in sig.parameters:
            p = "input"
        else:
            p = "contents"
    except Exception:
        p = "contents"
    pref = getattr(settings, "GEMINI_EMBED_MODELS", ["text-embedding-004"])
    models: List[str] = []
    for m in pref:
        models.extend([m] if "/" in m else [m, f"models/{m}"])
    def parse(resp: Any) -> Optional[List[float]]:
        try:
            emb = getattr(resp, "embedding", None)
            if emb is not None:
                vals = getattr(emb, "values", None)
                if vals:
                    return list(vals)
        except Exception:
            pass
        try:
            embs = getattr(resp, "embeddings", None)
            if embs:
                first = embs[0] if len(embs) else None
                if first is not None:
                    vals = getattr(first, "values", None)
                    if vals:
                        return list(vals)
        except Exception:
            pass
        if isinstance(resp, dict):
            try:
                vals = resp.get("embedding", {}).get("values")
                if vals:
                    return list(vals)
            except Exception:
                pass
            try:
                first = (resp.get("embeddings") or [None])[0]
                if isinstance(first, dict) and "values" in first:
                    return list(first["values"])
            except Exception:
                pass
        return None
    errors: List[str] = []
    for model in models:
        try:
            vecs: List[List[float]] = []
            for t in texts:
                resp = c.models.embed_content(model=model, **{p: t})
                v = parse(resp)
                if not v:
                    raise RuntimeError("임베딩 응답 파싱 실패")
                vecs.append(v)
            LAST_EMBED_META.update({"param": p, "model": model, "dim": len(vecs[0])})
            return vecs
        except Exception as e:
            errors.append(f"{model} via {p}: {e}")
            continue
    raise RuntimeError("임베딩 실패: " + " | ".join(errors))

# URL/텍스트 유틸
_LINK_RE = re.compile(r"https?://[^\s\]\)]+", re.IGNORECASE)

def extract_links_from_text(text: str, max_n: int = 6):
    urls, seen = [], set()
    for m in _LINK_RE.finditer(text or ""):
        u = m.group(0).rstrip(".,);")
        if u not in seen:
            urls.append(u)
            seen.add(u)
        if len(urls) >= max_n:
            break
    return urls

def _slug(s: str, n=60) -> str:
    s = re.sub(r"[^0-9A-Za-z가-힣\-_. ]+", "", s or "")
    s = re.sub(r"\s+", "-", s).strip("-")
    return s[:n] or "doc"

def _sha(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8", "ignore")).hexdigest()[:16]

def _iso(dt) -> str:
    try:
        if isinstance(dt, datetime):
            return dt.isoformat()
        if not dt:
            return ""
        try:
            return parsedate_to_datetime(dt).isoformat()
        except Exception:
            return datetime.fromisoformat(str(dt).replace("Z", "+00:00")).isoformat()
    except Exception:
        return ""

_URL_MD = re.compile(r"\[[^\]]+\]\((https?://[^\s)]+)\)")
_URL_RAW = re.compile(r"(https?://[^\s<>\]\)\"']+)")

def _extract_urls(text: str) -> List[str]:
    if not text:
        return []
    urls: List[str] = []
    try:
        urls += _URL_MD.findall(text)
    except Exception:
        pass
    try:
        urls += _URL_RAW.findall(text)
    except Exception:
        pass
    out: List[str] = []
    seen = set()
    for u in urls:
        u = u.strip().rstrip(").,]")
        if not u.lower().startswith(("http://", "https://")):
            continue
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out

def _chunk_text(text: str, size=1600, overlap=200):
    t = (text or "").strip()
    if not t:
        return []
    out = []
    i = 0
    n = len(t)
    while i < n:
        j = min(i + size, n)
        out.append(t[i:j])
        if j == n:
            break
        i = j - overlap
    return out

# 뉴스 검색(RSS) + 본문 크롤링
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
ACCEPT_LANG = "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7"

def _search_news_rss(query: str, top_k: int):
    import feedparser
    url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=ko&gl=KR&ceid=KR:ko"
    feed = feedparser.parse(url)
    arts = []
    for e in feed.get("entries", [])[:top_k]:
        link = e.get("link", "")
        src = (e.get("source") or {}).get("title", "") or urlparse(link).netloc
        arts.append({
            "title": e.get("title", ""),
            "url": link,
            "source": src,
            "published_at": (e.get("published") or e.get("updated") or ""),
            "snippet": e.get("summary", ""),
        })
    return arts

def _resolve_redirect(url: str, timeout: int = 12):
    try:
        r = requests.get(url, headers={"User-Agent": UA, "Accept-Language": ACCEPT_LANG}, timeout=timeout, allow_redirects=True)
        return r.url or url, r.text
    except Exception:
        return url, None

def _readability_text(html: str) -> str:
    try:
        from readability import Document
        from lxml import html as lhtml
        frag = Document(html).summary(html_partial=True)
        return (lhtml.fromstring(frag).text_content() or "").strip()
    except Exception:
        return ""

def _fetch_article_text(url: str, timeout: int = 12, min_chars: int = 400) -> str:
    try:
        final, pre_html = _resolve_redirect(url, timeout=timeout)
        import trafilatura
        html_src = pre_html or trafilatura.fetch_url(final, timeout=timeout)
        text = trafilatura.extract(html_src, output_format="txt", include_links=False, include_comments=False, favor_recall=True, no_fallback=False) if html_src else ""
        text = (text or "").strip()
        if len(text) < min_chars:
            if not html_src:
                html_src = requests.get(final, headers={"User-Agent": UA, "Accept-Language": ACCEPT_LANG}, timeout=timeout).text
            alt = _readability_text(html_src or "")
            if len(alt) > len(text):
                text = alt
        return text if len(text) >= min_chars else ""
    except Exception:
        return ""

def _crawl_news_bodies(news: list, max_workers: int = 6):
    out = [dict(n) for n in (news or [])]
    if not out:
        return out
    def job(n):
        u = (n.get("url") or "").strip()
        n["news_body"] = _fetch_article_text(u, timeout=12, min_chars=int(getattr(settings, "MIN_NEWS_BODY_CHARS", 400)))
        return n
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(job, n): i for i, n in enumerate(out)}
        for f in as_completed(futs):
            i = futs[f]
            try:
                out[i] = f.result()
            except Exception:
                out[i]["news_body"] = ""
    return out

# Google 임베딩을 Chroma embedding_function에 바인딩
class GoogleGenAIEmbeddingFunction:
    def __init__(self, _model_hint: Optional[str] = None):
        self.model_hint = _model_hint
    def __call__(self, texts: List[str]) -> List[List[float]]:
        return _embed_texts(texts)

def _current_embed_dim() -> int:
    try:
        v = _embed_texts(["dim_probe"])[0]
        return len(v)
    except Exception:
        return -1

# Chroma helpers
def _chroma_client():
    chromadb = importlib.import_module("chromadb")
    Path(settings.CHROMA_DB_DIR).mkdir(parents=True, exist_ok=True)
    PersistentClient = getattr(chromadb, "PersistentClient", None)
    if PersistentClient:
        return PersistentClient(path=settings.CHROMA_DB_DIR)
    from chromadb.config import Settings as _S
    return chromadb.Client(_S(chroma_db_impl="duckdb+parquet", persist_directory=settings.CHROMA_DB_DIR))

def _safe_get_collection_name_matching_dim(base_name: str, want_dim: int):
    client = _chroma_client()
    try:
        col = client.get_or_create_collection(
            name=base_name,
            embedding_function=GoogleGenAIEmbeddingFunction()
        )
        col_dim = -1
        try:
            peek = col.get(limit=1, include=["embeddings"])
            if peek and peek.get("embeddings"):
                col_dim = len(peek["embeddings"][0])
        except Exception:
            pass
        if col_dim in (-1, None) or col_dim == want_dim:
            return col, base_name
        else:
            alt_name = f"{base_name}"
            alt = client.get_or_create_collection(
                name=alt_name,
                embedding_function=GoogleGenAIEmbeddingFunction()
            )
            return alt, alt_name
    except Exception:
        try:
            col = client.get_collection(name=base_name)
            return col, base_name
        except Exception as e:
            raise e

def _current_embed_dim() -> int:
    try:
        v = _embed_texts(["__dim_probe__"])[0]
        return len(v)
    except Exception:
        return -1

def _chroma_collection():
    c = _chroma_client()
    base = settings.CHROMA_COLLECTION
    want_dim = _current_embed_dim()
    cur_dim = -1
    try:
        col = c.get_or_create_collection(name=base)
        try:
            got = col.get(limit=1, include=["embeddings"])
            embs = got.get("embeddings") or []
            if embs and embs[0]:
                cur_dim = len(embs[0])
        except Exception:
            pass
        if cur_dim in (-1, None) or cur_dim == want_dim:
            return col
    except Exception:
        pass
    alt = f"{base}_{want_dim}"
    log.warning(f"[Chroma] 컬렉션 '{base}'(dim={cur_dim}) != 현재 임베딩 dim={want_dim} → '{alt}' 사용")
    return c.get_or_create_collection(name=alt)

def _chroma_upsert(ids: List[str], docs: List[str], metas: List[Dict[str, Any]], embs: List[List[float]]):
    col = _chroma_collection()
    if hasattr(col, "upsert"):
        return col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
    try:
        col.delete(ids=ids)
    except Exception:
        pass
    return col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)

def _chroma_count(col=None) -> int:
    try:
        col = col or _chroma_collection()
        if hasattr(col, "count"):
            return int(col.count())
        data = col.get(limit=1_000_000)
        return len(data.get("ids") or [])
    except Exception:
        return 0

def _chroma_query_with_embeddings(col, query: str, topk: int, sources_filter: Optional[Dict] = None):
    q_emb = _embed_texts([query])[0]
    try:
        if sources_filter:
            return col.query(
                query_embeddings=[q_emb],
                n_results=max(1, int(topk)),
                include=["documents", "metadatas", "distances"],
                where=sources_filter
            )
        return col.query(
            query_embeddings=[q_emb],
            n_results=max(1, int(topk)),
            include=["documents", "metadatas", "distances"],
        )
    except TypeError:
        return col.query(query_embeddings=[q_emb], n_results=max(1, int(topk)))

# 모델 답변 + 관련 뉴스
def gemini_answer_with_news(question: str):
    lang = _detect_lang(question)
    prompt = _mk_news_prompt(question, lang)
    answer = _ask_gemini(prompt, model=None)
    try:
        topk = int(getattr(settings, "NEWS_TOPK", 5))
        news = _search_news_rss(question, topk)
    except Exception:
        news = []
    if not news:
        urls = extract_links_from_text(answer, max_n=5)
        news = [
            {
                "title": u,
                "url": u,
                "source": urlparse(u).netloc,
                "published_at": "",
                "snippet": "",
            } for u in urls
        ]
    return (answer or "").strip(), news

# 인덱싱
def _indexto_chroma_safe(query: str, answer: str, news: List[Dict[str, Any]]):
    if not getattr(settings, "WEB_INGEST_TO_CHROMA", os.environ.get("WEB_INGEST_TO_CHROMA", "1") not in ("0", "false", "False")):
        return None
    size = int(getattr(settings, "EMBED_CHUNK_SIZE", os.environ.get("EMBED_CHUNK_SIZE", "1600")))
    overlap = int(getattr(settings, "EMBED_CHUNK_OVERLAP", os.environ.get("EMBED_CHUNK_OVERLAP", "200")))
    now = datetime.utcnow().isoformat()
    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []
    a_chunks = _chunk_text(answer, size=size, overlap=overlap)
    base_a = f"answer:{_sha(query)}"
    for i, ch in enumerate(a_chunks):
        if not ch.strip():
            continue
        ids.append(f"{base_a}:{i}")
        docs.append(ch)
        metas.append({"source": "web_answer", "title": "웹검색 답변", "question": query, "ingested_at": now})
    news_summaries: List[Dict[str, Any]] = []
    min_chars = int(getattr(settings, "MIN_NEWS_BODY_CHARS", 400))
    for art in (news or []):
        url = (art.get("final_url") or art.get("url") or "").strip()
        title = (art.get("title") or "").strip() or (urlparse(url).netloc if url else "뉴스")
        body = (art.get("news_body") or "").strip()
        if not (url and body and len(body) >= min_chars):
            news_summaries.append({"title": title or url or "뉴스", "url": url, "chunks": 0})
            continue
        chunks = _chunk_text(body, size=size, overlap=overlap)
        base = f"news:{_slug(title)}:{_sha(url)}"
        cnt = 0
        for i, ch in enumerate(chunks):
            if not ch.strip():
                continue
            ids.append(f"{base}:{i}")
            docs.append(ch)
            metas.append({
                "source": "news",
                "url": url,
                "title": title,
                "source_name": art.get("source", ""),
                "published_at": art.get("published_at", ""),
                "ingested_at": now,
            })
            cnt += 1
        news_summaries.append({"title": title, "url": url, "chunks": cnt})
    link_summaries: List[Dict[str, Any]] = []
    link_total_chunks = 0
    if getattr(settings, "CRAWL_ANSWER_LINKS", os.environ.get("CRAWL_ANSWER_LINKS", "1") not in ("0", "false", "False")):
        max_links = int(getattr(settings, "ANSWER_LINK_MAX", os.environ.get("ANSWER_LINK_MAX", "5")))
        timeout_s = int(getattr(settings, "ANSWER_LINK_TIMEOUT", os.environ.get("ANSWER_LINK_TIMEOUT", "12")))
        urls = _extract_urls(answer)[:max(0, max_links)]
        for u in urls:
            body = _fetch_article_text(u, timeout=timeout_s)
            cnt = 0
            if body:
                chunks = _chunk_text(body, size=size, overlap=overlap)
                base = f"anslink:{_slug(urlparse(u).netloc)}:{_sha(u)}"
                for i, ch in enumerate(chunks):
                    if not ch.strip():
                        continue
                    ids.append(f"{base}:{i}")
                    docs.append(ch)
                    metas.append({"source": "answer_link", "url": u, "question": query, "ingested_at": now})
                    cnt += 1
                link_total_chunks += cnt
            link_summaries.append({"url": u, "chunks": cnt})
    clean = [(i, d, m) for i, d, m in zip(ids, docs, metas) if d and d.strip()]
    if not clean:
        return {
            "inserted": 0,
            "answer_chunks": 0,
            "news_total_chunks": 0,
            "answer_link_total_chunks": 0,
            "news_items": news_summaries,
            "answer_links": link_summaries,
            "collection": settings.CHROMA_COLLECTION,
            "dir": settings.CHROMA_DB_DIR,
            "ingested_at": now,
            "note": "인덱싱할 데이터가 없습니다.",
        }
    ids, docs, metas = map(list, zip(*clean))
    embs = _embed_texts(docs)
    _chroma_upsert(ids=ids, docs=docs, metas=metas, embs=embs)
    ans_chunks = sum(1 for m in metas if m.get("source") == "web_answer")
    news_chunks = sum(1 for m in metas if m.get("source") == "news")
    return {
        "inserted": len(ids),
        "answer_chunks": ans_chunks,
        "news_total_chunks": news_chunks,
        "answer_link_total_chunks": link_total_chunks,
        "news_items": news_summaries,
        "answer_links": link_summaries,
        "collection": settings.CHROMA_COLLECTION,
        "dir": settings.CHROMA_DB_DIR,
        "ingested_at": now,
    }

# 뷰: 홈
def home(request):
    mode = (request.GET.get("mode") or "").strip().lower()
    q = (request.GET.get("q") or "").strip()
    ingest = request.GET.get("ingest") == "1"
    if request.method == "GET" and not request.GET:
        request.session.pop("gemini_state", None)
        request.session.pop("rag_state", None)
        ctx = {
            "model_name_gemini": getattr(settings, "GEMINI_MODEL_DIRECT", None) or _gemini_model(),
            "model_name_rag": getattr(settings, "GEMINI_MODEL_RAG", None) or _gemini_model(),
            "q_gemini": "",
            "gemini_answer": "",
            "gemini_error": "",
            "news_list": [],
            "ingest_result": "",
            "ingest_error": "",
            "q_rag": "",
            "rag_answer": "",
            "rag_error": "",
            "rag_sources": [],
        }
        resp = render(request, "news.html", ctx)
        resp["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp["Pragma"] = "no-cache"
        resp["Expires"] = "0"
        return resp
    gemini_state = request.session.get("gemini_state") or {}
    rag_state = request.session.get("rag_state") or {}
    q_gemini_saved = gemini_state.get("q", "")
    q_rag_saved = rag_state.get("q", "")
    gemini_answer_text = gemini_state.get("answer", "")
    news_list = gemini_state.get("news", [])
    gemini_error = ""
    ingest_result = ""
    ingest_error = ""
    rag_answer_text = rag_state.get("answer", "")
    rag_sources = rag_state.get("sources", [])
    rag_error = ""
    if mode == "gemini" and q:
        try:
            gemini_answer_text, news_list = gemini_answer_with_news(q)
            request.session["gemini_state"] = {
                "q": q,
                "answer": gemini_answer_text,
                "news": [
                    {
                        "title": n.get("title", ""),
                        "url": n.get("url", ""),
                        "source": n.get("source", ""),
                        "published_at": n.get("published_at", ""),
                        "snippet": n.get("snippet", ""),
                    } for n in (news_list or [])
                ],
            }
            request.session.modified = True
        except Exception as e:
            gemini_error = f"Gemini 오류: {e}"
        try:
            AUTO_INGEST_AFTER_GEMINI = getattr(
                settings, "AUTO_INGEST_AFTER_GEMINI",
                os.environ.get("AUTO_INGEST_AFTER_GEMINI", "1").lower() not in ("0", "false", "no")
            )
        except Exception:
            AUTO_INGEST_AFTER_GEMINI = True
        if not gemini_error and (AUTO_INGEST_AFTER_GEMINI or ingest):
            try:
                news_with_bodies = _crawl_news_bodies(news_list, max_workers=6) if news_list else []
                res = _indexto_chroma_safe(q, gemini_answer_text, news_with_bodies)
                inserted = res.get("inserted", 0) if res else 0
                total_chunks = res.get("news_total_chunks", 0) if res else 0
                ingest_result = f"{inserted}개 저장, 뉴스 청크 {total_chunks}개"
            except Exception as e:
                ingest_error = f"ingest 오류: {e}"
        q_gemini = q
    else:
        q_gemini = q_gemini_saved
    if mode == "rag" and q:
        try:
            topk = max(1, int(getattr(settings, "RAG_QUERY_TOPK", 5)))
            fallback_topk = max(topk + 5, int(getattr(settings, "RAG_FALLBACK_TOPK", 12)))
            rag_answer_text, hits = _rag_answer_best_effort(q, initial_topk=topk, fallback_topk=fallback_topk)
            rag_sources = [
                f"[{i+1}] {(h['meta'].get('title') or h['meta'].get('url') or '문서')} · "
                f"{h['meta'].get('source_name') or h['meta'].get('source') or ''}".strip(" ·")
                for i, h in enumerate(hits)
            ]
            request.session["rag_state"] = {
                "q": q,
                "answer": rag_answer_text,
                "sources": rag_sources,
            }
            request.session.modified = True
            q_rag = q
        except Exception as e:
            rag_error = f"RAG 오류: {e}"
            q_rag = q
    else:
        q_rag = q_rag_saved
    ctx = {
        "model_name_gemini": getattr(settings, "GEMINI_MODEL_DIRECT", None) or _gemini_model(),
        "model_name_rag": getattr(settings, "GEMINI_MODEL_RAG", None) or _gemini_model(),
        "q_gemini": q_gemini,
        "gemini_answer": gemini_answer_text,
        "gemini_error": gemini_error,
        "news_list": news_list,
        "ingest_result": ingest_result,
        "ingest_error": ingest_error,
        "q_rag": q_rag,
        "rag_answer": rag_answer_text,
        "rag_error": rag_error,
        "rag_sources": rag_sources,
    }
    resp = render(request, "news.html", ctx)
    resp["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp["Pragma"] = "no-cache"
    resp["Expires"] = "0"
    return resp

# 뷰: API - 검색
@csrf_exempt
@require_http_methods(["POST"])
def api_search(request: HttpRequest):
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except Exception:
        return _fail("유효한 JSON이 아닙니다.")
    mode = (payload.get("mode") or "web").strip().lower()
    query = (payload.get("query") or "").strip()
    model = (payload.get("model") or "").strip() or None
    if not query:
        return _fail("query가 비었습니다.")
    if mode == "web":
        # 질의 언어에 맞게 직접 질의 전달 (모델이 자동 처리)
        answer = _ask_gemini(query, model=model)
        news, crawl_err = [], None
        try:
            topk = int(getattr(settings, "NEWS_TOPK", os.environ.get("NEWS_TOPK", "5")))
            news = _search_news_rss(query, topk)
            news = _crawl_news_bodies(news, max_workers=4)
        except Exception as e:
            crawl_err = str(e)
        ingest_summary, ingest_error = None, None
        try:
            ingest_summary = _indexto_chroma_safe(query, answer, news)
        except Exception as e:
            ingest_error = str(e)
        safe_news = [
            {
                "title": n.get("title", ""),
                "url": n.get("url", ""),
                "source": n.get("source", ""),
                "published_at": n.get("published_at", ""),
                "snippet": n.get("snippet", ""),
            } for n in (news or [])
        ]
        ok_cnt = sum(1 for n in news if (n.get("news_body") or ""))
        total = len(news or [])
        crawl_err = (crawl_err or "")
        crawl_err = f"{crawl_err} | news bodies: {ok_cnt}/{total}".strip(" |")
        return _ok({
            "mode": "web",
            "model": (model or _gemini_model()),
            "text": answer,
            "news": safe_news,
            "ingest": ingest_summary,
            "warnings": {"crawl": crawl_err, "ingest": ingest_error},
        })
    elif mode == "rag":
        try:
            col = _chroma_collection()
            if getattr(settings, "RAG_AUTO_SEED_IF_EMPTY", True) and _chroma_count(col) == 0:
                _rag_seed_internal(level=True)
            topk = max(1, int(getattr(settings, "RAG_QUERY_TOPK", 5)))
            res = _chroma_query_with_embeddings(col, query, topk)
            docs = (res.get("documents") or [[]])[0]
            metas = (res.get("metadatas") or [[]])[0]
            ids = (res.get("ids") or [[]])[0]
            dists = (res.get("distances") or [[]])[0]
            hits = []
            for i, doc in enumerate(docs):
                if not doc:
                    continue
                snip = (doc[:500] if isinstance(doc, str) else str(doc)).replace("\n", " ").strip()
                m = metas[i] if i < len(metas) else {}
                score = float(dists[i]) if (dists and i < len(dists) and dists[i] is not None) else None
                hits.append({"id": ids[i] if i < len(ids) else "", "score": score, "meta": m, "snippet": snip})
            if not hits:
                reason = (
                    f"검색 결과 없음 (collection='{settings.CHROMA_COLLECTION}', "
                    f"dir='{settings.CHROMA_DB_DIR}', count={_chroma_count(col)})"
                )
                return _ok({"mode": "rag", "model": (model or _gemini_model()), "text": "[검색 결과 없음]", "hits": [], "reason": reason})
            context = "\n\n".join(f"[{i+1}] {h['snippet']}" for i, h in enumerate(hits))
            lang = _detect_lang(query)
            prompt = _make_rag_prompt(query, context, lang)
            text = _ask_gemini(prompt, model=model)
            return _ok({"mode": "rag", "model": (model or _gemini_model()), "text": text, "hits": hits})
        except Exception as e:
            return _fail(f"RAG 검색 실패: {e}")
    return _fail(f"알 수 없는 mode: {mode}")

# 진단/유틸 엔드포인트들
@require_http_methods(["GET"])
def api_ping(request: HttpRequest):
    return _ok({"pong": "Pong!"})

@require_http_methods(["GET"])
def api_config(request: HttpRequest):
    p = settings.CHROMA_DB_DIR
    exists = os.path.isdir(p)
    writable = False
    write_error = None
    try:
        Path(p).mkdir(parents=True, exist_ok=True)
        test_path = Path(p) / ".write_test"
        with open(test_path, "w", encoding="utf-8") as f:
            f.write("ok")
        writable = True
        try:
            test_path.unlink()
        except Exception:
            pass
    except Exception as e:
        write_error = str(e)
    return _ok({
        "WEB_INGEST_TO_CHROMA": bool(getattr(settings, "WEB_INGEST_TO_CHROMA", True)),
        "CHROMA_DB_DIR": p,
        "dir_exists": exists,
        "dir_writable": writable,
        "dir_write_error": write_error,
        "CHROMA_COLLECTION": getattr(settings, "CHROMA_COLLECTION", ""),
        "GEMINI_API_KEY_set": bool(getattr(settings, "GEMINI_API_KEY", None) or os.environ.get("GEMINI_API_KEY")),
        "embedding_models_probe": getattr(settings, "GEMINI_EMBED_MODELS", ["text-embedding-004"]),
    })

@require_http_methods(["GET"])
def api_diag(request: HttpRequest):
    steps = []
    key_ok = bool(getattr(settings, "GEMINI_API_KEY", None) or os.environ.get("GEMINI_API_KEY"))
    steps.append({"name": "api_key", "ok": key_ok})
    if not key_ok:
        return _ok({"ok": False, "steps": steps})
    try:
        txt = _ask_gemini("ping")[:30]
        steps.append({"name": "chat", "ok": True, "text": txt})
    except Exception as e:
        steps.append({"name": "chat", "ok": False, "error": str(e)})
        return _ok({"ok": False, "steps": steps})
    tried_models = getattr(settings, "GEMINI_EMBED_MODELS", ["text-embedding-004"])
    try:
        vecs = _embed_texts(["hello world"])
        dim = len(vecs[0]) if vecs else 0
        steps.append({
            "name": "embed",
            "ok": True,
            "dim": dim,
            "tried": tried_models,
            "used_param": LAST_EMBED_META.get("param"),
            "used_model": LAST_EMBED_META.get("model"),
        })
    except Exception as e:
        steps.append({"name": "embed", "ok": False, "error": str(e), "tried": tried_models})
        return _ok({"ok": False, "steps": steps})
    try:
        v = vecs[0]
        col = _chroma_collection()
        col.add(ids=["diag:1"], documents=["diag"], metadatas=[{"source": "diag"}], embeddings=[v])
        steps.append({"name": "chroma_add", "ok": True, "dir": settings.CHROMA_DB_DIR, "collection": settings.CHROMA_COLLECTION})
        try:
            col.delete(ids=["diag:1"])
        except Exception:
            pass
    except Exception as e:
        steps.append({"name": "chroma_add", "ok": False, "error": str(e)})
        return _ok({"ok": False, "steps": steps})
    return _ok({"ok": True, "steps": steps})

@require_http_methods(["GET"])
def api_chroma_verify(request: HttpRequest):
    try:
        q = (request.GET.get("question") or request.GET.get("q") or "").strip()
        if not q:
            return _fail("question 파라미터가 필요합니다. 예: /api/chroma_verify?question=당신의_질문")
        col = _chroma_collection()
        data = None
        err_where = None
        try:
            data = col.get(where={"question": q}, include=["metadatas", "documents", "ids"])
        except Exception as e:
            err_where = str(e)
        if not data:
            try:
                try:
                    data = col.get(include=["metadatas", "documents", "ids"])
                except TypeError:
                    data = col.get()
            except Exception as e:
                return _fail("Chroma get() 실패", {"reason": str(e)})
            _ids = data.get("ids") or []
            _docs = data.get("documents") or []
            _metas = data.get("metadatas") or []
            ids, docs, metas = [], [], []
            for i, m in enumerate(_metas):
                try:
                    if isinstance(m, dict) and (m.get("question") == q):
                        ids.append(_ids[i] if i < len(_ids) else "")
                        docs.append(_docs[i] if i < len(_docs) else "")
                        metas.append(m)
                except Exception:
                    continue
            data = {"ids": ids, "documents": docs, "metadatas": metas, "_where_error": err_where}
        def _flatten(v):
            if v and isinstance(v, list) and len(v) == 1 and isinstance(v[0], list):
                return v[0]
            return v or []
        ids = _flatten(data.get("ids"))
        docs = _flatten(data.get("documents"))
        metas = _flatten(data.get("metadatas"))
        total = len(ids)
        ans = link = news = other = 0
        for m in metas:
            if not isinstance(m, dict):
                other += 1
                continue
            src = m.get("source")
            if src == "web_answer":
                ans += 1
            elif src == "answer_link":
                link += 1
            elif src == "news":
                news += 1
            else:
                other += 1
        sample = []
        for i in range(min(5, total)):
            snippet = (docs[i] or "")[:160].replace("\n", " ") if i < len(docs) else ""
            sample.append({
                "id": ids[i] if i < len(ids) else "",
                "source": (metas[i].get("source") if (i < len(metas) and isinstance(metas[i], dict)) else ""),
                "url": (metas[i].get("url") if (i < len(metas) and isinstance(metas[i], dict)) else ""),
                "snippet": snippet
            })
        return _ok({
            "verify": {
                "question": q,
                "total": total,
                "answer_chunks": ans,
                "answer_link_chunks": link,
                "news_chunks": news,
                "other_chunks": other,
                "collection": settings.CHROMA_COLLECTION,
                "dir": settings.CHROMA_DB_DIR,
                "sample": sample
            },
            "debug": {
                "where_error": data.get("_where_error"),
                "shapes": {"ids": type(ids).__name__, "docs": type(docs).__name__, "metas": type(metas).__name__}
            }
        })
    except Exception as e:
        return _fail("검증 처리 실패", {"exception": str(e)})

# RAG 시드/진단
def _rag_seed_internal(level: bool = False):
    col = _chroma_collection()
    docs = [
        "새 DB의 첫 문서입니다. 이것은 RAG 동작 점검용 샘플 텍스트입니다.",
        "두 번째 문서입니다. RAG 검색이 정상 동작하는지 확인하세요.",
    ]
    ids = ["seed:doc1", "seed:doc2"]
    metas = [{"source": "seed", "title": "doc1"}, {"source": "seed", "title": "doc2"}]
    embs = _embed_texts(docs)
    if hasattr(col, "upsert"):
        col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
    else:
        try:
            col.delete(ids=ids)
        except Exception:
            pass
        col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
    return {"seeded": len(ids)}

@csrf_exempt
@require_http_methods(["POST"])
def api_rag_seed(request: HttpRequest):
    try:
        info = _rag_seed_internal(level=True)
        col = _chroma_collection()
        return _ok({"seed": info, "count": _chroma_count(col)})
    except Exception as e:
        return _fail(f"RAG 시드 실패: {e}")

@require_http_methods(["GET"])
def api_rag_diag(request: HttpRequest):
    try:
        col = _chroma_collection()
        count = _chroma_count(col)
        sample = []
        try:
            got = col.get(limit=3, include=["documents", "metadatas", "ids"])
            ids = got.get("ids") or []
            docs = got.get("documents") or []
            metas = got.get("metadatas") or []
            for i in range(min(3, len(ids))):
                sample.append({
                    "id": ids[i],
                    "snippet": (docs[i] or "")[:120].replace("\n", " "),
                    "meta": metas[i] if i < len(metas) else {},
                })
        except Exception:
            pass
        return _ok({"dir": settings.CHROMA_DB_DIR, "collection": settings.CHROMA_COLLECTION, "count": count, "sample": sample})
    except Exception as e:
        return _fail(f"RAG 진단 실패: {e}")

# 외부에서 쓰는 래퍼
def ask_gemini(prompt: str, model: Optional[str] = None) -> str:
    return _ask_gemini(prompt, model=model)

def embed_texts(texts: List[str]) -> List[List[float]]:
    return _embed_texts(texts)

# 별도 엔드포인트
@csrf_exempt
@require_http_methods(["GET", "POST"])
def web_qa_view(request: HttpRequest):
    q = None
    model = None
    if request.method == "GET":
        q = (request.GET.get("q") or request.GET.get("query") or request.GET.get("question") or "").strip()
        model = (request.GET.get("model") or "").strip() or None
    else:
        try:
            payload = json.loads(request.body.decode("utf-8") or "{}")
        except Exception:
            payload = request.POST
        q = (payload.get("query") or payload.get("q") or payload.get("question") or "").strip()
        model = (payload.get("model") or "").strip() or None
    if not q:
        return _fail("query가 비었습니다.")
    # 직접 질의 전달 → 입력 언어에 맞춰 생성됨
    answer = _ask_gemini(q, model=model)
    news, crawl_err = [], None
    try:
        topk = int(getattr(settings, "NEWS_TOPK", os.environ.get("NEWS_TOPK", "5")))
        news = _search_news_rss(q, topk)
        news = _crawl_news_bodies(news, max_workers=4)
    except Exception as e:
        crawl_err = str(e)
    ingest_summary, ingest_error = None, None
    try:
        ingest_summary = _indexto_chroma_safe(q, answer, news)
    except Exception as e:
        ingest_error = str(e)
    safe_news = [
        {
            "title": n.get("title", ""),
            "url": n.get("url", ""),
            "source": n.get("source", ""),
            "published_at": n.get("published_at", ""),
            "snippet": n.get("snippet", ""),
        } for n in (news or [])
    ]
    return _ok({
        "mode": "web",
        "model": (model or _gemini_model()),
        "text": answer,
        "news": safe_news,
        "ingest": ingest_summary,
        "warnings": {"crawl": crawl_err, "ingest": ingest_error},
    })

@csrf_exempt
@require_http_methods(["GET", "POST"])
def rag_qa_view(request: HttpRequest):
    q = None
    model = None
    if request.method == "GET":
        q = (request.GET.get("q") or request.GET.get("query") or request.GET.get("question") or "").strip()
        model = (request.GET.get("model") or "").strip() or None
    else:
        try:
            payload = json.loads(request.body.decode("utf-8") or "{}")
        except Exception:
            payload = request.POST
        q = (payload.get("query") or payload.get("q") or payload.get("question") or "").strip()
        model = (payload.get("model") or "").strip() or None
    if not q:
        return _fail("query가 비었습니다.")
    try:
        col = _chroma_collection()
        if getattr(settings, "RAG_AUTO_SEED_IF_EMPTY", True) and _chroma_count(col) == 0:
            _rag_seed_internal(level=True)
        topk = max(1, int(getattr(settings, "RAG_QUERY_TOPK", 5)))
        res = _chroma_query_with_embeddings(col, q, topk)
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        ids = (res.get("ids") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        hits = []
        for i, doc in enumerate(docs):
            if not doc:
                continue
            snip = (doc[:500] if isinstance(doc, str) else str(doc)).replace("\n", " ").strip()
            m = metas[i] if i < len(metas) else {}
            score = float(dists[i]) if (dists and i < len(dists) and dists[i] is not None) else None
            hits.append({"id": ids[i] if i < len(ids) else "", "score": score, "meta": m, "snippet": snip})
        if not hits:
            reason = (
                f"검색 결과 없음 (collection='{settings.CHROMA_COLLECTION}', "
                f"dir='{settings.CHROMA_DB_DIR}', count={_chroma_count(col)})"
            )
            return _ok({"mode": "rag", "model": (model or _gemini_model()), "text": "[검색 결과 없음]", "hits": [], "reason": reason})
        context = "\n\n".join(f"[{i+1}] {h['snippet']}" for i, h in enumerate(hits))
        lang = _detect_lang(q)
        prompt = _make_rag_prompt(q, context, lang)
        text = _ask_gemini(prompt, model=model)
        return _ok({"mode": "rag", "model": (model or _gemini_model()), "text": text, "hits": hits})
    except Exception as e:
        return _fail(f"RAG 검색 실패: {e}")

# 데모 템플릿
def result_view(request):
    model_answer = request.GET.get("answer") or "여기에 모델 생성 답변을 넣어 주세요."
    news_list = [
        {
            "title": "점심식사 후 마시는 커피, 뇌에 '이런' 영향 미친다",
            "url": "https://example.com/news/1",
            "source": "헬스조선",
            "published_at": "Tue, 22 Jul 2025 07:00:00 GMT",
            "snippet": "연구팀은 식후 카페인이 인지 기능에 미치는 ...",
        },
    ]
    return render(request, "chroma", {
        "model_name": "gemini-2.0-flash",
        "model_answer": model_answer,
        "news_list": news_list,
        "now": timezone.now().strftime("%Y-%m-%d %H:%M"),
    })

# 뉴스만 수집/본문 크롤링 후 Chroma에 저장
@require_http_methods(["GET"])
def api_news_ingest(request: HttpRequest):
    q = (request.GET.get("q") or request.GET.get("query") or "").strip()
    if not q:
        return _fail("q 파라미터 필요: /api/news_ingest?q=질문")

    try:
        # 1) 뉴스 RSS 수집 + 본문 크롤링
        topk = int(getattr(settings, "NEWS_TOPK", os.environ.get("NEWS_TOPK", "5")))
        news = _search_news_rss(q, topk)
        news = _crawl_news_bodies(news, max_workers=6)

        # 2) 인덱싱: 답변은 비우고(news만 저장)
        ingest_summary = _indexto_chroma_safe(q, answer="", news=news)

        # 3) 클라이언트에 안전한 메타만 반환
        safe_news = [
            {
                "title": n.get("title", ""),
                "url": n.get("url", ""),
                "source": n.get("source", ""),
                "published_at": n.get("published_at", ""),
                "snippet": n.get("snippet", ""),
            }
            for n in (news or [])
        ]
        return _ok({"query": q, "news": safe_news, "insgest": ingest_summary})
    except Exception as e:
        return _fail(f"뉴스 인덱싱 실패: {e}")
    
# ---------- RAG 베스트-에포트 파이프라인 유틸 ----------

def _parse_hits_from_res(res):
    def _pick(v):
        return v[0] if (isinstance(v, list) and v and isinstance(v[0], list)) else (v or [])
    docs  = _pick(res.get("documents"))
    metas = _pick(res.get("metadatas"))
    ids   = _pick(res.get("ids")) if "ids" in res else [""] * len(docs)
    dists = _pick(res.get("distances"))
    hits = []
    for i, doc in enumerate(docs):
        if not doc:
            continue
        snip = (doc[:800] if isinstance(doc, str) else str(doc)).replace("\n", " ").strip()
        m = metas[i] if i < len(metas) else {}
        score = float(dists[i]) if (dists and i < len(dists) and dists[i] is not None) else None
        hits.append({"id": ids[i] if i < len(ids) else "", "score": score, "meta": m, "snippet": snip})
    return hits

def _rag_answer_best_effort(question: str, initial_topk: int = 5, fallback_topk: int = 12):
    """
    1) 1차 검색(topk) → 컨텍스트 생성 → 답변
    2) 답변이 빈약(짧음/특정 문구 포함)하면:
       - 키워드 확장(LLM) → 더 넓은 topk로 재검색 → 재답변
    """
    col = _chroma_collection()
    sources_filter = getattr(settings, "RAG_SOURCES_FILTER", None)

    # --- 1차 검색
    try:
        res = _chroma_query_with_embeddings(col, question, initial_topk, sources_filter=sources_filter)
    except TypeError:
        res = _chroma_query_with_embeddings(col, question, initial_topk)

    hits = _parse_hits_from_res(res)
    context = "\n\n".join(f"[{i+1}] {h['snippet']}" for i, h in enumerate(hits))
    lang = _detect_lang(question)
    ans = _ask_gemini(_make_rag_prompt(question, context, lang), model=None)

    def _weak(a: str) -> bool:
        t = (a or "").strip().lower()
        # 너무 짧거나 '없음/없다/not found/no aparece' 류의 빈약 응답
        return (not t) or (len(t) < 60) or ("본문에 없음" in t) or ("not found" in t) or ("no aparece" in t) or ("no se encuentra" in t)

    if not hits or _weak(ans):
        # --- 2차: 키워드 확장 + 더 넓은 topk
        try:
            if lang == "ko":
                kw_prompt = f"아래 질문의 한국어 핵심 키워드를 쉼표로 8~12개만 나열해줘. 설명 없이 키워드만.\n질문: {question}"
            elif lang == "es":
                kw_prompt = f"Da de 8 a 12 palabras clave esenciales en español para la siguiente pregunta, separadas por comas, sin explicación.\nPregunta: {question}"
            else:
                kw_prompt = f"List 8–12 essential English keywords for the question below, separated by commas, with no explanations.\nQuestion: {question}"
            kw = _ask_gemini(kw_prompt, model=None)
        except Exception:
            kw = ""
        expanded_q = (question + " " + (kw or "")).strip()
        try:
            res2 = _chroma_query_with_embeddings(col, expanded_q, fallback_topk, sources_filter=None)
        except TypeError:
            res2 = _chroma_query_with_embeddings(col, expanded_q, fallback_topk)

        hits2 = _parse_hits_from_res(res2)
        if hits2:
            context2 = "\n\n".join(f"[{i+1}] {h['snippet']}" for i, h in enumerate(hits2))
            ans2 = _ask_gemini(_make_rag_prompt(question, context2, lang), model=None)
            if not ans or len((ans2 or "").strip()) > len((ans or "").strip()):
                return ans2, hits2

    return ans, hits
