"""Open benchmark of Keiro /search/lite on SimpleQA, one level at a time.

  L0  lite only            titles + snippets, exactly what the endpoint returns
  L1  lite + crawl         fetch lite's top CRAWL_K pages with the crawler below
  L2  lite + crawl + pick  BM25 picks the best ~BUDGET tokens of passages, no model

Per question and level: is the gold answer there (normalized substring), and
if not, does an LLM judge find it in a variant form (same rules as our deep
benchmark). Timings per stage and token counts are recorded.

The crawler, fully shown:
  tier 1  curl_cffi GET with Chrome's TLS fingerprint, trafilatura extraction
  tier 2  headless Chromium (Playwright) for JS pages and pages tier 1 got walled on
Extraction runs in worker processes (lxml is not safe across threads).

The open recipe (each step a switch, all off by default):
  FILTER=1   drop results with no readable text (video, chat, translate, search pages)
  VARIANT=1  a second lite call on the keyword-only question, merged by rank
  WIKI=1     Wikipedia's own search, merged by rank
  CRAWL_K=8  crawl the top 8 of the merged list

Env: KEIRO_API_KEY, OPENROUTER_API_KEY, N (500), SEED (42), OFFSET (1000),
     CRAWL_K (5), BUDGET (1000), CONC (4), QIDS, JUDGE_MODEL, OUT (/out/results.json)
"""
import asyncio
import csv
import json
import math
import os
import random
import re
import time
import urllib.request
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import trafilatura
from curl_cffi.requests import AsyncSession

env = os.getenv
KEIRO_KEY = os.environ["KEIRO_API_KEY"]
OPENROUTER_KEY = os.environ["OPENROUTER_API_KEY"]
LITE_URL = env("LITE_URL", "https://api.keirolabs.cloud/api/v2/search/lite")
N, SEED, OFFSET = int(env("N", "500")), int(env("SEED", "42")), int(env("OFFSET", "1000"))
CRAWL_K, BUDGET, CONC = int(env("CRAWL_K", "5")), int(env("BUDGET", "1000")), int(env("CONC", "4"))
JUDGE_MODEL = env("JUDGE_MODEL", "minimax/minimax-m3.1-flash")
CSV = env("SIMPLEQA_CSV", "/data/simple_qa_test_set.csv")
OUT = env("OUT", "/out/results.json")
PAGE_DEADLINE = 9
# pipeline options (all off = the plain L1 measured first)
FILTER = env("FILTER", "0") == "1"    # drop results with no readable text (video, chat, translate, search pages)
VARIANT = env("VARIANT", "0") == "1"  # second lite call on the keyword-only question, merged by rank
WIKI = env("WIKI", "0") == "1"        # Wikipedia's own search, merged by rank
QIDS = env("QIDS", "")               # file of question texts: run only those (for testing on misses)

# ─── Scoring ─────────────────────────────────────────────────────────────────


def norm(s: str) -> str:
    s = re.sub(r"\b(a|an|the)\b", " ", s.lower())
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s)).strip()


def has(texts: list[str], gold: str) -> bool:
    g = norm(gold)
    return any(t and g in norm(t) for t in texts)


def windows(texts: list[str], gold: str, n: int = 6) -> list[str]:
    """Text around the gold answer's rarest words: what the judge reads."""
    words = [w for w in norm(gold).split() if len(w) >= 3 or w.isdigit()] or norm(gold).split()
    texts = [t for t in texts if t]
    if not words or not texts:
        return []
    rare = sorted(words, key=lambda w: sum(t.lower().count(w) for t in texts) or 1e9)[:2]
    out = []
    for t in texts:
        low = t.lower()
        for w in rare:
            if (i := low.find(w)) >= 0:
                out.append(t[max(0, i - 400):i + 400])
                break
        if len(out) >= n:
            break
    return out


async def judge(s: AsyncSession, q: str, gold: str, texts: list[str]) -> bool:
    ex = "\n---\n".join(windows(texts, gold))[:12000]
    if not ex:
        return False
    prompt = (f"Question: {q}\nGold answer: {gold}\n\nExcerpts:\n{ex}\n\n"
              "Does any excerpt explicitly state the gold answer? Count variant forms: date order, "
              "abbreviations, numerals vs words, transliterations, paraphrases, every part of a compound "
              "answer stated. Do NOT count answers you would have to compute or infer, partial answers, "
              "or a different value. Reply with exactly YES or NO.")
    for _ in range(3):
        try:
            r = await s.post("https://openrouter.ai/api/v1/chat/completions", timeout=60,
                             headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
                             json={"model": JUDGE_MODEL, "temperature": 0, "max_tokens": 800, "reasoning": {"effort": "minimal"},
                                   "messages": [{"role": "user", "content": prompt}]})
            words = re.findall(r"\b(YES|NO)\b", (r.json()["choices"][0]["message"]["content"] or "").upper())
            return bool(words) and words[-1] == "YES"  # last verdict: the model may restate the rules first
        except Exception:
            await asyncio.sleep(2)
    return False


def tokens(texts: list[str]) -> int:
    return sum(len(t) for t in texts) // 4

# ─── Lite ────────────────────────────────────────────────────────────────────


async def lite(s: AsyncSession, q: str) -> list[dict]:
    for attempt in range(4):
        r = await s.post(LITE_URL, json={"query": q, "maxResults": 10}, timeout=20,
                         headers={"Authorization": f"Bearer {KEIRO_KEY}"})
        if r.status_code == 200:
            return r.json().get("results") or []
        if r.status_code in (429, 502, 503, 504):
            await asyncio.sleep(3 * (attempt + 1))
            continue
        raise RuntimeError(f"lite {r.status_code}: {r.text[:120]}")
    raise RuntimeError("lite unavailable after retries")

_NO_TEXT = re.compile(r"(^|\.)(youtube\.com|youtu\.be|tiktok\.com|whatsapp\.com|instagram\.com|facebook\.com|"
                      r"pinterest\.[a-z.]+|x\.com|twitter\.com|yandex\.[a-z.]+|bing\.com)$|^translate\.google\.|^(www\.)?google\.[a-z.]+$")


def host(url: str) -> str:
    return (re.match(r"https?://([^/:?#]+)", url or "") or [None, ""])[1].lower()


def keyword_variant(q: str) -> str:
    t = [w for w in dict.fromkeys(re.findall(r"[a-z0-9]{3,}", q.lower())) if w not in _STOP]
    return " ".join(t) if len(t) >= 2 else ""


async def wiki(s: AsyncSession, q: str) -> list[dict]:
    try:
        r = await s.get("https://en.wikipedia.org/w/api.php", timeout=8, params={
            "action": "query", "list": "search", "srsearch": q, "srlimit": 3, "format": "json"})
        return [{"url": "https://en.wikipedia.org/wiki/" + h["title"].replace(" ", "_"), "title": h["title"],
                 "snippet": re.sub(r"<[^>]+>", "", h.get("snippet") or "")} for h in r.json()["query"]["search"]]
    except Exception:
        return []


async def gather_results(s: AsyncSession, q: str) -> tuple[list[dict], list[dict]]:
    """(lite as returned, the list the pipeline crawls from)."""
    base = await lite(s, q)
    lists = [base]
    kv = keyword_variant(q) if VARIANT else ""
    extra = await asyncio.gather(*([lite(s, kv)] if kv else []), *([wiki(s, kv or q)] if WIKI else []),
                                 return_exceptions=True)
    lists += [x for x in extra if isinstance(x, list)]
    seen, merged = set(), []
    for rank in range(max(map(len, lists))):
        for lst in lists:
            if rank < len(lst) and (u := lst[rank].get("url")):
                k = u.split("#")[0].rstrip("/")
                if k not in seen and not (FILTER and _NO_TEXT.search(host(u))):
                    seen.add(k)
                    merged.append(lst[rank])
    return base, merged


# ─── Crawler ─────────────────────────────────────────────────────────────────

_WALLS = ("just a moment", "captcha", "access denied", "unusual traffic", "verify you are human",
          "attention required", "enable javascript", "not a bot", "security check", "bad gateway")


def usable(text: str) -> bool:
    return len(text) >= 400 and not any(w in text[:800].lower() for w in _WALLS)


def extract(html: str) -> str:
    return (trafilatura.extract(html, favor_recall=True, include_tables=True) or "").strip()


_pool = None


async def extract_async(html: str) -> str:
    global _pool
    if _pool is None:
        _pool = ProcessPoolExecutor(4)
    try:
        return await asyncio.get_running_loop().run_in_executor(_pool, extract, html[:3_000_000])
    except BrokenProcessPool:
        _pool = None
        return ""


_ctx = None
_ctx_lock = asyncio.Lock()
_tabs = asyncio.Semaphore(8)


async def browser():
    global _ctx
    async with _ctx_lock:
        if _ctx is None:
            from playwright.async_api import async_playwright
            pw = await async_playwright().start()
            b = await pw.chromium.launch(args=["--disable-blink-features=AutomationControlled"])
            _ctx = await b.new_context(user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"))

            async def block(route):
                if route.request.resource_type in ("image", "media", "font", "stylesheet"):
                    await route.abort()
                else:
                    await route.continue_()
            await _ctx.route("**/*", block)
        return _ctx


async def tier_http(s: AsyncSession, url: str):
    r = await s.get(url, timeout=6, allow_redirects=True)
    if r.status_code >= 400 or "html" not in (r.headers.get("content-type") or "html"):
        return None
    text = await extract_async(r.text)
    return text if usable(text) else None


async def tier_browser(url: str):
    async with _tabs:
        page = await (await browser()).new_page()
        try:
            await page.goto(url, timeout=7000, wait_until="domcontentloaded")
            text = await extract_async(await page.content())
            if not usable(text):
                await page.wait_for_timeout(1500)
                text = (await page.inner_text("body") or "").strip()
            return text if usable(text) else None
        finally:
            await page.close()


async def crawl(s: AsyncSession, url: str) -> tuple[str, str]:
    async def ladder():
        for name, tier in (("http", lambda: tier_http(s, url)), ("browser", lambda: tier_browser(url))):
            try:
                if text := await tier():
                    return text[:100_000], name
            except Exception:
                continue
        return "", "blocked"
    try:
        return await asyncio.wait_for(ladder(), PAGE_DEADLINE)
    except Exception:
        return "", "timeout"

# ─── L2: BM25 passage pick ───────────────────────────────────────────────────

_STOP = set("""what whats which who whom whose when where why how did does do was were is are the of in
on at to for and that this from with by as it its be been provide me you can could would give name tell""".split())


def bm25_pick(q: str, pages: list[str], budget: int, size: int = 1200) -> list[str]:
    """Split pages into overlapping passages, BM25-rank them against the query
    (statistics from this query's own passages), pack best-first into budget."""
    qt = [w for w in re.findall(r"[a-z0-9]+", q.lower()) if w not in _STOP and len(w) > 1]
    ps = [p[i:i + size] for p in pages if p for i in range(0, max(1, len(p) - size // 2), size // 2)]
    if not ps or not qt:
        return []
    toks = [re.findall(r"[a-z0-9]+", p.lower()) for p in ps]
    avg = sum(map(len, toks)) / len(toks)
    df = Counter(w for t in toks for w in set(t))
    def score(t):
        tf = Counter(t)
        return sum(math.log(1 + (len(ps) - df[w] + 0.5) / (df[w] + 0.5)) * tf[w] * 2.2
                   / (tf[w] + 1.2 * (0.25 + 0.75 * len(t) / avg)) for w in qt if tf[w])
    out, taken, used = [], [], 0
    for i in sorted(range(len(ps)), key=lambda i: -score(toks[i])):
        if used + len(ps[i]) // 4 > budget:
            continue
        mine = set(toks[i])
        if any(len(mine & o) / max(1, len(mine | o)) >= 0.6 for o in taken):
            continue  # skip near-copies of a passage already taken
        out.append(ps[i])
        taken.append(mine)
        used += len(ps[i]) // 4
    return out

# ─── Run ─────────────────────────────────────────────────────────────────────


async def one(s: AsyncSession, row: dict) -> dict:
    q, gold = row["problem"].strip(), row["answer"].strip()
    t0 = time.time()
    res, pool = await gather_results(s, q)
    t1 = time.time()
    top = pool[:CRAWL_K]
    pages = await asyncio.gather(*[crawl(s, r["url"]) for r in top])
    t2 = time.time()
    texts = [p[0] for p in pages]
    picked = bm25_pick(q, texts, BUDGET)
    t3 = time.time()

    l0 = [f"{r.get('title', '')} {r.get('snippet', '')}" for r in res]
    l1 = l0 + [f"{r.get('title', '')} {r.get('snippet', '')}" for r in pool] + texts
    rec = {"q": q, "gold": gold, "results": len(res), "urls": [r.get("url") for r in res],
           "tiers": [p[1] for p in pages],
           "ms": {"lite": int((t1 - t0) * 1000), "crawl": int((t2 - t1) * 1000), "pick": int((t3 - t2) * 1000)},
           "tokens": {"L0": tokens(l0), "L1": tokens(l1), "L2": tokens(picked)}}
    for lvl, tx in (("L0", l0), ("L1", l1), ("L2", picked)):
        raw = has(tx, gold)
        rec[lvl] = {"raw": raw, "judged": raw or await judge(s, q, gold, tx)}
    # first lite rank whose snippet or crawled page holds the answer
    rec["rank"] = next((i for i, r in enumerate(top) if has([texts[i]], gold)), None)
    return rec


async def main():
    global N
    if not os.path.exists(CSV):
        urllib.request.urlretrieve("https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv", CSV)
    rows = list(csv.DictReader(open(CSV, newline="", encoding="utf-8")))
    random.seed(SEED)
    sample = random.sample(rows, OFFSET + N)[OFFSET:]
    if QIDS:
        want = set(json.load(open(QIDS)))
        sample = [r for r in sample if r["problem"].strip() in want]
        N = len(sample)
    print(f"{len(sample)} SimpleQA questions (seed {SEED}, offset {OFFSET}), crawl top {CRAWL_K}, budget {BUDGET}", flush=True)
    recs, sem = [], asyncio.Semaphore(CONC)
    async with AsyncSession(impersonate="chrome", max_clients=64) as s:
        async def run(row):
            async with sem:
                try:
                    rec = await one(s, row)
                except Exception as e:
                    rec = {"q": row["problem"], "gold": row["answer"], "error": str(e)[:200]}
            recs.append(rec)
            if "error" in rec:
                print(f"[{len(recs)}/{N}] ERROR {rec['error']}", flush=True)
            else:
                print(f"[{len(recs)}/{N}] L0={rec['L0']['judged']!s:5} L1={rec['L1']['judged']!s:5} "
                      f"L2={rec['L2']['judged']!s:5} {sum(rec['ms'].values()):6}ms | {rec['q'][:60]}", flush=True)
            if len(recs) % 50 == 0:
                json.dump({"records": recs}, open(OUT, "w"))
        await asyncio.gather(*[run(r) for r in sample])

    ok = [r for r in recs if "error" not in r]
    n = len(recs)
    pct = lambda k: sorted(sum(r["ms"].values()) if k == "total" else r["ms"][k] for r in ok)
    summary = {
        "n": n, "errors": n - len(ok), "seed": SEED, "offset": OFFSET, "crawl_k": CRAWL_K, "budget": BUDGET,
        "filter": FILTER, "variant": VARIANT, "wiki": WIKI,
        "levels": {lvl: {"raw": sum(r[lvl]["raw"] for r in ok), "judged": sum(r[lvl]["judged"] for r in ok),
                         "judged_pct": round(100 * sum(r[lvl]["judged"] for r in ok) / n, 1),
                         "avg_tokens": round(sum(r["tokens"][lvl] for r in ok) / max(1, len(ok)))}
                   for lvl in ("L0", "L1", "L2")},
        "answer_in_top1": sum(r["rank"] == 0 for r in ok),
        "answer_in_top3": sum(r["rank"] is not None and r["rank"] < 3 for r in ok),
        "latency_ms": {k: {"p50": (v := pct(k))[len(v) // 2], "p90": v[int(len(v) * 0.9)]}
                       for k in ("lite", "crawl", "pick", "total")} if ok else {},
        "crawl_tiers": dict(Counter(t for r in ok for t in r["tiers"])),
    }
    json.dump({"summary": summary, "records": recs}, open(OUT, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
