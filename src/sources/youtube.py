"""
YouTube source loader.
Fetches transcript, chunks it, and tags each chunk with video metadata + timestamp citation.

Strategy (most → least reliable on cloud platforms):
  1. youtube-transcript-api  — fast, direct; blocked by YouTube on some cloud IPs
  2. yt-dlp                  — mimics a real browser; bypasses most IP blocks
  3. Invidious proxy          — requests go through a third-party server, never from
                               our blocked cloud IP — most reliable on HF/Render/Railway
"""
from __future__ import annotations
import json
import os
import re
import ssl
import urllib.parse
import urllib.request
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config

# ── SSL fix for cloud platforms ────────────────────────────────────────────────
# Point requests/urllib3 to certifi's up-to-date bundle.
try:
    import certifi
    _CERT = certifi.where()
    os.environ.setdefault("REQUESTS_CA_BUNDLE", _CERT)
    os.environ.setdefault("SSL_CERT_FILE", _CERT)
except ImportError:
    _CERT = None


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _extract_video_id(url: str) -> str:
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"(?:youtu\.be\/)([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract video ID from URL: {url}")


def _format_timestamp(seconds: float) -> str:
    seconds = int(seconds)
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _build_ssl_ctx() -> ssl.SSLContext:
    if _CERT:
        return ssl.create_default_context(cafile=_CERT)
    return ssl.create_default_context()


# ──────────────────────────────────────────────────────────────────────────────
# Method 1: youtube-transcript-api
# ──────────────────────────────────────────────────────────────────────────────

def _fetch_via_transcript_api(video_id: str) -> List[dict]:
    """
    Returns list of {text, start} dicts.
    Raises on any failure so the caller can try the next method.
    """
    from youtube_transcript_api import YouTubeTranscriptApi
    api = YouTubeTranscriptApi()
    entries = api.fetch(video_id)
    result = []
    for e in entries:
        text  = e.text  if hasattr(e, "text")  else e["text"]
        start = e.start if hasattr(e, "start") else e["start"]
        result.append({"text": text, "start": float(start)})
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Method 2: yt-dlp (browser-like — works on most cloud platforms)
# ──────────────────────────────────────────────────────────────────────────────

def _parse_json3(data: dict) -> List[dict]:
    """Parse YouTube json3 subtitle format into [{text, start}]."""
    entries = []
    for event in data.get("events", []):
        segs = event.get("segs", [])
        if not segs:
            continue
        text = "".join(s.get("utf8", "") for s in segs).strip()
        if text and text != "\n":
            start = event.get("tStartMs", 0) / 1000.0
            entries.append({"text": text, "start": start})
    return entries


def _parse_vtt(vtt_text: str) -> List[dict]:
    """Parse WebVTT subtitle text into [{text, start}]."""
    entries = []
    lines = vtt_text.splitlines()
    i = 0
    while i < len(lines):
        # Look for timestamp lines: 00:00:00.000 --> 00:00:05.000
        if "-->" in lines[i]:
            start_str = lines[i].split("-->")[0].strip()
            # Parse HH:MM:SS.mmm or MM:SS.mmm
            parts = start_str.replace(",", ".").split(":")
            try:
                if len(parts) == 3:
                    start = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
                else:
                    start = int(parts[0]) * 60 + float(parts[1])
            except ValueError:
                i += 1
                continue
            i += 1
            text_lines = []
            while i < len(lines) and lines[i].strip():
                # Strip VTT tags like <00:00:00.000><c>text</c>
                clean = re.sub(r"<[^>]+>", "", lines[i]).strip()
                if clean:
                    text_lines.append(clean)
                i += 1
            if text_lines:
                entries.append({"text": " ".join(text_lines), "start": start})
        else:
            i += 1
    return entries


def _fetch_subtitle_url(url_str: str) -> List[dict]:
    """Fetch a subtitle URL and parse it (json3 or vtt)."""
    ctx = _build_ssl_ctx()
    req = urllib.request.Request(
        url_str,
        headers={"User-Agent": "Mozilla/5.0 (compatible; python-requests/2.32)"},
    )
    with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
        raw = resp.read().decode("utf-8", errors="replace")

    # json3 format starts with '{'
    if raw.strip().startswith("{"):
        return _parse_json3(json.loads(raw))
    # Otherwise assume VTT
    return _parse_vtt(raw)


def _fetch_via_ytdlp(video_id: str, url: str) -> Tuple[List[dict], str]:
    """
    Use yt-dlp to get subtitle URLs, then fetch them directly.
    Returns (transcript_entries, video_title).
    """
    import yt_dlp  # type: ignore

    ydl_opts = {
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
        # Ask yt-dlp to fetch subtitle info without writing files
        "writesubtitles": False,
        "writeautomaticsub": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    title = info.get("title", f"YouTube Video ({video_id})")

    manual_subs = info.get("subtitles", {})
    auto_subs   = info.get("automatic_captions", {})

    # Prefer manual English captions, fall back to auto-generated
    for lang in ["en", "en-US", "en-GB", "en-CA"]:
        for sub_dict in [manual_subs, auto_subs]:
            cap_list = sub_dict.get(lang)
            if not cap_list:
                continue
            # Prefer json3, then vtt, then any
            for preferred_ext in ["json3", "vtt"]:
                for fmt in cap_list:
                    if fmt.get("ext") == preferred_ext:
                        entries = _fetch_subtitle_url(fmt["url"])
                        if entries:
                            return entries, title
            # Fall back to first available format
            if cap_list:
                entries = _fetch_subtitle_url(cap_list[0]["url"])
                if entries:
                    return entries, title

    raise ValueError(
        f"No English subtitles found for this video via yt-dlp. "
        f"The video may have subtitles disabled or only non-English captions."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Method 3: Invidious proxy (most reliable on cloud — bypasses IP blocks)
# ──────────────────────────────────────────────────────────────────────────────

# Public Invidious instances — tried in order, first healthy one wins.
# These are community-run YouTube frontends; requests come FROM their servers,
# so YouTube's cloud-IP block never applies to us.
_INVIDIOUS_INSTANCES = [
    "https://inv.nadeko.net",
    "https://invidious.io.lol",
    "https://yewtu.be",
    "https://invidious.nerdvpn.de",
    "https://inv.tux.pizza",
]


def _invidious_request(url: str, timeout: int = 10) -> bytes:
    ctx = _build_ssl_ctx()
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; RAG-app/1.0)",
            "Accept":     "application/json, text/vtt, */*",
        },
    )
    with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
        return resp.read()


def _fetch_via_invidious(video_id: str) -> Tuple[List[dict], str]:
    """
    Fetch captions via a public Invidious instance.
    Returns (transcript_entries, video_title).

    The Invidious API endpoints used:
      GET /api/v1/captions/{videoId}          → lists available caption tracks
      GET /api/v1/captions/{videoId}?label=X  → returns the VTT for that track
      GET /api/v1/videos/{videoId}            → returns video metadata (title etc.)
    """
    last_error = "No Invidious instances responded"

    for instance in _INVIDIOUS_INSTANCES:
        try:
            # ── 1. Get caption track list ──────────────────────────────────────
            caps_raw  = _invidious_request(f"{instance}/api/v1/captions/{video_id}")
            caps_data = json.loads(caps_raw)
            captions  = caps_data.get("captions", [])

            if not captions:
                last_error = f"{instance}: no captions available for this video"
                continue

            # ── 2. Pick the best English track ────────────────────────────────
            chosen = None
            for c in captions:
                label = c.get("label", "").lower()
                code  = c.get("languageCode", "")
                if "english" in label or code.startswith("en"):
                    # Prefer auto-generated only as a last resort
                    if chosen is None or "auto" not in label:
                        chosen = c
            if chosen is None:
                chosen = captions[0]  # Take whatever is there

            # ── 3. Fetch the VTT for that track ───────────────────────────────
            label_param = urllib.parse.quote(chosen["label"])
            vtt_raw     = _invidious_request(
                f"{instance}/api/v1/captions/{video_id}?label={label_param}"
            )
            entries = _parse_vtt(vtt_raw.decode("utf-8", errors="replace"))
            if not entries:
                last_error = f"{instance}: VTT parsed but no entries found"
                continue

            # ── 4. Get video title ─────────────────────────────────────────────
            try:
                meta  = json.loads(_invidious_request(f"{instance}/api/v1/videos/{video_id}"))
                title = meta.get("title", f"YouTube Video ({video_id})")
            except Exception:
                title = f"YouTube Video ({video_id})"

            return entries, title

        except Exception as exc:
            last_error = f"{instance}: {exc}"
            continue  # Try next instance

    raise ValueError(f"All Invidious instances failed. Last error: {last_error}")


# ──────────────────────────────────────────────────────────────────────────────
# Title helper (oembed — no API key needed)
# ──────────────────────────────────────────────────────────────────────────────

def _get_title(video_id: str, url: str) -> str:
    try:
        ctx = _build_ssl_ctx()
        oembed = f"https://www.youtube.com/oembed?url={url}&format=json"
        with urllib.request.urlopen(oembed, context=ctx, timeout=5) as resp:
            return json.loads(resp.read()).get("title", f"YouTube Video ({video_id})")
    except Exception:
        return f"YouTube Video ({video_id})"


# ──────────────────────────────────────────────────────────────────────────────
# Document builder
# ──────────────────────────────────────────────────────────────────────────────

def _entries_to_documents(
    entries: List[dict],
    title: str,
    video_id: str,
    url: str,
) -> List[Document]:
    """Convert raw transcript entries into chunked LangChain Documents."""
    # Build full text + char→time map
    full_text = ""
    time_map: List[Tuple[int, float]] = []
    for e in entries:
        offset = len(full_text)
        time_map.append((offset, e["start"]))
        full_text += e["text"] + " "

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.MAX_CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(full_text)

    docs = []
    char_cursor = 0
    for chunk in chunks:
        chunk_start_sec = 0.0
        for char_offset, start_sec in time_map:
            if char_offset <= char_cursor:
                chunk_start_sec = start_sec
            else:
                break

        ts     = _format_timestamp(chunk_start_sec)
        ts_sec = int(chunk_start_sec)
        ts_url = f"https://www.youtube.com/watch?v={video_id}&t={ts_sec}s"

        docs.append(Document(
            page_content=chunk,
            metadata={
                "source_type":   "youtube",
                "source_url":    url,
                "video_id":      video_id,
                "title":         title,
                "timestamp":     ts,
                "timestamp_url": ts_url,
                "citation":      f"[YouTube] {title} @ {ts} — {ts_url}",
            },
        ))
        char_cursor += len(chunk)

    return docs


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def _web_search_fallback(video_id: str, url: str) -> List[Document]:
    """
    Last resort when all transcript methods fail on cloud platforms.

    Gets the video title via oembed, then reuses the app's existing web_search()
    function (DuckDuckGo / Tavily — whichever is configured) to find web content
    about the video. Returns Documents tagged source_type='youtube_web_fallback'
    so citations make clear these came from web sources, not the raw transcript.
    """
    from src.sources.web_search import web_search

    # oembed works from cloud IPs — gives us the real video title
    title = _get_title(video_id, url)

    # Single focused query is more reliable than three parallel ones
    query = f"{title} explained summary key points"

    raw_docs = web_search(query, max_results=5)

    if not raw_docs:
        raise ValueError(
            f"Web search returned no results for '{title}'. "
            "Try pasting a direct web URL or uploading a PDF instead."
        )

    # Re-tag documents so the UI shows the YouTube context
    docs: List[Document] = []
    for d in raw_docs:
        src_url = d.metadata.get("source_url", "")
        docs.append(Document(
            page_content=d.page_content,
            metadata={
                **d.metadata,
                "source_type": "youtube_web_fallback",
                "video_id":    video_id,
                "citation": (
                    f"[YouTube→Web] \"{title}\" — "
                    f"{d.metadata.get('title', src_url)}"
                ),
            },
        ))

    return docs


def load_youtube(url: str) -> List[Document]:
    """
    Load a YouTube video transcript and return chunked Documents with citations.

    Four-method cascade (most → least reliable on cloud platforms):
      1. youtube-transcript-api  — fast, works locally and some cloud IPs
      2. yt-dlp                  — browser-like, bypasses many IP blocks
      3. Invidious proxy         — routes through a third-party server
      4. Web search fallback     — searches DuckDuckGo for content about the
                                   video when all transcript methods are blocked;
                                   always works, degrades gracefully
    """
    video_id = _extract_video_id(url)
    entries: List[dict] = []
    title   = f"YouTube Video ({video_id})"
    errors  = []

    # ── Attempt 1: youtube-transcript-api ─────────────────────────────────────
    try:
        from youtube_transcript_api import YouTubeTranscriptApi  # noqa: F401
        entries = _fetch_via_transcript_api(video_id)
        title   = _get_title(video_id, url)
    except Exception as e:
        errors.append(f"transcript-api: {e}")

    # ── Attempt 2: yt-dlp ─────────────────────────────────────────────────────
    if not entries:
        try:
            import yt_dlp  # noqa: F401
            entries, title = _fetch_via_ytdlp(video_id, url)
        except ImportError:
            errors.append("yt-dlp: not installed")
        except Exception as e:
            errors.append(f"yt-dlp: {e}")

    # ── Attempt 3: Invidious proxy ─────────────────────────────────────────────
    if not entries:
        try:
            entries, title = _fetch_via_invidious(video_id)
        except Exception as e:
            errors.append(f"invidious: {e}")

    # ── Got transcript entries → build documents ───────────────────────────────
    if entries:
        return _entries_to_documents(entries, title, video_id, url)

    # ── Attempt 4: Web search fallback (always works on cloud) ────────────────
    # All transcript methods failed (YouTube blocking cloud IP).
    # Fall back to searching the web for content about this video.
    try:
        return _web_search_fallback(video_id, url)
    except Exception as e:
        errors.append(f"web-fallback: {e}")

    # ── Truly nothing worked ───────────────────────────────────────────────────
    raise ValueError(
        f"Could not load YouTube video {url}. All methods failed.\n"
        + "\n".join(f"  • {e}" for e in errors)
    )
