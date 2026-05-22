"""
YouTube source loader.
Fetches transcript, chunks it, and tags each chunk with video metadata + timestamp citation.
"""
from __future__ import annotations
import os
import re
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config

# ── SSL fix for cloud platforms (HuggingFace Spaces, Render, etc.) ─────────────
# YouTube's CDN rejects connections with outdated CA bundles.
# Point requests/urllib3 to certifi's up-to-date bundle if env vars not already set.
try:
    import certifi
    _CERT_FILE = certifi.where()
    os.environ.setdefault("REQUESTS_CA_BUNDLE", _CERT_FILE)
    os.environ.setdefault("SSL_CERT_FILE", _CERT_FILE)
except ImportError:
    pass


def _extract_video_id(url: str) -> str:
    """Extract YouTube video ID from any youtube.com or youtu.be URL."""
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
    """Convert seconds to HH:MM:SS or MM:SS string."""
    seconds = int(seconds)
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def load_youtube(url: str) -> List[Document]:
    """
    Load a YouTube video transcript and return chunked Documents.

    Each document carries metadata:
      - source_type: "youtube"
      - source_url: original video URL
      - video_id: YouTube video ID
      - title: video title (if available)
      - timestamp: start time of the chunk (HH:MM:SS)
      - timestamp_url: direct link to that moment in the video
      - citation: human-readable citation string
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        raise ImportError("Run: pip install youtube-transcript-api")

    video_id = _extract_video_id(url)

    try:
        # v0.6+ uses an instance and .fetch(); older versions used the class method.
        # We support both to be safe.
        api = YouTubeTranscriptApi()
        transcript_list = api.fetch(video_id)
    except Exception as e:
        err = str(e).lower()
        if "disabled" in err:
            raise ValueError(f"Transcripts are disabled for this video: {url}")
        if "no transcript" in err or "could not retrieve" in err:
            raise ValueError(f"No transcript found for video (video may be private, age-restricted, or transcript unavailable): {url}")
        if "ssl" in err or "eof" in err or "max retries" in err or "connection" in err:
            raise ValueError(
                f"Network error reaching YouTube (SSL/connection issue on cloud host). "
                f"Try a different video, or ingest a PDF/URL instead. Details: {e}"
            )
        raise ValueError(f"Failed to fetch transcript: {e}")

    # Try to get video title via oembed (no API key needed)
    title = f"YouTube Video ({video_id})"
    try:
        import urllib.request, json
        oembed_url = f"https://www.youtube.com/oembed?url={url}&format=json"
        with urllib.request.urlopen(oembed_url, timeout=5) as resp:
            data = json.loads(resp.read())
            title = data.get("title", title)
    except Exception:
        pass

    # Combine transcript entries into text blocks with start times.
    # v0.6+ returns objects with .text / .start attributes; older versions return dicts.
    full_text = ""
    time_map: List[tuple[int, float]] = []  # (char_offset, start_seconds)
    for entry in transcript_list:
        text  = entry.text  if hasattr(entry, "text")  else entry["text"]
        start = entry.start if hasattr(entry, "start") else entry["start"]
        offset = len(full_text)
        time_map.append((offset, start))
        full_text += text + " "

    # Chunk the text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.MAX_CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(full_text)

    # Find best timestamp for each chunk
    docs = []
    char_cursor = 0
    for chunk in chunks:
        # Find the last time_map entry before this chunk's position
        chunk_start_seconds = 0.0
        for char_offset, start_sec in time_map:
            if char_offset <= char_cursor:
                chunk_start_seconds = start_sec
            else:
                break

        ts = _format_timestamp(chunk_start_seconds)
        ts_seconds = int(chunk_start_seconds)
        ts_url = f"https://www.youtube.com/watch?v={video_id}&t={ts_seconds}s"

        docs.append(Document(
            page_content=chunk,
            metadata={
                "source_type": "youtube",
                "source_url": url,
                "video_id": video_id,
                "title": title,
                "timestamp": ts,
                "timestamp_url": ts_url,
                "citation": f"[YouTube] {title} @ {ts} — {ts_url}",
            }
        ))
        char_cursor += len(chunk)

    return docs
