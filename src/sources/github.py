"""
GitHub repository source loader.
Clones/fetches a repo and indexes code files with file-path citations.
"""
from __future__ import annotations
import os
import tempfile
from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
import config

# File extensions to index and their Language enum for smart splitting
LANGUAGE_MAP = {
    ".py":    Language.PYTHON,
    ".js":    Language.JS,
    ".ts":    Language.JS,
    ".jsx":   Language.JS,
    ".tsx":   Language.JS,
    ".java":  Language.JAVA,
    ".cpp":   Language.CPP,
    ".c":     Language.C,
    ".go":    Language.GO,
    ".rs":    Language.RUST,
    ".rb":    Language.RUBY,
    ".md":    Language.MARKDOWN,
    ".html":  Language.HTML,
    ".txt":   None,
}

MAX_FILE_SIZE_BYTES = 500_000  # Skip files larger than 500KB


def load_github(repo_url: str, branch: str = "main") -> List[Document]:
    """
    Load a GitHub repository and return chunked Documents.

    Args:
        repo_url:  Full GitHub URL, e.g. https://github.com/langchain-ai/langchain
        branch:    Branch to clone (default: main)

    Metadata per chunk:
      - source_type: "github"
      - repo_url: original repo URL
      - file_path: relative file path within repo
      - language: detected language
      - citation: "[GitHub] owner/repo — path/to/file.py"
    """
    try:
        from git import Repo
    except ImportError:
        raise ImportError("Run: pip install gitpython")

    # Parse owner/repo from URL
    parts = repo_url.rstrip("/").split("/")
    owner, repo_name = parts[-2], parts[-1].replace(".git", "")
    repo_label = f"{owner}/{repo_name}"

    # Build authenticated URL if token provided
    clone_url = repo_url
    if config.GITHUB_TOKEN:
        protocol_end = repo_url.find("://") + 3
        clone_url = f"{repo_url[:protocol_end]}{config.GITHUB_TOKEN}@{repo_url[protocol_end:]}"

    docs = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"Cloning {repo_label} (branch: {branch})...")
        Repo.clone_from(clone_url, tmp_dir, branch=branch, depth=1)

        for root, dirs, files in os.walk(tmp_dir):
            # Skip hidden dirs and common non-code dirs
            dirs[:] = [
                d for d in dirs
                if not d.startswith(".") and d not in {"node_modules", "__pycache__", ".git", "dist", "build", "venv"}
            ]

            for fname in files:
                ext = os.path.splitext(fname)[-1].lower()
                if ext not in LANGUAGE_MAP:
                    continue

                full_path = os.path.join(root, fname)
                rel_path = os.path.relpath(full_path, tmp_dir)

                if os.path.getsize(full_path) > MAX_FILE_SIZE_BYTES:
                    continue

                try:
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read().strip()
                    if not content:
                        continue
                except Exception:
                    continue

                language = LANGUAGE_MAP[ext]
                if language:
                    splitter = RecursiveCharacterTextSplitter.from_language(
                        language=language,
                        chunk_size=config.MAX_CHUNK_SIZE,
                        chunk_overlap=config.CHUNK_OVERLAP,
                    )
                else:
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=config.MAX_CHUNK_SIZE,
                        chunk_overlap=config.CHUNK_OVERLAP,
                    )

                chunks = splitter.split_text(content)
                github_file_url = f"https://github.com/{repo_label}/blob/{branch}/{rel_path.replace(os.sep, '/')}"

                for i, chunk in enumerate(chunks):
                    docs.append(Document(
                        page_content=chunk,
                        metadata={
                            "source_type": "github",
                            "repo_url": repo_url,
                            "repo_label": repo_label,
                            "file_path": rel_path,
                            "file_url": github_file_url,
                            "language": ext.lstrip("."),
                            "chunk_index": i,
                            "citation": f"[GitHub] {repo_label} — {rel_path}",
                        }
                    ))

    return docs
