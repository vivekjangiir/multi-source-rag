"""
One-command launcher for Multi-Source RAG.
Run:  python start.py

  1. Checks / installs missing core packages
  2. Starts the FastAPI server on http://localhost:8000
  3. Opens the chat UI in your browser
"""
import subprocess, sys, time, webbrowser, os

REQUIRED = [
    "fastapi", "uvicorn", "aiofiles", "python-dotenv",
    "langchain", "langchain-core", "langchain-community",
    "langchain-text-splitters", "langgraph",
    "langchain-groq", "langchain-huggingface",
    "sentence-transformers", "langchain-chroma", "chromadb",
    "duckduckgo-search", "youtube-transcript-api",
    "pymupdf", "docx2txt", "beautifulsoup4", "requests",
    "pydantic", "python-multipart",
]

def check_and_install():
    print("Checking dependencies...")
    missing = []
    for pkg in REQUIRED:
        mod = pkg.replace("-", "_").split("[")[0]
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"Installing: {', '.join(missing)}")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--quiet", *missing
        ])
        print("Done.\n")
    else:
        print("All packages present.\n")

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    check_and_install()

    print("Starting server at http://localhost:8000 ...")
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app",
         "--host", "0.0.0.0", "--port", "8000", "--reload"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    # Wait for server to be ready
    import urllib.request, urllib.error
    for _ in range(20):
        try:
            urllib.request.urlopen("http://localhost:8000/health", timeout=1)
            break
        except Exception:
            time.sleep(0.5)

    print("Opening browser...")
    webbrowser.open("http://localhost:8000")
    print("\nRAG app running. Press Ctrl+C to stop.\n")

    try:
        server.wait()
    except KeyboardInterrupt:
        server.terminate()
        print("\nStopped.")

if __name__ == "__main__":
    main()
