import subprocess
import time
import sys


def start_backend():
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend:app", "--reload"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )


def start_frontend():
    return subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "frontend.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )


if __name__ == "__main__":
    print("🚀 Starting AI RAG System...")

    backend_process = start_backend()

    # small delay so backend starts first
    time.sleep(3)

    frontend_process = start_frontend()

    print("✅ Backend running on http://localhost:8000")
    print("✅ Frontend running on Streamlit")

    try:
        backend_process.wait()
        frontend_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        backend_process.terminate()
        frontend_process.terminate()