import subprocess

def start_server():
    subprocess.run(["uvicorn", "fasapi_app:app", "--host", "127.0.0.1", "--port", "8000"])

if __name__ == "__main__":
    start_server()
