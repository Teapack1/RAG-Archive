import subprocess

result = subprocess.run("sqlite3 --version", shell=True, capture_output=True, text=True)
print(result.stdout.strip())

print("Starting the FastAPI server...")
subprocess.run("uvicorn fast_app:app --host 0.0.0.0 --port 7860", shell=True)
