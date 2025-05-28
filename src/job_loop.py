import subprocess
import time
import datetime

while True:
    try:
        print(f"[{datetime.datetime.now()}] Starting detect.py")
        result = subprocess.run(
            ["/app/venv/bin/python", "/app/logo-detection/src/detect.py"],
            capture_output=True,
            text=True
        )

        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        if result.returncode != 0:
            print(f"detect.py exited with code {result.returncode}")

    except Exception as e:
        print(f"Error running detect.py: {str(e)}")

    time.sleep(30)
