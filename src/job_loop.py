import subprocess
import time
import datetime

while True:
    try:
        start_time = time.time()
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

        # 処理時間を計算
        execution_time = time.time() - start_time
        print(f"Execution time: {execution_time:.2f} seconds")
        
        # 処理が10秒未満の場合、残りの時間を待機
        wait_time = max(60 - execution_time, 0)
        if wait_time > 0:
            print(f"Waiting for {wait_time:.2f} seconds")
            time.sleep(wait_time)
        
    except Exception as e:
        print(f"Error running detect.py: {str(e)}")
        time.sleep(60)  # エラー時は通常通り30秒待機
