from logo_detector import run_logo_detection
import json

if __name__ == "__main__":
    print("Running logo detection in standalone mode...")
    
    results = run_logo_detection(dev_mode=False)  # テスト画像（mock）で動作
    print("Detection results:")
    print(json.dumps(results, indent=2, ensure_ascii=False))
