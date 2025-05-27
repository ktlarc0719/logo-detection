from flask import Flask, request, jsonify
import subprocess

app = Flask(__name__)

# ========== ①: /run-detect ==========

@app.route('/run-detect', methods=['GET'])
def run_detect():
    try:
        result = subprocess.run(
            ["/app/venv/bin/python", "/app/src/detect.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return jsonify({
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ========== ②: /webhook/git-pull ==========

@app.route('/webhook/git-pull', methods=['POST'])
def webhook_git_pull():
    try:
        data = request.get_json()
        if data.get("ref") == "refs/heads/main":
            result = subprocess.run(
                ["git", "pull"],
                cwd="/app",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            return jsonify({
                "message": "git pull executed",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }), 200
        else:
            return jsonify({"message": "ref does not match, no action taken"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ========== アプリ起動 ==========

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
