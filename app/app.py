from flask import Flask, request, jsonify, render_template
import os
import docker
import uuid
import json

app = Flask(__name__, static_url_path='/static', static_folder='static')

# Make sure the shared folder paths are consistent across both Flask & TensorFuzz
app.config['UPLOAD_FOLDER'] = '/shared/uploads'
app.config['RESULTS_FOLDER'] = '/shared/results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Initialize Docker client
client = docker.from_env()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save file inside the shared volume
    unique_id = str(uuid.uuid4())
    filename = f"{unique_id}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # Run TensorFuzz container
        container = client.containers.run(
            "nnfuzz_tensorfuzz",
            command=["python", "/app/fuzz_runner.py", f"/shared/uploads/{filename}"],
            volumes={'nnfuzz_shared': {'bind': '/shared', 'mode': 'rw'}},
            detach=True
        )

        # Wait for completion and retrieve logs
        result = container.wait()
        logs = container.logs().decode('utf-8')
        container.remove()

        # Parse results
        with open(os.path.join(app.config['RESULTS_FOLDER'], 'result.json'), 'r') as f:
            result_data = json.load(f)

        # Convert any paths in results to URLs
        if 'coverage_graph' in result_data:
            result_data['coverage_graph'] = f"/shared/results/{os.path.basename(result_data['coverage_graph'])}"

        return jsonify(result_data)

    except Exception as e:
        return jsonify({
            "error": f"Fuzzing failed: {str(e)}",
            "details": logs if 'logs' in locals() else None
        }), 500

# List uploaded files (to verify uploads are working)
@app.route('/files', methods=['GET'])
def list_files():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    return jsonify({"uploaded_files": files}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)