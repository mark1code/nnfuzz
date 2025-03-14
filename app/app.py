from flask import Flask, request, jsonify, render_template, redirect, url_for
import os
import docker
import uuid
import json
import matplotlib.pyplot as plt
import io
import base64
import numpy as np

app = Flask(__name__, static_url_path='/static', static_folder='static')

# shared folders for uploads and results
app.config['UPLOAD_FOLDER'] = '/shared/uploads'
app.config['RESULTS_FOLDER'] = '/shared/results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Initialize Docker client
client = docker.from_env()

def create_coverage_graph(result_data):
    coverage_log = result_data.get("results", {}).get("coverage", {}).get("log", [])
    if not coverage_log:
        return None

    try:
        # Get iterations and coverage values
        iterations = [entry["iteration"] for entry in coverage_log]
        coverage_values = [entry["coverage"] for entry in coverage_log]
    except (KeyError, TypeError):
        return None

    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.plot(iterations, coverage_values, marker='o', linestyle='-', color='blue', label="Coverage")
    plt.xlabel("Iteration")
    plt.ylabel("Coverage")
    plt.title("Coverage Over Iterations")
    plt.legend()
    plt.grid(True)

    # Save plot to base64
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()
    return image_base64

def create_objectives_graph(result_data):
    fuzzer_type = result_data.get("fuzzer", "unknown")
    objectives_found = result_data.get("results", {}).get("objectives_found", 0)
    
    # If no objectives were found, don't show this graph
    if objectives_found <= 0:
        return None
    
    try:
        # use fuzz_log if tensorfuzz or deephunter
        fuzz_log = result_data.get("logs", {}).get("fuzz_log", [])
        objective_events = []
        
        if fuzz_log:
            # Extract objective events from fuzz log
            for entry in fuzz_log:
                if entry.get("status") == "objective found":
                    objective_events.append({
                        "iteration": entry.get("iteration", 0),
                        "coverage": entry.get("coverage", 0)
                    })
        
        # otherwise syntheise if dlfuzz
        if not objective_events and objectives_found > 0:
            coverage_log = result_data.get("results", {}).get("coverage", {}).get("log", [])
            if coverage_log and len(coverage_log) > 1:
                # for DLFuzz and other fuzzers without specific objective timing,
                # have to distribute objectives across timeline evenly
                total_iterations = coverage_log[-1]["iteration"]
                if total_iterations > 0:
                    step = total_iterations / (objectives_found + 1)
                    
                    # find coverage at various points
                    for i in range(1, objectives_found + 1):
                        iteration = int(i * step)
                        
                        # Find closest coverage point
                        closest_idx = 0
                        min_diff = float('inf')
                        for j, entry in enumerate(coverage_log):
                            diff = abs(entry["iteration"] - iteration)
                            if diff < min_diff:
                                min_diff = diff
                                closest_idx = j
                        
                        coverage_at_point = coverage_log[closest_idx]["coverage"]
                        objective_events.append({
                            "iteration": iteration,
                            "coverage": coverage_at_point
                        })
                        
        if not objective_events:
            return None
            
        iterations = [entry["iteration"] for entry in objective_events]
        coverage_at_objectives = [entry["coverage"] for entry in objective_events]
        
        # Create the plot
        plt.figure(figsize=(10, 5))
        plt.scatter(iterations, coverage_at_objectives, marker='X', s=100, color='red', label="Objective Found")
        
        # Add coverage line if available for context
        coverage_log = result_data.get("results", {}).get("coverage", {}).get("log", [])
        if coverage_log:
            all_iterations = [entry["iteration"] for entry in coverage_log]
            all_coverage = [entry["coverage"] for entry in coverage_log]
            plt.plot(all_iterations, all_coverage, linestyle='-', color='blue', alpha=0.5, label="Coverage")
        
        plt.xlabel("Iteration")
        plt.ylabel("Coverage")
        title = f"Objectives Found During Fuzzing ({objectives_found} total)"
        plt.title(title)
        plt.legend()
        plt.grid(True)
        
        # save plot to base64
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close()
        return image_base64
    except Exception as e:
        print(f"Error creating objectives graph: {e}")
        return None

def format_time_elapsed(seconds):
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours"

def create_error_distribution_graph(result_data):
    errors = result_data.get("logs", {}).get("errors", [])
    if not errors or len(errors) == 0:
        return None
    
    try:
        # Categorise errors by type
        error_types = {}
        for error in errors:
            # Extract error type
            error_type = error.split(':')[0].strip()
            if not error_type:
                error_type = "Unknown"
            
            if error_type in error_types:
                error_types[error_type] += 1
            else:
                error_types[error_type] = 1
        
        if not error_types:
            return None
            
        # Create the plot
        plt.figure(figsize=(8, 8))
        plt.pie(error_types.values(), labels=error_types.keys(), autopct='%1.1f%%',
                shadow=True, startangle=90)
        plt.axis('equal')
        plt.title(f"Error Distribution ({sum(error_types.values())} errors)")
        
        # Save plot to base64
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close()
        return image_base64
    except Exception as e:
        print(f"Error creating error distribution graph: {e}")
        return None

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

    # Save the file
    unique_id = str(uuid.uuid4())
    filename = f"{unique_id}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Check for selected fuzzer
    selected_fuzzer = request.form.get("fuzzer")
    if not selected_fuzzer:
        return jsonify({"error": "No fuzzer selected"}), 400

    # Clean up prev results
    result_file = os.path.join(app.config['RESULTS_FOLDER'], 'result.json')
    if os.path.exists(result_file):
        os.remove(result_file)

    try:
        # Use preset commands for each fuzzer
        if selected_fuzzer == "tensorfuzz":
            container_image = "nnfuzz_tensorfuzz"
            command = [
                "python", "/app/tensorfuzz_fuzzer.py",
                f"/shared/uploads/{filename}"
            ]
            
        elif selected_fuzzer == "dlfuzz":
            container_image = "nnfuzz_dlfuzz"
            command = [
                "python", "/app/dlfuzz_fuzzer.py",
                f"/shared/uploads/{filename}",
                "--dataset", "MNIST",
                "--model_name", "model1"
            ]
            
        elif selected_fuzzer == "deephunter":
            container_image = "nnfuzz_deephunter"
            command = [
                "python", "/app/deephunter_fuzzer.py",
                f"/shared/uploads/{filename}"
            ]
        else:
            return jsonify({"error": "Invalid fuzzer selected"}), 400

        # Environment variables
        environment = {
            "PYTHONUNBUFFERED": "1"
        }

        # Run the container
        print(f"Running container: {container_image}")
        print(f"Command: {' '.join(command)}")
        
        container = client.containers.run(
            container_image,
            command=command,
            volumes={'nnfuzz_shared': {'bind': '/shared', 'mode': 'rw'}},
            environment=environment,
            detach=True
        )
        
        # exit at completion
        result = container.wait()
        logs = container.logs().decode('utf-8')
        container.remove()
        
        # Check for errors
        if result["StatusCode"] != 0:
            error_result = {
                "error": f"Fuzzing container exited with status {result['StatusCode']}",
                "logs": logs,
                "fuzzer": selected_fuzzer,
                "model": filename
            }
            
            with open(result_file, 'w') as f:
                json.dump(error_result, f)

        # Check if result file was created
        if not os.path.exists(result_file):
            error_result = {
                "error": "Fuzzing completed but no results were generated",
                "logs": logs,
                "fuzzer": selected_fuzzer,
                "model": filename
            }
            
            with open(result_file, 'w') as f:
                json.dump(error_result, f)

        return redirect(url_for('results'))
    
    except Exception as e:
        error_msg = f"Fuzzing failed: {str(e)}"
        
        # error result
        error_result = {
            "error": error_msg,
            "logs": logs if 'logs' in locals() else "No logs available",
            "fuzzer": selected_fuzzer,
            "model": filename
        }
        
        with open(result_file, 'w') as f:
            json.dump(error_result, f)
            
        return redirect(url_for('results'))

@app.route('/results')
def results():
    result_file = os.path.join(app.config['RESULTS_FOLDER'], 'result.json')
    if not os.path.exists(result_file):
        return render_template("results.html", error="No fuzzing results available.")
    
    try:
        with open(result_file, 'r') as f:
            result_data = json.load(f)
        
        # Check for errors
        if "error" in result_data:
            return render_template("results.html", 
                                   error=result_data["error"],
                                   result=result_data)
        
        # Format time elapsed for display
        timing_data = result_data.get("results", {}).get("timing", {})
        duration_seconds = timing_data.get("duration_seconds", 0)
        formatted_time = format_time_elapsed(duration_seconds)
        
        if "results" not in result_data:
            result_data["results"] = {}
        if "timing" not in result_data["results"]:
            result_data["results"]["timing"] = {}
        result_data["results"]["timing"]["formatted_time"] = formatted_time
        
        # Generate visualization graphs
        graphs = {
            "coverage_graph": create_coverage_graph(result_data),
            "objectives_graph": create_objectives_graph(result_data),
            "error_distribution_graph": create_error_distribution_graph(result_data)
        }
        
        return render_template(
            "results.html", 
            result=result_data,
            **graphs  # all graphs to template
        )
    except Exception as e:
        error_msg = f"Error processing results: {str(e)}"
        
        # no result data available, provide fallback
        fallback_result = {
            "fuzzer": "Error",
            "iterations": 0,
            "results": {
                "coverage": {"overall": 0, "log": []},
                "objectives_found": 0,
                "timing": {"duration_seconds": 0, "formatted_time": "0 seconds"}
            }
        }
        
        return render_template("results.html", error=error_msg, result=fallback_result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)