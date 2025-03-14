#!/usr/bin/env python
import os
import sys
import json
import subprocess
import argparse
import time
import re

def run_dlfuzz(strategy, threshold, num_neurons, output_dir, num_mutations, model_name, dataset, logs):
    # Use the dataset argument directly to determine folder to use
    folder = dataset  
    current_dir = os.getcwd()
    target_dir = os.path.join(current_dir, folder)
    
    if not os.path.exists(target_dir):
        error_msg = "Error: Directory {} does not exist.".format(target_dir)
        logs.append(error_msg)
        return {"error": error_msg, "output": None, "coverage_data": []}
    
    os.chdir(target_dir)
    logs.append("Changed directory to: {}".format(target_dir))
    
    # command, arguments should match those expected by gen_diff.py.
    cmd = [
        "python", "gen_diff.py",
        strategy,
        str(threshold),
        str(num_neurons),
        output_dir,
        str(num_mutations),
        model_name
    ]
    logs.append("Running DLFuzz command: " + " ".join(cmd))
    
    coverage_data = []
    start_time = time.time()
    
    try:
        # Run DLFuzz with real-time output processing
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        
        current_iteration = 0
        for line in process.stdout:
            line = line.strip()
            logs.append(line)
            
            # Parse coverage information from the output
            if "covered neurons percentage" in line:
                try:
                    coverage_match = re.search(r'covered neurons percentage \d+ neurons ([\d.]+)', line)
                    if coverage_match:
                        coverage_value = float(coverage_match.group(1))
                        coverage_data.append({
                            "iteration": current_iteration,
                            "coverage": coverage_value
                        })
                        current_iteration += 1
                except Exception as e:
                    logs.append("Error parsing coverage: {}".format(str(e)))
        
        # Wait for process completion
        process.wait()
        output = "\n".join(logs)
        
        # If no coverage data was collected, add at least one point
        if not coverage_data and "covered neurons percentage" in output:
            try:
                coverage_match = re.search(r'covered neurons percentage \d+ neurons ([\d.]+)', output)
                if coverage_match:
                    coverage_value = float(coverage_match.group(1))
                    coverage_data.append({
                        "iteration": 0,
                        "coverage": coverage_value
                    })
            except Exception:
                pass
        
    except subprocess.CalledProcessError as e:
        error_msg = e.output.decode("utf-8") if hasattr(e, 'output') else str(e)
        logs.append("DLFuzz failed, error: " + error_msg)
        output = "Error: " + error_msg
        return {"error": error_msg, "output": output, "coverage_data": coverage_data}
    
    # original directory
    os.chdir(current_dir)
    logs.append("Returned to directory: {}".format(current_dir))
    
    return {"output": output, "coverage_data": coverage_data}

def main():
    parser = argparse.ArgumentParser(description="DLFuzz Fuzzer Runner")
    parser.add_argument("model_path", help="Path to the neural network model (for compatibility)")
    parser.add_argument("--strategy", default="[2]", help="Neuron selection strategy (default: [2])")
    parser.add_argument("--threshold", type=float, default=0.5, help="Activation threshold (default: 0.5)")
    parser.add_argument("--num_neurons", type=int, default=5, help="Number of neurons to cover (default: 5)")
    parser.add_argument("--output_dir", default="0602", help="Output folder for adversarial examples (default: 0602)")
    parser.add_argument("--num_mutations", type=int, default=5, help="Number of mutations per seed (default: 5)")
    parser.add_argument("--model_name", default="model1", help="Name of the model under test (e.g., vgg16 or model1)")
    parser.add_argument("--dataset", default="MNIST", help="Dataset to use: MNIST or ImageNet")
    
    args = parser.parse_args()
    
    logs = []
    start_time = time.time()
    
    try:
        # run and capture the output
        dlfuzz_result = run_dlfuzz(
            args.strategy,
            args.threshold,
            args.num_neurons,
            args.output_dir,
            args.num_mutations,
            args.model_name,
            args.dataset,
            logs
        )
        
        #useful metrics
        dlfuzz_output = dlfuzz_result.get("output", "")
        objectives_found = 0
        final_coverage = 0.0
        
        # additional metrics
        if "adversial num" in dlfuzz_output:
            match = re.search(r'adversial num = (\d+)', dlfuzz_output)
            if match:
                objectives_found = int(match.group(1))
        
        # final coverage
        coverage_data = dlfuzz_result.get("coverage_data", [])
        if coverage_data:
            final_coverage = coverage_data[-1]["coverage"]
        else:
            # extract final coverage from the output
            match = re.search(r'covered neurons percentage \d+ neurons ([\d.]+)', dlfuzz_output)
            if match:
                final_coverage = float(match.group(1))
        
        end_time = time.time()
        duration = end_time - start_time
        
        # build JSON
        result = {
            "fuzzer": "dlfuzz",
            "model": args.model_path,
            "iterations": len(coverage_data) if coverage_data else 1,  # use number of coverage points as iteration count
            "parameters": {
                "strategy": args.strategy,
                "threshold": args.threshold,
                "num_neurons": args.num_neurons,
                "output_dir": args.output_dir,
                "num_mutations": args.num_mutations,
                "model_name": args.model_name,
                "dataset": args.dataset
            },
            "results": {
                "objectives_found": objectives_found,
                "coverage": {
                    "overall": final_coverage,
                    "log": coverage_data
                },
                "timing": {
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration_seconds": duration
                }
            },
            "logs": {
                "debug": logs,
                "errors": []
            },
            "additional_info": {
                "raw_output": "DLFuzz run completed.",
                "dlfuzz_output": dlfuzz_output
            }
        }
    
    except Exception as e:
        error_msg = str(e)
        logs.append("DLFuzz failed with error: " + error_msg)

        # ensure the results folder exists error JSON
        results_dir = "/shared/results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # write error details to result.json
        error_result = {
            "error": "Fuzzing failed",
            "details": error_msg,
            "logs": logs
        }

        with open(os.path.join(results_dir, "result.json"), "w") as f:
            json.dump(error_result, f, indent=2)

        print(json.dumps(error_result))
        sys.exit(1)  # error, exit
    
    # ensure shared results directory exists.
    results_dir = "/shared/results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    result_file = os.path.join(results_dir, "result.json")
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)
    
    print(json.dumps(result))
    
if __name__ == "__main__":
    main()
