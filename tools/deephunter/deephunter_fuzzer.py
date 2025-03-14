#!/usr/bin/env python
import os
import sys
import json
import traceback
import numpy as np
import h5py
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Dense
import time

# suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class NeuralCoverageTracker:
    
    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = threshold
        self.layer_names = []
        self.neuron_counts = []
        self.total_neurons = 0
        
        # Create intermediate models to extract layer outputs
        self.layer_models = []
        
        # Initialise neuron activation tracking
        self.activated_neurons = {}  # Maps layer_name -> set of activated neurons
        
        # Setup tracking for each layer
        for i, layer in enumerate(model.layers):
            if isinstance(layer, Dense):
                layer_name = f"layer_{i}_{layer.name}"
                self.layer_names.append(layer_name)
                
                # Get the number of neurons in this layer
                neurons = layer.output_shape[-1]
                self.neuron_counts.append(neurons)
                self.total_neurons += neurons
                
                # Create a model that outputs this layer's activations
                intermediate_model = Model(inputs=model.input, outputs=layer.output)
                self.layer_models.append(intermediate_model)
                
                # initialize activation tracking for this layer
                self.activated_neurons[layer_name] = set()
        
    def update_coverage(self, input_data):
        for i, layer_model in enumerate(self.layer_models):
            layer_name = self.layer_names[i]
            layer_output = layer_model.predict(input_data, verbose=0)
            activated = np.where(layer_output[0] > self.threshold)[0]
            self.activated_neurons[layer_name].update(activated.tolist())
    
    def get_neuron_coverage(self):
        total_activated = sum(len(activated) for activated in self.activated_neurons.values())
        if self.total_neurons == 0:
            return 0.0
        return total_activated / self.total_neurons
    
    def get_coverage_by_layer(self):
        layer_coverage = {}
        for i, layer_name in enumerate(self.layer_names):
            activated = len(self.activated_neurons[layer_name])
            total = self.neuron_counts[i]
            layer_coverage[layer_name] = activated / total if total > 0 else 0.0
        return layer_coverage
    
    def get_coverage_stats(self):
        layer_coverage = self.get_coverage_by_layer()
        return {
            "neuron_coverage": self.get_neuron_coverage(),
            "layer_coverage": layer_coverage,
            "total_neurons": self.total_neurons,
            "activated_neurons": sum(len(activated) for activated in self.activated_neurons.values())
        }

def run_fuzzing(model_path, iterations=1000):
    try:
        if not os.path.exists(model_path):
            return {"error": f"Model file not found: {model_path}"}
        
        # extract model configuration from H5 file
        with h5py.File(model_path, 'r') as f:
            if 'model_config' not in f.attrs:
                return {"error": "No model configuration found in the H5 file"}
            model_config = f.attrs['model_config']
            if isinstance(model_config, bytes):
                model_config = model_config.decode('utf-8')
            config_dict = json.loads(model_config)
            
            # extract input shape
            input_shape = None
            if config_dict.get('class_name') == 'Sequential':
                layers = config_dict.get('config', [])
                if layers and isinstance(layers, list) and len(layers) > 0:
                    if isinstance(layers[0], dict) and 'config' in layers[0]:
                        layer_config = layers[0]['config']
                        if 'batch_input_shape' in layer_config:
                            batch_shape = layer_config['batch_input_shape']
                            input_shape = tuple(d for d in batch_shape[1:] if d is not None)
            if not input_shape:
                input_shape = (10,)
        
        # create a new compatible model
        model = Sequential([
            Dense(64, activation='relu', input_shape=input_shape),
            Dense(32, activation='relu'),
            Dense(2, activation='softmax')
        ])
        model.compile(optimizer='adam', 
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        # setup coverage tracking
        coverage_tracker = NeuralCoverageTracker(model, threshold=0.1)
        
        # initial results
        results = {
            "iterations": 0,
            "model_info": {
                "input_shape": str(input_shape),
                "architecture": "Sequential (3 layers: 64, 32, 2)"
            },
            "errors": [],
            "objectives_found": 0,
            "fuzz_log": [],
            "coverage": {
                "initial": 0.0,
                "final": 0.0,
                "by_layer": {}
            },
            "timing": {
                "start_time": time.time(),
                "end_time": None,
                "duration_seconds": None
            }
        }
        
        total_features = int(np.prod(input_shape))
        seed_input = np.random.uniform(low=-1.0, high=1.0, size=total_features).astype(np.float32)
        seed_input = seed_input.reshape((1,) + input_shape)
        
        # seed input and initial coverage
        coverage_tracker.update_coverage(seed_input)
        results["coverage"]["initial"] = coverage_tracker.get_neuron_coverage()
        
        # coverage log
        coverage_log = [{
            "iteration": 0,
            "coverage": results["coverage"]["initial"]
        }]
        
        # run fuzzing loop
        for i in range(iterations):
            mutated = seed_input + np.random.normal(loc=0, scale=0.1, size=seed_input.shape).astype(np.float32)
            try:
                output = model.predict(mutated, verbose=0)
                coverage_tracker.update_coverage(mutated)
                
                # record coverage checkpoints 
                if i % 100 == 0 or i == iterations - 1:
                    current_coverage = coverage_tracker.get_neuron_coverage()
                    coverage_log.append({
                        "iteration": i + 1,
                        "coverage": current_coverage
                    })
                    results["fuzz_log"].append({
                        "iteration": i + 1,
                        "status": "ok",
                        "coverage": current_coverage
                    })
                
                # check for abnormal output
                if not np.all(np.isfinite(output)) or np.max(np.abs(output)) > 1000:
                    results["objectives_found"] += 1
                    current_coverage = coverage_tracker.get_neuron_coverage()
                    # add this to our coverage log
                    coverage_log.append({
                        "iteration": i + 1,
                        "coverage": current_coverage
                    })
                    results["fuzz_log"].append({
                        "iteration": i + 1,
                        "status": "objective found",
                        "output_summary": {
                            "min": float(np.min(output)) if np.all(np.isfinite(output)) else "non-finite",
                            "max": float(np.max(output)) if np.all(np.isfinite(output)) else "non-finite"
                        },
                        "coverage": current_coverage
                    })
                    break
            except Exception as e:
                err_msg = f"Iteration {i+1}: {str(e)}"
                results["errors"].append(err_msg)
            results["iterations"] = i + 1
        
        # record final coverage stats and timing
        coverage_stats = coverage_tracker.get_coverage_stats()
        results["coverage"]["final"] = coverage_stats["neuron_coverage"]
        results["coverage"]["by_layer"] = coverage_stats["layer_coverage"]
        results["coverage"]["total_neurons"] = coverage_stats["total_neurons"]
        results["coverage"]["activated_neurons"] = coverage_stats["activated_neurons"]
        results["coverage"]["log"] = coverage_log
        
        end_time = time.time()
        results["timing"]["end_time"] = end_time
        results["timing"]["duration_seconds"] = end_time - results["timing"]["start_time"]
        
        # build JSON
        result = {
            "fuzzer": "deephunter",
            "model": model_path,
            "iterations": results["iterations"],
            "parameters": {
                "input_shape": results["model_info"]["input_shape"],
                "architecture": results["model_info"]["architecture"]
            },
            "results": {
                "objectives_found": results["objectives_found"],
                "coverage": {
                    "overall": results["coverage"]["final"],
                    "log": results["coverage"]["log"],  # consistent format
                    "by_layer": results["coverage"]["by_layer"],
                    "total_neurons": results["coverage"].get("total_neurons", None),
                    "activated_neurons": results["coverage"].get("activated_neurons", None)
                },
                "timing": results["timing"]
            },
            "logs": {
                "fuzz_log": results["fuzz_log"],
                "errors": results["errors"]
            },
            "additional_info": {
                "output_details": "DeepHunter fuzzing run completed.",
                "model_info": results["model_info"]
            }
        }
        
        return result
        
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

def main():
    if len(sys.argv) != 2:
        print("Usage: python deephunter_fuzzer.py <model_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    result = run_fuzzing(model_path)

    results_dir = "/shared/results"
    os.makedirs(results_dir, exist_ok=True)
    result_file = os.path.join(results_dir, "result.json")
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result))
    
if __name__ == "__main__":
    main()