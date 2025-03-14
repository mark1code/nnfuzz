#!/usr/bin/env python
import os
import sys
import json
import traceback
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

from tensorfuzz.lib import corpus
from tensorfuzz.lib.sample_functions import uniform_sample_function

RESULTS_DIR = "/shared/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# custom to avoid FLANN problem
class CustomInputCorpus:
    def __init__(self, seed_corpus_elements, sample_function, threshold=0.1):
        self.corpus = seed_corpus_elements.copy()
        self.sample_function = sample_function
        self.threshold = threshold
        self.log = [f"Created CustomInputCorpus with {len(self.corpus)} elements"]
        
    def maybe_add_to_corpus(self, corpus_element):
        self.corpus.append(corpus_element)
        return True
        
    def sample_input(self):
        if not self.corpus:
            raise ValueError("Corpus is empty")
        return np.random.choice(self.corpus)

def run_fuzzing(model_path):
    logs = []
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        # c
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        K.set_session(sess)
        
        # load model
        model = load_model(model_path)
        model._make_predict_function()
        sess.run(tf.compat.v1.global_variables_initializer())
        
        # get model input shape and compute expected features
        input_shape = model.input_shape[1:]
        expected_features = int(np.prod(input_shape))
        logs.append(f"Input shape: {input_shape}, expected features: {expected_features}")
        
        # create input tensor placeholder and compute logits
        input_tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, expected_features), name="input_tensor")
        logits = model(input_tensor)
        if isinstance(logits, dict):
            logs.append("Model output is a dict; extracting first value.")
            logits = list(logits.values())[0]
        
        tensor_map = {
            "input": [input_tensor],
            "coverage": [logits],
            "metadata": [logits]
        }
        
        # Custom fetch function for TensorFuzz
        def fetch_function(input_batches):
            if not input_batches or len(input_batches) == 0:
                raise ValueError("Empty input batches provided to fetch_function")
            arrays = []
            for i, elem in enumerate(input_batches[0]):
                try:
                    inp = elem.data
                    if not isinstance(inp, np.ndarray):
                        inp = np.array(inp, dtype=np.float32)
                    inp = inp.flatten()
                    if inp.size != expected_features:
                        if inp.size > expected_features:
                            inp = inp[:expected_features]
                        else:
                            inp = np.pad(inp, (0, expected_features - inp.size), 'constant')
                    arrays.append(inp)
                    if i == 0:
                        logs.append(f"First corpus element shape: {inp.shape}")
                except Exception as e:
                    logs.append(f"ERROR in fetch_function processing element {i}: {e}")
                    arrays.append(np.zeros(expected_features, dtype=np.float32))
            data_stack = np.stack(arrays, axis=0)
            logs.append(f"batch shape: {data_stack.shape}")
            feed_dict = {tensor_map["input"][0]: data_stack}
            coverage_out, metadata_out = sess.run(
                [tensor_map["coverage"][0], tensor_map["metadata"][0]],
                feed_dict=feed_dict
            )
            return [coverage_out], [metadata_out]
        
        def metadata_function(metadata_batches):
            return metadata_batches[0]
        
        def coverage_function(batches):
            return batches[0]
        
        # Objective function
        def objective_function(corpus_element):
            if corpus_element.metadata is None:
                return False
            logits_val = corpus_element.metadata
            if np.any(~np.isfinite(logits_val)):
                logs.append("Found non-finite output")
                return True
            if np.any(np.abs(logits_val) > 1000):
                logs.append("Found extremely large output")
                return True
            return False
        
        # Modified mutation function with explicit numeric casting
        def custom_mutation_function(corpus_element):
            try:
                mutations = []
                data = corpus_element.data
                for _ in range(10):
                    mutated_data = np.array(data, dtype=np.float32).copy()
                    noise = np.random.normal(0, 0.1, size=mutated_data.shape)
                    mutated_data += noise
                    mutated_data = np.clip(mutated_data, -10.0, 10.0)
                    mutated_data = mutated_data.astype(np.float32)
                    new_element = corpus.CorpusElement(
                        mutated_data,
                        None,  # coverage
                        None,  # metadata
                        parent=corpus_element
                    )
                    mutations.append(new_element)
                return mutations
            except Exception as e:
                logs.append(f"ERROR in mutation_function: {e}")
                traceback.print_exc()
                return [corpus_element]
        
        # generate seed input
        seed_input = np.random.uniform(low=-1.0, high=1.0, size=expected_features).astype(np.float32)
        logs.append(f"Seed input shape: {seed_input.shape}")
        
        dummy_input = np.zeros((1, expected_features), dtype=np.float32)
        dummy_input[0] = seed_input
        
        feed_dict = {tensor_map["input"][0]: dummy_input}
        coverage_out, metadata_out = sess.run(
            [tensor_map["coverage"][0], tensor_map["metadata"][0]],
            feed_dict=feed_dict
        )
        
        seed_element = corpus.CorpusElement(
            seed_input,
            coverage_out[0] if coverage_out.ndim > 1 else coverage_out,
            metadata_out[0] if metadata_out.ndim > 1 else metadata_out,
            parent=None
        )
        
        input_corpus = CustomInputCorpus(
            [seed_element],
            uniform_sample_function,
            threshold=0.1
        )
        
        # coverage tracking fuzzer
        class CustomFuzzer:
            def __init__(self, corpus, coverage_function, metadata_function, 
                         objective_function, mutation_function, fetch_function):
                self.corpus = corpus
                self.coverage_function = coverage_function
                self.metadata_function = metadata_function
                self.objective_function = objective_function
                self.mutation_function = mutation_function
                self.fetch_function = fetch_function
                self.objectives_found = []
                self.coverage_log = []
                self.fuzz_log = []
            
            def run_iteration(self):
                corpus_element = self.corpus.sample_input()
                mutations = self.mutation_function(corpus_element)
                coverage_batches, metadata_batches = self.fetch_function([mutations])
                
                # For normalized coverage, use a scalar value from logits
                # Extract a scalar coverage value (e.g., max activation)
                if coverage_batches[0].ndim > 1:
                    coverage_scalar = float(np.max(coverage_batches[0]))
                else:
                    coverage_scalar = float(coverage_batches[0][0])
                
                # add to coverage log
                self.coverage_log.append({
                    "iteration": len(self.coverage_log),
                    "coverage": coverage_scalar
                })
                
                for i, mutation in enumerate(mutations):
                    mutation.coverage = coverage_batches[0][i]
                    mutation.metadata = metadata_batches[0][i]
                    if self.objective_function(mutation):
                        self.objectives_found.append(mutation)
                        self.fuzz_log.append({
                            "iteration": len(self.coverage_log),
                            "status": "objective found",
                            "coverage": coverage_scalar
                        })
                    self.corpus.maybe_add_to_corpus(mutation)
                return len(self.objectives_found) > 0
        
        # start recording time
        start_time = time.time()
        
        fuzzer_obj = CustomFuzzer(
            input_corpus,
            coverage_function,
            metadata_function,
            objective_function,
            custom_mutation_function,
            fetch_function
        )
        
        results = {"iterations_completed": 0, "errors": []}
        logs.append("Starting fuzzing for 1000 iterations")
        try:
            for i in range(1000):
                logs.append(f"Iteration {i+1}/1000")
                found_objective = fuzzer_obj.run_iteration()
                results["iterations_completed"] = i + 1
                if found_objective:
                    logs.append("Found objective! Stopping.")
                    break
                if (i + 1) % 100 == 0:  # Log every 100
                    corpus_size = len(fuzzer_obj.corpus.corpus)
                    logs.append(f"Progress: {i + 1}/1000 iterations completed. Corpus size: {corpus_size}")
        except Exception as e:
            err_msg = f"Error during fuzzing: {str(e)}"
            logs.append(err_msg)
            traceback.print_exc()
            results["errors"].append(err_msg)
        
        # get final coverage
        final_coverage = 0
        if fuzzer_obj.coverage_log:
            final_coverage = fuzzer_obj.coverage_log[-1]["coverage"]
        
        # build JSON
        result = {
            "fuzzer": "tensorfuzz",
            "model": model_path,
            "iterations": results["iterations_completed"],
            "parameters": {
                "input_shape": str(input_shape),
                "expected_features": expected_features
            },
            "results": {
                "corpus_size": len(fuzzer_obj.corpus.corpus),
                "objectives_found": len(fuzzer_obj.objectives_found),
                "coverage": {
                    "log": fuzzer_obj.coverage_log,
                    "overall": final_coverage
                },
                "timing": {
                    "start_time": start_time,
                    "end_time": time.time(),
                    "duration_seconds": time.time() - start_time
                }
            },
            "logs": {
                "fuzz_log": fuzzer_obj.fuzz_log,
                "errors": results["errors"],
                "debug": logs
            },
            "additional_info": {
                "interesting_inputs": [],
                "raw_output": "TensorFuzz run completed."
            }
        }

        with open(os.path.join(RESULTS_DIR, "result.json"), "w") as f:
            json.dump(result, f, indent=2)
        
        return result
    # write exceptions to JSON
    except Exception as e:
        traceback.print_exc()
        result_data = {"error": str(e)}
        with open(os.path.join(RESULTS_DIR, "result.json"), "w") as f:
            json.dump(result_data, f)
        return result_data

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tensorfuzz_fuzzer.py <model_path>")
        sys.exit(1)
    result = run_fuzzing(sys.argv[1])
    print(json.dumps(result))