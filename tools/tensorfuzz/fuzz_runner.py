import sys
import json
import tensorflow as tf
from tensorfuzz.fuzz import fuzzer
from tensorfuzz.coverage import coverage
from tensorfuzz.mutation import mutation

def run_fuzzing(model_path):
    try:
        # 1. Load model
        model = tf.keras.models.load_model(model_path)
        
        # 2. Configure fuzzer
        cov = coverage.DeepGrapeCoverage(model)
        mut = mutation.ImageMutation()
        
        # 3. Initialize with model-appropriate parameters
        fuzz = fuzzer.Fuzzer(
            coverage_function=cov,
            mutation_function=mut,
            model=model,
            total_inputs_to_fuzz=1000,
            mutations_per_corpus_item=10,
            input_shape=model.input_shape[1:]  # Automatically get input size
        )
        
        # 4. Run fuzzing
        fuzz.run()
        
        # 5. Return standardized results
        return {
            'coverage': fuzz.coverage_history,
            'vulnerabilities': len(fuzz.found_errors),
            'errors': [str(e) for e in fuzz.found_errors]
        }
    
    except Exception as e:
        return {'error': str(e)}

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fuzz_runner.py <model_path>")
        sys.exit(1)
    
    result = run_fuzzing(sys.argv[1])
    print(json.dumps(result))