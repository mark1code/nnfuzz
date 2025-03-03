import os
import sys
import json
import tensorflow as tf
from tensorfuzz.lib import fuzzer
from tensorfuzz.lib import corpus
from tensorfuzz.lib.coverage_functions import all_logit_coverage_function
from tensorfuzz.lib.mutation_functions import do_basic_mutations
from tensorfuzz.lib.sample_functions import uniform_sample_function

# Ensure the results directory exists
RESULTS_DIR = "/shared/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Dynamic custom object function
def custom_objects_fallback(obj_name):
    def dummy_function(*args, **kwargs):
        return 0  # Return a neutral value that won't impact training
    print(f"Warning: Unknown custom object '{obj_name}' replaced with a dummy function.")
    return dummy_function

def run_fuzzing(model_path):
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU

        # Attempt to load the model with dynamic unknown object handling
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={obj_name: custom_objects_fallback(obj_name) for obj_name in ["perplexity", "cross_entropy"]}
        )

        # Define coverage function
        coverage_function = all_logit_coverage_function

        # Generate random seed input
        input_shape = model.input_shape[1:]
        with tf.Session() as sess:
            seed_input = sess.run(tf.random.uniform(shape=(1,) + input_shape, minval=0, maxval=1))


        # Create seed corpus
        seed_corpus = corpus.seed_corpus_from_numpy_arrays(
            [[seed_input]],
            coverage_function,
            lambda x: (x, x),
            lambda x: (x, x)
        )

        # Initialize fuzzer
        fuzz = fuzzer.Fuzzer(
            corpus.InputCorpus(seed_corpus, uniform_sample_function, 0.1, "kdtree"),
            coverage_function,
            lambda x: False,  # No objective function for now
            do_basic_mutations,
            lambda x: (x, x)  # Dummy fetch function
        )

        fuzz.run()

        result_data = {
            'coverage': fuzz.coverage_history,
            'vulnerabilities': len(fuzz.found_errors),
            'errors': [str(e) for e in fuzz.found_errors]
        }

        # Save results to JSON
        result_path = os.path.join(RESULTS_DIR, "result.json")
        with open(result_path, "w") as f:
            json.dump(result_data, f)

        return result_data

    except Exception as e:
        result_data = {'error': str(e)}

        # Save failure message to JSON
        result_path = os.path.join(RESULTS_DIR, "result.json")
        with open(result_path, "w") as f:
            json.dump(result_data, f)

        return result_data

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fuzz_runner.py <model_path>")
        sys.exit(1)

    result = run_fuzzing(sys.argv[1])
    print(json.dumps(result))
