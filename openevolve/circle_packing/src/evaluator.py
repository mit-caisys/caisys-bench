"""
Evaluator for circle packing example (n=26) with improved timeout handling
"""

import hashlib
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

from openevolve.evaluation_result import EvaluationResult

THIS_FILE_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
ORACLE = str(THIS_FILE_DIR / "oracle.json")


class TimeoutError(Exception):
    pass


def get_result(program_path: str, result: EvaluationResult) -> EvaluationResult:
    if not os.path.exists(ORACLE):
        with open(ORACLE, "w") as f:
            json.dump({}, f)

    with open(ORACLE, "r") as f:
        data = json.load(f)

    with open(program_path, "rb") as f:
        content = f.read()

    file_hash = hashlib.sha256(content).hexdigest()
    if file_hash not in data:
        data[file_hash] = {}
        data[file_hash]["metrics"] = result.metrics
        data[file_hash]["artifacts"] = result.artifacts
        with open(ORACLE, "w") as f:
            json.dump(data, f, indent=4)

    return EvaluationResult(
        metrics=data[file_hash]["metrics"],
        artifacts=data[file_hash]["artifacts"],
    )


def timeout_handler(signum, frame):
    """Handle timeout signal"""
    raise TimeoutError("Function execution timed out")


def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle

    Returns:
        True if valid, False otherwise
    """
    n = centers.shape[0]

    # Check for NaN values
    if np.isnan(centers).any():
        return False, "NaN values detected in circle centers"

    if np.isnan(radii).any():
        return False, "NaN values detected in circle radii"

    # Check if radii are nonnegative and not nan
    for i in range(n):
        if radii[i] < 0:
            return False, f"Circle {i} has negative radius {radii[i]}"
        elif np.isnan(radii[i]):
            return False, f"Circle {i} has nan radius"

    # Check if circles are inside the unit square
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-6 or x + r > 1 + 1e-6 or y - r < -1e-6 or y + r > 1 + 1e-6:
            return (
                False,
                f"Circle {i} at ({x}, {y}) with radius {r} is outside the unit square",
            )

    # Check for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-6:  # Allow for tiny numerical errors
                return (
                    False,
                    f"Circles {i} and {j} overlap: dist={dist}, r1+r2={radii[i]+radii[j]}",
                )

    return True, ""


def run_with_timeout(program_path, timeout_seconds=20):
    """
    Run the program in a separate process with timeout
    using a simple subprocess approach

    Args:
        program_path: Path to the program file
        timeout_seconds: Maximum execution time in seconds

    Returns:
        centers, radii, sum_radii tuple from the program
    """
    # Create a temporary file to execute
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
        # Write a script that executes the program and saves results
        script = f"""
import sys
import numpy as np
import os
import pickle
import traceback

# Add the directory to sys.path
sys.path.insert(0, os.path.dirname('{program_path}'))

# Debugging info
print(f"Running in subprocess, Python version: {{sys.version}}")
print(f"Program path: {program_path}")

try:
    # Import the program
    spec = __import__('importlib.util').util.spec_from_file_location("program", '{program_path}')
    program = __import__('importlib.util').util.module_from_spec(spec)
    spec.loader.exec_module(program)
    
    # Run the packing function
    print("Calling run_packing()...")
    centers, radii, sum_radii = program.run_packing()
    print(f"run_packing() returned successfully: sum_radii = {{sum_radii}}")

    # Save results to a file
    results = {{
        'centers': centers,
        'radii': radii,
        'sum_radii': sum_radii
    }}

    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {temp_file.name}.results")
    
except Exception as e:
    # If an error occurs, save the error instead
    print(f"Error in subprocess: {{str(e)}}")
    traceback.print_exc()
    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump({{'error': str(e)}}, f)
    print(f"Error saved to {temp_file.name}.results")
"""
        temp_file.write(script.encode())
        temp_file_path = temp_file.name

    results_path = f"{temp_file_path}.results"

    try:
        # Run the script with timeout
        process = subprocess.Popen(
            [sys.executable, temp_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            exit_code = process.returncode

            # Always print output for debugging purposes
            print(f"Subprocess stdout: {stdout.decode()}")
            if stderr:
                print(f"Subprocess stderr: {stderr.decode()}")

            # Still raise an error for non-zero exit codes, but only after printing the output
            if exit_code != 0:
                raise RuntimeError(f"Process exited with code {exit_code}")

            # Load the results
            if os.path.exists(results_path):
                with open(results_path, "rb") as f:
                    results = pickle.load(f)

                # Check if an error was returned
                if "error" in results:
                    raise RuntimeError(f"Program execution failed: {results['error']}")

                return results["centers"], results["radii"], results["sum_radii"]
            else:
                raise RuntimeError("Results file not found")

        except subprocess.TimeoutExpired:
            # Kill the process if it times out
            process.kill()
            process.wait()
            raise TimeoutError(f"Process timed out after {timeout_seconds} seconds")

    finally:
        # Clean up temporary files
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        if os.path.exists(results_path):
            os.unlink(results_path)


def evaluate(program_path: str) -> EvaluationResult:
    result = _evaluate(program_path)
    return get_result(program_path, result)


def _evaluate(program_path: str) -> EvaluationResult:
    """
    Evaluate the program by running it once and checking the sum of radii

    Args:
        program_path: Path to the program file

    Returns:
        EvaluationResult, consisting of metrics and artifacts
    """
    # Target value from the paper
    TARGET_VALUE = 2.635  # AlphaEvolve result for n=26

    try:
        # For constructor-based approaches, a single evaluation is sufficient
        # since the result is deterministic
        start_time = time.time()

        # Use subprocess to run with timeout
        centers, radii, reported_sum = run_with_timeout(
            program_path, timeout_seconds=600  # Single timeout
        )

        end_time = time.time()
        eval_time = end_time - start_time

        # Ensure centers and radii are numpy arrays
        if not isinstance(centers, np.ndarray):
            centers = np.array(centers)
        if not isinstance(radii, np.ndarray):
            radii = np.array(radii)

        # Check for NaN values before validation
        if np.isnan(centers).any() or np.isnan(radii).any():
            message = "NaN values detected in solution"
            print(message)
            return EvaluationResult(
                metrics={
                    "sum_radii": 0.0,
                    "target_ratio": 0.0,
                    "validity": 0.0,
                    "eval_time": float(time.time() - start_time),
                    "combined_score": 0.0,
                },
                artifacts={"error": message},
            )

        # Validate solution
        valid, packing_message = validate_packing(centers, radii)
        if not valid:
            print(packing_message)

        # Check shape and size
        shape_valid = centers.shape == (26, 2) and radii.shape == (26,)
        if not shape_valid:
            message = f"Invalid shapes: centers={centers.shape}, radii={radii.shape}, expected (26, 2) and (26,)"
            print(message)
            if packing_message == "":
                packing_message = message
            else:
                packing_message += f"; {message}"
            valid = False

        # Calculate sum
        sum_radii = np.sum(radii) if valid else 0.0

        # Make sure reported_sum matches the calculated sum
        if abs(sum_radii - reported_sum) > 1e-6:
            print(
                f"Warning: Reported sum {reported_sum} doesn't match calculated sum {sum_radii}"
            )

        # Target ratio (how close we are to the target)
        target_ratio = sum_radii / TARGET_VALUE if valid else 0.0

        # Validity score
        validity = 1.0 if valid else 0.0

        # Combined score - higher is better
        combined_score = target_ratio * validity

        print(
            f"Evaluation: valid={valid}, sum_radii={sum_radii:.6f}, target={TARGET_VALUE}, ratio={target_ratio:.6f}, time={eval_time:.2f}s"
        )

        return EvaluationResult(
            metrics={
                "sum_radii": float(sum_radii),
                "target_ratio": float(target_ratio),
                "validity": float(validity),
                "eval_time": float(eval_time),
                "combined_score": float(combined_score),
            },
            artifacts={"error": packing_message},
        )

    except TimeoutError as e:
        print(f"Evaluation timed out: {str(e)}")
        return EvaluationResult(
            metrics={
                "sum_radii": 0.0,
                "target_ratio": 0.0,
                "validity": 0.0,
                "eval_time": 0.0,
                "combined_score": 0.0,
            },
            artifacts={"error": "Timeout during evaluation"},
        )
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        traceback.print_exc()
        return EvaluationResult(
            metrics={
                "sum_radii": 0.0,
                "target_ratio": 0.0,
                "validity": 0.0,
                "eval_time": 0.0,
                "combined_score": 0.0,
            },
            artifacts={"error": str(e)},
        )


# For testing
if __name__ == "__main__":
    if len(sys.argv) > 1:
        result = evaluate(sys.argv[1])
        print(f"Score: {result.metrics['combined_score']:.4f}")
        print(f"Sum Radii: {result.metrics['sum_radii']:.4f}")
        print(f"Time: {result.metrics['eval_time']:.4f}")
        print(f"Error: {result.artifacts['error']}")
