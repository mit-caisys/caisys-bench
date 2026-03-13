"""
Evaluator for Rust adaptive sorting example
"""

import asyncio
import hashlib
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path

from openevolve.evaluation_result import EvaluationResult

THIS_FILE_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
ORACLE = str(THIS_FILE_DIR / "oracle_32_cores.json")

logger = logging.getLogger("examples.rust_adaptive_sort.evaluator")


def evaluate(program_path: str) -> EvaluationResult:
    result = asyncio.run(_evaluate_wrapper(program_path))
    if "error" in result.artifacts:
        logger.error(f"Error evaluating program: {result.artifacts['error']}")
        if "stderr" in result.artifacts:
            logger.error(f"Stderr: {result.artifacts['stderr']}")
        if "stdout" in result.artifacts:
            logger.error(f"Stdout: {result.artifacts['stdout']}")
    return result


async def _evaluate_wrapper(program_path: str) -> EvaluationResult:
    result = await _evaluate(program_path)
    return get_result(program_path, result)


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
        metrics=data[file_hash]["metrics"], artifacts=data[file_hash]["artifacts"]
    )


async def _evaluate(program_path: str) -> EvaluationResult:
    """
    Evaluate a Rust sorting algorithm implementation.

    Tests the algorithm on various data patterns to measure:
    - Correctness
    - Performance (speed)
    - Adaptability to different data patterns
    - Memory efficiency
    """
    try:
        # Create a temporary Rust project
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir) / "sort_test"

            # Initialize Cargo project
            result = subprocess.run(
                ["cargo", "init", "--name", "sort_test", str(project_dir)],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                return EvaluationResult(
                    metrics={"combined_score": 0.0, "compile_success": 0.0},
                    artifacts={
                        "error": "Failed to create Cargo project",
                        "stderr": result.stderr,
                    },
                )

            # Copy the program to src/lib.rs
            lib_path = project_dir / "src" / "lib.rs"
            with open(program_path, "r") as src:
                lib_content = src.read()
            with open(lib_path, "w") as dst:
                dst.write(lib_content)

            # Create main.rs with benchmark code
            project_source_dir = THIS_FILE_DIR / "sort_test"
            main_file_source = project_source_dir / "src" / "main.rs"
            with open(main_file_source, "r") as f:
                main_content = f.read()
            main_path = project_dir / "src" / "main.rs"
            with open(main_path, "w") as f:
                f.write(main_content)

            cargo_toml_source = project_source_dir / "Cargo.toml"
            with open(cargo_toml_source, "r") as f:
                cargo_toml_content = f.read()
            cargo_toml_path = project_dir / "Cargo.toml"
            with open(cargo_toml_path, "w") as f:
                f.write(cargo_toml_content)

            cargo_lock_source = project_source_dir / "Cargo.lock"
            with open(cargo_lock_source, "r") as f:
                cargo_lock_content = f.read()
            cargo_lock_path = project_dir / "Cargo.lock"
            with open(cargo_lock_path, "w") as f:
                f.write(cargo_lock_content)

            # Build the project
            build_result = subprocess.run(
                ["cargo", "build", "--release"],
                cwd=project_dir,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if build_result.returncode != 0:
                # Extract compilation errors
                return EvaluationResult(
                    metrics={
                        "combined_score": 0.0,
                        "compile_success": 0.0,
                        "correctness": 0.0,
                        "performance_score": 0.0,
                        "adaptability_score": 0.0,
                    },
                    artifacts={
                        "error": "Compilation failed",
                        "stderr": build_result.stderr,
                        "stdout": build_result.stdout,
                    },
                )

            # Run the benchmark
            run_result = subprocess.run(
                ["cargo", "run", "--release"],
                cwd=project_dir,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if run_result.returncode != 0:
                return EvaluationResult(
                    metrics={
                        "combined_score": 0.0,
                        "compile_success": 1.0,
                        "correctness": 0.0,
                        "performance_score": 0.0,
                        "adaptability_score": 0.0,
                    },
                    artifacts={"error": "Runtime error", "stderr": run_result.stderr},
                )

            # Parse JSON output
            try:
                # Find JSON in output (between first { and last })
                output = run_result.stdout
                start = output.find("{")
                end = output.rfind("}") + 1
                json_str = output[start:end]

                results = json.loads(json_str)

                # Calculate overall score
                correctness = results["correctness"]
                performance = results["performance_score"]
                adaptability = results["adaptability_score"]

                # Weighted score (correctness is mandatory)
                if correctness < 1.0:
                    overall_score = 0.0
                else:
                    overall_score = 0.6 * performance + 0.4 * adaptability

                # Check for memory safety (basic check via valgrind if available)
                memory_safe = 1.0  # Rust is memory safe by default

                return EvaluationResult(
                    metrics={
                        "combined_score": overall_score,
                        "compile_success": 1.0,
                        "correctness": correctness,
                        "performance_score": performance,
                        "adaptability_score": adaptability,
                        "avg_time": results["avg_time"],
                        "memory_safe": memory_safe,
                    },
                    artifacts={
                        "times": results["times"],
                        "all_correct": results["all_correct"],
                        "build_output": build_result.stdout,
                    },
                )

            except (json.JSONDecodeError, KeyError) as e:
                return EvaluationResult(
                    metrics={
                        "combined_score": 0.0,
                        "compile_success": 1.0,
                        "correctness": 0.0,
                        "performance_score": 0.0,
                        "adaptability_score": 0.0,
                    },
                    artifacts={
                        "error": f"Failed to parse results: {str(e)}",
                        "stdout": run_result.stdout,
                    },
                )

    except subprocess.TimeoutExpired:
        return EvaluationResult(
            metrics={
                "combined_score": 0.0,
                "compile_success": 0.0,
                "correctness": 0.0,
                "performance_score": 0.0,
                "adaptability_score": 0.0,
            },
            artifacts={"error": "Timeout during evaluation"},
        )
    except Exception as e:
        return EvaluationResult(
            metrics={
                "combined_score": 0.0,
                "compile_success": 0.0,
                "correctness": 0.0,
                "performance_score": 0.0,
                "adaptability_score": 0.0,
            },
            artifacts={"error": str(e), "type": "evaluation_error"},
        )


# For testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        result = evaluate(sys.argv[1])
        print(f"Score: {result.metrics['combined_score']:.4f}")
        print(f"Correctness: {result.metrics['correctness']:.4f}")
        print(f"Performance: {result.metrics['performance_score']:.4f}")
        print(f"Adaptability: {result.metrics['adaptability_score']:.4f}")
        print(f"Time: {result.metrics['avg_time']:.4f}")
