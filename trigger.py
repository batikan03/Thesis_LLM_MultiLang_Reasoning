import subprocess
import sys
from pathlib import Path


def run_script(description: str, script_path: Path, args=None):
    print(f"----- Running: {description} --------")
    if not script_path.is_file():
        print(f" {script_path} does not exist.")
        return

    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(str(a) for a in args)

    result = subprocess.run(
        cmd,
        cwd=script_path.parent,
    )

    if result.returncode != 0:
        print(f" '{description}' failed with code {result.returncode}")
        sys.exit(result.returncode)
    else:
        print(f"Finished: {description} ")


def main():
    base = Path(__file__).resolve().parent

    confidence_root = (
        base
        / "confidence_results"
        / "new_results_200d"
        / "confidence_results_RQ1_newresults"
    )

    scripts_in_order = [
        # 1) LLM pipelines (generate model outputs)
        ("RQ1 fully multilingual  pipeline",
         base / "LLM_Pipelines" / "pipeline_for_MMLU_RQ1.py",
         None),

        # 2) Evaluation tools (decoder metrics, F1, etc.)
        ("Results reader",
         base / "evaluation_tools" / "readerforresults_alternative.py",
         None),

        ("Decoder confidence pipeline",
         base / "evaluation_tools" / "decoderconfidencepipeline.py",
         None),

        # 3) Graph making tools
        ("Graph generator",
         base / "graph_making_tools" / "graphilator.py",
         [confidence_root]), 

        ("Instability graph maker",
         base / "graph_making_tools" / "instability_graph_maker.py",
         None),
    ]

    for desc, script, args in scripts_in_order:
        run_script(desc, script, args)


if __name__ == "__main__":
    main()
