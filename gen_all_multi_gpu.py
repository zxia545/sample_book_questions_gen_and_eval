import os
import argparse
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

"""
Wrapper script to launch multiple vLLM servers on different GPUs
and call an existing gen_all.py script for generation.
Assumes gen_all.py defines a CLI matching:
  python gen_all.py --input_file <in1,in2,...> \
                    --output_file <out1,out2,...> \
                    --api_base <api> --model_name <model> \
                    [--max_tokens N] [--temperature T] [--threads K]
"""

def process_model(model_name, model_path, input_files, output_dir,
                  port_start, max_tokens, temperature, threads, gpu_queue):
    gpu_id = gpu_queue.get()
    try:
        # Prepare env for subprocess
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        port = port_start + gpu_id
        api_base = f"http://localhost:{port}"

        # Launch vLLM server on assigned GPU
        vllm_cmd = [
            "vllm", "serve",
            "--model", model_path,
            "--port", str(port)
        ]
        print(f"[INFO] Starting vLLM for {model_name} on GPU {gpu_id} (port {port})")
        proc = subprocess.Popen(vllm_cmd, env=env)
        time.sleep(5)  # allow server to initialize

        # Prepare comma-separated input and output lists
        in_str = ",".join(input_files)
        out_files = [f"{model_name}_{os.path.basename(f)}" for f in input_files]
        out_str = ",".join([os.path.join(output_dir, o) for o in out_files])

        # Call gen_all.py once per model with combined inputs/outputs
        gen_cmd = [
            "python", "gen_all.py",
            "--input_file", in_str,
            "--output_file", out_str,
            "--api_base", api_base,
            "--model_name", model_name,
            "--max_tokens", str(max_tokens),
            "--temperature", str(temperature),
            "--threads", str(threads)
        ]
        print(f"[INFO] Generating for {model_name}: {in_str} -> {out_str}")
        subprocess.run(gen_cmd, env=env, check=True)
    finally:
        print(f"[INFO] Stopping vLLM for {model_name} on GPU {gpu_id}")
        proc.terminate()
        proc.wait()
        gpu_queue.put(gpu_id)


def main():
    parser = argparse.ArgumentParser(description="Run gen_all.py across multiple models on separate GPUs.")
    parser.add_argument("--models_dir", required=True, help="Directory of model subfolders.")
    parser.add_argument("--input_files", required=True,
                        help="Comma-separated list of JSONL input files.")
    parser.add_argument("--output_dir", required=True, help="Directory for generated outputs.")
    parser.add_argument("--port_start", type=int, default=8000,
                        help="Base port for vLLM servers.")
    parser.add_argument("--max_tokens", type=int, default=256,
                        help="Max tokens per call.")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature.")
    parser.add_argument("--threads", type=int, default=10,
                        help="Worker threads for gen_all.")
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3,4,5,6,7",
                        help="Comma-separated GPU IDs to use.")
    args = parser.parse_args()

    input_files = [f.strip() for f in args.input_files.split(",")]
    model_names = [d for d in os.listdir(args.models_dir)
                   if os.path.isdir(os.path.join(args.models_dir, d))]
    gpu_ids = [int(x) for x in args.gpu_ids.split(",")]

    os.makedirs(args.output_dir, exist_ok=True)
    gpu_queue = Queue()
    for gid in gpu_ids:
        gpu_queue.put(gid)

    with ThreadPoolExecutor(max_workers=len(model_names)) as executor:
        for m in model_names:
            mp = os.path.join(args.models_dir, m)
            executor.submit(
                process_model,
                m, mp, input_files, args.output_dir,
                args.port_start, args.max_tokens,
                args.temperature, args.threads, gpu_queue
            )

if __name__ == "__main__":
    main()
