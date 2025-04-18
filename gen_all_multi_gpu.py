import os
import argparse
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

from utils import start_vllm_server, stop_vllm_server, chat_completion, write_jsonl, read_jsonl

# Mapping from dataset prefix to tailored system prompts
SYSTEM_PROMPTS = {
    "anthropology": "You are an Expert Anthropologist. Your task is to provide a thorough and accurate response.",
    "econ": "You are an Expert Economist. Your task is to provide a thorough and accurate response.",
    "law": "You are an Expert Lawyer. Your task is to provide a thorough and accurate response.",
    "phi": "You are an Expert Philosopher. Your task is to provide a thorough and accurate response."
}

def gen_answers(input_file, output_file, api_base, model_name,
                max_tokens=256, temperature=0.7, threads=10):
    """
    Generates answers for different datasets using tailored system prompts.
    """
    base_name = os.path.basename(input_file)
    domain = base_name.split("-")[0].lower()
    system_prompt = SYSTEM_PROMPTS.get(
        domain,
        "You are an Expert in your field. Your task is to provide a thorough and accurate response."
    )

    input_data_list = read_jsonl(input_file)
    output_data_list = []

    def process_data(data_item):
        question = data_item.get("question", "")
        choices = data_item.get("choices", [])
        if choices:
            question += "\nOptions:\n" + "\n".join(choices)
        prompt = ("Ensure that your answer is precise and complete, "
                  "covering all important aspects of the question:\n" + question)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        response = chat_completion(
            api_base=api_base,
            model_name=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        data_item["llm_answer"] = response
        return data_item

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(process_data, item) for item in input_data_list]
        for future in futures:
            output_data_list.append(future.result())

    write_jsonl(output_file, output_data_list)
    print(f"[INFO] Completed {output_file}")


def process_model(model_name, model_path, input_files, output_dir,
                  port_start, max_tokens, temperature, threads, gpu_queue):
    gpu_id = gpu_queue.get()
    try:
        # Restrict this process to the assigned GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        port = port_start + gpu_id
        api_base = f"http://localhost:{port}"
        print(f"[INFO] Launching vLLM for {model_name} on GPU {gpu_id}, port {port}")
        pid = start_vllm_server(model_path, model_name, port, gpu_id)
        try:
            for input_file in input_files:
                out_file = os.path.join(output_dir, f"{model_name}_{os.path.basename(input_file)}")
                gen_answers(
                    input_file, out_file, api_base, model_name,
                    max_tokens=max_tokens, temperature=temperature, threads=threads
                )
        finally:
            stop_vllm_server(pid)
            print(f"[INFO] Stopped vLLM for {model_name} (GPU {gpu_id})")
    finally:
        gpu_queue.put(gpu_id)


def main():
    parser = argparse.ArgumentParser(
        description="Generate answers across multiple models concurrently on separate GPUs."
    )
    parser.add_argument(
        "--models_dir", type=str, required=True,
        help="Directory containing subfolders for each model."
    )
    parser.add_argument(
        "--input_files", type=str, required=True,
        help="Comma-separated list of input JSONL files."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save output files."
    )
    parser.add_argument(
        "--port_start", type=int, default=8000,
        help="Starting port number for vLLM servers."
    )
    parser.add_argument(
        "--max_tokens", type=int, default=256,
        help="Maximum tokens per generation."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature."
    )
    parser.add_argument(
        "--threads", type=int, default=10,
        help="Threads per model for batch processing."
    )
    parser.add_argument(
        "--gpu_ids", type=str, default="0,1,2,3,4,5,6,7",
        help="Comma-separated GPU IDs to use."
    )

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
        for model_name in model_names:
            model_path = os.path.join(args.models_dir, model_name)
            executor.submit(\
                process_model,
                model_name, model_path, input_files, args.output_dir,
                args.port_start, args.max_tokens,
                args.temperature, args.threads, gpu_queue
            )

if __name__ == "__main__":
    main()
