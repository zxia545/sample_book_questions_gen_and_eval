from utils import start_vllm_server, stop_vllm_server, chat_completion, write_jsonl, read_jsonl
import argparse
from concurrent.futures import ThreadPoolExecutor
import os

# Mapping from dataset prefix to tailored system prompts
SYSTEM_PROMPTS = {
    "anthropology": "You are an Expert Anthropologist. Your task is to provide a thorough and accurate response.",
    "econ": "You are an Expert Economist. Your task is to provide a thorough and accurate response.",
    "law": "You are an Expert Lawyer. Your task is to provide a thorough and accurate response.",
    "phi": "You are an Expert Philosopher. Your task is to provide a thorough and accurate response."
}

def gen_answers(input_file, output_file, api_base, model_name, max_tokens=256, temperature=0.7, threads=10):
    """
    Generates answers for different datasets using tailored system prompts.
    The dataset type is derived from the input file name (assumes file name starts with the domain prefix).
    If a data item has type "multi-choice" or "single-choice", its question is combined with its choices.
    """
    # Derive the domain from the input filename (e.g. "anthropology-ancient_society.jsonl" -> "anthropology")
    base_name = os.path.basename(input_file)
    domain = base_name.split("-")[0].lower()
    system_prompt = SYSTEM_PROMPTS.get(
        domain,
        "You are an Expert in your field. Your task is to provide a thorough and accurate response."
    )
    
    input_data_list = read_jsonl(input_file)
    output_data_list = []

    def process_data(data_item, api_base, model_name, max_tokens=256, temperature=0.7):
        # Get the main question text.
        question = data_item.get("question", "")
        
        # Check if this is a multi-choice or single-choice question.
        # q_type = data_item.get("type", "").lower()
        # if q_type in ["multi-choice", "single-choice"]:
        choices = data_item.get("choices", [])
        if choices:
            # Append the choices to the question prompt.
            question += "\nOptions:\n" + "\n".join(choices)
        
        # Combine with an instruction similar to the math script.
        prompt = "Ensure that your answer is precise and complete, covering all important aspects of the question:\n" + question
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # Call the chat completion helper.
        response = chat_completion(api_base=api_base, model_name=model_name, messages=messages,
                                   max_tokens=max_tokens, temperature=temperature)
        data_item["llm_answer"] = response
        return data_item

    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [
            executor.submit(process_data, data_item, api_base, model_name, max_tokens, temperature)
            for data_item in input_data_list
        ]
        for future in futures:
            output_data_list.append(future.result())

    write_jsonl(output_file, output_data_list)
    print(f"[INFO] Generation complete. Results saved to {output_file}.")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate answers for various datasets using vLLM.")
    parser.add_argument("--input_file", type=str, help="Path to the input JSONL file.")
    parser.add_argument("--output_file", type=str, help="Path to the output JSONL file.")
    parser.add_argument("--api_base", type=str, help="Base URL for the OpenAI API.")
    parser.add_argument("--model_name", type=str, help="Name of the model to use.")
    parser.add_argument("--max_tokens", type=int, default=256, help="Maximum number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model.")
    parser.add_argument("--port", type=int, default=8000, help="Port to host the model on.")
    parser.add_argument("--gpu", type=int, default=1, help="Number of GPUs to use.")
    parser.add_argument("--threads", type=int, default=10, help="Number of threads to use for generation.")
    
    args = parser.parse_args()
    
    
    if args.model_path:
        process_id = start_vllm_server(args.model_path, args.model_name, args.port, args.gpu)
        
        
        if ',' in args.input_file and ',' in args.output_file:
            input_files = args.input_file.split(',').strip()
            output_files = args.output_file.split(',').strip()
            for i in range(len(input_files)):
                gen_answers(input_files[i], output_files[i], args.api_base, args.model_name,
                            args.max_tokens, args.temperature, args.threads)
        else:
            gen_answers(args.input_file, args.output_file, args.api_base, args.model_name,
                        args.max_tokens, args.temperature, args.threads)
        stop_vllm_server(process_id)
    else:
        if ',' in args.input_file and ',' in args.output_file:
            input_files = args.input_file.split(',').strip()
            output_files = args.output_file.split(',').strip()
            for i in range(len(input_files)):
                gen_answers(input_files[i], output_files[i], args.api_base, args.model_name,
                            args.max_tokens, args.temperature, args.threads)
        else:
            gen_answers(args.input_file, args.output_file, args.api_base, args.model_name,
                        args.max_tokens, args.temperature, args.threads)
