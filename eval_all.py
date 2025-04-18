from utils import start_vllm_server, stop_vllm_server, chat_completion, write_jsonl, read_jsonl
import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor

def scorer(response):
    response = response.lower()
    if "the answer is correct" in response or "the answer is approximated but should be correct" in response:
        return True
    else:
        return False

# Original system message for judge, single-choice and multi-choice evaluations.
check_sys_msg = """You are a helpful AI assistant. You will use your coding and language skills to verify the answer.
You are given:
    1. A problem.
    2. A reply with the answer to the problem.
    3. A ground truth answer.
Please do the following:
1. Extract the answer in the reply: "The answer is <answer extracted>".
2. Check whether the answer in the reply matches the ground truth answer. When comparison is not obvious (for example, 3*\\sqrt(6) and 7.348), you may write code to check the answer and wait for the user to execute the code.
3. After everything is done, please choose a reply from the following options:
    - "The answer is correct."
    - "The answer is approximated but should be correct. Correct Answer: <ground truth answer> | Answer extracted: <answer extracted>."
    - "The answer is incorrect. Correct Answer: <ground truth answer> | Answer extracted: <answer extracted>."
    - "The reply doesn't contain an answer." """

def extract_rating(response):
    """
    Extract a rating (1 to 5) from the LLM response.
    Assumes the response contains a phrase like "Rating: X" (case insensitive).
    """
    m = re.search(r'Rating\s*:\s*([1-5])', response, re.IGNORECASE)
    if m:
        return int(m.group(1))
    else:
        return None

def eval_jsonl(path_to_jsonl, api_base, model_name, max_tokens=256, temperature=0.7, threads=10, output_file=None):
    def process_data(data_item, api_base, model_name, max_tokens=256, temperature=0.7):
        reference_answer = data_item.get("answer", "")
        llm_answer = data_item.get("llm_answer", "")
        question = data_item.get("question", "")
        q_type = data_item.get("type", "").lower()
        
        # For single-choice or multi-choice, append choices if present.
        if q_type in ["single-choice", "multi-choice"]:
            choices = data_item.get("choices")
            if choices:
                question += "\nOptions:\n" + choices
        
        user_prompt = "Problem: " + question + f"\n\nReply: {llm_answer}\n\nGround truth answer: " + reference_answer
        if q_type in ["open", "fill"]:
            user_prompt = "Problem: " + question + f"\n\nReply: {llm_answer}\n\nGround truth answer: " + reference_answer + "Ground turth explanation: " + data_item.get("explanation", "")

        # For judge, single-choice and multi-choice types, use the original prompt.
        if q_type in ["judge", "single-choice", "multi-choice"]:
            messages = [
                {"role": "system", "content": check_sys_msg},
                {"role": "user", "content": user_prompt}
            ]
            response = chat_completion(api_base=api_base, model_name=model_name, messages=messages,
                                       max_tokens=max_tokens, temperature=temperature)
            eval_result = scorer(response)
            return {
                "question": question,
                "llm_answer": llm_answer,
                "reference_answer": reference_answer,
                "eval_feedback": response,
                "eval_result": eval_result,
                "type": q_type
            }
        # For fill and open types, use a rating prompt.
        elif q_type in ["fill", "open"]:
            rating_sys_msg = """You are a helpful AI assistant. Your task is to evaluate the quality of a given answer by comparing it with the ground truth with it explanation.
            You are provided with:
                1. A problem statement.
                2. A reply containing the answer to the problem.
                3. The ground truth answer.
                4. The ground truth explanation.
            Please perform the following steps:
            1. Extract the answer from the reply.
            2. Compare the extracted answer with the ground truth answer.
            3. Evaluate the quality of the reply based on its correctness and completeness and consider the ground truth explanation.
            4. Assign a rating from 1 to 5 based on the following criteria:
                - **Rating 5:** The reply is almost identical to the ground truth. The answer is complete, accurate, and uses nearly the same formulation.
                - **Rating 4:** The reply is mostly correct, with only minor errors or omissions that do not impact the overall correctness.
                - **Rating 3:** The reply is partially correct. It demonstrates some understanding but includes noticeable errors or missing key details.
                - **Rating 2:** The reply is largely incorrect or incomplete, showing significant misunderstandings of the problem.
                - **Rating 1:** The reply is completely off; it does not address the problem or is entirely incorrect.
            4. Output your evaluation in the exact format: "Rating: X. Explanation: <your explanation>." 
            Ensure that your explanation clearly justifies the assigned rating.
            """

            messages = [
                {"role": "system", "content": rating_sys_msg},
                {"role": "user", "content": user_prompt}
            ]
            response = chat_completion(api_base=api_base, model_name=model_name, messages=messages,
                                       max_tokens=max_tokens, temperature=temperature)
            rating = extract_rating(response)
            return {
                "question": question,
                "llm_answer": llm_answer,
                "reference_answer": reference_answer,
                "eval_feedback": response,
                "eval_rating": rating,
                "type": q_type
            }
        # Fallback to judge prompt if type is unspecified or unrecognized.
        else:
            messages = [
                {"role": "system", "content": check_sys_msg},
                {"role": "user", "content": user_prompt}
            ]
            response = chat_completion(api_base=api_base, model_name=model_name, messages=messages,
                                       max_tokens=max_tokens, temperature=temperature)
            eval_result = scorer(response)
            return {
                "question": question,
                "llm_answer": llm_answer,
                "reference_answer": reference_answer,
                "eval_feedback": response,
                "eval_result": eval_result,
                "type": q_type
            }
    
    win_counter = 0
    data_list = list(read_jsonl(path_to_jsonl))
    total_counter = len(data_list)
    file_name = os.path.splitext(os.path.basename(path_to_jsonl))[0]
    
    # Default output file location if not provided.
    if output_file is None:
        output_file = os.path.join("eval_results", file_name + "_eval.jsonl")
    
    output_list = []
    
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(process_data, data_item, api_base, model_name, max_tokens, temperature)
                   for data_item in data_list]
        for future in futures:
            result_json = future.result()
            # For judge-style evaluation, count correct answers.
            if "eval_result" in result_json:
                win_counter += int(result_json.get("eval_result"))
            output_list.append(result_json)
   
    write_jsonl(output_file, output_list)
    print(f'[INFO] Evaluation results have been saved to {output_file}')
    if total_counter > 0:
        print(f'[INFO] Accuracy: {win_counter/total_counter*100:.2f}%')
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the LLM on a jsonl file')
    parser.add_argument('--api_base', type=str, default="https://api.openai.com", help='API base URL')
    parser.add_argument('--model_name', type=str, default="text-davinci-003", help='Model name')
    parser.add_argument('--path_to_jsonl_list', type=str, help='List of paths to the input JSONL files')
    parser.add_argument('--max_tokens', type=int, default=256, help='Max tokens')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model')
    parser.add_argument('--port', type=int, default=8000, help='Port')
    parser.add_argument('--gpu', type=int, default=1, help='GPU')
    parser.add_argument('--threads', type=int, default=10, help='Threads')
    parser.add_argument('--output_file_list', type=str, default=None, help='List of output file paths')
    
    args = parser.parse_args()
    
    # When a model path is provided, we start the vLLM server.
    if args.model_path:
        process_id = start_vllm_server(args.model_path, args.model_name, args.port, args.gpu)
        path_json_list = args.path_to_jsonl_list.split(',')
        output_file_list = args.output_file_list.split(',')
        for path_to_jsonl, output_path in zip(path_json_list, output_file_list):
            eval_jsonl(path_to_jsonl, args.api_base, args.model_name, args.max_tokens,
                       args.temperature, args.threads, output_path)
        stop_vllm_server(process_id)
    else:
        path_json_list = args.path_to_jsonl_list.split(',')
        output_file_list = args.output_file_list.split(',')
        for path_to_jsonl, output_path in zip(path_json_list, output_file_list):
            eval_jsonl(path_to_jsonl, args.api_base, args.model_name, args.max_tokens,
                       args.temperature, args.threads, output_path)
