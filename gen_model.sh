python gen_all.py \
--input_file=./dataset/anthropology-ancient_society.jsonl,./dataset/anthropology-race_of_man.jsonl,./dataset/econ-econ_v2.jsonl,./dataset/law-law_v2.jsonl,./dataset/phi-phi_v2.jsonl,./dataset/anthropology-man_past.jsonl,./dataset/econ-econ_v1.jsonl,./dataset/law-law_v1.jsonl,./dataset/phi-phi_v1.jsonl,./dataset/phi-phi_v3.jsonl \
--output_file=./llama3.1_output/anthropology-ancient_society.jsonl,./llama3.1_output/anthropology-race_of_man.jsonl,./llama3.1_output/econ-econ_v2.jsonl,./llama3.1_output/law-law_v2.jsonl,./llama3.1_output/phi-phi_v2.jsonl,./llama3.1_output/anthropology-man_past.jsonl,./llama3.1_output/econ-econ_v1.jsonl,./llama3.1_output/law-law_v1.jsonl,./llama3.1_output/phi-phi_v1.jsonl,./llama3.1_output/phi-phi_v3.jsonl \
--port=8010 \
--max_tokens=512 \
--gpu=8 \
--threads=88 \
--model_name=davinci \
--api_base=http://localhost:8010 \