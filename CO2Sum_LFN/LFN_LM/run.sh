all_split_num=8
now_index=1
output_file_path=./result

CUDA_VISIBLE_DEVICES=0 python3 -u main.py \
-input_file_path ./xsum_external.jsonl \
-out_file_path ${output_file_path}/xsum_external \
-stop_word_file_path ../stopword_large.txt \
-model_path ./gpt2-distil \
-top_k 30 \
-all_split ${all_split_num} \
-now_split_index ${now_index} \
-num_workers 8 \
-process_data_type jsonl \
-cuda > ${output_file_path}/log/test-log-${all_split_num}-${now_index}-$(date "+%m-%d-%H-%M-%S")

