from rouge_google import rouge_n
from tqdm import tqdm
import json
import jsonlines
import sys

write_path = sys.argv[1]
src_path = sys.argv[2]
tgt_path = sys.argv[3]

data = []
writer = jsonlines.open(write_path, "w")

with open(src_path, "r") as f_src, open(tgt_path, "r") as f_tgt:
    for line_src, line_tgt in tqdm(zip(f_src, f_tgt)):
        list_src = line_src.strip().split(". ")
        list_tgt = line_tgt.strip().split(". ")
        l_src = len(list_src) - 1
        oracle_part = ""
        next_part = ""
        exist_idx = set()
        sample_data = dict()
        sample_data["list_src"] = list_src
        sample_data["list_tgt"] = list_tgt
        oracle_idx = []
        next_idx = []
        for sentence_tgt in list_tgt:
            max_id = 0
            max_rouge = 0.0
            for idx, sentence_src in enumerate(list_src):
                r1f, r1p, r1r = rouge_n([sentence_tgt], [sentence_src], 1)
                if r1f > max_rouge:
                    max_rouge = r1f
                    max_id = idx
            oracle_idx.append(max_id)
            next_idx.append(min(max_id + 1, l_src))
        sample_data["oracle_idx"] = oracle_idx
        sample_data["next_idx"] = next_idx
        writer.write(sample_data)
writer.close()