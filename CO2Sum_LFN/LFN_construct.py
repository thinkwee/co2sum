from os import error, register_at_fork
from random import choice, random
import numpy as np
from tqdm import tqdm
import faiss
import sys

src_path = sys.argv[1]
tgt_path = sys.argv[2]
fragment_path = sys.argv[3]
output_path = sys.argv[4]
log_path = sys.argv[5]

stopwords = set()

replace_ratio = 0.15

with open("./stopword_large.txt", "r") as f:
    for line in f:
        stopwords.add(line.lower().strip())


def init_faiss():
    vocab = set()
    with open(src_path, "r") as f:
        for idx, line in tqdm(enumerate(f)):
            for w in line.strip().split(" "):
                if w not in stopwords:
                    vocab.add(w.lower())

    words = []
    embeddings = []
    idx2word = dict()
    word2idx = dict()
    idx_count = 0

    with open("./glove.6B.50d.txt", "r") as f:
        for idx, line in tqdm(enumerate(f)):
            line = line.strip().split(" ")
            word = line[0].lower()
            if word in vocab:
                words.append(word)
                embedding_sample = [eval(item) for item in line[1:]]
                embeddings.append(embedding_sample)
                idx2word[idx_count] = word
                word2idx[word] = idx_count
                idx_count += 1

    embeddings_np = np.asarray(embeddings).astype('float32')
    del (embeddings)
    index = faiss.IndexFlatL2(50)
    print(index.is_trained)
    index.add(embeddings_np)
    print(index.ntotal)

    return word2idx, idx2word, index, embeddings_np


word2idx, idx2word, index, embeddings_np = init_faiss()
no_replace_count = 0


def check_pure_word(word):
    return "." not in word and '"' not in word and "," not in word and "?" not in word and "'" not in word and "!" not in word


def replace(line_src, line_tgt, line_info):
    flag = 1

    list_src = [w for w in line_src.strip().split(" ") if len(w) > 0]
    vocab_src = set(list_src)
    list_tgt = line_tgt.strip().split(" ")
    info_set = line_info.strip().split(" ")
    replace_set = set()
    for w in info_set:
        if w.lower() not in stopwords:
            replace_set.add(w.lower())

    replace_set = list(replace_set)
    query2idx = {word: idx for idx, word in enumerate(replace_set)}
    query_idx = [word2idx[word] if word in word2idx else 0 for word in replace_set]
    query_embedding = np.asarray([embeddings_np[idx] for idx in query_idx])
    error_flag = False
    replace_log = dict()
    replace_log["replace_w"] = []
    try:
        D, query_result = index.search(query_embedding, 8)
        query_result_final = []
        for result in query_result:
            result_final = [idx_result for idx_result in result[3:] if idx2word[idx_result] in vocab_src]
            if len(result_final) == 0:
                result_final = result[2:]
            query_result_final.append(result_final)

        list_modified = []
    except ValueError:
        error_flag = True

    if error_flag:
        pos = int(random() * len(list_tgt))
        pos = min(pos, len(list_tgt) - 1)
        w = list_tgt[pos]
        pos_src = int(random() * len(list_src))
        pos_src = min(pos, len(list_src) - 1)
        replace_word = list_src[pos_src]
        list_modified = list_tgt
        list_modified[pos] = replace_word
        replace_log["replace_w"].append(w + " --> " + replace_word)
    else:
        replace_count = 0
        for w in list_tgt:
            w_lower = w.lower()
            if w_lower in replace_set and check_pure_word(
                    w_lower) and replace_ratio > random() and replace_count < 100:
                if w_lower in word2idx:
                    replace_word_idx = choice(query_result_final[query2idx[w_lower]])
                    replace_word = idx2word[replace_word_idx]
                    if not check_pure_word(replace_word):
                        replace_word_idx = choice(query_result_final[query2idx[w_lower]])
                        replace_word = idx2word[replace_word_idx]
                else:
                    pos = int(random() * len(list_src))
                    pos = min(pos, len(list_src) - 1)
                    replace_word = list_src[pos]
                if replace_word != w:
                    replace_log["replace_w"].append(w + " --> " + replace_word)
                    replace_count += 1
                    list_modified.append(replace_word)
                else:
                    list_modified.append(w)
                flag = 0
            else:
                list_modified.append(w)
        if flag:
            pos = int(random() * len(list_tgt))
            pos = min(pos, len(list_tgt) - 1)
            w = list_tgt[pos]
            if w.lower() in word2idx and w.lower() in query2idx and check_pure_word(w.lower()):
                replace_word_idx = choice(query_result_final[query2idx[w.lower()]])
                replace_word = idx2word[replace_word_idx]
                if not check_pure_word(replace_word):
                    replace_word_idx = choice(query_result_final[query2idx[w.lower()]])
                    replace_word = idx2word[replace_word_idx]
            else:
                count = 0
                while not check_pure_word(w):
                    pos = int(random() * len(list_tgt))
                    pos = min(pos, len(list_tgt) - 1)
                    w = list_tgt[pos]
                    count += 1
                    if count == 10:
                        break
                if count == 10:
                    pos = int(random() * len(list_tgt))
                    pos = min(pos, len(list_tgt) - 1)
                    w = list_tgt[pos]

                pos_src = int(random() * len(list_src))
                pos_src = min(pos, len(list_src) - 1)
                replace_word = list_src[pos_src]
                count = 0
                while not check_pure_word(replace_word) and replace_word != w:
                    pos_src = int(random() * len(list_src))
                    pos_src = min(pos, len(list_src) - 1)
                    replace_word = list_src[pos_src]
                    count += 1
                    if count == 10:
                        break
                if count == 10:
                    pos_src = int(random() * len(list_src))
                    pos_src = min(pos, len(list_src) - 1)
                    replace_word = list_src[pos_src]

            list_modified = list_tgt
            list_modified[pos] = replace_word
            replace_log["replace_w"].append(w + " !!-->!! " + replace_word)

    line_modified = " ".join(list_modified)

    return flag, line_modified, replace_log


fw = open(output_path, "w")
fw_log = open(log_path, "w")
with open(src_path, "r") as f_src, open(tgt_path, "r") as f_tgt, open(fragment_path, "r") as f_info:
    for idx, (line_src, line_tgt, line_info) in tqdm(enumerate(zip(f_src, f_tgt, f_info))):
        flag, line_modified, replace_log = replace(line_src, line_tgt, line_info)
        if flag:
            no_replace_count += 1

        fw.write(line_modified + "\n")
        fw_log.write(str(replace_log) + "\n")
fw.close()
fw_log.write(str(no_replace_count) + "\n")
fw_log.close()
