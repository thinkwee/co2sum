import json
import os

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
from transformers import GPT2Tokenizer


class GetCandidateSentence():
    def __init__(self, stop_word_path, top_k):
        self.stopwords = set()
        with open(stop_word_path, "r") as f:
            for line in f:
                self.stopwords.add(line.strip())
        self.top_k = top_k

    def get_candiates(self, sentence):
        delete_set = []
        current_span = []
        list_line = sentence.split(" ")
        limit_length = 0
        for idx, word in enumerate(list_line):
            if word.lower() not in self.stopwords:
                limit_length += 1
                delete_set.append([idx])
                current_span.append(idx)
                if len(current_span) > 1:
                    delete_set.append(current_span)
                if len(current_span) == 3:
                    current_span = []
            else:
                current_span = []
        if len(current_span) > 1:
            delete_set.append(current_span)
        sampled_idx = [i for i in range(len(delete_set))]
        sampled_idx = np.random.choice(sampled_idx, min(10, len(sampled_idx)), replace=False)
        sampled_idx = [set(delete_set[i]) for i in sampled_idx]
        sampled_idx_2 = [sampled_idx[i] | sampled_idx[j] for i in range(len(sampled_idx)) for j in range(i + 1, len(sampled_idx))]
        sampled_idx_2 = list(np.random.choice(sampled_idx_2, min(10, len(sampled_idx_2)), replace=False))
        sampled_idx_3 = [sampled_idx[i] | sampled_idx_2[j] for i in range(len(sampled_idx)) for j in range(len(sampled_idx_2))]
        final_sampled = sampled_idx + sampled_idx_2 + sampled_idx_3
        final_sampled.sort(key=lambda x: -len(x))

        idx = 0
        result = []
        deduplicate = set()
        limit_length = limit_length // 2
        for pos in final_sampled:
            if len(pos) > limit_length:
                continue
            span = " ".join([list_line[i] for i in range(len(list_line)) if i not in set(pos) and list_line[i] not in self.stopwords])
            if span not in deduplicate:
                deduplicate.add(span)
                result.append(span)
                idx += 1
                if idx == self.top_k:
                    break
        return result


class SentenceDataset(Dataset):
    def __init__(self, opt):
        self.use_cuda = opt.cuda
        self.rem_words = opt.rem_words
        self.candidate_generate = GetCandidateSentence(opt.stop_word_file_path, opt.top_k)
        self.GPT2_tokenizer = GPT2Tokenizer.from_pretrained(opt.model_path + '/')
        self.GPT2_tokenizer.pad_token = self.GPT2_tokenizer.eos_token

        self.tgt_text_list = []
        self.s_next_list = []

        if opt.process_data_type == 'json':
            all_data_list = None
            with open(opt.input_file_path, 'r', encoding='utf-8') as f:
                all_data_list = json.load(f)
            split_num = len(all_data_list) // opt.all_split
            self.start_index = (opt.now_split_index - 1) * split_num
            if opt.now_split_index == opt.all_split:
                all_data_list = all_data_list[self.start_index:]
            else:
                all_data_list = all_data_list[self.start_index:self.start_index + split_num]

            print('load result from last save file: ', opt.out_file_path)
            print('...')
            if os.path.exists(opt.out_file_path):
                with open(opt.out_file_path, 'r', encoding='utf-8') as f:
                    file_content = f.readlines()
                    process_num = len(file_content) - 1
                all_data_list = all_data_list[process_num:]
                print('load finish, load num: ', process_num)
            else:
                print('no last save file, will start new ...')
            for all_data in tqdm(all_data_list):
                self.tgt_text_list.append(all_data['list_tgt'])
                self.s_next_list.append([all_data['list_src'][id] for id in all_data['next_idx']])

        elif opt.process_data_type == 'jsonl':
            all_data_list = None
            with open(opt.input_file_path, 'r', encoding='utf-8') as f:
                all_data_list = f.readlines()
            split_num = len(all_data_list) // opt.all_split
            self.start_index = (opt.now_split_index - 1) * split_num
            if opt.now_split_index == opt.all_split:
                all_data_list = all_data_list[self.start_index:]
            else:
                all_data_list = all_data_list[self.start_index:self.start_index + split_num]

            print('load result from last save file: ', opt.out_file_path)
            print('...')
            if os.path.exists(opt.out_file_path):
                with open(opt.out_file_path, 'r', encoding='utf-8') as f:
                    file_content = f.readlines()
                    process_num = len(file_content) - 1
                all_data_list = all_data_list[process_num:]
                print('load finish, load num: ', process_num)
            else:
                print('no last save file, will start new ...')
            for all_data in tqdm(all_data_list):
                all_data = json.loads(all_data)
                self.tgt_text_list.append(all_data['list_tgt'])
                self.s_next_list.append([all_data['list_src'][id] for id in all_data['next_idx']])
        else:
            raise Exception('only accept json or jsonl')

    def __len__(self):

        return len(self.tgt_text_list)

    def __getitem__(self, index):
        try:
            tgt_texts = self.tgt_text_list[index]
            s_nexts = self.s_next_list[index]
            tgt_all_text_list = []
            tgt_all_text_ids_list = []
            tgt_all_text_mask_list = []
            tgt_all_text_mask_sum_list = []
            condition_all_text_ids_list = []
            condition_all_text_mask_list = []
            condition_all_text_mask_except_tgt_list = []
            condition_all_text_mask_except_tgt_sum_list = []

            for tgt_index in range(len(tgt_texts)):
                if len(tgt_texts[tgt_index].strip().strip('\n').split()) < self.rem_words:
                    continue
                tgt_input_text_list = [tgt_texts[tgt_index]]
                tgt_text_condidates = self.candidate_generate.get_candiates(tgt_texts[tgt_index])
                tgt_input_text_list.extend(tgt_text_condidates)
                tokenizer_result = self.GPT2_tokenizer(tgt_input_text_list, padding=True, return_tensors='pt', truncation=True, max_length=600)
                tgt_all_text_list.append(tgt_input_text_list)
                tgt_all_text_ids_list.append(tokenizer_result.input_ids)
                tgt_all_text_mask_list.append(tokenizer_result.attention_mask)
                tgt_all_text_mask_sum_list.append(tokenizer_result.attention_mask.sum(dim=-1))

                condition_input_text_list = [text + ' . ' + s_nexts[tgt_index] for text in tgt_input_text_list]
                tokenizer_result = self.GPT2_tokenizer(condition_input_text_list, padding=True, return_tensors='pt', truncation=True, max_length=600)
                condition_all_text_ids_list.append(tokenizer_result.input_ids)
                condition_all_text_mask_list.append(tokenizer_result.attention_mask)

                condition_mask_sum_list_temp = tokenizer_result.attention_mask.clone().detach()
                for condition_mask_index in range(tokenizer_result.attention_mask.size(0)):
                    condition_mask_sum_list_temp[condition_mask_index, 0:tgt_all_text_mask_sum_list[-1][condition_mask_index] + 1] = 0
                condition_all_text_mask_except_tgt_list.append(condition_mask_sum_list_temp)
                condition_all_text_mask_except_tgt_sum_list.append(condition_mask_sum_list_temp.sum(dim=-1))

        except Exception as e:
            print(index)
            print(e)

        return index, tgt_all_text_list, tgt_all_text_ids_list, tgt_all_text_mask_list, tgt_all_text_mask_sum_list, \
               condition_all_text_ids_list, condition_all_text_mask_list, \
               condition_all_text_mask_except_tgt_sum_list, condition_all_text_mask_except_tgt_list


def get_data_loader(opt):
    dataset = SentenceDataset(opt)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=opt.num_workers, pin_memory=False)

    return data_loader