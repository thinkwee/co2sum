import torch
from model import LanguageModel
import time


def run_process(opt, data_loader):

    language_model = LanguageModel(opt)

    result_tgt_text_list = []

    with torch.no_grad():
        for index, data in enumerate(data_loader):
            index, tgt_all_text_list, tgt_all_text_ids_list, tgt_all_text_mask_list, tgt_all_text_mask_sum_list, \
            condition_all_text_ids_list, condition_all_text_mask_list, \
            condition_all_text_mask_except_tgt_sum_list, condition_all_text_mask_except_tgt_list = data

            if opt.cuda:
                for cuda_index in range(len(tgt_all_text_ids_list)):
                    tgt_all_text_ids_list[cuda_index] = tgt_all_text_ids_list[cuda_index].cuda()
                    tgt_all_text_mask_list[cuda_index] = tgt_all_text_mask_list[cuda_index].cuda()
                    tgt_all_text_mask_sum_list[cuda_index] = tgt_all_text_mask_sum_list[cuda_index].cuda()
                    condition_all_text_ids_list[cuda_index] = condition_all_text_ids_list[cuda_index].cuda()
                    condition_all_text_mask_list[cuda_index] = condition_all_text_mask_list[cuda_index].cuda()
                    condition_all_text_mask_except_tgt_sum_list[cuda_index] = condition_all_text_mask_except_tgt_sum_list[cuda_index].cuda()
                    condition_all_text_mask_except_tgt_list[cuda_index] = condition_all_text_mask_except_tgt_list[cuda_index].cuda()

            result_tgt_text_list.append(
                language_model(tgt_all_text_list, tgt_all_text_ids_list, tgt_all_text_mask_list, tgt_all_text_mask_sum_list, condition_all_text_ids_list,
                               condition_all_text_mask_list, condition_all_text_mask_except_tgt_sum_list, condition_all_text_mask_except_tgt_list))

            if (index + 1) % 10000 == 0:
                print(index, ', time: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                with open(opt.out_file_path, 'a+', encoding='utf-8') as f:
                    for tgt_text_index in range(len(result_tgt_text_list)):
                        f.write('\n' + '. '.join(result_tgt_text_list[tgt_text_index]))
                    result_tgt_text_list = []

        with open(opt.out_file_path, 'a+', encoding='utf-8') as f:
            for tgt_text_index in range(len(result_tgt_text_list)):
                f.write('\n' + '. '.join(result_tgt_text_list[tgt_text_index]))
            result_tgt_text_list = []
