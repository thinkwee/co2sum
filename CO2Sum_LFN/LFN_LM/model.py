import torch.nn.modules as nn
from transformers import GPT2LMHeadModel
import torch


class LanguageModel(nn.Module):
    def __init__(self, opt):
        super(LanguageModel, self).__init__()

        self.GPT2_model = GPT2LMHeadModel.from_pretrained(opt.model_path + '/')
        if opt.cuda:
            self.GPT2_model = self.GPT2_model.cuda()
        self.GPT2_model.eval()

    def forward(self, tgt_all_text_list, tgt_all_text_ids_list, tgt_all_text_mask_list, tgt_all_text_mask_sum_list, condition_all_text_ids_list,
                condition_all_text_mask_list, condition_all_text_mask_except_tgt_sum_list, condition_all_text_mask_except_tgt_list):

        result_text_list = []
        for index in range(len(tgt_all_text_ids_list)):
            tgt_text_list = tgt_all_text_list[index]
            tgt_text_ids = tgt_all_text_ids_list[index]
            tgt_text_mask = tgt_all_text_mask_list[index]
            tgt_text_mask_sum = tgt_all_text_mask_sum_list[index]
            condition_text_ids = condition_all_text_ids_list[index]
            condition_text_mask = condition_all_text_mask_list[index]
            condition_text_mask_except_tgt_sum = condition_all_text_mask_except_tgt_sum_list[index]
            condition_text_mask_except_tgt = condition_all_text_mask_except_tgt_list[index]

            tgt_text_list_logits = self.GPT2_model(input_ids=tgt_text_ids, attention_mask=tgt_text_mask).logits
            condition_text_logits = self.GPT2_model(input_ids=condition_text_ids, attention_mask=condition_text_mask).logits

            tgt_text_list_log_score = torch.log_softmax(tgt_text_list_logits, dim=-1).gather(dim=-1, index=tgt_text_ids.unsqueeze(-1)).squeeze(-1)
            tgt_text_list_log_score.masked_fill_(tgt_text_mask == 0, 0)
            tgt_text_list_log_score = tgt_text_list_log_score.sum(dim=-1) / tgt_text_mask_sum

            condition_text_log_score = torch.log_softmax(condition_text_logits, dim=-1).gather(dim=-1, index=condition_text_ids.unsqueeze(-1)).squeeze(-1)
            condition_text_log_score.masked_fill_(condition_text_mask_except_tgt == 0, 0)
            condition_text_log_score = condition_text_log_score.sum(dim=-1) / condition_text_mask_except_tgt_sum

            chose_tgt_text_list_index = tgt_text_list_log_score > tgt_text_list_log_score[0][0]
            chose_tgt_text_list_index[0][0] = True

            condition_text_log_score += chose_tgt_text_list_index
            max_score, max_index = torch.max(condition_text_log_score, dim=1)
            result_text_list.append(tgt_text_list[max_index][0])

        return result_text_list