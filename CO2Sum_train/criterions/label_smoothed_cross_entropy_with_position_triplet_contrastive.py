import math
import logging

from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion
from fairseq import metrics, utils

from collections import deque

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def label_smoothed_nll_loss(lprobs, target, epsilon, mask_neg=None, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if mask_neg is not None:
        mask_neg = mask_neg.contiguous().view(mask_neg.size(0) * mask_neg.size(1), 1)
        nll_loss.masked_fill_(mask_neg == 1, 0)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion("label_smoothed_cross_entropy_with_position_triplet_contrastive")
class LabelSmoothedCrossEntropyCriterionWithPositionTripletContrastive(LabelSmoothedCrossEntropyCriterion):
    def __init__(self, task, sentence_avg, label_smoothing, ignore_prefix_size=0, report_accuracy=False, contrastive_lambda=2.0, temperature=1.0, margin=1.0):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.contrastive_lambda = contrastive_lambda
        self.margin = margin
        self.temperature = temperature

    @staticmethod
    def add_args(parser):
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        parser.add_argument("--contrastive-lambda", type=float, default=2.0, help="The contrastive loss weight")
        parser.add_argument(
            "--margin",
            type=float,
            default=1.0,
        )
        parser.add_argument(
            "--temperature",
            type=float,
            default=1.0,
        )

    def swap_sample(self, sample):
        target = sample["target"]
        prev_output_tokens = sample["net_input"]["prev_output_tokens"]
        src_tokens = torch.cat((prev_output_tokens[:, :1], sample["net_input"]['src_tokens']), dim=-1)

        target_neg = sample["target_neg"]
        prev_output_tokens_neg = sample["net_input_neg"]["prev_output_tokens"]
        src_tokens_neg = torch.cat((prev_output_tokens_neg[:, :1], sample["net_input_neg"]['src_tokens']), dim=-1)

        return {
            "net_input": {
                "src_tokens": target.contiguous(),
                "src_lengths": (target != self.padding_idx).int().sum(dim=1),
                "prev_output_tokens": src_tokens[:, :-1].contiguous()
            },
            "net_input_neg": {
                "src_tokens": target_neg.contiguous(),
                "src_lengths": (target_neg != self.padding_idx).int().sum(dim=1),
                "prev_output_tokens": src_tokens_neg[:, :-1].contiguous()
            },
            'nsentences': sample['nsentences'],
            'ntokens': utils.item((src_tokens[:, 1:] != self.padding_idx).int().sum().data),
            "target": src_tokens[:, 1:].contiguous(),
            "id": sample["id"],
        }

    def _sentence_embedding(self, encoder_out, src_tokens):
        encoder_output = encoder_out.transpose(0, 1)
        mask = (src_tokens != self.padding_idx)
        encoder_embedding = (encoder_output * mask.unsqueeze(-1)).sum(dim=1) / mask.int().sum(dim=1).unsqueeze(-1)
        return encoder_embedding

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])

        encoder_out = model.encoder.forward(sample["net_input"]["src_tokens"], sample["net_input"]["src_lengths"])["encoder_out"][0]
        encoder_out = self._sentence_embedding(encoder_out, sample["net_input"]["src_tokens"])

        loss, nll_loss = self.compute_loss(model, net_output, sample, target_type="target", reduce=reduce)

        sample_size = (sample["target"].size(0) if self.sentence_avg else sample["ntokens"])
        nsentences = sample["target"].size(0)
        ntokens = sample["ntokens"]

        ntokens_neg = sample["ntokens_neg"] if sample["ntokens_neg"] is not None else 1
        mask_length_neg = sample["mask_length_neg"] if sample["mask_length_neg"] is not None else 1

        nll_loss_neg = nll_loss - nll_loss
        triplet_loss = nll_loss - nll_loss
        infoNce_loss = nll_loss - nll_loss
        if sample['target_neg'] is not None:
            net_output_neg = model(**sample["net_input_neg"])
            mask_neg = sample['mask_neg'].transpose(0, 1)
            loss_neg, nll_loss_neg = self.compute_loss(model, net_output_neg, sample, mask_neg=mask_neg, target_type="target_neg", reduce=reduce)
            pos_loss = nll_loss / ntokens
            neg_loss = nll_loss_neg / mask_length_neg if mask_length_neg > 0 else 0
            triplet_loss = max((pos_loss - neg_loss) + self.margin, triplet_loss)

            reverse_sample = self.swap_sample(sample)
            positive_encoder_out = model.encoder.forward(reverse_sample["net_input"]["src_tokens"], reverse_sample["net_input"]["src_lengths"])["encoder_out"][0]
            positive_encoder_out = self._sentence_embedding(positive_encoder_out, reverse_sample["net_input"]["src_tokens"])
            negative_encoder_out = model.encoder.forward(reverse_sample["net_input_neg"]["src_tokens"], reverse_sample["net_input_neg"]["src_lengths"])["encoder_out"][0]
            negative_encoder_out = self._sentence_embedding(negative_encoder_out, reverse_sample["net_input_neg"]["src_tokens"])

            infoNce_loss = self.get_infoNce_loss(encoder_out, positive_encoder_out, negative_encoder_out)

        #all_loss = loss + triplet_loss * self.contrastive_lambda * ntokens_neg + infoNce_loss * self.contrastive_lambda * ntokens / nsentences
        all_loss = loss + triplet_loss * self.contrastive_lambda * ntokens_neg + infoNce_loss * self.contrastive_lambda * ntokens

        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "nll_loss_neg": nll_loss_neg.data,
            "triplet_loss": triplet_loss.data,
            "infoNce_loss": infoNce_loss.data,
            "ntokens": ntokens,
            "ntokens_neg": ntokens_neg,
            "nsentences": nsentences,
            "sample_size": sample_size
        }

        return all_loss, sample_size, logging_output

    def get_infoNce_loss(self, encoder_out, positive_out, negative_out):
        l_pos = torch.sum(encoder_out * positive_out, dim=1).unsqueeze(-1)
        l_neg = torch.matmul(encoder_out, negative_out.transpose(0, 1))

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.temperature

        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        crossentropyloss = nn.CrossEntropyLoss()
        loss = crossentropyloss(logits, labels)

        return loss

    def similarity_function(self, ):
        return nn.CosineSimilarity(dim=-1)

    def get_lprobs_and_target(self, model, net_output, sample, target_type):
        def get_targets(target_type):
            return sample[target_type]

        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = get_targets(target_type)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size:, :].contiguous()
                target = target[:, self.ignore_prefix_size:].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size:, :, :].contiguous()
                target = target[self.ignore_prefix_size:, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, mask_neg=None, target_type=None, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample, target_type)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            mask_neg,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        super().reduce_metrics(logging_outputs)
        nsentences = utils.item(sum(log.get("nsentences", 0) for log in logging_outputs))
        infoNce_loss = utils.item(sum(log.get("infoNce_loss", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        ntokens_neg = utils.item(sum(log.get("ntokens_neg", 0) for log in logging_outputs))
        nll_loss_neg = utils.item(sum(log.get("nll_loss_neg", 0) for log in logging_outputs))
        triplet_loss = utils.item(sum(log.get("triplet_loss", 0) for log in logging_outputs))

        metrics.log_scalar("nll_loss_neg", nll_loss_neg / ntokens_neg / math.log(2), ntokens_neg, round=3)
        metrics.log_scalar("triplet_loss", triplet_loss / math.log(2), ntokens_neg, round=3)
        metrics.log_scalar(
            "infoNce_loss",
            infoNce_loss / nsentences / math.log(2),
            nsentences,
            round=3,
        )
