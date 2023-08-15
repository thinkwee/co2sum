# CO2Sum
-   Code for the paper [CO2Sum:Contrastive Learning for Factual-Consistent Abstractive Summarization](https://arxiv.org/pdf/2112.01147.pdf)
-   Include the negative sample constuction method LFN CO2Sum_LFN, and the training code CO2Sum_train

## CO2Sum_LFN
-   Code for the LFN algorithm introduced in the paper
-   The process of LFN consists of three steps:
    -   run get_next.py to get the context of summary
    -   run ./LFN_LM/run.sh to get the fact fragments based on the article, summary and context
    -   run LFN_construct.py to construct the negative samples based on the fact fragments

## CO2Sum_train
-   We develop our method based on the fairseq. Since there is no model architecture modified, you can just extend the criterion, data, tasks by setting --user-dir to CO2Sum_train then start training and inference by using default fairseq-train and fairseq-generate
-   The loss function of CoEnc and CoDec are described in the ./criterions/label_smoothed_cross_entropy_with_position_triplet_contrastive.py
-   The data loading process for ground truth summary and negative samples is described in ./data/language_position_triplet_dataset.py