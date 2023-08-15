import argparse
from data_process import get_data_loader
from run_process import run_process

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-input_file_path', type=str, default='./cnndm_external.json')
    parser.add_argument('-out_file_path', type=str, default='./result')
    parser.add_argument('-model_path', type=str, default='./distilgpt2')
    parser.add_argument('-rem_words', type=int, default=3)
    parser.add_argument('-top_k', type=int, default=10)
    parser.add_argument('-stop_word_file_path', type=str, default='./stopword_large.txt')
    parser.add_argument('-num_workers', type=int, default=0)
    parser.add_argument('-cuda', action='store_true')
    parser.add_argument('-all_split', type=int, default=16)
    parser.add_argument('-now_split_index', type=int, default=1)
    parser.add_argument('-process_data_type', type=str, default='json')

    opt = parser.parse_args()
    opt.out_file_path = opt.out_file_path + '-' + str(opt.all_split) + '-' + str(opt.now_split_index)
    print(opt)

    data_loader = get_data_loader(opt)

    run_process(opt, data_loader)

    print('finish')
