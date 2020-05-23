from code_parser import *
import pickle
import tqdm
import pandas as pd
import os
from gensim.models import KeyedVectors as word2vec
import sys

# vectors_text_path = '../data/target_vecs.txt'
# all_funcs_data_path = '../data/bcb_funcs_all.tsv'
# parser_path = '../data/java.so'


def process(all_funcs_data_path, parser_path, vectors_text_path, output_path):
    print("loading functions")
    all_funcs = pd.read_csv(all_funcs_data_path, delimiter="\t", header=None)

    print("preparing parser")
    parser = get_parser(parser_path)

    print("loading code2vec")
    code2vec = word2vec.load_word2vec_format(vectors_text_path, binary=False)

    print("processing functions")
    if not os.path.exists(output_path):
      os.mkdir(output_path)

    for index, row in tqdm.tqdm(all_funcs.iterrows()):
        processed_function = parse_program(row[1], parser, code2vec)
        with open(os.path.join(output_path, str(row[0])), 'wb') as f:
            pickle.dump(processed_function, f)


if __name__ == '__main__':

    args = sys.argv

    if len(args) == 5:
        process(args[1], args[2], args[3], args[4])
    else:
        print("args_required: all_funcs_data_path, parser_path, vectors_text_path, output_path")
