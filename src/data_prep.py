import os
import pickle
import sys

import pandas as pd
import tqdm
from gensim.models import KeyedVectors as word2vec
from sklearn.model_selection import train_test_split

from code_parser import *


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
        processed_function = nx.convert_node_labels_to_integers(processed_function)
        nx.write_gpickle(processed_function, os.path.join(output_path, str(row[0])))


def split_dataset(pairs_path, function_path, split_folder):
    with open(pairs_path, 'rb') as f:
        pair_ids = pickle.load(f).to_numpy()

    pair_ids_filtered = []
    for id1, id2, label in pair_ids:
        if os.path.exists(os.path.join(function_path, str(id1))) and os.path.exists(
                os.path.join(function_path, str(id2))):
            pair_ids_filtered.append([id1, id2, label])

    print(f"removed {len(pair_ids) - len(pair_ids_filtered)} elements")

    pair_ids_filtered = np.array(pair_ids_filtered)
    train_data, test_data = train_test_split(pair_ids_filtered, test_size=0.15, random_state=42,
                                             stratify=pair_ids_filtered[:, 2])
    train_data, valid_data = train_test_split(train_data, test_size=0.15, random_state=42, stratify=train_data[:, 2])

    print(f"train: {len(train_data)}, valid: {len(valid_data)}, test: {len(test_data)}")
    np.savez(os.path.join(split_folder, "train"), train_data)
    np.savez(os.path.join(split_folder, "valid"), valid_data)
    np.savez(os.path.join(split_folder, "test"), test_data)


if __name__ == '__main__':

    args = sys.argv

    if len(args) == 7:
        print(args)
        process(args[1], args[2], args[3], args[4])
        split_dataset(args[5], args[4], args[6])
    else:
        print(
            "args_required: all_funcs_data_path, parser_path, vectors_text_path, output_path, pairs_dataset, split_folder")
