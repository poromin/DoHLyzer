import gzip
import logging
import math
import os
import pickle

import ijson
import numpy as np
from sklearn.model_selection import train_test_split

import analyzer.utils as utils

def create_segments(clumps_list, segment_size):
    clumps_list2 = []
    for c in clumps_list:
        c2 = [
            utils.normalize(math.log10(max(1e-12, c[0])), data_min=-12, data_max=-2),
            utils.normalize(math.log10(max(1e-12, c[1])), data_min=-12, data_max=-2),
            utils.normalize(math.log10(min(1e4, c[2])), data_min=0.5, data_max=4),
            utils.normalize(math.log2(min(256, c[3])), data_min=0, data_max=8),
            c[4]
        ]
        clumps_list2.append(c2)

    while len(clumps_list2) < segment_size:
        clumps_list2.append([-1, -1, -1, -1, 0])

    return utils.nwise(clumps_list2, segment_size)

def load_json(file_path, label, segment_size, shuffle=True, max_count=0):
    logging.info(f"Loading {file_path}.")
    with (gzip.open(file_path, 'r') if file_path.endswith('gz') else open(file_path, 'r')) as json_file:
        items = ijson.items(json_file, 'item')
        segments = []

        for flow in items:
            if 0 < max_count < len(segments):
                break
            segments.extend(create_segments(flow, segment_size))

    if shuffle:
        np.random.shuffle(segments)

    return np.array(segments), np.full(len(segments), label)

def load_dataset(dir_path, segment_size, use_cache=True):
    cache_path = os.path.join(dir_path, f'cache-{segment_size}')
    if use_cache and os.path.exists(cache_path):
        logging.info("Using cached dataset.")
        return pickle.load(open(cache_path, 'rb'))

    doh_file = os.path.join(dir_path, 'mal-doh.json.gz')
    ndoh_file = os.path.join(dir_path, 'benign-doh.json.gz')

    if not (os.path.exists(doh_file) and os.path.exists(ndoh_file)):
        raise FileNotFoundError("Required files 'mal-doh.json.gz' or 'benign-doh.json.gz' are missing in the directory.")

    logging.info("Loading datasets.")
    doh_dataset = load_json(doh_file, 1, segment_size)
    ndoh_dataset = load_json(ndoh_file, 0, segment_size, max_count=len(doh_dataset[0]))

    logging.info("Combining datasets.")
    main_dataset = utils.combine(doh_dataset, ndoh_dataset)

    logging.info("Splitting dataset into training and testing sets.")
    dataset_tuple = train_test_split(*main_dataset)

    if use_cache:
        logging.info("Caching the dataset.")
        pickle.dump(dataset_tuple, open(cache_path, 'wb'))

    return dataset_tuple
