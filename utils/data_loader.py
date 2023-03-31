import json
import torch
import numpy as np
import copy
import itertools


def pad_sequence(seq_to_pad: list, length, pad_value=-1):
    """
    padding seq_to_pad to specific length with value -1
    :return: seq with len(seq) == length
    """
    if len(seq_to_pad) < length:
        seq_to_pad += [pad_value for _ in range(length - len(seq_to_pad))]

    return seq_to_pad


def get_k_hot_encoding(id_list: list, num_classes: int):
    """
    get k-hot encoding based on the input ids
    :param id_list: list, list of ids, shape (input_items_num, )
    :param num_classes:
    :return:
        k_hot_encoding[i] = 1 if i in id_list, else 0
    """
    k_hot_encoding = torch.zeros(num_classes)
    if len(id_list) > 0:
        k_hot_encoding[id_list] = 1
    return k_hot_encoding


class CustomizedDataLoader(object):
    """
    Customized Data Loader
    """

    def __init__(self, data_path: str):
        """
        :param data_path: batched json file
        """
        self.data_path = data_path
        print(f"creating data loader from path {data_path}")
        with open(data_path, 'r') as file:
            # self.data, batches of sets [[user i's set_j, user u's set_v, ...], ...]
            self.data = json.load(file)

    def load_data(self):
        """
        load data
        :return:
            batches_sets
            n_users: total number of users
            n_items: total number of items
        """
        # get users and items num
        user_ids_set, item_ids_set = set(), set()
        for batch_sets in self.data:
            user_ids_set = user_ids_set.union(set([batch_set['user_id'] for batch_set in batch_sets]))
            item_ids_set = item_ids_set.union(set(itertools.chain.from_iterable([batch_set['items_id'] for batch_set in batch_sets])))

        num_users, num_items = len(user_ids_set), len(item_ids_set)

        # batched sets data
        batches_sets = []
        for batch_sets in self.data:
            # each user only has one set at a time, so set_time and set_delta_t denote user_time and user_delta_t in fact
            # batch_sets [{"user_id": 6, "items_id": [1161, 3394], "set_time": "2018-03-01", "set_semantic_time_feat": [2018, 3, 1, 3, 0],
            #              "set_delta_t": 0, "items_delta_t": [0, 0], "set_type": "train", "next_set": [1161]}, ...]

            """
               batch_length, shape (batch_size, ), tensor
               batch_set_type, shape (batch_size, ), np.ndarray
               batch_user_id, shape (batch_size, ), tensor
               batch_items_id, shape (batch_size, max_set_size), tensor
               batch_set_semantic_time_feat, shape (batch_size, semantic_time_feat_input_dim), Tensor
               batch_user_delta_t, shape (batch_size, ), Tensor
               batch_items_delta_t, shape (batch_size, max_set_size), Tensor
               batch_set_truth, shape (batch_size, num_items), Tensor (calculate loss)
            """
            batch_length, batch_set_type, batch_user_id, batch_items_id, batch_set_truth = [], [], [], [], []

            # get maximal size of set in batch_sets (for padding)
            max_set_size = max([len(batch_set['items_id']) for batch_set in batch_sets])

            for batch_set in batch_sets:
                # length of each set
                batch_length.append(len(batch_set['items_id']))
                # type of each set, train, validate or test
                batch_set_type.append(batch_set['set_type'])
                # user id and items id
                batch_user_id.append(batch_set['user_id'])
                batch_items_id.append(pad_sequence(seq_to_pad=copy.deepcopy(batch_set['items_id']), length=max_set_size, pad_value=-1))

                batch_set_truth.append(get_k_hot_encoding(id_list=batch_set['next_set'], num_classes=num_items))

            batches_sets.append(
                (
                 torch.tensor(batch_length), np.array(batch_set_type), torch.tensor(batch_user_id),
                 torch.tensor(batch_items_id), torch.stack(batch_set_truth, dim=0).float()
                 )
            )

        return batches_sets, num_users, num_items
