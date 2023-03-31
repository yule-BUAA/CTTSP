import pandas as pd
import json
from tqdm import tqdm
from collections import defaultdict
import datetime


class Set(object):
    """
    Class for a single Set
    """

    def __init__(self, user_id: int, items_id: list, set_time: datetime.date or int):
        # id for user and items
        self.user_id = user_id  # user id
        self.items_id = items_id  # items id
        # set_time
        self.set_time = set_time  # set appearing time, datetime.date or int
        # set_type, str
        self.set_type = None  # None, 'train', 'validate' or 'test'
        # next items, denotes ground truth
        self.next_set = list()  # next basket items

    def to_json(self) -> dict:
        """
        convert Set object to json format
        :return:
        """
        if isinstance(self.set_time, datetime.date):
            time_str = self.set_time.strftime("%Y-%m-%d")
        else:
            time_str = self.set_time
        return {
            'user_id': self.user_id,
            'items_id': self.items_id,
            'set_time': time_str,
            'set_type': self.set_type,
            'next_set': self.next_set
        }


def get_frequent_items(transaction_df: pd.DataFrame, frequency_rate: float, key: str = 'product_id'):
    """
    get frequent items based on frequency_rate

    :param transaction_df: pd.DataFrame
    :param frequency_rate
    :param key
    :return:
        new_df: pd.DataFrame['product_id',...]
    """
    value_counts = transaction_df[key].value_counts()
    total_number = len(transaction_df)
    sum_number = 0
    item_list = []
    for index in tqdm(value_counts.index):
        if sum_number / total_number >= frequency_rate:
            break
        sum_number += value_counts[index]
        item_list.append(index)

    new_df = transaction_df[transaction_df[key].isin(item_list)]
    return new_df


def save_as_json(data: dict or list, path: str):
    """
    save data as json file with path

    :param data:
    :param path:
    :return:
    """
    with open(path, "w") as file:
        file.write(json.dumps(data))
        file.close()
        print(f'{path} writes successfully.')


def reindex_items_users(users_baskets: list, items_map_dic_path: str, users_map_dic_path: str):
    """
    reindex item id and user id in baskets

    :param users_baskets: list, [[user_1's Basket_1, user_1's Basket_2, ...], [user_2's Basket_1, user_2's Basket_2, ...], ...]
    :param items_map_dic_path:
    :param users_map_dic_path:
    :return:
    """
    # reindex item id
    items_list = []
    for user_baskets in users_baskets:
        for basket in user_baskets:
            items_list.extend(basket.items_id)

    unique_item_id_list = list(set(items_list))
    unique_item_id_list.sort()

    # generate item reindex mapping
    item_id_map_dic = defaultdict(int)
    for index, value in enumerate(unique_item_id_list):
        item_id_map_dic[value] = index
    save_as_json(item_id_map_dic, items_map_dic_path)

    # reindex item
    for user_baskets in users_baskets:
        for basket in user_baskets:
            for index, item in enumerate(basket.items_id):
                basket.items_id[index] = item_id_map_dic[item]

    # reindex user id
    user_id_map_dic = defaultdict(int)
    for index, user_baskets in enumerate(users_baskets):
        # [Basket,...]
        user_id_map_dic[user_baskets[0].user_id] = index
        for basket in user_baskets:
            basket.user_id = index
    save_as_json(user_id_map_dic, users_map_dic_path)


def get_next_set_info(user_baskets: list):
    """
    get next set information of each set
    :param user_baskets: [Basket,...], time ascending
    :return:
    """
    for i in range(0, len(user_baskets) - 1):
        user_baskets[i].next_set = user_baskets[i + 1].items_id


def set_batch(baskets: list):
    """
    batch baskets with Set-batch

    :param baskets: [Basket,...], the streaming baskets, ascending by time
    :return:
    """
    set_batches = []

    # record the current position of users and items
    user_current_position_marker = {}
    item_current_position_marker = {}

    def get_user_current_position(user_id: int) -> int:
        if user_current_position_marker.get(user_id) is None:
            return -1
        else:
            return user_current_position_marker[user_id]

    def get_item_current_position(item_id: int) -> int:
        if item_current_position_marker.get(item_id) is None:
            return -1
        else:
            return item_current_position_marker[item_id]

    def update_position(basket, index):
        user_current_position_marker[basket.user_id] = index
        for item_id in basket.items_id:
            item_current_position_marker[item_id] = index

    for basket in baskets:

        current_position = max(max([get_item_current_position(item_id) for item_id in basket.items_id]),
                               get_user_current_position(basket.user_id),
                               -1)
        # position to insert (current position add 1）
        batch_to_insert = current_position + 1

        update_position(basket, batch_to_insert)

        if batch_to_insert >= len(set_batches):
            set_batches.append([])
        set_batches[batch_to_insert].append(basket)

    return set_batches
