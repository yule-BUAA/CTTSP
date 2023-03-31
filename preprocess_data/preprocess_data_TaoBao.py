import pandas as pd
import random
from datetime import datetime
import os
from tqdm import tqdm

from utils_preprocess_data import Set
from utils_preprocess_data import get_frequent_items, reindex_items_users, get_next_set_info, save_as_json, set_batch

frequency_rate = 0.8
min_baskets_length = 4
max_baskets_length = 20
max_basket_boundary = 5


def read_file(data_path: str) -> pd.DataFrame:
    """
    Read original csv files from the data folder.

    Args:
        data_path : the path of the data.

    Returns:
        Transactions in pd.DataFrame.
    """
    transaction_df = pd.read_csv(data_path, header=None)
    transaction_df.columns = ['customer_id', 'product_id', 'subclass', 'behavior', 'date_time']
    transaction_df = transaction_df[['customer_id', 'subclass', 'behavior', 'date_time']]

    # behavior consists of 'pv'(click), 'buy', 'cart' and 'fav'
    # only buy behavior
    transaction_df = transaction_df[transaction_df['behavior'] == "buy"]

    transaction_df['date_time'] = pd.to_datetime(transaction_df['date_time'], unit='s').astype(str)
    # year-month-day
    transaction_df['date_time'] = transaction_df['date_time'].map(lambda x: x.split(' ')[0])
    transaction_df = transaction_df.sort_values(by='date_time')

    return transaction_df


def generate_baskets(transaction_df: pd.DataFrame, items_map_dic_path: str, users_map_dic_path: str):
    """
    generate baskets

    :param transaction_df: pd.DataFrame['customer_id', 'subclass', 'behavior', 'date_time']
    :param items_map_dic_path: save path of items id mapping dictionary
    :param users_map_dic_path: save path of users id mapping dictionary
    :return:
        users_baskets: [[Basket,...],...]
    """
    random.seed(0)

    # [[user_1's Basket_1, user_1's Basket_2, ...], [user_2's Basket_1, user_2's Basket_2, ...], ...]
    users_baskets = []
    for user_id, user in tqdm(transaction_df.groupby(['customer_id'])):
        baskets = []  # [Basket,...]
        for day, trans in user.groupby(['date_time']):  # select by user and day
            product_index_list = list(set(trans['subclass'].tolist()))
            date_time = datetime.strptime(day, "%Y-%m-%d").date()
            basket = Set(user_id, product_index_list, set_time=date_time)
            baskets.append(basket)

        # sort by time
        baskets = sorted(baskets, key=lambda basket: basket.set_time, reverse=False)

        if len(baskets) < min_baskets_length:
            # drop too short sequence
            continue
        if len(baskets) > max_baskets_length:
            # trim over-length sequence
            baskets = baskets[:random.randint(
                max_baskets_length - max_basket_boundary, max_baskets_length)]

        users_baskets.append(baskets)

    """
    reindex item id and user id
    """
    reindex_items_users(users_baskets, items_map_dic_path, users_map_dic_path)

    """
    print info
    """
    items_set, set_count, item_count = set(), 0, 0
    for user_baskets in users_baskets:
        set_count += len(user_baskets)
        for basket in user_baskets:
            item_count += len(basket.items_id)
            items_set = items_set.union(basket.items_id)

    # statistics of the dataset
    print(f'statistic: ')
    print(f'number of user: {len(users_baskets)}')
    print(f'number of item: {len(items_set)}')
    print(f'number of set: {set_count}')
    print(f'number of item per set: {item_count / set_count}')
    print(f'number of set per user: {set_count / len(users_baskets)}')
    print(f'date start from {transaction_df["date_time"].min()}, end at {transaction_df["date_time"].max()}')

    return users_baskets


def generate_data(users_baskets: list, out_set_batch_path: str):
    """
    1. separate train / validate / test set
    get next basket info as ground truth
    2. apply Set-batch for parallel training

    :param users_baskets: input data, [[Basket,...],...]
    :param out_set_batch_path: output set batch file path
    :return:
    """

    for index, user_baskets in enumerate(users_baskets):
        # sort baskets by time
        user_baskets = sorted(user_baskets, key=lambda basket: basket.set_time, reverse=False)
        # user_baskets[:-3] -> train (user_baskets[1:-2] as ground truth);
        # user_baskets[-3] -> validate  (user_baskets[-2] as ground truth);
        # user_baskets[-2] -> test (user_baskets[-1] as ground truth);
        for basket in user_baskets[:-3]:
            basket.set_type = 'train'
        user_baskets[-3].set_type = 'validate'
        user_baskets[-2].set_type = 'test'

        # get next-set (ground truth) for each set
        get_next_set_info(user_baskets)

        users_baskets[index] = user_baskets

    """
    flatten and sort by time with ascending order
    """
    streaming_baskets = []
    for user_baskets in users_baskets:
        streaming_baskets.extend(user_baskets)
    # sort basket by day with ascending order
    streaming_baskets = sorted(streaming_baskets, key=lambda basket: basket.set_time, reverse=False)

    """
    use set-batch
    """
    basket_batches = set_batch(streaming_baskets)

    for index, basket_batch in enumerate(basket_batches):
        basket_batches[index] = [basket.to_json() for basket in basket_batch]
    save_as_json(basket_batches, out_set_batch_path)


if __name__ == "__main__":
    data_path = f"../original_data/TaoBao_Userbehavior/UserBehavior.csv"

    dataset_name = 'TaoBao'

    root_path = f'../dataset/{dataset_name}'
    os.makedirs(root_path, exist_ok=True)

    items_map_dic_path = f'{root_path}/{dataset_name}_items_map_dic.json'
    users_map_dic_path = f'{root_path}/{dataset_name}_users_map_dic.json'
    out_set_batch_path = f'{root_path}/{dataset_name}.json'

    print('Reading files ...\n')
    transaction_df = read_file(data_path)

    print('Removing not frequent items ...\n')
    transaction_df = get_frequent_items(transaction_df, frequency_rate=frequency_rate, key='subclass')

    users_baskets = generate_baskets(transaction_df, items_map_dic_path, users_map_dic_path)

    print(f'Generating {dataset_name}.json file ...\n')
    generate_data(users_baskets, out_set_batch_path)
