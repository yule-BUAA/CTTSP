import torch
import numpy as np
import torch.nn as nn
import warnings
import os
import shutil
from tqdm import tqdm
import json
import logging
import time
import sys

from utils.load_config import get_attribute, config
from utils.utils import create_optimizer, create_lr_scheduler, convert_to_gpu, set_random_seed, get_n_params
from utils.metrics import get_metric, get_all_metric
from utils.data_loader import CustomizedDataLoader
from utils.EarlyStopping import EarlyStopping
from model.CTTSP import CTTSP


def evaluate(model: nn.Module, batches_sets: tuple, num_users: int, num_items: int, logger: logging.Logger):
    """
    evaluate model
    :param model:
    :param batches_sets:
    :param num_users:
    :param num_items:
    :param logger:
    :return:
    """

    # reset memory at each epoch
    model.reset_memory()
    model.eval()

    train_metrics, val_metrics, test_metrics = [], [], []
    train_loss_list, val_loss_list, test_loss_list = [], [], []
    users_history_items = convert_to_gpu(torch.zeros(num_users, num_items), device=get_attribute('device'))

    tqdm_loader = tqdm(batches_sets, ncols=175)
    for batch, batch_sets in enumerate(tqdm_loader):
        train_loss, val_loss, test_loss = None, None, None
        """
        batch_sets-> tuple(
           batch_length, shape (batch_size, ), tensor
           batch_set_type, shape (batch_size, ), np.ndarray
           batch_user_id, shape (batch_size, ), tensor
           batch_items_id, shape (batch_size, max_set_size), tensor
           batch_set_truth, shape (batch_size, num_items), Tensor (calculate loss)
           )
        """
        batch_length, batch_set_type, batch_user_id, batch_items_id, batch_set_truth = batch_sets

        batch_user_id, batch_items_id, batch_set_truth = convert_to_gpu(batch_user_id, batch_items_id, batch_set_truth,
                                                                        device=get_attribute('device'))

        # split set_batch for training, validating and testing
        train_idx = torch.tensor(np.where(batch_set_type == 'train')[0])
        val_idx = torch.tensor(np.where(batch_set_type == 'validate')[0])
        test_idx = torch.tensor(np.where(batch_set_type == 'test')[0])

        # update users_history_items using train_idx, val_idx and test_idx, note that test_idx does not affect the user, since the ground truth is in 'next_set'
        index = torch.cat([train_idx, val_idx, test_idx])
        if len(index) > 0:
            for set_user_id, set_items_id, set_length in zip(batch_user_id[index], batch_items_id[index],
                                                             batch_length[index]):
                set_items_id = set_items_id[:set_length]
                users_history_items[set_user_id][set_items_id] += 1

        # feed a batch into model
        batch_set_predict, batch_user_memory, batch_items_memory = model(batch_length=batch_length,
                                                                         batch_user_id=batch_user_id,
                                                                         batch_items_id=batch_items_id,
                                                                         users_history_items=users_history_items)

        # if current batch contains at least one train set
        if len(train_idx) > 0:
            train_batch_set_predict, train_batch_set_truth = batch_set_predict[train_idx], batch_set_truth[train_idx]
            train_loss = loss_func(train_batch_set_predict, train_batch_set_truth)

            train_loss_list.append(train_loss.item())
            train_metrics.append(get_metric(y_true=train_batch_set_truth, y_pred=train_batch_set_predict))

        # if current batch contains at least one validate set
        if len(val_idx) > 0:
            val_batch_set_predict, val_batch_set_truth = batch_set_predict[val_idx], batch_set_truth[val_idx]
            val_loss = loss_func(val_batch_set_predict, val_batch_set_truth)

            val_loss_list.append(val_loss.item())
            val_metrics.append(get_metric(y_true=val_batch_set_truth, y_pred=val_batch_set_predict))

        # if current batch contains at least one test set
        if len(test_idx) > 0:
            test_batch_set_predict, test_batch_set_truth = batch_set_predict[test_idx], batch_set_truth[test_idx]
            test_loss = loss_func(test_batch_set_predict, test_batch_set_truth)

            test_loss_list.append(test_loss.item())
            test_metrics.append(get_metric(y_true=test_batch_set_truth, y_pred=test_batch_set_predict))

        # update memory
        if len(index) > 0:
            model.update_memory(batch_length=batch_length[index],
                                batch_user_id=batch_user_id[index],
                                batch_items_id=batch_items_id[index],
                                batch_user_memory=batch_user_memory[index],
                                batch_items_memory=batch_items_memory[index])

        tqdm_loader.set_description(
            f'batch: {batch + 1}, train loss: {train_loss.item() if train_loss is not None else None}, '
            f'val loss: {val_loss.item() if val_loss is not None else None}, test loss: {test_loss.item() if test_loss is not None else None}')

    train_metric = get_all_metric(metric_list=train_metrics)
    val_metric = get_all_metric(metric_list=val_metrics)
    test_metric = get_all_metric(metric_list=test_metrics)

    logger.info(f"train loss {torch.Tensor(train_loss_list).mean()}, valid loss: {torch.Tensor(val_loss_list).mean()}, "
                f"test loss: {torch.Tensor(test_loss_list).mean()}, \n"
                f"train metric: {train_metric}, \nvalid metric: {val_metric}, \ntest metric: {test_metric}")
    return train_metric, val_metric, test_metric


if __name__ == "__main__":
    """
    init dataloader, paths and logger
    """
    warnings.filterwarnings('ignore')

    set_random_seed(seed=get_attribute('seed'))

    batches_sets, num_users, num_items = CustomizedDataLoader(get_attribute('data_path')).load_data()

    model = CTTSP(num_users=num_users,
                  num_items=num_items,
                  embedding_dimension=get_attribute('embedding_dimension'),
                  dropout=get_attribute('dropout'),
                  continuous_time_probability_importance=get_attribute('continuous_time_probability_importance'),
                  user_perspective_importance=get_attribute('user_perspective_importance'))

    model = convert_to_gpu(model, device=get_attribute('device'))

    optimizer = create_optimizer(model=model, optimizer_name=get_attribute('optimizer'),
                                 learning_rate=get_attribute('learning_rate'),
                                 weight_decay=get_attribute('weight_decay'))

    scheduler = create_lr_scheduler(optimizer=optimizer, learning_rate=get_attribute('learning_rate'),
                                    t_max=get_attribute('scheduler_iter'))

    save_model_folder = f"../save_model_folder/{get_attribute('dataset_name')}/{get_attribute('model_name')}"
    shutil.rmtree(save_model_folder, ignore_errors=True)
    os.makedirs(save_model_folder, exist_ok=True)

    # set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    os.makedirs(f"../logs/{get_attribute('dataset_name')}/{get_attribute('model_name')}", exist_ok=True)
    # create file handler that logs debug and higher level messages
    fh = logging.FileHandler(
        f"../logs/{get_attribute('dataset_name')}/{get_attribute('model_name')}/{str(time.time())}.log")
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f'config -> {json.dumps(config, indent=4)}')
    logger.info(model)
    logger.info(
        f'Model #Params: {get_n_params(model) * 4} B, {get_n_params(model) * 4 / 1024} KB, {get_n_params(model) * 4 / 1024 / 1024} MB.')

    early_stopping = EarlyStopping(patience=get_attribute('patience'), save_model_folder=save_model_folder,
                                   save_model_name=get_attribute('model_name'), logger=logger)

    loss_func = nn.MultiLabelSoftMarginLoss(reduction="mean")

    """
    training and validating model
    """
    for epoch in range(get_attribute('epochs')):
        # reset memory at each epoch
        model.reset_memory()
        model.train()

        users_history_items = convert_to_gpu(torch.zeros(num_users, num_items), device=get_attribute('device'))

        tqdm_loader = tqdm(batches_sets, ncols=175)
        for batch, batch_sets in enumerate(tqdm_loader):
            train_loss, val_loss, test_loss = None, None, None
            """
            batch_sets-> tuple(
               batch_length, shape (batch_size, ), tensor
               batch_set_type, shape (batch_size, ), np.ndarray
               batch_user_id, shape (batch_size, ), tensor
               batch_items_id, shape (batch_size, max_set_size), tensor
               batch_set_truth, shape (batch_size, num_items), Tensor (calculate loss)
               )
            """
            batch_length, batch_set_type, batch_user_id, batch_items_id, batch_set_truth = batch_sets

            batch_user_id, batch_items_id, batch_set_truth = convert_to_gpu(batch_user_id, batch_items_id,
                                                                            batch_set_truth,
                                                                            device=get_attribute('device'))

            # split set_batch for training, validating and testing
            train_idx = torch.tensor(np.where(batch_set_type == 'train')[0])
            val_idx = torch.tensor(np.where(batch_set_type == 'validate')[0])
            test_idx = torch.tensor(np.where(batch_set_type == 'test')[0])

            # update users_history_items using train_idx, val_idx and test_idx, note that test_idx does not affect the user, since the ground truth is in 'next_set'
            # do not include set_type is null, since the corresponding items_id is in the test set, should not be utilized
            index = torch.cat([train_idx, val_idx, test_idx])
            if len(index) > 0:
                for set_user_id, set_items_id, set_length in \
                        zip(batch_user_id[index], batch_items_id[index], batch_length[index]):
                    set_items_id = set_items_id[:set_length]
                    users_history_items[set_user_id][set_items_id] += 1

            # feed a batch into model
            batch_set_predict, batch_user_memory, batch_items_memory = model(batch_length=batch_length,
                                                                             batch_user_id=batch_user_id,
                                                                             batch_items_id=batch_items_id,
                                                                             users_history_items=users_history_items)

            # if current batch contains at least one train set
            if len(train_idx) > 0:
                train_batch_set_predict, train_batch_set_truth = batch_set_predict[train_idx], batch_set_truth[train_idx]
                train_loss = loss_func(train_batch_set_predict, train_batch_set_truth)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                scheduler.step()

            # update memory
            if len(index) > 0:
                model.update_memory(batch_length=batch_length[index],
                                    batch_user_id=batch_user_id[index],
                                    batch_items_id=batch_items_id[index],
                                    batch_user_memory=batch_user_memory[index],
                                    batch_items_memory=batch_items_memory[index])

            # detach the gradiant of memory
            model.memory_detach()

            tqdm_loader.set_description(f'epoch: {epoch + 1}, batch: {batch + 1}, '
                                        + f'learning rate: {optimizer.param_groups[0]["lr"]}, '
                                        + f'train loss: {train_loss.item() if train_loss is not None else None}')

        logger.info(f'evaluating model at epoch {epoch + 1}...')
        eval_train_metric, eval_val_metric, eval_test_metric = evaluate(model, batches_sets, num_users, num_items,
                                                                        logger=logger)

        validate_ndcg = np.mean([eval_val_metric[key] for key in eval_val_metric if key.startswith(f"ndcg_")])
        early_stop = early_stopping.step([('ndcg', validate_ndcg, True)], model)

        if early_stop:
            break

    """
    testing model
    """
    # load best model and calculate final metrics
    early_stopping.load_checkpoint(model)

    logger.info('calculating final metrics...')

    eval_train_metric, eval_val_metric, eval_test_metric = evaluate(model, batches_sets, num_users, num_items, logger=logger)

    save_result_folder = f"../results/{get_attribute('dataset_name')}"
    os.makedirs(save_result_folder, exist_ok=True)
    save_result_path = f"{save_result_folder}/{get_attribute('model_name')}.json"

    with open(save_result_path, 'w') as file:
        scores_str = json.dumps({"train": eval_train_metric, "validate": eval_val_metric, "test": eval_test_metric},
                                indent=4)
        file.write(scores_str)
        logger.info(f'result saves at {save_result_path} successfully.')

    sys.exit()
