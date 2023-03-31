import torch
import torch.nn as nn


class PredictionLayer(nn.Module):
    """
    Prediction Layer
    """

    def __init__(self, num_items: int, embedding_dimension: int, dropout: float, continuous_time_probability_importance: float):
        """
        :param num_items: number of items
        :param embedding_dimension: number of embedding dimension
        :param dropout: dropout rate
        :param continuous_time_probability_importance: float, importance of continuous_time probability
        """
        super(PredictionLayer, self).__init__()

        self.num_items = num_items
        self.embedding_dimension = embedding_dimension

        self.continuous_time_probability_importance = continuous_time_probability_importance

        self.dropout = nn.Dropout(dropout)

        self.item_memory_fc_projection = nn.Linear(embedding_dimension, embedding_dimension)

    def forward(self, items_memory: torch.Tensor, batch_length: torch.tensor, batch_user_id: torch.tensor,
                batch_items_id: torch.tensor, users_history_items: torch.Tensor, batch_user_memory: torch.Tensor,
                batch_items_memory: torch.Tensor, batch_items_personalized_probability: torch.Tensor):
        """
        :param items_memory: nn.Parameter, no grad (num_items, embedding_dimension), Tensor, previous items memory
        :param batch_length: shape (batch_size, ), tensor
        :param batch_user_id: shape (batch_size, ), tensor
        :param batch_items_id: shape (batch_size, max_set_size), tensor
        :param users_history_items: history items of each user, shape (num_users, num_items), Tensor
        :param batch_user_memory: shape (batch_size, embedding_dimension), Tensor, current users memory
        :param batch_items_memory: shape (batch_size, max_set_size, embedding_dimension), Tensor, current items memory
        :param batch_items_personalized_probability: items personalized probability, shape (batch_size, num_items), Tensor
        :return:
            batch_set_predict: shape (batch_size, num_items)
        """

        batch_set_predict = []

        # shape (batch_size, embedding_dimension)
        batch_user_memory = self.dropout(batch_user_memory)
        # shape (batch_size, max_set_size, embedding_dimension)
        batch_items_memory = self.dropout(batch_items_memory)

        # batch_user_history_items, shape (batch_size, num_items)
        batch_user_history_items = users_history_items[batch_user_id]
        # iterate over each set in the batch
        for set_user_memory, set_items_memory, set_items_id, set_length, set_user_history_items, set_items_personalized_probability in \
                zip(batch_user_memory, batch_items_memory, batch_items_id, batch_length, batch_user_history_items, batch_items_personalized_probability):
            # set_items_id, shape (num_actual_items, )
            set_items_id = set_items_id[:set_length]
            # set_items_memory, shape (num_actual_items, embedding_dimension)
            set_items_memory = set_items_memory[:set_length]

            # exclude historical items that appear in the current set
            set_user_interacted_items = torch.nonzero(set_user_history_items).squeeze(dim=1)
            # history_items_id, shape (num_history_items, )
            history_items_id = torch.tensor(list(set(set_user_interacted_items.tolist()) - set(set_items_id.tolist())))

            if len(history_items_id) > 0:
                # history_items_id, shape (num_history_items, embedding_dimension)
                batch_history_items_memory = self.item_memory_fc_projection(items_memory[history_items_id])

            # set_items_continuous_time_probability, shape (num_actual_items, )
            set_items_continuous_time_probability = torch.sum(set_user_memory * set_items_memory, dim=1)
            if len(history_items_id) > 0:
                # set_history_items_continuous_time_probability, shape (num_history_items, )
                set_history_items_continuous_time_probability = torch.sum(set_user_memory * batch_history_items_memory, dim=1)

            # one-hot occurrence of items that have interacted with users, shape (num_items, )
            beta = torch.zeros(self.num_items).to(users_history_items.device)
            beta[set_user_interacted_items] = 1
            # set_predict, shape (num_items, )
            set_predict = (1 - beta * self.continuous_time_probability_importance) * set_items_personalized_probability
            set_predict[set_items_id] = set_predict[set_items_id] + self.continuous_time_probability_importance * set_items_continuous_time_probability
            if len(history_items_id) > 0:
                set_predict[history_items_id] = set_predict[history_items_id] + self.continuous_time_probability_importance * set_history_items_continuous_time_probability

            batch_set_predict.append(set_predict)

        # batch_set_predict, shape (batch_size, num_items)
        batch_set_predict = torch.stack(batch_set_predict, dim=0)

        return batch_set_predict
