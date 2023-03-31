import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PersonalizedProbabilityCalculator(nn.Module):
    """
    Personalized Probability Calculator
    """

    def __init__(self, num_items: int, embedding_dimension: int, dropout: float, user_perspective_importance: float):
        """
        :param num_items: int, number of items
        :param embedding_dimension: int, number of embedding dimension
        :param dropout: float, dropout rate
        :param user_perspective_importance: float, user perspective importance
        """
        super(PersonalizedProbabilityCalculator, self).__init__()

        self.num_items = num_items
        self.embedding_dimension = embedding_dimension
        self.dropout = dropout
        self.user_perspective_importance = user_perspective_importance

        self.dropout = nn.Dropout(p=dropout)

        self.leaky_relu_func = nn.LeakyReLU(negative_slope=0.2)

        self.fc_projection = nn.Linear(embedding_dimension, embedding_dimension, bias=True)

        # user embedding
        self.user_embedding = nn.Parameter(torch.randn([1, embedding_dimension]), requires_grad=True)
        # # items embedding
        self.items_embedding = nn.Parameter(torch.randn([num_items, embedding_dimension]), requires_grad=True)

    def pad_users_history_items(self, batch_user_id: torch.tensor, users_history_items: torch.Tensor):
        """
        pad historical interacted items of users in the batch to the maximal length
        :param batch_user_id: shape (batch_size, ), tensor
        :param users_history_items: history items of each user, shape (num_users, num_items), Tensor
        :return:    batch_user_history_items_length, tensor
                    batch_user_history_items_id, tensor, each list contains the item idx
        """
        # list of user historical items ids in the batch
        batch_user_history_items_id = []
        for user_history_items in users_history_items[batch_user_id]:
            user_history_items_id = user_history_items.nonzero().squeeze(dim=1)
            user_history_items_id_list = []
            for user_history_item_id in user_history_items_id:
                user_history_items_id_list += [int(user_history_item_id)] * int(user_history_items[user_history_item_id])
            batch_user_history_items_id.append(user_history_items_id_list)

        # list, shape (batch_size, )
        batch_user_history_items_length = [len(user_history_items_id) for user_history_items_id in batch_user_history_items_id]
        # get maximal number of historical items in the batched users (for padding)
        max_history_item_num = max(batch_user_history_items_length)
        for index, user_history_items_id in enumerate(batch_user_history_items_id):
            batch_user_history_items_id[index] = user_history_items_id + [-1] * (max_history_item_num - batch_user_history_items_length[index])

        return torch.tensor(batch_user_history_items_length), torch.tensor(batch_user_history_items_id)

    def forward(self, batch_user_id: torch.tensor, users_history_items: torch.Tensor):
        """
        first pad historical interacted items of users in the batch to the maximal length
        :param batch_user_id: shape (batch_size, ), tensor
        :param users_history_items: history items of each user, shape (num_users, num_items), Tensor
        :return:
            batch_items_static_scores: shape (batch_size, num_items)
        """

        # batch_user_history_items_length, list, shape (batch_size, )
        # batch_user_history_items_id, list, shape (batch_size, max_history_item_num)
        batch_user_history_items_length, batch_user_history_items_id = self.pad_users_history_items(batch_user_id=batch_user_id, users_history_items=users_history_items)

        # shape (batch_size, max_history_item_num, embedding_dimension)
        batch_user_history_items_embedding = self.dropout(self.items_embedding[batch_user_history_items_id])

        # # shape (num_items, embedding_dimension)
        items_embedding = self.dropout(self.items_embedding)

        # shape (num_items, embedding_dimension)
        item_queries_embedding = items_embedding
        # shape (1, num_items, embedding_dimension),  shape (1, embedding_dimension) repeat num_items
        user_queries_embedding = self.dropout(self.user_embedding.unsqueeze(dim=1).repeat(1, self.num_items, 1))

        # (num_items, embedding_dimension) einsum (batch_size, max_history_item_num, embedding_dimension) -> (batch_size, num_items, max_history_item_num)
        item_attention = torch.einsum('if,bnf->bin', item_queries_embedding, batch_user_history_items_embedding)
        # (1, num_items, embedding_dimension) einsum (batch_size, max_history_item_num, embedding_dimension) -> (batch_size, 1, num_items, max_history_item_num)
        user_attention = torch.einsum('qif,bnf->bqin', user_queries_embedding, batch_user_history_items_embedding)

        # mask based on batch_length, shape (batch_size, num_items, max_history_item_num)
        item_attention_mask = torch.zeros_like(item_attention)
        for node_idx, user_history_items_length in enumerate(batch_user_history_items_length):
            item_attention_mask[node_idx][:, user_history_items_length:] = - np.inf

        item_attention = item_attention + item_attention_mask
        item_attention = self.leaky_relu_func(item_attention)

        # shape (batch_size, num_items, max_history_item_num)
        item_attention_scores = F.softmax(item_attention, dim=-1)

        # mask based on batch_length, shape (batch_size, 1, num_items, max_history_item_num)
        user_attention_mask = torch.zeros_like(user_attention)
        for node_idx, user_history_items_length in enumerate(batch_user_history_items_length):
            user_attention_mask[node_idx][:, :, user_history_items_length:] = - np.inf

        user_attention = user_attention + user_attention_mask
        user_attention = self.leaky_relu_func(user_attention)

        # shape (batch_size, 1, num_items, max_history_item_num)  squeeze to (batch_size, num_items, max_history_item_num)
        user_attention_scores = F.softmax(user_attention, dim=-1).squeeze(dim=1)

        # shape (batch_size, num_items, max_history_item_num)
        attention_scores = (1 - self.user_perspective_importance) * item_attention_scores + self.user_perspective_importance * user_attention_scores

        # (batch_size, num_items, max_history_item_num) bmm (batch_size, max_history_item_num, embedding_dimension) -> (batch_size, num_items, embedding_dimension)
        batch_items_personalized_embedding = torch.bmm(attention_scores, batch_user_history_items_embedding)
        batch_items_personalized_embedding = self.dropout(batch_items_personalized_embedding)

        # shape (batch_size, num_items, embedding_dimension)
        batch_items_personalized_embedding = self.fc_projection(batch_items_personalized_embedding)

        # batch_items_personalized_probability, shape (batch_size, num_items)
        # sum((batch_size, num_items, embedding_dimension) * (num_items, embedding_dimension), dim=-1) -> (batch_size, num_items)
        batch_items_personalized_probability = (batch_items_personalized_embedding * items_embedding).sum(dim=-1)

        return batch_items_personalized_probability
