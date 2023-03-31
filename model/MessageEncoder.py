import torch
import torch.nn as nn
import numpy as np

class MessageEncoder(nn.Module):
    """
    Message Encoder
    """

    def __init__(self, embedding_dimension: int):
        """
        :param embedding_dimension: number of embedding dimension
        """
        super(MessageEncoder, self).__init__()

        self.embedding_dimension = embedding_dimension

        self.user_Q, self.user_K, self.user_V = nn.Linear(embedding_dimension, embedding_dimension), \
                                                nn.Linear(embedding_dimension, embedding_dimension), \
                                                nn.Linear(embedding_dimension, embedding_dimension)
        self.item_Q, self.item_K, self.item_V = nn.Linear(embedding_dimension, embedding_dimension), \
                                                nn.Linear(embedding_dimension, embedding_dimension), \
                                                nn.Linear(embedding_dimension, embedding_dimension)

    def forward(self, users_memory: torch.Tensor, items_memory: torch.Tensor, batch_length: torch.tensor,
                batch_user_id: torch.tensor, batch_items_id: torch.tensor):
        """
        :param users_memory: nn.Parameter, no grad (num_users, embedding_dimension), Tensor, previous users memory
        :param items_memory: nn.Parameter, no grad (num_items, embedding_dimension), Tensor, previous items memory
        :param batch_length: shape (batch_size, ), tensor
        :param batch_user_id: shape (batch_size, ), tensor
        :param batch_items_id: shape (batch_size, max_set_size), tensor
        :return:
            batch_user_message: shape (batch_size, embedding_dimension)
            batch_items_message: shape (batch_size, max_set_size, embedding_dimension)
        """

        # batch_user_memory, shape (batch_size, embedding_dimension)
        batch_user_memory = users_memory[batch_user_id]
        # batch_items_memory, shape (batch_size, max_set_size, embedding_dimension)
        batch_items_memory = items_memory[batch_items_id]

        # user message calculation

        # user_attention_scores, shape (batch_size, max_set_size)
        user_attention_scores = torch.matmul(self.user_Q(batch_user_memory).unsqueeze(dim=1),
                                             self.user_K(batch_items_memory).transpose(1, 2)).squeeze(dim=1) / np.sqrt(self.embedding_dimension)

        # user_attention_mask, shape (batch_size, max_set_size)
        user_attention_mask = torch.zeros_like(user_attention_scores)
        for index, set_length in enumerate(batch_length):
            user_attention_mask[index, set_length:] = - np.inf

        # user_attention_scores, shape (batch_size, max_set_size)
        user_attention_scores = user_attention_scores + user_attention_mask
        user_attention_scores = torch.softmax(user_attention_scores, dim=1)
        # batch_user_message, shape (batch_size, embedding_dimension)
        batch_user_message = torch.bmm(user_attention_scores.unsqueeze(dim=1), self.user_V(batch_items_memory)).squeeze(dim=1)

        # item message calculation
        # item_attention_scores, shape (batch_size, max_set_size, max_set_size)
        item_attention_scores = torch.matmul(self.item_Q(batch_items_memory),
                                             self.item_K(batch_items_memory).transpose(1, 2)) / np.sqrt(self.embedding_dimension)

        # item_attention_mask, shape (batch_size, max_set_size, max_set_size)
        item_attention_mask = torch.zeros_like(item_attention_scores)
        for index, set_length in enumerate(batch_length):
            item_attention_mask[index, :, set_length:] = - np.inf

        # item_attention_scores, shape (batch_size, max_set_size, max_set_size)
        item_attention_scores = item_attention_scores + item_attention_mask
        item_attention_scores = torch.softmax(item_attention_scores, dim=2)
        # batch_items_message, shape (batch_size, max_set_size, embedding_dimension)
        batch_items_message = torch.bmm(item_attention_scores, self.item_V(batch_items_memory))

        # batch_user_message, shape (batch_size, 2 * embedding_dimension)
        batch_user_message = torch.cat([batch_user_memory, batch_user_message], dim=1)
        # batch_items_message, shape (batch_size, max_set_size, 2 * embedding_dimension)
        batch_items_message = torch.cat([batch_user_memory.unsqueeze(dim=1).repeat(1, batch_items_message.shape[1], 1), batch_items_message], dim=2)

        return batch_user_message, batch_items_message
