import torch
import torch.nn as nn


class GatedUpdater(nn.Module):
    """
    Gated Updater
    """

    def __init__(self, message_dimension: int, embedding_dimension: int):
        """
        :param message_dimension: number of message dimension
        :param embedding_dimension: number of embedding dimension
        """
        super(GatedUpdater, self).__init__()

        self.message_dimension = message_dimension
        self.embedding_dimension = embedding_dimension

        self.message_projection = nn.Linear(message_dimension, embedding_dimension, bias=True)
        self.hidden_state_projection = nn.Linear(embedding_dimension, embedding_dimension, bias=True)

        self.message_query = nn.Parameter(torch.randn([embedding_dimension, embedding_dimension]), requires_grad=True)
        self.hidden_state_query = nn.Parameter(torch.randn([embedding_dimension, embedding_dimension]), requires_grad=True)

    def forward(self, batch_message: torch.Tensor, batch_hidden_state: torch.Tensor):
        """
        :param batch_message: shape (batch_size, message_dimension)
        :param batch_hidden_state: shape (batch_size, embedding_dimension)
        :return:
            batch_updated_state: shape (batch_size, embedding_dimension)
        """
        # batch_message, shape (batch_size, embedding_dimension)
        batch_message = self.message_projection(batch_message)
        # batch_hidden_state, shape (batch_size, embedding_dimension)
        batch_hidden_state = self.hidden_state_projection(batch_hidden_state)

        # message_scores, shape (batch_size, embedding_dimension)
        message_scores = torch.matmul(batch_message, self.message_query)
        # hidden_state_scores, shape (batch_size, embedding_dimension)
        hidden_state_scores = torch.matmul(batch_hidden_state, self.hidden_state_query)
        # attention_scores, shape (batch_size, embedding_dimension, 2)
        attention_scores = torch.softmax(torch.stack([message_scores, hidden_state_scores], dim=2), dim=2)
        # batch_updated_state, shape (batch_size, embedding_dimension)
        batch_updated_state = torch.sum(attention_scores * torch.stack([batch_message, batch_hidden_state], dim=2), dim=2)

        # batch_updated_state, shape (batch_size, embedding_dimension)
        batch_updated_state = torch.tanh(batch_updated_state)

        return batch_updated_state


class MemoryUpdater(nn.Module):
    """
    Memory Updater
    """

    def __init__(self, message_dimension: int, embedding_dimension: int):
        """
        :param message_dimension: number of message dimension
        :param embedding_dimension: number of embedding dimension
        """
        super(MemoryUpdater, self).__init__()

        self.message_dimension = message_dimension
        self.embedding_dimension = embedding_dimension

        self.user_memory_module = GatedUpdater(message_dimension=message_dimension, embedding_dimension=embedding_dimension)
        self.item_memory_module = GatedUpdater(message_dimension=message_dimension, embedding_dimension=embedding_dimension)

    def forward(self, users_memory: torch.Tensor, items_memory: torch.Tensor, batch_user_id: torch.tensor,
                batch_items_id: torch.tensor, batch_user_message: torch.Tensor, batch_items_message: torch.Tensor):
        """
        :param users_memory: nn.Parameter, no grad (num_users, embedding_dimension), Tensor, previous users memory
        :param items_memory: nn.Parameter, no grad (num_items, embedding_dimension), Tensor, previous items memory
        :param batch_user_id: shape (batch_size, ), tensor
        :param batch_items_id: shape (batch_size, max_set_size), tensor
        :param batch_user_message: shape (batch_size, message_dimension)
        :param batch_items_message: shape (batch_size, max_set_size, message_dimension)
        :return:
            batch_user_memory: shape (batch_size, embedding_dimension), current users memory
            batch_items_memory: shape (batch_size, max_set_size, embedding_dimension), current items memory
        """

        # batch_user_memory, shape (batch_size, embedding_dimension)
        batch_user_memory = users_memory[batch_user_id]
        # batch_items_memory, shape (batch_size, max_set_size, embedding_dimension)
        batch_items_memory = items_memory[batch_items_id]

        # batch_user_memory, shape (batch_size, embedding_dimension)
        batch_user_memory = self.user_memory_module(batch_user_message, batch_user_memory)

        # input format of GRU is (batch, input_size), so we reshape the first two dimension and feed into GRU
        batch_items_message = batch_items_message.reshape(batch_items_id.shape[0] * batch_items_id.shape[1], self.message_dimension)
        batch_items_memory = batch_items_memory.reshape(batch_items_id.shape[0] * batch_items_id.shape[1], self.embedding_dimension)
        # batch_items_memory, shape (batch_size * max_set_size, embedding_dimension)
        batch_items_memory = self.item_memory_module(batch_items_message, batch_items_memory)
        # batch_items_memory, shape (batch_size, max_set_size, embedding_dimension)
        batch_items_memory = batch_items_memory.reshape(batch_items_id.shape[0], batch_items_id.shape[1], self.embedding_dimension)

        return batch_user_memory, batch_items_memory
