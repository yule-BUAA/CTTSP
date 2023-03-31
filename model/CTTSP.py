import torch
import torch.nn as nn
from model.MessageEncoder import MessageEncoder
from model.MemoryUpdater import MemoryUpdater
from model.PersonalizedProbabilityCalculator import PersonalizedProbabilityCalculator
from model.PredictionLayer import PredictionLayer


class CTTSP(nn.Module):
    """
    CTTSP Model
    """

    def __init__(self, num_users: int, num_items: int, embedding_dimension: int, dropout: float,
                 continuous_time_probability_importance: float, user_perspective_importance: float):
        """
        :param num_users: int, number of users
        :param num_items: int, number of items
        :param embedding_dimension: int, number of embedding dimension
        :param dropout: float, dropout rate
        :param continuous_time_probability_importance: float, continuous_time probability importance
        :param user_perspective_importance: float, user perspective importance
        """
        super(CTTSP, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dimension = embedding_dimension
        self.dropout = dropout
        self.continuous_time_probability_importance = continuous_time_probability_importance
        self.user_perspective_importance = user_perspective_importance

        self.personalized_probability_calculator = PersonalizedProbabilityCalculator(num_items=num_items,
                                                                                     embedding_dimension=embedding_dimension,
                                                                                     dropout=dropout,
                                                                                     user_perspective_importance=user_perspective_importance)

        self.message_aggregator = MessageEncoder(embedding_dimension=embedding_dimension)

        self.memory_updater = MemoryUpdater(message_dimension=2 * embedding_dimension,
                                            embedding_dimension=embedding_dimension)

        self.prediction_layer = PredictionLayer(num_items=num_items,
                                                embedding_dimension=embedding_dimension,
                                                dropout=dropout,
                                                continuous_time_probability_importance=continuous_time_probability_importance)

        # user memory and item memory, no grad and should be reset when each epoch begins
        self.users_memory = nn.Parameter(torch.zeros([num_users, embedding_dimension]), requires_grad=False)
        self.items_memory = nn.Parameter(torch.zeros([num_items, embedding_dimension]), requires_grad=False)

    def reset_memory(self):
        """
        reset memory to zero at the beginning of each epoch
        """
        self.users_memory.data = self.users_memory.new_zeros(self.users_memory.shape)
        self.items_memory.data = self.items_memory.new_zeros(self.items_memory.shape)

    def update_memory(self, batch_length: torch.tensor, batch_user_id: torch.tensor, batch_items_id: torch.tensor,
                      batch_user_memory: torch.Tensor, batch_items_memory: torch.Tensor):
        """
        :param batch_length, shape (batch_size, ), tensor
        :param batch_user_id, shape (batch_size, ), tensor
        :param batch_items_id, shape (batch_size, ), tensor
        :param batch_user_memory: shape (batch_size, embedding_dimension), Tensor, current users memory
        :param batch_items_memory: shape (batch_size, max_set_size, embedding_dimension), Tensor, current users memory
        """
        for set_user_id, set_items_id, set_length, set_user_memory, set_items_memory in \
                zip(batch_user_id, batch_items_id, batch_length, batch_user_memory, batch_items_memory):
            # set_items_id, shape (num_actual_items, )
            set_items_id = set_items_id[:set_length]
            # set_items_memory, shape (num_actual_items, embedding_dimension)
            set_items_memory = set_items_memory[:set_length]
            # set memory data
            self.users_memory.data[set_user_id] = set_user_memory
            self.items_memory.data[set_items_id] = set_items_memory

    def memory_detach(self):
        """
        detach memory, removing its gradient, to avoid memory-backward in the next epoch
        """
        self.users_memory.detach_()
        self.items_memory.detach_()

    def forward(self, batch_length: torch.tensor, batch_user_id: torch.tensor, batch_items_id: torch.tensor,
                users_history_items: torch.Tensor):
        """
        :param batch_length: shape (batch_size, ), tensor
        :param batch_user_id: shape (batch_size, ), tensor
        :param batch_items_id: shape (batch_size, max_set_size), tensor
        :param users_history_items: shape (num_users, num_items), Tensor item occurrence of each user,
        :return:
            batch_set_predict: shape (batch_size, n_items)
            batch_user_memory: shape (batch_size, embedding_dimension)
            batch_items_memory: shape (batch_size, max_set_size, embedding_dimension)
        """

        # personalized probability from dual perspectives
        # batch_items_personalized_probability shape -> (batch_size, num_items)
        batch_items_personalized_probability = self.personalized_probability_calculator(batch_user_id=batch_user_id,
                                                                                        users_history_items=users_history_items)

        # message aggregate for users and items
        # batch_user_message shape -> (batch_size, embedding_dimension)
        # batch_items_message shape -> (batch_size, max_set_size, embedding_dimension)
        batch_user_message, batch_items_message = self.message_aggregator(users_memory=self.users_memory,
                                                                          items_memory=self.items_memory,
                                                                          batch_length=batch_length,
                                                                          batch_user_id=batch_user_id,
                                                                          batch_items_id=batch_items_id)

        # update memory for users and items
        # batch_user_memory shape -> (batch_size, embedding_dimension)
        # batch_items_memory shape -> (batch_size, max_set_size, embedding_dimension)
        batch_user_memory, batch_items_memory = self.memory_updater(users_memory=self.users_memory,
                                                                    items_memory=self.items_memory,
                                                                    batch_user_id=batch_user_id,
                                                                    batch_items_id=batch_items_id,
                                                                    batch_user_message=batch_user_message,
                                                                    batch_items_message=batch_items_message)

        # prediction layer
        # batch_set_predict shape -> (batch_size, num_items)
        batch_set_predict = self.prediction_layer(items_memory=self.items_memory,
                                                  batch_length=batch_length,
                                                  batch_user_id=batch_user_id,
                                                  batch_items_id=batch_items_id,
                                                  users_history_items=users_history_items,
                                                  batch_user_memory=batch_user_memory,
                                                  batch_items_memory=batch_items_memory,
                                                  batch_items_personalized_probability=batch_items_personalized_probability)

        return batch_set_predict, batch_user_memory, batch_items_memory
