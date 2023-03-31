import os
import json
import torch

abs_path = os.path.join(os.path.dirname(__file__), "config.json")
with open(abs_path) as file:
    config = json.load(file)


def get_attribute(attribute_name: str, default_value=None):
    """
    get configs
    :param attribute_name: config key
    :param default_value: None
    :return:
    """
    try:
        return config[attribute_name]
    except KeyError:
        return default_value

# dataset specified settings
config.update(config[f"{get_attribute('dataset_name')}"])
config.pop('DC')
config.pop('JingDong')
config.pop('TaFeng')
config.pop('TaoBao')

config['data_path'] = f"{os.path.dirname(os.path.dirname(__file__))}/dataset/{get_attribute('dataset_name')}/{get_attribute('dataset_name')}.json"
config['device'] = f'cuda:{get_attribute("cuda")}' if torch.cuda.is_available() and get_attribute("cuda") >= 0 else 'cpu'
