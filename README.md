# Continuous-Time User Preference Modelling for Temporal Sets Prediction

The description of "Continuous-Time User Preference Modelling for Temporal Sets Prediction" 
is [available here](). 

### Original data:
The original data could be downloaded from [here](https://drive.google.com/file/d/1KCUsQ57DPfn7gldjqWBa4GvIZNi_UhJy/view?usp=sharing). 
You can download the data and then put the data files in the ```./original_data``` folder.


### To run the code:
  
  1. run ```./preprocess_data/preprocess_data_{dataset_name}.py``` to preprocess the original data, 
     where ```dataset_name``` can be JingDong, DC, TaFeng and TaoBao. 
     We also provide the preprocessed datasets at [here](https://drive.google.com/file/d/18wKh5QsCAbQakkMuZ5u3gBTW-S2wND8E/view?usp=sharing), 
     which should be put in the ```./dataset``` folder.
     
  2. run ```./train/train_CTTSP.py``` to train the model on different datasets using the configuration in ```./utils/config.json```.

  3. run ```./evaluate/evaluate_CTTSP.py``` to evaluate the model. 
     Please make sure the ```config``` in ```evaluate_CTTSP.py``` keeps identical to that in the model training process.


## Environments:
- [PyTorch 1.8.1](https://pytorch.org/)
- [tqdm](https://github.com/tqdm/tqdm)
- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)


## Hyperparameter settings:
Hyperparameters can be found in ```./utils/config.json``` file, and you can adjust them when training the model on different datasets.

| Hyperparameters  | JingDong  | DC  | TaFeng  | TaoBao |
| -------    | ------- | -------  | -------  | -------  |
| learning rate  | 0.001  | 0.001  | 0.001  |  0.001   |
| dropout rate | 0.2  | 0.2  | 0.15  |  0.05   |
| embedding dimension  | 64  | 64  | 64  |  32   |
| user perspective importance  | 0.9  | 0.5  | 0.05 |  0.9  |
| continuous-time probability importance  | 0.9  | 0.0  | 0.7  |  0.7   |
