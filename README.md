# Exploring Euphemism Detection in Few-Shot and Zero-Shot Settings

This is the official code for the paper "Exploring Euphemism Detection in Few-Shot and Zero-Shot Settings" accepted at EMNLP FigLang Workshop 2022 under the Euphemism Detection Shared Task. 

## Dataset
We use the dataset from the Euphemism Detection Shared Task, which is available [here](https://codalab.lisn.upsaclay.fr/competitions/5726#results). For creating the few-shot and zero-shot datasets, please refer to the details in the paper. We provide sample splits in `few` and `zero`. 

## Training and Evaluation
To train, 
```bash
python train.py --availability zero --category employment
```
This will create a saved model for every checkpoint at the specified output path. See `train.py` for specific arguments.

To evaluate, 
```bash
python evaluate.py --availability zero --category employment --model_path (checkpoint-path)/pytorch_model.bin
```