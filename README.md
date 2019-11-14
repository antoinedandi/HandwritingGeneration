# Handwriting Generation

This repo contains a Pytorch implementation of the paper
[Generating Sequences With Recurrent Neural Networks](https://arxiv.org/pdf/1308.0850.pdf) by Alex Graves

## Requirements
* Python >= 3.5 (3.6 recommended)
* PyTorch == 1.2
* tqdm
* tensorboard >= 1.14

## Models
* **Unconditional Handwriting Generation**

![Unconditional Handwriting](saved_figures/uncond_handwriting.png)

* **Conditional Handwriting Generation**

![Conditional Handwriting](saved_figures/cond_handwriting.png)

* **Handwriting Recognition using a Seq2Seq with attention model**


## Usage

### To train a model:

##### Unconditional Handwriting Generation
Modify the `config_unconditional.json` as needed and run:
 ```
 python train.py -c config_unconditional.json
 ```

##### Conditional Handwriting Generation
Modify the `config_conditional.json` as needed and run:
 ```
 python train.py -c config_conditional.json
 ```

##### Handwriting Recognition
Modify the `config_conditional.json` as needed and run:
  ```
  python train.py -c config_recognition.json
  ```

### To evaluate a model:
You can test trained model by running `experiments.py` passing path to the trained checkpoint by `--resume` argument.
Example:
  ```
  python experiments.py -r path/to/trained_checkpoint
  ```

### Resuming from checkpoints
You can resume from a previously saved checkpoint by:
  ```
  python train.py --resume path/to/checkpoint
  ```

## Folder Structure
  ```
  HandwritingGeneration/
  │
  ├── README.md
  │
  ├── train.py - main script to start training
  ├── experiments.py - evaluation of trained model
  │
  ├── config_unconditional.json  - holds configuration for training an unconditional handwriting model
  ├── config_conditional.json    - holds configuration for training a conditional handwriting model
  ├── config_recognition.json    - holds configuration for training a handwriting recognition model
  ├── parse_config.py - class to handle config file and cli options
  │
  ├── base/ - abstract base classes
  │   ├── base_data_loader.py
  │   ├── base_model.py
  │   └── base_trainer.py
  │
  ├── data_loader/ 
  │   └── data_loaders.py  - Class to handle the loading of the data
  │
  ├── data/ - default directory for storing input data
  │
  ├── model/ - models, losses, and metrics
  │   ├── custom layers/
  │   │   ├── lstm_with_gaussian_attention.py   - Used in conditional handwriting generation
  │   │   └── seq2seq_modules.py                - Used in handwriting recognition
  │   ├── models.py
  │   ├── metric.py
  │   └── loss.py
  │
  ├── notebooks/
  │   ├── examples.ipynb
  │   └── results.ipynb
  │
  ├── saved/
  │   ├── models/ - trained models are saved here
  │   └── log/ - logdir for tensorboard and logging output
  │
  ├── saved_figures/ - saved images of generated handwriting
  │
  ├── trainer/ - trainers
  │   └── trainer.py
  │
  ├── logger/ - module for tensorboard visualization and logging
  │   ├── visualization.py
  │   ├── logger.py
  │   └── logger_config.json
  │  
  └── utils/ - utility functions
      └── util.py
      
  ```