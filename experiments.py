import argparse
import shutil
import torch
import model.models as module_arch
import data_loader.data_loader as module_data
from data_loader.data_loader import HandWritingDataset
from parse_config import ConfigParser
from utils import plot_stroke, read_json


#######################################################################################################################
# Functions used in the "results" notebook
#######################################################################################################################

def generate_unconditionally(config_fn='../saved/models/UnconditionalHandwriting/1114_083533/config.json',
                             resume='../saved/models/UnconditionalHandwriting/1114_083533/model_best.pth'):
    # Parsing the config
    config = read_json(config_fn)
    config = ConfigParser(config, resume)
    # set up device and data_loader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = HandWritingDataset('../data')
    # build model architecture and load weights
    model = config.init_obj('arch', module_arch, char2idx=dataset.char2idx, device=device)
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    # prepare model for inference
    model = model.to(device)
    model.eval()
    # Generation of unconditional handwriting
    with torch.no_grad():
        sampled_stroke = model.generate_unconditional_sample()
    # Clean notebooks folder
    shutil.rmtree('saved')
    return sampled_stroke


def generate_conditionally(text, config_fn='../saved/models/ConditionalHandwriting/1114_101215/config.json',
                           resume='../saved/models/ConditionalHandwriting/1114_101215/model_best.pth'):
    # Parsing the config
    config = read_json(config_fn)
    config = ConfigParser(config, resume)
    # set up device and data_loader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = HandWritingDataset('../data')
    # build model architecture and load weights
    model = config.init_obj('arch', module_arch, char2idx=dataset.char2idx, device=device)
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    # prepare model for inference
    model = model.to(device)
    model.eval()
    # Generation of unconditional handwriting
    with torch.no_grad():
        sampled_stroke = model.generate_conditional_sample(text)
    # Clean notebooks folder
    shutil.rmtree('saved')
    return sampled_stroke


def recognize_stroke(stroke, config_fn='../saved/models/Seq2SeqHandwritingRecognition/1114_091246/config.json',
                     resume='../saved/models/Seq2SeqHandwritingRecognition/1114_091246/model_best.pth'):
    # Parsing the config
    config = read_json(config_fn)
    config = ConfigParser(config, resume)
    # set up device and data_loader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = HandWritingDataset('../data')
    # build model architecture and load weights
    model = config.init_obj('arch', module_arch, char2idx=dataset.char2idx, device=device)
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    # prepare model for inference
    model = model.to(device)
    model.eval()
    # Generation of unconditional handwriting
    with torch.no_grad():
        predicted_seq = model.recognize_sample(stroke)
        predicted_text = dataset.tensor2sentence(torch.tensor(predicted_seq))
    # Clean notebooks folder
    shutil.rmtree('saved')
    return predicted_text

#######################################################################################################################


def main(config):
    logger = config.get_logger('experiments')

    # setup the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)

    # build model architecture
    model = config.init_obj('arch', module_arch, char2idx=data_loader.dataset.char2idx, device=device)
    logger.info(model)

    # Loading the weights of the model
    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    with torch.no_grad():

        if str(model).startswith('Unconditional'):
            sampled_stroke = model.generate_unconditional_sample()
            plot_stroke(sampled_stroke)

        elif str(model).startswith('Conditional'):
            sampled_stroke = model.generate_conditional_sample('hello world')
            plot_stroke(sampled_stroke)

        elif str(model).startswith('Seq2Seq'):
            sent, stroke = data_loader.dataset[21]
            predicted_seq = model.recognize_sample(stroke)
            print('real text:      ', data_loader.dataset.tensor2sentence(sent))
            print('predicted text: ', data_loader.dataset.tensor2sentence(torch.tensor(predicted_seq)))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='handwriting model')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')

    config = ConfigParser.from_args(args)
    main(config)
