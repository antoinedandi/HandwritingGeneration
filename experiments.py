import argparse
import torch
import model.models as module_arch
import data_loader.data_loader as module_data
from parse_config import ConfigParser
from utils import plot_stroke


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
