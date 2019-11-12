import argparse
import torch
import data_loader.data_loader as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.models as module_arch
from parse_config import ConfigParser


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=12,
        shuffle=False,
        validation_split=0.0,
        num_workers=0
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_idx, (_, _, strokes, strokes_mask) in enumerate(data_loader):
            strokes, strokes_mask = strokes.to(device), strokes_mask.to(device)
            batch_size = strokes.size(0)

            # Compute the loss
            model.hidden = model.init_hidden(batch_size)
            output_network = model(strokes)
            gaussian_params = model.compute_gaussian_parameters(output_network)
            loss = criterion(gaussian_params, strokes, strokes_mask)
            print('loss :', loss)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
