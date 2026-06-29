import os
from Utils.arguments import load_json_config
from Utils.misc import get_save_path, seed_everything, save_args
import logging
import sys
from Utils.misc import initialize_logger
import socket
from Utils.trainer import TrainUtilsPaired


logger = logging.getLogger()


if __name__ == '__main__':
    config = load_json_config('/path/to/config.json')

    save_path = get_save_path(config)
    initialize_logger(save_path, 'log')
    seed_everything()
    save_args(config, save_path)

    logger.info(f'Training on [{socket.gethostname()}]')
    logger.info(f'Result output: {save_path}')

    trainer = TrainUtilsPaired(config, save_path)
    trainer.train()