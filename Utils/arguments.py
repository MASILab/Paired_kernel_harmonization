import argparse
import yaml
import os
import logging
import json


logger = logging.getLogger()


def load_json_config(config_file):
    f = open(config_file)
    config = json.load(f)

    # Add default configurations here.

    return config
