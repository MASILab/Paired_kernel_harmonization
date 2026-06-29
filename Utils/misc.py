import os
import sys
import json
import random
import logging
from typing import Dict, Union, Optional
from types import SimpleNamespace
import torch
import numpy as np
import json
import yaml
import pandas as pd
import glob
import pprint
import traceback
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
from Utils.arguments import load_json_config


logger = logging.getLogger()


def load_yaml_config(yaml_config):
    logger.info(f'Read yaml file {yaml_config}')
    f = open(yaml_config, 'r').read()
    config = yaml.safe_load(f)

def seed_everything(seed=123):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_save_path(config):
    save_dir = os.path.join(config['env']['project_dir'], 'train_output')
    os.makedirs(save_dir, exist_ok=True)
    num_existing_dirs = len(os.listdir(save_dir))
    save_path = os.path.join(save_dir, "run_{}".format(num_existing_dirs))
    os.makedirs(save_path, exist_ok=True)
    return save_path


def save_args(config, save_path: str):
    args_file_path = os.path.join(save_path, "args.json")
    with open(args_file_path, "w") as file:
        json.dump(config, file, indent=4)


def save_state(
        epoch: int,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        optim_g: torch.optim.Optimizer,
        optim_d: torch.optim.Optimizer,
        path: str,
        filename: str = "best_model.tar",
):
    old_checkpoint_files = list(
        filter(lambda x: "checkpoint" in x, os.listdir(path))
    )

    state_dict = {
        "epoch": epoch,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "optim_g": optim_g.state_dict(),
        "optim_d": optim_d.state_dict(),
    }
    file_path = os.path.join(path, filename)
    logger.info("Save current state to {}".format(filename))
    torch.save(state_dict, file_path)

    for file in old_checkpoint_files:
        os.remove(os.path.join(path, file))


def load_dataset_indices(load_path: str, file_name: str = "indices.json"):
    with open(os.path.join(load_path, file_name), "r") as file:
        indices = json.load(file)
    return indices


def load_state(path: str, map_location=None):
    loaded_state = torch.load(path, map_location=map_location)
    logger.info(
        "Loaded state from {} saved at epoch {}".format(path, loaded_state["epoch"])
    )
    return loaded_state


def load_args(run_path):
    run_args = json.load(open(os.path.join(run_path, "args.json")))
    return SimpleNamespace(**run_args)


class AverageMeter:
    """
    AverageMeter implements a class which can be used to track a metric over the entire training process.
    (see https://github.com/CuriousAI/mean-teacher/)
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """
        Resets all class variables to default values
        """
        self.val = 0
        self.vals = []
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates class variables with new value and weight
        """
        self.val = val
        self.vals.append(val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        """
        Implements format method for printing of current AverageMeter state
        """
        return "{self.val:{format}} ({self.avg:{format}})".format(
            self=self, format=format
        )


class AverageMeterSet:
    """
    AverageMeterSet implements a class which can be used to track a set of metrics over the entire training process
    based on AverageMeters (Source: https://github.com/CuriousAI/mean-teacher/)
    """
    def __init__(self):
        self.meters = {}

    def __getitem__(self, key):
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, postfix=""):
        return {name + postfix: meter.val for name, meter in self.meters.items()}

    def averages(self, postfix="/avg"):
        return {name + postfix: meter.avg for name, meter in self.meters.items()}

    def sums(self, postfix="/sum"):
        return {name + postfix: meter.sum for name, meter in self.meters.items()}

    def counts(self, postfix="/count"):
        return {name + postfix: meter.count for name, meter in self.meters.items()}


def set_device(config):
    if torch.cuda.is_available() and (config['env']['device'] == 'cuda'):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return device


def initialize_logger(save_path: str, log_file: str):
    logger = logging.getLogger()
    logging.basicConfig(
        filename=os.path.join(save_path, log_file),
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)


class DatasetUtils:
    @staticmethod
    def get_pid_date_from_case_str(case_str):
        raise NotImplementedError

    @staticmethod
    def get_case_str_from_pid_date(pid, date):
        raise NotImplementedError


class VLSPUtils(DatasetUtils):
    @staticmethod
    def get_pid_date_from_case_str(case_str):
        return int(case_str[:8]), int(case_str[12:20])

    @staticmethod
    def get_case_str_from_pid_date(pid, date):
        return f'{pid:08d}time{date}'


class NLSTUtils(DatasetUtils):
    @staticmethod
    def get_pid_date_from_case_str(case_str):
        return int(case_str[:6]), int(case_str[10:14])

    @staticmethod
    def get_case_str_from_pid_date(pid, date):
        return f'{pid}time{date}'


def get_dataset_utils(dataset_name):
    if dataset_name == 'vlsp':
        return VLSPUtils()
    elif dataset_name == 'nlst':
        return NLSTUtils()
    else:
        raise NotImplementedError


# From: https://github.com/theRealSuperMario/supermariopy/blob/master/scripts/tflogs2pandas.py
def tflog2pandas(path: str) -> pd.DataFrame:
    """convert single tensorflow log file to pandas DataFrame
    Parameters
    ----------
    path : str
        path to tensorflow log file
    Returns
    -------
    pd.DataFrame
        converted dataframe
    """
    DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0,  # 0 means load all
        "histograms": 1,
    }
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data


def save_img_stack_hdf5_grp(target_grp, img_stack, ds_name):
    if ds_name in target_grp:
        del target_grp[ds_name]

    chunk_shape = list(img_stack.shape)
    chunk_shape[0] = 1
    chunk_shape = tuple(chunk_shape)
    target_grp.create_dataset(
        ds_name,
        data=img_stack,
        chunks=chunk_shape,
        compression='gzip'
    )


class AnalyzeTrainingCurveMergeFinal:
    def __init__(self, initial_save_path, finetune_save_path, out_dir):
        self.initial_save_path = initial_save_path
        self.finetune_save_path = finetune_save_path
        self.out_dir = out_dir

    def get_merged_run_df(self):
        initial_config = load_json_config(os.path.join(self.initial_save_path, 'args.json'))
        finetune_config = load_json_config(os.path.join(self.finetune_save_path, 'args.json'))

        initial_epoch = initial_config['model']['num_epoch']
        finetune_epoch = finetune_config['model']['num_epoch']

        initial_analyzer = AnalyzeTrainingCurve(initial_epoch, self.initial_save_path)
        finetune_analyzer = AnalyzeTrainingCurve(finetune_epoch, self.finetune_save_path)

        initial_run_df = initial_analyzer.get_run_data_df()
        finetune_run_df = finetune_analyzer.get_run_data_df()
        finetune_run_df['step'] = finetune_run_df['step'] + initial_epoch

        merged_df = pd.concat([initial_run_df, finetune_run_df], ignore_index=True)

        return merged_df, initial_epoch, finetune_epoch

    def analyze_training_curve(self):
        merged_df, initial_epoch, finetune_epoch = self.get_merged_run_df()

        train_curve_png = os.path.join(self.out_dir, 'train_curve.png')
        breakdown_train_png = os.path.join(self.out_dir, 'breakdown_train.png')
        breakdown_valid_png = os.path.join(self.out_dir, 'breakdown_valid.png')
        AnalyzeTrainingCurve.plot_training_validation_curve(
            initial_epoch + finetune_epoch, merged_df, train_curve_png)
        AnalyzeTrainingCurve.plot_breakdown_loss_curve(
            initial_epoch + finetune_epoch, 'train', merged_df, breakdown_train_png)
        AnalyzeTrainingCurve.plot_breakdown_loss_curve(
            initial_epoch + finetune_epoch, 'valid', merged_df, breakdown_valid_png)

    def get_minial_valid_epoch(self):
        merged_df, _, _ = self.get_merged_run_df()
        valid_total_df = merged_df.loc[merged_df['metric'] == 'Loss/valid_total_loss']
        valid_total_df = valid_total_df.sort_values(by=['value'])
        min_epoch = valid_total_df.iloc[0]['step']
        min_val = valid_total_df.iloc[0]['value']

        print(f'Min valid epoch: {min_epoch} ({min_val})')


class AnalyzeTrainingCurve:
    def __init__(self, cur_epoch, save_path):
        self.epoch = cur_epoch
        self.save_path = save_path

    def get_run_data_df(self):
        tflog_path = glob.glob(f'{self.save_path}/events.out.tfevents*')[0]
        run_data_df = tflog2pandas(tflog_path)
        return run_data_df

    def analyze_training_curve(self):
        run_data_df = self.get_run_data_df()

        out_dir = os.path.join(self.save_path, 'training_curve')
        os.makedirs(out_dir, exist_ok=True)

        # 1. Get the training-validation curve - check if model converge
        train_curve_png = os.path.join(out_dir, 'train_curve.png')
        self.plot_training_validation_curve(self.epoch, run_data_df, train_curve_png)
        # train_curve_split_png = os.path.join(out_dir, 'train_curve_split.png')
        # self.plot_traininig_validation_curve_diff_scale(self.epoch, run_data_df, train_curve_split_png)

        # 2. Get the breakdown loss curves
        breakdown_train_png = os.path.join(out_dir, 'breakdown_train.png')
        self.plot_breakdown_loss_curve(self.epoch, 'train', run_data_df, breakdown_train_png)
        # breakdown_valid_png = os.path.join(out_dir, 'breakdown_valid.png')
        # self.plot_breakdown_loss_curve(self.epoch, 'valid', run_data_df, breakdown_valid_png)

        # 3. Save the dataframe in case we need to look at the raw data
        loss_break_csv = os.path.join(out_dir, 'loss_table.csv')
        print(f'Save to {loss_break_csv}')
        run_data_df.to_csv(loss_break_csv, index=False)

    def get_minial_valid_epoch(self):
        run_df = self.get_run_data_df()
        valid_total_df = run_df.loc[run_df['metric'] == 'Loss/valid_total_loss']
        valid_total_df = valid_total_df.sort_values(by=['value'])
        min_epoch = valid_total_df.iloc[0]['step']
        min_val = valid_total_df.iloc[0]['value']

        print(f'Min valid epoch: {min_epoch} ({min_val})')

    @staticmethod
    def plot_breakdown_loss_curve(epoch, mode, run_data_df, out_png):
        # Get tag name directly from the dataframe
        tag_list = list(set(run_data_df['metric'].to_list()))
        tag_list = [tag for tag in tag_list if 'Loss/' in tag]
        tag_key_list = [tag.replace(f'Loss/{mode}_', '').replace('_loss', '') for tag in tag_list]
        show_item_dict = {}
        for tag_key, tag in zip(tag_key_list, tag_list):
            show_item_dict[tag_key] = tag

        fig, ax = plt.subplots(figsize=(7, 4))
        for show_item in show_item_dict:
            tag_name = show_item_dict[show_item]
            step_list, value_list = AnalyzeTrainingCurve._get_step_value_pair(run_data_df, tag_name, epoch)
            ax.plot(step_list, value_list, label=show_item, alpha=0.7)

        ax.legend(loc='best')
        ax.set_xlim(0, epoch + 5)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_yscale('log')

        print(f'Save to {out_png}')
        plt.savefig(out_png, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()

    @staticmethod
    def plot_traininig_validation_curve_diff_scale(epoch, run_data_df, out_png):
        show_item_dict = {
            'Training': 'Loss/train_total_loss',
            'Validation': 'Loss/valid_total_loss'
        }
        fig, ax1 = plt.subplots(figsize=(7, 4))

        loss_value_full_list = []
        step_list, value_list = AnalyzeTrainingCurve._get_step_value_pair(run_data_df, show_item_dict['Training'], epoch)
        ax1.plot(step_list, value_list, alpha=0.7, color='blue')
        loss_value_full_list += value_list

        ax1.set_xlim(0, epoch + 5)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Total Loss (training)')

        ax2 = ax1.twinx()
        step_list, value_list = AnalyzeTrainingCurve._get_step_value_pair(run_data_df, show_item_dict['Validation'], epoch)
        ax2.plot(step_list, value_list, alpha=0.7, color='orange')
        loss_value_full_list += value_list
        ax2.set_ylabel('Total Loss (validation)')

        print(f'Save to {out_png}')
        plt.savefig(out_png, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()

    @staticmethod
    def plot_training_validation_curve(epoch, run_data_df, out_png):
        show_item_dict = {
            'Training': 'Loss/train_total_loss'
            # 'Validation': 'Loss/valid_total_loss'
        }
        fig, ax = plt.subplots(figsize=(7, 4))
        loss_value_full_list = []
        for show_item in show_item_dict:
            tag_name = show_item_dict[show_item]
            step_list, value_list = AnalyzeTrainingCurve._get_step_value_pair(run_data_df, tag_name, epoch)
            ax.plot(step_list, value_list, label=show_item, alpha=0.7)
            loss_value_full_list += value_list

        ax.legend(loc='best')
        ax.set_xlim(0, epoch + 5)
        # val_min, val_max = np.min(loss_value_full_list), np.max(loss_value_full_list)
        # val_range = val_max - val_min
        # show_extend_ratio = 0.1
        # show_extend_val = show_extend_ratio * val_range
        # ax.set_ylim(val_min - show_extend_val, val_max + show_extend_val)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total Loss')
        ax.set_yscale('log')

        print(f'Save to {out_png}')
        plt.savefig(out_png, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()

    @staticmethod
    def _get_step_value_pair(df, tag, n_step):
        tag_df = df.loc[(df['metric'] == tag) & (df['step'] <= n_step)]
        tag_df.sort_values(by='step', inplace=True)
        step_list = tag_df['step'].to_list()
        value_list = tag_df['value'].to_list()

        return step_list, value_list


class AnalyzeTrainingCurveV1Pipeline:
    def __init__(self, config, save_path):
        self.config = config
        self.save_path = save_path

    def plot_breakdown_loss_curve(self, mode, epoch_spec=None):
        tflog_path = glob.glob(f'{self.save_path}/events.out.tfevents*')[0]
        run_data_df = tflog2pandas(tflog_path)

        epoch = self.config['model']['num_epoch'] - 1 if epoch_spec is None else epoch_spec
        show_item_dict = {}
        for loss_term in ['valid', 'hole', 'tv', 'perc', 'style']:
            show_item_dict[loss_term] = f'Loss/{mode}_{loss_term}_loss'

        fig, ax = plt.subplots(figsize=(7, 4))
        for show_item in show_item_dict:
            tag_name = show_item_dict[show_item]
            loss_coef = self.config['model'][f'{show_item}_coef']
            step_list, value_list = AnalyzeTrainingCurve._get_step_value_pair(run_data_df, tag_name, epoch)
            value_list = [loss_coef * val for val in value_list]
            ax.plot(step_list, value_list, label=show_item, alpha=0.7)

        ax.legend(loc='best')
        ax.set_xlim(0, epoch + 5)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_yscale('log')

        out_png = os.path.join(self.save_path, f'loss_breakdown_{mode}_{epoch}.png')
        print(f'Save to {out_png}')
        plt.savefig(out_png, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()


def to_items(dic):
    return dict(map(_to_item, dic.items()))


def _to_item(item):
    return item[0], item[1].item()


def read_file_contents_list(file_name):
    print(f'Reading from file list txt {file_name}', flush=True)
    with open(file_name) as file:
        lines = [line.rstrip('\n') for line in file]
        print(f'Number items: {len(lines)}')
        return lines


def load_json(in_path):
    with open(in_path) as json_file:
        # logger.info(f'Load {in_path}')
        load_data = json.load(json_file)
        return load_data