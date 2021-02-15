from typing import Optional
import os
from pydantic import BaseModel
import yaml, pathlib, os
from pydantic import validator, Field
import torch

class LogConfigs(BaseModel):
    log_dir: str
    log_file_name: str


class ProjectConfigs(BaseModel):
    PROJECT_DIR: str = pathlib.Path(__file__).resolve().parents[1]
    PROJECT_NAME: str = "news"
    VERSION: str = "beta1"


class DataConfigs(BaseModel):
    test_date_rate: float = 0.2
    DATASET: str = "datasets/sogounews"
    stop_words: str = 'HIT_stop_words.txt'
    LABEL_NAME_FILE: str = "label_names.txt"
    TRAIN_CORPUS: str = "train.txt"
    TEST_CORPUS: str = "test.txt"
    TRAIN_LABEL: str = "train_labels.txt"
    TEST_LABEL: str = "test_labels.txt"
    out_file: str = 'out.txt'
    final_model: str = 'final_model.pt'
    bad_case_data: str = "bad_case.csv"

    @validator("bad_case_data", pre=False, always=True)
    def set_bad_case_data(cls, v, values):
        project_path = pathlib.Path(__file__).resolve().parents[1]
        values['DATASET'] = os.path.join(project_path, values['DATASET'])
        v = os.path.join(values['DATASET'], v)
        return v


class TrainArgs(BaseModel):
    pretrained_weights_path: str
    MAX_LEN: int = 200
    TRAIN_BATCH: int = 32
    ACCUM_STEP: int = 2
    EVAL_BATCH: int = 128
    category_vocab_size: int = 100
    top_pred_num: int = 50
    CUDA_DEVICE_ORDER: str = '0'
    CUDA_VISIBLE_DEVICES: str = '1,2,3,4'
    GPUS: int = 4
    MCP_EPOCH: int = 3
    SELF_TRAIN_EPOCH: int = 1
    dist_port: int = 12345
    update_interval: int = 50
    match_threshold: int = 20
    early_stop: str = 'store_true'
    device: Optional[str] = Field(default='')

    @validator("device", pre=False, always=True)
    def set_device(cls, v, values):
        # os.environ["CUDA_DEVICE_ORDER"] = values['CUDA_DEVICE_ORDER']
        # os.environ["CUDA_VISIBLE_DEVICES"] = values['CUDA_VISIBLE_DEVICES']
        return v


class Configs(BaseModel):
    log: LogConfigs
    project: ProjectConfigs
    data: DataConfigs
    train_args: TrainArgs


def read_yaml(file_path):
    with open(file_path) as f:
        return yaml.safe_load(f)


project_path = pathlib.Path(__file__).resolve().parents[1]
curr_conf_path = os.path.join(project_path, 'config')

configs_yaml = os.path.join(curr_conf_path, 'configs.yaml')

configs = Configs(**read_yaml(str(configs_yaml)))

# LOGS.log.debug(configs.dict())
