import os
import sys
import glob

import torch
import torch.nn as nn
import torch.multiprocessing
import pytorch_lightning as pt
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import random

from arguments import dump_parameters
from utilities.file_utils import Utils as utils

from models.qa_transformer import QA_Transfomer
from data.bioasq_dataset import BioASQDataset


def get_checkpoint(params):

    path = os.path.join(utils.output_path, params.load_checkpoint,
                        'lightning_logs', 
                        'version_{}'.format(params.version), 
                        'checkpoints')

    all_checkpoints = list(glob.glob(os.path.join(path, '*.ckpt')))
    epoch2path = {int(c.split('/')[-1].split('.')[0].split('=')[1].split('-')[0]): c for c in all_checkpoints}

    return epoch2path


def main(params):

    torch.manual_seed(params.random_seed)
    torch.cuda.manual_seed(params.random_seed)
    np.random.seed(params.random_seed)
    random.seed(params.random_seed)

    params_filename = os.path.join(utils.output_path, 'experiment_params.log')
    experiment_dir = utils.path_exists(os.path.join(utils.output_path, params.experiment_id), True)
    model = QA_Transfomer(params, BioASQDataset)

    if params.load_checkpoint:
        epoch2path = get_checkpoint(params)
        epoch_keys = sorted(list(epoch2path.keys()))
        epoch_key = -1 if self.params.epoch_key not in epoch2path else self.params.epoch_key
        resume_from_checkpoint = epoch2path[epoch_key]        

        print("Checkpoint: {}".format(resume_from_checkpoint))
        checkpoint = torch.load(resume_from_checkpoint)
        model.load_state_dict(checkpoint['state_dict'])

    else:
        resume_from_checkpoint = None

    ckpt_metric = 'validation_f1_score' 
    checkpoint_callback = ModelCheckpoint(monitor=ckpt_metric, mode='max')

    trainer = pt.Trainer(default_root_dir=experiment_dir, weights_save_path=experiment_dir,
                         max_epochs=params.num_epoch, checkpoint_callback=checkpoint_callback,
                         distributed_backend=backend, gpus=params.gpus)

    if params.do_learn:
        trainer.fit(model)

    if params.evaluate:
        trainer.test(model)


if __name__ == "__main__":
    params = dump_parameters()
    main(params)
