import abc
import os
from itertools import chain

import torch
import torch.optim as optim
import pytorch_lightning as pt
from transformers import AdamW
from torch.utils.data import DataLoader

from utilities.file_utils import Utils as utils
from utilities import data_helpers as dh
from utilities import evaluate as metric_functions


class LightningTrainer(pt.LightningModule):

    @abc.abstractmethod
    def _model_forward(self, batch):
        pass

    def init_metric_func(self):
        self.metric = {'f1_score': metric_functions.f1_score, 'em': metric_functions.exact_match_score}

    def init_experiment_paths(self):

        self.experiment_parameters_path = utils.path_exists(os.path.join(utils.output_path, 'parameters'), True)
        self.experiment_log_filename = os.path.join(utils.output_path, 'experiments.log')
        self.prediction_path = utils.path_exists(os.path.join(utils.output_path, 'predictions'), True)
        self.experiment_prediction_filename = os.path.join(self.prediction_path, '{}.xlsx'.format(self.params.experiment_id))


    def init_dataset(self):

        self.validation_dataset, self.test_dataset, self.train_dataset = None, None, None

        if self.params.do_learn:
            self.log.info('Loading Training Set')
            self.train_dataset = self.dataset_class('train', self.tokenizer)
            self.validation_dataset = self.dataset_class('test-dev', tokenizer = self.tokenizer)
        
        if self.params.evaluate:
            if self.params.use_test_dataset:
                self.test_dataset = self.dataset_class('test', tokenizer = self.tokenizer)
            else:
                self.test_dataset = self.dataset_class('test-dev', tokenizer = self.tokenizer)
        

    def configure_optimizers(self):

        optimizer = AdamW(
            chain(self.parameters(), self.loss.parameters()), lr=self.params.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=300, eta_min=0.0001)

        return [optimizer], [scheduler]


    def train_dataloader(self):

        data_loader = DataLoader(self.train_dataset,
                                 batch_size=self.params.batch_size,
                                 shuffle=True,
                                 num_workers=self.params.num_workers,
                                 pin_memory=self.params.pin_memory)

        return data_loader


    def val_dataloader(self):

        data_loader = DataLoader(self.validation_dataset,
                                  batch_size=self.params.batch_size,
                                  shuffle=False,
                                  pin_memory=self.params.pin_memory,
                                  num_workers=self.params.num_workers)

        return data_loader


    def test_dataloader(self):
        
        data_loader = DataLoader(self.test_dataset,
                                  batch_size=self.params.batch_size,
                                  shuffle=False,
                                  pin_memory=self.params.pin_memory,
                                  num_workers=self.params.num_workers)

        return data_loader


    def get_decoded_prediction(self, input_ids, start_score, end_score):


        ans_tokens = input_ids[torch.argmax(start_score) : torch.argmax(end_score) + 1]
        answer_tokens = self.tokenizer.convert_ids_to_tokens(ans_tokens, skip_special_tokens = True)

        return answer_tokens


    def training_step(self, batch, batch_idx):

        input_ids, _, _, _, answer_text, _ = batch
        loss, start_scores, end_scores = self._model_forward(batch)

        max_start_scores = torch.argmax(start_scores, dim = 1)
        max_end_scores = torch.argmax(end_scores, dim = 1)

        self.log('loss', loss)
        return_dict = {'loss': loss}

        return_dict['gt_answer_text'] = answer_text
        return_dict['pred_answer_text'] = [self.get_decoded_prediction(inp, s, e) for inp, s, e in zip(input_ids, max_start_scores, max_end_scores)]

        return return_dict


    def validation_step(self, batch, batch_idx):

        input_ids, _, _, _, answer_text, _ = batch
        loss, start_scores, end_scores = self._model_forward(batch)

        max_start_scores = torch.argmax(start_scores, dim = 1)
        max_end_scores = torch.argmax(end_scores, dim = 1)

        self.log('val_loss', loss.item())
        return_dict = {'val_loss': loss.item()}

        return_dict['gt_answer_text'] = answer_text
        return_dict['pred_answer_text'] = [self.get_decoded_prediction(inp, s, e) for inp, s, e in zip(input_ids, max_start_scores, max_end_scores)]

        return return_dict


    def test_step(self, batch, batch_idx):

        input_ids, _, answer_span, possible_ans, answer_text, idx = batch
        _, start_scores, end_scores = self._model_forward(batch)

        max_start_scores = torch.argmax(start_scores, dim = 1)
        max_end_scores = torch.argmax(end_scores, dim = 1)

        return_dict = {}

        return_dict['gt_answer_span'] = answer_span
        return_dict['gt_answer_text'] = answer_text
        return_dict['possible_span'] = possible_ans
        
        return_dict['pred_answer_span'] = [tuple(s, e) for s, e in zip(max_start_scores, max_end_scores)]
        return_dict['pred_answer_text'] = [self.get_decoded_prediction(inp, s, e) for inp, s, e in zip(input_ids, max_start_scores, max_end_scores)]
        return_dict['indices'] = idx

        return return_dict

    def training_epoch_end(self, outputs):

        avg_loss = torch.stack([x['loss'] for out in outputs for x in out]).mean()
        self.log('train_loss', avg_loss)

        all_gt_text = [x['gt_answer_text'] for out in outputs for x in out]
        all_pred_text = [x['pred_answer_text'] for out in outputs for x in out]

        for metric_name, metric_func in self.metric.items():
            metric_out = torch.tensor(metric_func(all_pred_text, all_gt_text))
            self.log('train_{}'.format(metric_name), metric_out, prog_bar = True)
            

    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack([x['val_loss'] for out in outputs for x in out]).mean()
        self.log('validation_loss', avg_loss)

        all_gt_text = [x['gt_answer_text'] for out in outputs for x in out]
        all_pred_text = [x['pred_answer_text'] for out in outputs for x in out]

        for metric_name, metric_func in self.metric.items():
            metric_out = torch.tensor(metric_func(all_pred_text, all_gt_text))
            self.log('validation_{}'.format(metric_name), metric_out, prog_bar = True)


        # select at random another set of data for next epoch
        self.validation_dataset.resample_data()

    def create_predictions_excel(self, outputs, all_gt_text, all_pred_text):
        
        all_gt_idx = [x['gt_answer_span'] for out in outputs for x in out]
        all_pred_idx = [x['pred_answer_span'] for out in outputs for x in out]
        all_poss_span = [x['possible_span'] for out in outputs for x in out]
        all_indices = [x['index'] for out in outputs for x in out]
        prediction_metadata = {}

        for i in range(len(all_indices)):

            id_ = self.test_dataset.ids[i]

            if id_ not in prediction_metadata:
                prediction_metadata[id_] = {}

            sample = self.test_dataset.data[id_]
            prediction_metadata[id_]['id'] = id_
            prediction_metadata[id_]['question'] = sample['question']
            prediction_metadata[id_]['gt_answer_text'] = all_gt_text[i]
            prediction_metadata[id_]['pred_answer_text'] = all_pred_text[i]
            prediction_metadata[id_]['context_text'] = sample['context']
            prediction_metadata[id_]['possible_span'] = all_poss_span
            prediction_metadata[id_]['gt_answer_span'] = '{}-{}'.format(all_gt_idx[i][0], all_gt_idx[i][1]) 
            prediction_metadata[id_]['pred_answer_span'] = '{}-{}'.format(all_pred_idx[i][0], all_pred_idx[i][1])

        df = dh.dict2dataframe(prediction_metadata)
        df.to_excel(self.experiment_prediction_filename)

             
    def test_epoch_end(self, outputs):

        all_gt_text = [x['gt_answer_text'] for out in outputs for x in out]
        all_pred_text = [x['pred_answer_text'] for out in outputs for x in out]

        results = {}

        for metric_name, metric_func in self.metric.items():
            metric_out = torch.tensor(metric_func(all_pred_text, all_gt_text))
            self.log('test_{}'.format(metric_name), metric_out, prog_bar = True)

            results[metric_name] = metric_out

        results_string = dh.dict2string(results, '{} version {}'.format(self.params.experiment_id, self.logger.version))
        mode = 'a' if os.path.exists(self.experiment_log_filename) else 'w'

        with open(self.experiment_log_filename, mode) as f:
            f.write(results_string)

        # print(results_string)

        if self.params.create_predictions_file:
            self.create_predictions_excel(outputs, all_gt_text, all_pred_text)



