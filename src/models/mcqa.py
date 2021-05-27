import torch
import numpy as np 
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModelForMultipleChoice


class MCQA(pt.LightningModule):
    
    def __init__(self, params, dataset, dest_path):
        
        self.dataset = dataset
        self.params = params
        self.num_choices = self.params.num_choices
         
        self.tokenizer = AutoTokenizer.from_pretrained(self.params.tokenizer_name)
        self.transfomer = AutoModelForMultipleChoice.from_pretrained(self.params.model_checkpoint, num_labels = self.num_choices)
        
        self.train_dataset, self.test_dataset, self.validation_dataset = None, None, None
       
        if self.params.do_learn:
            self.train_dataset = dataset('train', self.tokenizer, self.params.sanity)
            self.validation_dataset = dataset('validation', self.tokenizer, self.params.sanity)
        if self.params.evaluate:
            self.test_dataset = dataset('test', self.tokenizer, self.params.sanity) 
        if params.evaluated_dev:
            self.test_dataset = dataset('validation', self.tokenizer, self.params.sanity)
           
        if os.path.join(dest_path) == False:
            os.mkdir(dest_path)
            
        self.dest_path = dest_path
        self.experiment_log_filename = os.path.join(self.dest_path, 'experiments.log') 
        self.experiment_prediction_filename = os.path.join(self.dest_path, 'predictions.xlsx')
        
        
    def forward(self, batch):
        
        inputs, attention_mask, labels, _ = batch 
        
        out = self.transformer(inputs, attention_mask = attention_mask, labels = labels)
        loss, logits = out[0], out[1]
        
        # do extra stuff
        
        return loss, logits
   


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
    
    
    def training_step(self, batch):
        
        _, _, labels, _ = batch
        loss, logits = self.forward(batch)
        pred = torch.argmax(logits, dim = 1)
        self.log('loss', loss)
        return_dict = {'loss': loss}
        return_dict['gt_answer'] = labels 
        return_dict['pred_answer'] = pred
        
        return return_dict
        
    def validation_step(self, batch):
        
        _, _, labels, _ = batch
        loss, logits = self.forward(batch)
        pred = torch.argmax(logits, dim = 1)
        self.log('val_loss', loss)
        return_dict = {'val_loss': loss}
        return_dict['gt_answer'] = labels 
        return_dict['pred_answer'] = pred
        
        return return_dict
        
    def test_step(self, batch):
        
        _, _, labels, text_data = batch
        loss, logits = self.forward(batch)
        pred = torch.argmax(logits, dim = 1)
        self.log('test_loss', loss)
        return_dict = {}
        
        for t in text_data:
            if len(return_dict) == 0:
                return_dict = {k: [] for k in t.keys()}
            
            for k, v in t.items():
                return_dict[k].append(v)
        
        return_dict = {'test_loss': loss}
        return_dict['gt_answer'] = labels 
        return_dict['pred_answer'] = pred
        
        
        return return_dict
    
    
    def get_accuracy(self, gt, pred):
        return torch.sum(gt == pred) / pred.size()[0] 
    
    
    def training_epoch_end(self, outputs):

        avg_loss = torch.stack([x['loss'] for out in outputs for x in out]).mean()
        self.log('train_loss', avg_loss)

        all_gt = torch.stack([x['gt_answer'] for out in outputs for x in out])
        all_pred = torch.stack([x['pred_answer'] for out in outputs for x in out])
        accuracy = self.get_accuracy(all_gt, all_pred)
        self.log('train_accuracy', accuracy, prog_bar = True)

        
    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack([x['val_loss'] for out in outputs for x in out]).mean()
        self.log('val_loss', avg_loss)

        all_gt = torch.stack([x['gt_answer'] for out in outputs for x in out])
        all_pred = torch.stack([x['pred_answer'] for out in outputs for x in out])
        accuracy = self.get_accuracy(all_gt, all_pred)
        self.log('val_accuracy', accuracy, prog_bar = True)
        
    def create_predictions_excel(self, all_preds, outputs):
          
        all_passages = [t['passage'] for t in outputs]
        all_questions = [t['question'] for t in outputs]
        all_options = ['-opt-'.join(t['choices']) for t in outputs]
        all_answers = [t['answer_text'] for t in outputs]
        all_pred_text = [t['choices'][i] for i, t in zip(all_preds, outputs)]
        all_indices = [x['index'] for out in outputs for x in out]
        
        prediction_metadata = {}

        for i in range(len(all_indices)):

            id_ = all_indices[i]

            if id_ not in prediction_metadata:
                prediction_metadata[id_] = {}

            sample = self.test_dataset.data[id_]
            prediction_metadata[id_]['id'] = id_
            prediction_metadata[id_]['question'] = all_questions[i]
            prediction_metadata[id_]['choices'] = all_options[i]
            prediction_metadata[id_]['gt_answer_text'] = all_answers[i]
            prediction_metadata[id_]['pred_answer_text'] = all_pred_text[i]
            prediction_metadata[id_]['context_text'] = all_passages[i]

        df = dh.dict2dataframe(prediction_metadata)
        df.to_excel(self.experiment_prediction_filename)
        
        
    def test_epoch_end(self, outputs):
        
        avg_loss = torch.stack([x['test_loss'] for out in outputs for x in out]).mean()
        self.log('test_loss', avg_loss)

        all_gt = torch.stack([x['gt_answer'] for out in outputs for x in out])
        all_pred = torch.stack([x['pred_answer'] for out in outputs for x in out])
        accuracy = self.get_accuracy(all_gt, all_pred)
        self.log('test_accuracy', accuracy, prog_bar = True)
       
        params = vars(self.params)
        params['accuracy'] = accuracy
        params['avg_loss'] = avg_loss
            
        results_string = dh.dict2string(params, '{} version {}'.format(self.params.experiment_id, self.logger.version))

        mode = 'a' if os.path.exists(self.experiment_log_filename) else 'w'
        with open(self.experiment_log_filename, mode) as f:
            f.write(results_string)
        
        if self.params.create_prediction_file:
            self.create_predictions_excel(all_pred, outputs)
            
        
