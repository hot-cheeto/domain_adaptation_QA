
import torch 
import torch.nn.functional as F

from utilities.file_utils import Utils as utils
from models.lightning_trainer import LightningTrainer
from transformers import AutoConfig, AutoModelForQuestionAnswering
from transformers import AutoTokenizer


class QA_Transfomer(LightningTrainer):

    def __init__(self, params, dataset):

        self.params = params
        self.dataset_class = dataset
        self.tokenizer = AutoTokenizer.from_pretrained(self.params.tokenizer_name, return_token_type_ids = True)
        self.transfomer = AutoModelForQuestiongAnswering.from_pretrained(self.params.model_name) 

        self.init_experiment_paths()
        self.init_dataset()

    def forward(self, inputs):

        out = self.transformer(inputs['input_ids'], attention_mask = inputs['attention_mask'])
        start_scores, end_scores = out[0], out[1]

        return start_scores, end_scores


    def _model_forward(self, batch, return_loss = True):

        inputs, _, start_idx, answer_text = batch
        end_idx = torch.tensor([s + len(t) for s, t in zip(start_idx, answer_text)]) 

        start_scores, end_scores = self.forward(inputs)

        if return_loss:
            loss = self.loss(start_scores, end_scores, start_idx, end_idx)
            return loss, start_scores, end_scores
        
     
        return start_scores, end_scores


