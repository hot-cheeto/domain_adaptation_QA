
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
        # sometimes huggingface gives me trouble downloading weights directly 
        self.model_weight_path = utils.path_exists(os.path.join(utils.weight_path , self.model_weight_name))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_weight_path, return_token_type_ids = True)
        self.transfomer = AutoModelForQuestiongAnswering.from_pretrained(self.model_weight_path) 

        self.init_experiment_paths()
        self.init_dataset()
        self.init_metric_func()

    def forward(self, input_ids, attention_mask, answer_span):
        
        out = self.transformer(input_ids, attention_mask = attention_mask, start_positions = answer_span[:, 0],  end_positions = answer_span[:,1])
        loss, start_scores, end_scores = out[0], out[1], out[2]

        return loss, start_scores, end_scores


    def _model_forward(self, batch, return_loss = True):

        input_ids, attention_mask, answer_span, answer_text, _ = batch
        loss, start_scores, end_scores = self.forward(input_ids, attention_mask, answer_span)

        return loss, start_scores, end_scores


