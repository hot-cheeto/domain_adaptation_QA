import os
import torch 
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function

from utilities.file_utils import Utils as utils
from models.lightning_trainer import LightningTrainer
from transformers import AutoConfig, AutoModelForQuestionAnswering
from transformers import AutoTokenizer

class GradientReversalFunction(Function):
    """Code from: https://github.com/jvanvugt/pytorch-domain-adaptation/blob/master/utils.py
       This is a Gradient Reversal Layer from "Unsupervised Domain Adaptation by Backpropagation"

       A torch.autograd.Function has to be used here, as it is this Class which records operations 
       during the forward pass and a means of calculating the gradients in the backward pass.
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        """Identity Function, the Reversal layer doesn't do anything special in the forward pass.
        """
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        """Now in the backward pass, we will reverse our gradients and scale them by lambda.
        """
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(nn.Module):
    """Code from: https://github.com/jvanvugt/pytorch-domain-adaptation/blob/master/utils.py
       This Module is what is used within our models to reverse the Gradient.
    """
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x, lambda_ =  None):
        curr_lambda = lambda_ if lambda_ != None else self.lambda_
        return GradientReversalFunction.apply(x, curr_lambda)


class LayerNormalizedProjection(nn.Module):
    """LayerNormalization MLP used in "Learning Invariant Representations of Social Media Users" 
       after the pooling and concatenation of the Encoder layers. 

       Note, this MLP looks like a good general purpose classification MLP as well.
    """
    def __init__(self, dim=512, activation=nn.ELU(), out_dim=None):
        super(LayerNormalizedProjection, self).__init__()

        out_dim = out_dim if out_dim else dim
        self.proj = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim, bias=False),
                activation,
                nn.Linear(dim, out_dim, bias=False))

        # Only add a LayerNorm in the end when we're projecting to an embedding
        if out_dim > 1:
            self.proj = nn.Sequential(self.proj, nn.LayerNorm(out_dim))


    def forward(self, x):
        return self.proj(x)


class QA_Transfomer(LightningTrainer):

    def __init__(self, params, dataset):

        super(QA_Transfomer, self).__init__()
        self.params = params
        self.dataset_class = dataset

        self.dann = params.dann

        # sometimes huggingface gives me trouble downloading weights directly 
        self.model_weight_path = utils.path_exists(os.path.join(utils.weight_path , 'bert-base-uncased'))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_weight_path, return_token_type_ids = True)
        self.bert_transfomer = AutoModelForQuestionAnswering.from_pretrained(self.model_weight_path, return_dict = True, output_hidden_states = True) 
        self.ques_cxt_dim = 768

        if self.dann:
          
          self.domain_mlp= LayerNormalizedProjection(self.ques_cxt_dim, 
                                                       out_dim = 1, 
                                                       activation = nn.ReLU(inplace = True))
            
          self.grad_rev = GradientReversal(lambda_ = 1.0)
          self.sigmoid = nn.Sigmoid()
          self.domain_loss = nn.BCELoss()


        self.init_experiment_paths()
        self.init_dataset()
        self.init_metric_func()
   

    def lambda_scheduler(self):
        p = float(self.current_epoch + 1) / float(self.params.num_epoch)
        return 2. / (1. + np.exp(-10 * p)) - 1

    
    def forward(self, input_ids, attention_mask, answer_span, domain_label = None):
      
        out = self.bert_transfomer(input_ids, attention_mask = attention_mask, start_positions = answer_span[:, 0],  end_positions = answer_span[:,1]) 
        loss, start_scores, end_scores = out[0], out[1], out[2]

        if self.dann and domain_label != None:

            embedded_ques_cxt = out[-1][-1][:, 0, :]
            domain_out = self.grad_rev(embedded_ques_cxt, self.lambda_scheduler())
            domain_out = self.domain_mlp(domain_out)
            domain_out = self.sigmoid(domain_out).squeeze()

            domain_loss = self.domain_loss(domain_out, domain_label)

            loss += domain_loss
            
        
        return loss, start_scores, end_scores


    def _model_forward(self, batch):

        if self.dann:
          input_ids, attention_mask, answer_span, answer_text, domain_label, _ = batch

        else:
          input_ids, attention_mask, answer_span, answer_text, _ = batch
          domain_label = None
        
        input_ids = torch.stack(input_ids).reshape((-1, self.params.max_seq_length))
        attention_mask = torch.stack(attention_mask).reshape((-1, self.params.max_seq_length))

        loss, start_scores, end_scores = self.forward(input_ids, attention_mask, answer_span, domain_label)

        return loss, start_scores, end_scores


