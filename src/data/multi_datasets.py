import torch
from torch.utils.data.dataset import Dataset

class Multi_Dataset(Dataset):

    def __init__(self, scr_dataset, target_dataset, dataset_type, params, tokenizer = None,
                  sanity = False, max_seq_length = 384, notebook = False):
 
        self.dataset_type = dataset_type 
        self.notebook = notebook
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.params = params
        self.target_prob_sample = self.params.target_prob_sample
        
        # for now a hack
        target_dataset_name = 'test-dev' if self.dataset_type == 'validation' or self.dataset_type == 'test' else self.dataset_type
        oversample = True if self.dataset_type == 'train' else False
        self.target_dataset = target_dataset(target_dataset_name, 
                                             tokenizer = self.tokenizer, 
                                             oversample = oversample, 
                                             max_seq_len = self.max_seq_length)

        self.scr_dataset = scr_dataset(self.dataset_type, self.tokenizer, max_seq_length = self.max_seq_length)

        self.number_examples = self.scr_dataset.number_examples
        
        if self.dataset_type  == 'train':
            # bc we are oversampling the target dataset
            self.target_dataset.number_examples = self.number_examples


    def __len__(self):
        return self.number_examples


    def __getitem__(self, index):

      if random.random() < self.target_prob_sample and self.dataset_type == 'train':
        
        return self.target_dataset[index], 0

      return self.scr_dataset[index], 1



