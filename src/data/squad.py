import torch
from torch.utils.data.dataset import Dataset
from datasets import load_dataset

class SQuAD(Dataset):

    def __init__(self, dataset_type, tokenizer, sanity = False, max_seq_length = 384, notebook = False):

        self.dataset_type = dataset_type 
        self.notebook = notebook
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self.data = load_dataset('squad', split = self.dataset_type)
        self.number_examples = 100 if sanity else self.data.num_rows 


    def __len__(self):
        return self.number_examples


    def __getitem__(self, index):

        
        context = self.data['context'][index]
        question = self.data['question'][index]
        answer = self.data['answers']

        answer_text = answer['text'][0]
        answer_start = answer['answer_start'][0]

        answer_span = tuple([answer_start, answer_start + len(answer_text)])
        
        if self.tokenizer != None:
            answer_span = torch.tensor([answer_start, answer_start + len(answer_text)]).long()
            input_encoded = self.tokenizer.encode_plus(question, context,
                                                    add_special_tokens = True, 
                                                    max_length = self.max_seq_length, 
                                                    pad_to_max_length = True, 
                                                    truncation = True, )



            return input_encoded['input_ids'], input_encoded['attention_mask'], answer_span, answer_text, index

        # for analysis
        return question, answer_text, answer_span, context 

