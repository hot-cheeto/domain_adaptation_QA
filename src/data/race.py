import torch
from torch.utils.data.dataset import Dataset
from datasets import load_dataset

class RACE(Dataset):

    def __init__(self, dataset_type, tokenizer, sanity = False, max_seq_length = 384, notebook = False):

        self.dataset_type = dataset_type 
        self.notebook = notebook
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self.data = load_dataset('race', 'all', split = self.dataset_type)
        self.number_examples = 100 if sanity else self.data.num_rows 
        self.options_letters = {'a': 0, 'b': 1, 'c': 2, 'd': 3}


    def __len__(self):
        return self.number_examples


    def encode_example(self, passage, question, option):

        text_a = passage
        text_b = question.replace("_", option)
        inputs = self.tokenizer.encode_plus(text_a, text_b,
                                            add_special_tokens = True,
                                            max_length = self.max_seq_length,
                                            pad_to_max_length = True,
                                            truncation = True,)


        return inputs


    def flatten_example(self, examples):

        inputs = {}
        for e in examples:
            if len(inputs) == 0:
                inputs = {k: [] for k in e.keys()}
            for k, v in e.items():
                inputs[k].append(v)

        inputs = {k: torch.tensor(v) for k, v in inputs.items()}

        return inputs


    def __getitem__(self, index):

        passage = self.data['article'][index]
        question = self.data['question'][index]
        options = self.data['options'][index]
        answer= self.data['answer'][index]
        text_answer = options[self.options_letters[answer.lower()]]

        text_data = {'passage': passage, 'question': question, 'options': options, 'answer': text_answer, 'idx': index}

        label_idx = torch.tensor(self.options_letters[answer.lower()]).long()
        examples = self.flatten_example([self.encode_example(passage, question, opt) for opt in options])

        return examples['input_ids'], examples['attention_mask'],  label_idx, text_data




