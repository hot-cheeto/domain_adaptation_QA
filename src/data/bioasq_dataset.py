import os
import torch
import numpy as np
import glob
import random

from torch.utils.data.dataset import Dataset
from utilities.file_utils import Utils as utils
from tqdm import tqdm 
from tqdm import tqdm_notebook 


class BioASQDataset(Dataset):

    def __init__(self, dataset_type, test_idx = 2, sanity = False, tokenizer = None, oversample = False, max_seq_len =  384, notebook = False):

        self.data_path = utils.path_exists(os.path.join(utils.data_path, 'BioASQ'))
        self.dataset_type = dataset_type 
        self.notebook = notebook
        self.max_seq_length= max_seq_len
        self.test_idx = test_idx
        self.sanity = sanity
        self.oversample = oversample

        self.empty = 0
        self.sub_data, self.sample_ids = {}, []

        self.data = self.load_data()
        self.ids = list(self.data.keys()) 
        
        if self.sanity:
          self.ids = self.ids[:10]

        self.num_examples = len(self.ids)
        self.tokenizer = tokenizer


    def resample_data(self):
        self.sample_ids = random.sample(self.ids, 100)
        self.sub_data = {k: v for k, v in self.data.items() if k in self.sample_ids}
        self.num_examples = 100


    def load_data(self):

        file_type = 'train' if self.dataset_type == 'test-dev' else self.dataset_type
        files = list(glob.glob(os.path.join(self.data_path, 'BioASQ-{}-*.json'.format(file_type))))
        files = [files[self.test_idx]] if self.dataset_type == 'test-dev' else files

        if self.dataset_type != 'test-dev':
            del files[self.test_idx]


        golden_files = [] if self.dataset_type == 'train' or self.dataset_type == 'test-dev' else {f.split('/')[-1].split('.')[0].split('_')[0].lower() : 
                        f for f in glob.glob(os.path.join(self.data_path, '*_golden.json'))}

        data = {}

        for curr_file in files:
 
            print('Loading and preprocessing file: {}'.format(curr_file))
            para_data= utils.read_json(path = curr_file)['data'][0]
            pbar = tqdm_notebook if self.notebook else tqdm

            for d in pbar(para_data['paragraphs']):
                context = d['context']

                if len(golden_files) > 0:
                    probid = ''.join(curr_file.split('/')[-1].split('-',3 )[-1].split('.')[0].split('-')).lower()
                    ques = utils.read_json(path = golden_files[probid])
                    id2qa = {d['id']: d['ideal_answer'][0] for d in ques['questions'] if 'ideal_answer' in d}
                
                for qa in d['qas']:

                    ans = qa['answers'] if 'answers' in qa else id2qa.get(qa['id'].split('_')[0], None)

                    if type(ans) == list:
                        ans = ans[0]['text']
                    
                    # exclude examples that are empty 
                    if len(qa['question'].split()) < 1 or ans == None or len(ans.split()) < 1 or len(context.split()) < 1:
                        self.empty += 1
                        continue

                    if qa['id'] in data: continue
                    if len(golden_files) > 0 and qa['id'].split('_')[0] not in id2qa: continue

                    data[qa['id']] = {'context': [], 'question': [], 'answers': []}
                    data[qa['id']]['question'] = qa['question']
                    data[qa['id']]['answers'] = qa['answers'] if 'answers' in qa else id2qa.get(qa['id'].split('_')[0], None)
                    data[qa['id']]['context'] = context


        return data


    def __len__(self):
        return self.num_examples


    def __getitem__(self, index):

        if self.oversample and self.do_learn:
          index = random.sample(range(self.num_examples), 1)[0]

        qa_sample = self.data[self.ids[index]]

        question, answers, context = qa_sample['question'], qa_sample['answers'], qa_sample['context']

        if self.dataset_type == 'train' or self.dataset_type == 'test-dev':
            answer_start = answers[0]['answer_start']
            answer_text = answers[0]['text']
        else:
            # hack due the qa in test set doesnt have a span sometimes they do but I am only using it to see what generates
            answer_text = answers
            answer_start = context.index(answer_text) if answer_text in context else 0
        
        answer_span = tuple([answer_start, answer_start + len(answer_text)])
       
        # if i am using a neural network 
        if self.tokenizer != None:
            answer_span = torch.tensor([answer_start, answer_start + len(answer_text)]).long()
            input_encoded = self.tokenizer.encode_plus(question, context,
                                                    add_special_tokens = True, 
                                                    max_length = self.max_seq_length, 
                                                    pad_to_max_length = True, 
                                                    truncation = True, )

            input_ids = torch.tensor(input_encoded['input_ids'])
            attention_mask = torch.tensor(input_encoded['attention_mask'])



            return input_ids, attention_mask, answer_span, answer_text, index


        # for analysis
        return question, answer_text, answer_span, context 







