import os
import torch
import numpy as np
import glob

from torch.utils.data.dataset import Dataset
from utilities.file_utils import Utils as utils
from tqdm import tqdm 
from tqdm import tqdm_notebook 


class BioASQDataset(Dataset):

    def __init__(self, dataset_type, tokenizer, notebook = False):

        self.data_path = utils.path_exists(os.path.join(utils.data_path, 'BioASQ'))
        self.dataset_type = dataset_type 
        self.use_test_dataset = use_test_dataset
        self.notebook = notebook

        self.data = self.load_data()
        self.ids = list(self.data.keys())
        self.tokenizer = tokenizer


    def load_data(self):

        files = list(glob.glob(os.path.join(self.data_path, 'BioASQ-{}-*.json'.format(self.dataset_type))))

        golden_files = [] if self.dataset_type == 'train' else {f.split('/')[-1].split('.')[0].split('_')[0].lower() : 
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

                    if qa['id'] in data: continue
                    if len(golden_files) > 0 and qa['id'].split('_')[0] not in id2qa: continue

                    data[qa['id']] = {'context': [], 'question': [], 'answers': []}
                    data[qa['id']]['question'] = qa['question']
                    data[qa['id']]['answers'] = qa['answers'] if 'answers' in qa else id2qa.get(qa['id'].split('_')[0], None)
                    data[qa['id']]['context'] = context

        return data


    def __len__(self):

        return len(self.ids)


    def find_answer_idx(self, answer_text, context):
        pass


    def __getitem__(self, index):

        qa_sample = self.data[self.ids[index]]

        question, answers, context = qa_sample['question'], qa_sample['answers'], qa_sample['context']

        if self.dataset_type == 'train':
            answer_idx = answers[0]['answer_start']
            answer_text = answers[0]['text']
        else:
            answer_text = answers 
            answer_idx = context.index(answer_text) if answer_text in context else -1

        input_encoded = self.tokenizer(question, context)


        return input_encoded, index, answer_idx, answer_text



