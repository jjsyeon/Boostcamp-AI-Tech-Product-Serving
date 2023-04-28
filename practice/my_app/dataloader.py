import os
import re
import pandas as pd

import pytorch_lightning as pl
import torch
import transformers
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from tqdm.auto import tqdm

# import utils

class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=[]):
        self.inputs = inputs
        self.targets = targets

    # 학습 및 추론 과정에서 데이터를 1개씩 꺼내오는 곳
    def __getitem__(self, idx):
        # 정답이 있다면 else문을, 없다면 if문을 수행합니다
        if len(self.targets) == 0:
            return torch.tensor(self.inputs[idx])
        else:
            return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])

    # 입력하는 개수만큼 데이터를 사용합니다
    def __len__(self):
        return len(self.inputs)


class Dataloader(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, shuffle, train_path, dev_path, test_path, predict_path, additional_preprocessing):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.additional_preprocessing = additional_preprocessing

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None
        
    
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, max_length=100)
        self.add_tokens = ["<PERSON>"]
        self.num_added_toks = self.tokenizer.add_tokens(self.add_tokens)

        self.target_columns = ['label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']
        
    def len_tokenizer(self):
        return len(self.tokenizer)
        

    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
            text = '[SEP]'.join([item[text_column] for text_column in self.text_columns])
            outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True)
            data.append(outputs['input_ids'])

        return data

    def preprocessing(self, data):
        # 안쓰는 컬럼을 삭제합니다.
        data = data.drop(columns=self.delete_columns)

        # 타겟 데이터가 없으면 빈 배열을 리턴합니다.
        try:
            targets = data[self.target_columns].values.tolist()
            
        except:
            targets = []
            
        # 텍스트 데이터를 전처리합니다.
        if self.additional_preprocessing:        
            sentence1 = data[self.text_columns[0]].values.tolist()
            sentence2 = data[self.text_columns[1]].values.tolist()
            
            data[self.text_columns[0]] = utils.preprop_sent(sentence1)
            data[self.text_columns[1]] = utils.preprop_sent(sentence2)
        
        inputs = self.tokenizing(data)

        return inputs, targets

    def setup(self, stage='fit'):
        if stage == 'fit':
            # 학습 데이터와 검증 데이터셋을 호출합니다
            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)

            # 학습데이터 준비
            train_inputs, train_targets = self.preprocessing(train_data)

            # 검증데이터 준비
            val_inputs, val_targets = self.preprocessing(val_data)

            # train 데이터만 shuffle을 적용해줍니다, 필요하다면 val, test 데이터에도 shuffle을 적용할 수 있습니다
            self.train_dataset = Dataset(train_inputs, train_targets)
            self.val_dataset = Dataset(val_inputs, val_targets)
        else:
            # 평가데이터 준비
            test_data = pd.read_csv(self.test_path)
            test_inputs, test_targets = self.preprocessing(test_data)
            self.test_dataset = Dataset(test_inputs, test_targets)

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs, predict_targets = self.preprocessing(predict_data)
            self.predict_dataset = Dataset(predict_inputs, [])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)

class stDataloader(pl.LightningDataModule):
    def __init__(self, sentence1, sentence2):
        super().__init__()
        self.model_name = 'snunlp/KR-ELECTRA-discriminator'
        self.batch_size = 1
        self.shuffle = False

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None
        
        self.sentence1, self.sentence2 = sentence1, sentence2
    
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name, max_length=100)
        self.add_tokens = ["<PERSON>"]
        self.num_added_toks = self.tokenizer.add_tokens(self.add_tokens)

        self.target_columns = ['label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']

    def len_tokenizer(self):
        return len(self.tokenizer)
        

    def tokenizing(self):

        def preprop_sent(sentence_lst):
            sentence_lst = [re.sub('!!+', '!!', sentence) for sentence in sentence_lst]
            sentence_lst = [re.sub('\?+' ,'?', sentence) for sentence in sentence_lst]
            sentence_lst = [re.sub('~+', '~', sentence) for sentence in sentence_lst]
            sentence_lst = [re.sub('\.\.+', '...', sentence) for sentence in sentence_lst]
            sentence_lst = [re.sub('ㅎㅎ+', 'ㅎㅎ', sentence) for sentence in sentence_lst]
            sentence_lst = [re.sub('ㅋㅋ+', 'ㅋㅋㅋ', sentence) for sentence in sentence_lst]
            sentence_lst = [re.sub('ㄷㄷ+', 'ㄷㄷ', sentence) for sentence in sentence_lst]
            sentence_lst = [re.sub('…', '...', sentence) for sentence in sentence_lst]
            return sentence_lst
        
        tokenizer = transformers.AutoTokenizer.from_pretrained('snunlp/KR-ELECTRA-discriminator', max_length=100)
        sentences = preprop_sent([self.sentence1, self.sentence2])
        text = '[SEP]'.join([sent for sent in sentences])

        data = tokenizer(text, add_special_tokens=True, truncation=True)
        
        return [data['input_ids']]
        
    def preprocessing(self):
        targets = []
        
        inputs = self.tokenizing()
        return inputs, targets

    def setup(self, stage='fit'):
        # if stage == 'fit':
        #     # 평가데이터 준비
        #     test_inputs, test_targets = self.tokenizing(test_data)
        #     self.test_dataset = Dataset(test_inputs, test_targets)

        predict_inputs, predict_targets = self.preprocessing()
        self.predict_dataset = Dataset(predict_inputs, [])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)