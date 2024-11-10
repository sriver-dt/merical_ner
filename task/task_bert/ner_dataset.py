import json
import os
from typing import TypeAlias, Union
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertConfig

FILE_LIKE: TypeAlias = Union[str, os.PathLike, Path]


class NerDataset(Dataset):
    def __init__(self, datas, targets):
        super(NerDataset, self).__init__()
        self.datas = datas
        self.targets = targets

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        data = self.datas[item]
        target = []
        for i, flag in enumerate(self.targets[item]):
            if flag[0] == 'B':
                target.append([i, i, flag[2:]])
            elif flag[0] == 'I':
                target[-1][1] = i
        return data, target


class CustomDataloader:
    def __init__(self, data_dir, bert_path, batch_size):
        self.class2idx = None
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.bert_configs = BertConfig.from_pretrained(bert_path)
        self.bert_max_len = self.bert_configs.max_position_embeddings
        
        self.token_cls = '[CLS]'
        self.token_sep = '[SEP]'

    def data_target_split(self, data_path):
        entity_categories = set()
        text_datas = []
        text = [self.token_cls, ]
        entity_targets = []
        text_token_entity = [self.token_cls, ]
        with data_path.open('r', encoding='utf-8') as file:
            for line in file.readlines():
                if line != '\n':
                    token, category = line.strip().split(' ')
                    text.append(token)
                    text_token_entity.append(category)
                    if category != 'O':
                        entity_categories.add(category)

                elif line == '\n':
                    # 超过最大长度做截断
                    if len(text) >= self.bert_max_len:
                        text = text[: self.bert_max_len - 1]
                        text_token_entity = text_token_entity[: self.bert_max_len - 1]
                    text.append(self.token_sep)
                    text_token_entity.append(self.token_sep)
                    text_datas.append(text)
                    entity_targets.append(text_token_entity)
                    text = [self.token_cls, ]
                    text_token_entity = [self.token_sep, ]
            entity_categories = list(entity_categories)
            entity_categories.insert(0, 'O')
        return text_datas, entity_targets, entity_categories

    @staticmethod
    def sequence_padding(sequence, max_len):
        masks = []
        for text in sequence:
            mask = np.ones(max_len)
            current_len = len(text)
            padding_length = max_len - current_len
            text.extend([0] * padding_length)
            mask[current_len:] = 0
            masks.append(mask.tolist())
        return sequence, masks

    def collate_fn(self, batch):
        batch_data, batch_target = zip(*batch)
        batch_y = []
        batch_token2id = []
        length = []
        for i in range(len(batch_data)):
            tokens = batch_data[i]
            tokens2id = self.bert_tokenizer.convert_tokens_to_ids(tokens)
            current_len = len(tokens2id)
            target = np.zeros(current_len)
            target[[0, -1]] = -1  # 将 '<CLS>' 和 '<SEP>' 标签类别设为-1
            if batch_target[i]:
                for start, end, flag in batch_target[i]:
                    target[start] = self.class2idx['B-' + flag]
                    target[start + 1: end + 1] = self.class2idx['I-' + flag]
            batch_token2id.append(tokens2id)
            batch_y.append(target.tolist())
            length.append(current_len)
        batch_max_len = max(length)
        batch_y, target_mask = self.sequence_padding(batch_y, max_len=batch_max_len)
        batch_y = torch.tensor(batch_y, dtype=torch.long)
        target_mask = torch.tensor(target_mask, dtype=torch.long)
        batch_y = batch_y + (target_mask - 1)  # 将填充部分标签置为-1
        batch_token2id, masks = self.sequence_padding(batch_token2id, max_len=batch_max_len)
        batch_token2id = torch.tensor(batch_token2id, dtype=torch.long)
        masks = torch.tensor(masks, dtype=torch.float32)
        return (batch_token2id, masks), batch_y

    def get_dataloader(self):
        # train_text_datas, train_entity_targets, entity_category = self.data_target_split(
        #     self.data_dir.joinpath('min_example.train')
        # )
        # test_text_datas, test_entity_targets, _ = self.data_target_split(self.data_dir.joinpath('min_example.dev'))

        train_text_datas, train_entity_targets, entity_category = self.data_target_split(
            self.data_dir.joinpath('example.train')
        )
        test_text_datas, test_entity_targets, _ = self.data_target_split(self.data_dir.joinpath('example.dev'))

        # 保存类别
        with open(self.data_dir.joinpath('entity_categories.json'), 'w', encoding='utf-8') as file:
            json.dump(entity_category, file, indent=4)
        self.class2idx = {entity_category[i]: i for i in range(len(entity_category))}

        train_dataset = NerDataset(train_text_datas, train_entity_targets)
        test_dataset = NerDataset(test_text_datas, test_entity_targets)

        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=True,
        )

        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size * 2,
            collate_fn=self.collate_fn,
            shuffle=False,
        )

        return train_dataloader, test_dataloader, len(entity_category)
