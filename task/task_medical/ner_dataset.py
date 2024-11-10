import json
import math
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
        target = self.targets[item]
        return data, target


class CustomDataloader:
    def __init__(self, data_dir, output_path, bert_path, batch_size, truncation=True):
        self.data_dir = data_dir
        self.output_path = output_path
        self.batch_size = batch_size
        self.truncation = truncation
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.bert_configs = BertConfig.from_pretrained(bert_path)
        self.bert_max_len = self.bert_configs.max_position_embeddings

        self.token_cls = '[CLS]'
        self.token_sep = '[SEP]'

        self.class2idx = None

    def data_target_split(self, datapath):
        text_datas = []
        entity_targets = []
        categories = set()
        with datapath.open('r', encoding='utf-8') as file:
            for line in file.readlines():
                line_data = json.loads(line)
                if self.truncation:
                    # 超过最大长度直接截断
                    # 获取实体数据
                    entities = []
                    for entity in line_data['entities']:
                        start_pos = entity['start_pos']
                        end_pos = entity['end_pos']
                        flag = entity['label_type'].strip()
                        categories.add(flag)
                        entities.append([start_pos, end_pos, flag])
                    entity_targets.append(entities)

                    # 获取文本数据
                    text = [self.token_cls, ]
                    # 将部分符号做替换
                    original_text = line_data['originalText'].lower()
                    for old, new in [('”', '"'), ("“", '"'), ("’", "'"), ("‘", "'"), ("`", "'"), ('—', '-')]:
                        original_text = original_text.replace(old, new)
                    for token in original_text:
                        text.append(token)
                    if len(text) >= self.bert_max_len:
                        text = text[: self.bert_max_len - 1]
                    text.append(self.token_sep)
                    text_datas.append(text)

                else:
                    # 分桶
                    text_len = len(line_data['originalText'])
                    buckets = math.ceil(text_len / (self.bert_max_len - 2))
                    bucket_len = text_len // buckets

                    entities_dict = sorted(line_data['entities'],
                                           key=lambda item: (item['start_pos'], -item['end_pos']))
                    si = 0
                    ei = bucket_len
                    each_slice = [[si, ei], ]
                    entities = []
                    for entity in entities_dict:
                        start_pos = entity['start_pos']
                        end_pos = entity['end_pos']
                        flag = entity['label_type']
                        categories.add(flag)

                        if start_pos >= bucket_len:
                            sub_buckets = (start_pos - si) // bucket_len
                            for _ in range(sub_buckets):
                                entity_targets.append(entities)
                                entities = []
                                si = ei
                                ei = (si + bucket_len) if (si + bucket_len) <= text_len else text_len
                                each_slice.append([si, ei])

                        if end_pos > ei:
                            entity_targets.append(entities)
                            entities = []
                            si = start_pos
                            ei = (si + bucket_len) if (si + bucket_len) <= text_len else text_len
                            each_slice[-1][-1] = si
                            each_slice.append([si, ei])
                        entities.append([start_pos - si, end_pos - si, flag])
                    entity_targets.append(entities)

                    original_text = line_data['originalText'].lower()
                    for old, new in [('”', '"'), ("“", '"'), ("’", "'"), ("‘", "'"), ("`", "'"), ('—', '-')]:
                        original_text = original_text.replace(old, new)
                    for s, e in each_slice:
                        text = [self.token_cls, ]
                        tokens = [t for t in original_text[s: e]]
                        text.extend(tokens)
                        text.append(self.token_sep)
                        text_datas.append(text)

        return text_datas, entity_targets, list(categories)

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
                    if end <= self.bert_max_len - 2:
                        target[start + 1] = self.class2idx['B-' + flag]
                        target[start + 2: end + 1] = self.class2idx['I-' + flag]
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
        train_text_datas, train_entity_targets, categories = self.data_target_split(
            self.data_dir.joinpath('training.txt'))
        test_text_datas, test_entity_targets, _ = self.data_target_split(self.data_dir.joinpath('test.json'))

        self.class2idx = {'O': 0}
        index = 1
        for label in categories:
            self.class2idx['B-' + label] = index
            self.class2idx['I-' + label] = index + 1
            index += 2
        with open(os.path.join(self.output_path, 'class2idx.json'), 'w', encoding='utf-8') as writer:
            json.dump(self.class2idx, writer, ensure_ascii=False, indent=4)

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

        return train_dataloader, test_dataloader, len(self.class2idx)
