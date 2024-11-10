from typing import Union

import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class NerBaseBert(nn.Module):
    def __init__(self, num_classes, model_path, bert_freeze=True):
        super(NerBaseBert, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.bert_tokenizer = BertTokenizer.from_pretrained(model_path)
        if bert_freeze:
            print('bert 模型参数冻结')
            for i, (name, param) in enumerate(self.bert.named_parameters()):
                if name == 'encoder.layer.10.attention.self.query.weight':
                    break
                param.requires_grad = False
                print(f'{name}冻结')
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, x, mask=None):
        x = self.bert(
            input_ids=x,
            attention_mask=mask,
            output_attentions=False,  # attention层的结果不返回
            output_hidden_states=False,  # 每层的输出不返回
            return_dict=True  # 以字典形式返回
        )

        last_hidden_state = x['last_hidden_state']
        output = self.fc(last_hidden_state)
        return output

    def predict(self, text: Union[str, list], device):
        if isinstance(text, str):
            x_tokens, x_idx = self.text_preprocessing(text, device)
            mask = None
        else:
            x_tokens, x_idx, mask = self.text_preprocessing(text, device)

        output = self.forward(x=x_idx, mask=mask)
        output = torch.argmax(output, dim=-1)
        if isinstance(text, str):
            return ((x_tokens, output[0, 1:-1].cpu().tolist()),)
        if isinstance(text, list):
            preds = []
            for i in range(len(output)):
                unmasked_len = int(torch.sum(mask[i]).cpu().item())
                pred = output[i, 1:unmasked_len - 1]
                preds.append(pred.tolist())
            return tuple(zip(x_tokens, preds))

    def text_preprocessing(self, text: Union[str, list], device=torch.device('cpu')):
        if isinstance(text, str):
            x_tokens, x_idx = self.str_preprocessing(text)
            x_idx = [x_idx]
            x_idx = torch.tensor(x_idx, dtype=torch.long, device=device)
            return x_tokens, x_idx
        else:
            max_len = max([len(i) for i in text])
            x_tokens = []
            x_idx = []
            mask = []
            for t in text:
                x_token, idx, m = self.str_preprocessing(t, max_len + 2)
                x_tokens.append(x_token)
                x_idx.append(idx)
                mask.append(m)
            x_idx = torch.tensor(x_idx, dtype=torch.long, device=device)
            mask = torch.tensor(mask, dtype=torch.float32, device=device)
            return x_tokens, x_idx, mask

    def str_preprocessing(self, string: str, max_len=None):
        x_tokens = [s for s in string]
        x_tokens.insert(0, '[CLS]')
        x_tokens.append('[SEP]')
        x_idx = self.bert_tokenizer.convert_tokens_to_ids(x_tokens)
        if max_len is not None:
            string_len = len(x_tokens)
            padding_len = max_len - string_len
            x_idx.extend([0] * padding_len)
            mask = np.zeros(max_len)
            mask[:string_len] = 1
            return x_tokens[1:-1], x_idx, mask.tolist()
        else:
            return x_tokens[1:-1], x_idx


# if __name__ == '__main__':
#     net = NerBaseBert(13)
#     x_ = ['患者2008年9月3日因“腹胀，发现腹部包块”在我院腹科行手术探查，',
#           '术中见盆腹腔肿物，与肠管及子宫关系密切']
#     r = net.predict(
#         text=x_,
#         device=torch.device('cpu')
#     )
