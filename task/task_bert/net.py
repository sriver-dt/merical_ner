import torch.nn as nn
from transformers import BertModel


class NerBaseBert(nn.Module):
    def __init__(self, num_classes, model_path, bert_freeze=True):
        super(NerBaseBert, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)
        if bert_freeze:
            print('bert 模型参数冻结')
            for i, (name, param) in enumerate(self.bert.named_parameters()):
                if name == 'encoder.layer.10.attention.self.query.weight':
                    break
                param.requires_grad = False
                print(f'{name}冻结')
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, x, mask):
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
