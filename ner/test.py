import os
from pathlib import Path
import warnings

import torch
import torch.nn as nn

from task.task_bert.ner_dataset import CustomDataloader
from task.task_bert.net import NerBaseBert
from train import Trainer
from optimizer import get_optimizer, get_scheduler

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

FILE_ROOT_DIR = os.path.dirname(__file__)
DATA_PATH = Path(os.path.join(FILE_ROOT_DIR, '../datas/china-people-daily-ner-corpus'))
BERT_PATH = Path(r'C:\Users\du\.cache\huggingface\hub\hub\bert-base-chinese')


def training():
    save_model_dir = os.path.join(FILE_ROOT_DIR, '../output/test')
    os.makedirs(save_model_dir, exist_ok=True)

    total_epoch = 5
    batch_size = 8
    lr = 0.00001
    bert_freeze = True
    device = 'cuda'
    early_stop = True
    early_stop_step = 5

    example_input = (
        torch.randint(0, 100, size=(2, 20)).to(dtype=torch.long),
        torch.rand(size=(2, 20)).to(dtype=torch.float32)
    )

    custom_dataloader = CustomDataloader(
        data_dir=DATA_PATH,
        bert_path=BERT_PATH,
        batch_size=batch_size,
    )
    train_dataloader, test_dataloader, num_classes = custom_dataloader.get_dataloader()

    net = NerBaseBert(num_classes=num_classes, model_path=BERT_PATH, bert_freeze=bert_freeze)
    optimizer = get_optimizer(net=net, lr=lr, optim_name='adamw')
    scheduler = get_scheduler(optimizer)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

    trainer = Trainer(
        net=net,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fc=loss_fn,
        optim=optimizer,
        scheduler=scheduler,
        total_epoch=total_epoch,
        save_model_dir=save_model_dir,
        example_input=example_input,
        eval_metrics=None,
        device=device,
        early_stop=early_stop,
        early_stop_step=early_stop_step
    )

    trainer.fit()


if __name__ == '__main__':
    training()
