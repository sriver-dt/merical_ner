import atexit
import logging
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


def get_accuracy():
    def accuracy(*args):
        acc = []
        for s, c in args:
            try:
                acc.append(c / s)
            except ZeroDivisionError:
                acc.append(c / (s + 1e-8))
        return acc

    return accuracy


def is_empty(path: Path):
    try:
        next(path.iterdir())
        return False
    except StopIteration:
        return True


class Trainer:
    def __init__(self, net: nn.Module, train_dataloader, test_dataloader, loss_fc, optim, scheduler,
                 total_epoch, save_model_dir: str, example_input, eval_metrics=None,
                 early_stop=True, early_stop_step=5, device='cpu'
                 ):
        super(Trainer, self).__init__()
        self.net = net
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.loss_fn = loss_fc
        self.optim = optim
        self.scheduler = scheduler
        self.device = torch.device('cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu')
        self.total_epoch = total_epoch
        self.eval_metric = eval_metrics if eval_metrics is not None else get_accuracy()
        self.best_score = 0
        self.start_epoch = 0
        self.save_model_dir = Path(save_model_dir)
        self.early_stop = early_stop
        self.early_stop_step = early_stop_step

        # 可视化
        self.summary_writer = SummaryWriter(log_dir=str(self.save_model_dir.joinpath('summary')))
        self.summary_writer.add_graph(self.net, input_to_model=example_input)
        self.global_step = 0
        atexit.register(self.close)

        # 模型恢复
        model_path_dict = {}
        if not self.save_model_dir.exists():
            self.save_model_dir.mkdir(parents=True)
        if self.save_model_dir.exists() and not is_empty(self.save_model_dir):
            model_path_dict = {model_path.parts[-1]: model_path for model_path in self.save_model_dir.iterdir()}
        if 'best.pkl' in model_path_dict.keys():
            model_path = model_path_dict['best.pkl']
        elif 'last.pkl' in model_path_dict.keys():
            model_path = model_path_dict['last.pkl']
        else:
            model_path = None
        if model_path is not None:
            model_state_dict = torch.load(f=model_path, map_location=torch.device('cpu'))
            missing_keys, unexpected_keys = self.net.load_state_dict(model_state_dict['model_state'])
            self.best_score = model_state_dict['best_score']
            self.start_epoch = model_state_dict['epoch']
            self.total_epoch += self.start_epoch
            print('正在进行模型恢复')
            print(f'missing_keys: {missing_keys}')
            print(f'unexpected_keys : {unexpected_keys}')

    def close(self):
        logging.info("close resources....")
        self.summary_writer.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __enter__(self):
        return self

    def fit(self):
        early_stop_count = 0
        self.net.to(device=self.device)
        for epoch in range(self.start_epoch, self.total_epoch):
            self.train(epoch)
            current_score = self.eval(epoch)
            if self.scheduler is not None:
                self.scheduler.step()
            self.save(epoch=epoch, name='last')
            if current_score >= self.best_score:
                self.best_score = current_score
                self.save(epoch=epoch, name='best')
                early_stop_count = 0
                continue
            early_stop_count += 1
            if early_stop_count == self.early_stop_step and self.early_stop:
                logging.info(f'提前停止：-----epoch {epoch}-----  best_score {self.best_score}')
                break

    def train(self, epoch):
        self.net.train()

        pbar = tqdm(range(len(self.train_dataloader)), desc='training...')
        for batch_idx, ((batch_x, batch_mask), batch_y) in enumerate(self.train_dataloader):
            self.global_step += 1
            batch_x = batch_x.to(device=self.device)
            batch_mask = batch_mask.to(device=self.device)
            batch_y = batch_y.to(device=self.device)
            output = self.net(batch_x, batch_mask)
            self.optim.zero_grad()
            # 损失处理
            loss = self.loss_fn(torch.permute(output, dims=(0, 2, 1)), batch_y)
            entity_loss = batch_y.gt(0).to(dtype=torch.long) * loss  # 实体损失
            other_loss = (1 - batch_y.gt(0).to(dtype=torch.long)) * loss  # 非实体损失
            final_loss = torch.sum(entity_loss + 0.6 * other_loss) / (torch.sum(batch_mask) - 2 * batch_x.shape[0])
            final_loss.backward()
            self.optim.step()

            pred = torch.argmax(output, dim=-1)

            # 整体预测正确统计
            sum_total = torch.numel(batch_y)
            corrects_total = torch.sum(pred.eq(batch_y)).cpu().item()

            # 非填充部分统计
            sum_unmask = torch.sum(batch_mask).cpu().item()
            corrects_unmask = torch.sum(pred.eq(batch_y) * batch_mask).cpu().item()

            # 实体部分统计
            sum_entity = torch.sum(batch_y.gt(0)).item()
            corrects_entity = torch.sum(pred.eq(batch_y) * batch_y.gt(0)).item()

            # 非实体部分统计
            sum_other = sum_unmask - sum_entity
            corrects_other = corrects_unmask - corrects_entity

            acc_total, acc_unmask, acc_entity, acc_other = self.eval_metric(
                (sum_total, corrects_total),
                (sum_unmask, corrects_unmask),
                (sum_entity, corrects_entity),
                (sum_other, corrects_other)
            )

            # 进度条信息
            pbar.set_description(f'train epoch:{epoch+1}/{self.total_epoch}')
            pbar.set_postfix(
                loss=round(final_loss.item(), 5),
                acc_unmask=round(acc_unmask, 3),
                acc_entity=round(acc_entity, 3),
                acc_other=round(acc_other, 3),
            )
            pbar.update(1)

            # 可视化
            self.summary_writer.add_scalar('loss', final_loss.item(), global_step=self.global_step)
            self.summary_writer.add_scalar('train_acc_total', acc_total, global_step=self.global_step)
            self.summary_writer.add_scalar('train_acc_unmask', acc_unmask, global_step=self.global_step)
            self.summary_writer.add_scalar('train_acc_entity', acc_entity, global_step=self.global_step)
            self.summary_writer.add_scalar('train_acc_other', acc_other, global_step=self.global_step)
            # if (batch_idx + 1) % 5 == 0:
            #     print(f'epoch: {epoch}/{self.total_epoch - 1} '
            #           f'{100.0 * (batch_idx + 1) / len(self.train_dataloader):.2f}% '
            #           f'loss: {loss.item():.5f} '
            #           f'acc_unmask: {acc_unmask:.3f} '
            #           f'acc_entity: {acc_entity:.3f} '
            #           f'acc_other: {acc_other:.3f}')

    def eval(self, epoch):
        self.net.eval()

        pbar = tqdm(range(len(self.test_dataloader)), desc='eval...')

        corrects_total, sum_total = 0, 0
        corrects_unmask, sum_unmask = 0, 0
        corrects_entity, sum_entity = 0, 0
        with torch.no_grad():
            for (batch_x, batch_mask), batch_y in self.test_dataloader:
                batch_x = batch_x.to(device=self.device)
                batch_mask = batch_mask.to(device=self.device)
                batch_y = batch_y.to(device=self.device)
                output = self.net(batch_x, batch_mask)

                pred = torch.argmax(output, dim=-1)

                sum_total = torch.numel(batch_y)
                corrects_total = torch.sum(pred.eq(batch_y)).cpu().item()

                sum_unmask += torch.sum(batch_mask).cpu().item()
                corrects_unmask += torch.sum(pred.eq(batch_y) * batch_mask).cpu().item()

                sum_entity += torch.sum(batch_y.gt(0)).item()
                corrects_entity += torch.sum(pred.eq(batch_y) * batch_y.gt(0)).item()

                sum_other = sum_unmask - sum_entity
                corrects_other = corrects_unmask - corrects_entity

                # 进度条信息
                pbar.update(1)

            acc_total, acc_unmask, acc_entity, acc_other = self.eval_metric(
                (sum_total, corrects_total),
                (sum_unmask, corrects_unmask),
                (sum_entity, corrects_entity),
                (sum_other, corrects_other)
            )

            print(f'eval epoch: {epoch} '
                  f'acc_total: {acc_total:.4f} '
                  f'acc_unmask: {acc_unmask:.4f} '
                  f'acc_entity: {acc_entity:.4f} '
                  f'acc_other: {acc_other:.4f}')

            self.summary_writer.add_scalar('eval_acc_total', acc_total, global_step=epoch)
            self.summary_writer.add_scalar('eval_acc_unmask', acc_unmask, global_step=epoch)
            self.summary_writer.add_scalar('eval_acc_entity', acc_entity, global_step=epoch)
            self.summary_writer.add_scalar('eval_acc_other', acc_other, global_step=epoch)
        return acc_entity

    def save(self, epoch, name):
        state_dict = {
            'epoch': epoch,
            'best_score': self.best_score,
            'model_state': self.net.state_dict()
        }
        torch.save(state_dict, f=self.save_model_dir.joinpath(f'{name}.pkl'))
