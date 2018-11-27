

import dataloader
import time
from utils import *
from network import *
from torchvision import  transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class mymodel(object):

    def __init__(self, net,
                 train_loader=None, test_loader=None,
                 epoch_nums=None,
                 checkpoint_path=None,
                 test_clips=3
                 ):
        self.net = net.to(device)

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.epoch_nums = epoch_nums
        self.checkpoint_path = checkpoint_path
        self.test_clips = test_clips

        self.epoch = 0
        self.best_prec1 = 0

    def load_state(self, state_dict, strict=False):
        print('start loading state_dict...')
        # if load state strict then must match every parameters between the net and the checkpoint
        # else can load the paramerters that matched and throw the parameters that don't matched

        if strict == True:
            self.net.load_state_dict(state_dict=state_dict)
        else:
            net_state_keys = list(self.net.state_dict().keys())
            not_matched_params = []
            for name, param in state_dict.items():
                print(name,param.shape)
                if name in self.net.state_dict().keys():
                    dst_param_shape = self.net.state_dict()[name].shape
                    if param.shape == dst_param_shape:
                        self.net.state_dict()[name].copy_(param)
                        net_state_keys.remove(name)
                    else:
                        not_matched_params.append(name)
                else:
                    not_matched_params.append(name)

            if not_matched_params:
                print('Failed to load {}'.format(not_matched_params))
            if net_state_keys:
                print('lack {} to load '.format(not_matched_params))

        print('load state_dict succeed...')
        return True

    def load_checkpoint(self, load_path, optimizer=False):
        assert os.path.exists(load_path), \
            'Failed to load {},file not exists'.format(load_path)
        checkpoint = torch.load(load_path)
        if 'state_dict' in checkpoint.keys():
            all_parmas_matched = self.load_state(checkpoint['state_dict'])
        else:
            all_parmas_matched = self.load_state(checkpoint)

        assert all_parmas_matched, 'Failed to load state_dict'

        if optimizer:
            if 'optimizer' in checkpoint.keys():
                optimizer.load_state_dict(checkpoint['optimizer'])
                print('optimizer load...')
            else:
                print(
                    'Failed to load optimizer,there is no optimizer in {}'.format(load_path)
                )

        if 'epoch' in checkpoint.keys():
            self.epoch = checkpoint['epoch']

    def save_checkpoint(self, is_best=False):
        save_path = self.checkpoint_path + "C3D_at_epoch{}.pth".format(self.epoch)

        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

        if is_best == True:
            save_path = self.checkpoint_path + "C3D_best_model.pth"
            torch.save(
                {
                    'epoch': self.epoch,
                    'state_dict': self.net.state_dict(),
                    'best_prec1': self.best_prec1,
                    'optimizer': self.optimizer.state_dict()
                }, save_path)
            print(
                'best model at {} opech has been saved to {}'.format(self.epoch, save_path)
            )
            return

        torch.save({
            'epoch': self.epoch,
            'state_dict': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, save_path)
        print('checkpoint (model & optimizer) has been saved to {}'.format(save_path))

    def _prepare(self):
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.003, momentum=0.9, weight_decay=0.0005)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        self.criterion = nn.CrossEntropyLoss()

    def run(self):
        self._prepare()

        for epoch in range(self.epoch, self.epoch_nums):

            self.epoch += 1
            print('Epoch:[{0}/{1}]\n[training stage]'.format(self.epoch, self.epoch_nums))
            self.train_1epoch()

            # validate after every training epoch
            prec1, val_loss = self.validate_1epoch()
            is_best = prec1 > self.best_prec1

            # update optimizer scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            self.save_checkpoint(is_best=False)
            # save model
            if is_best:
                self.best_prec1 = prec1
                self.save_checkpoint(is_best=True)

    def train_1epoch(self):

        # switch to train mode
        self.net.train()
        end = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        with torch.set_grad_enabled(True):
            # mini-batch training
            for i, (data, label) in enumerate(self.train_loader):
                # measure the data loading time
                data_time.update(time.time() - end)

                data = data.to(device)
                label = label.to(device)
                outputs = self.net(data)
                loss = self.criterion(outputs, label)

                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # measure accuracy and record loss
                prec1, prec5 = accuracy(outputs.data, label, topk=(1, 5))
                losses.update(loss.item(), data.size(0))
                top1.update(prec1.item(), data.size(0))
                top5.update(prec5.item(), data.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

        info = {'Epoch': [self.epoch],
                'Batch Time': [round(batch_time.avg, 3)],
                'Epoch Time': [round(batch_time.sum, 3)],
                'Data Time': [round(data_time.avg, 3)],
                'Loss': [round(losses.avg, 5)],
                'Prec@1': [round(top1.avg, 4)],
                'Prec@5': [round(top5.avg, 4)],
                'lr': self.optimizer.param_groups[0]['lr']
                }
        record_info(info, 'record/train.csv', 'train')

    def validate_1epoch(self):
        # switch to evaluate mode
        self.net.eval()
        end = time.time()

        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        with torch.set_grad_enabled(False):
            for i, (data, label, video_subpath) in enumerate(self.test_loader):
                # during test,there are more than one clips
                # an use thier avgrage scores as the final scores 

                preds = torch.zeros(label.shape[0], 101).to(device)
                label = label.to(device)
                for j in range(self.test_clips):
                    data_j = data[:, j, :, :, :, :].to(device)
                    # (batch_size,num_clips,C,F,W,H)

                    # compute output
                    output = self.net(data_j)

                    # sum the clips preds scores
                    preds += output

                batch_time.update(time.time() - end)
                end = time.time()
                prec1, prec5 = accuracy(preds, label, topk=(1, 5))
                loss = self.criterion(preds, label)

                losses.update(loss.item(), data.size(0))
                top1.update(prec1.item(), data.size(0))
                top5.update(prec5.item(), data.size(0))

        info = {'Epoch': [self.epoch],
                'Batch Time': [round(batch_time.avg, 3)],
                'Epoch Time': [round(batch_time.sum, 3)],
                'Loss': [round(losses.avg, 5)],
                'Prec@1': [round(top1.avg, 3)],
                'Prec@5': [round(top5.avg, 3)]}
        record_info(info, 'record/test.csv', 'test')
        return top1.avg, losses.avg


if __name__ == '__main__':

    sampler = dataloader.SequentialSampling(num=16, interval=1)
    testdata = dataloader.VideoIter(
        video_prefix='./raw/data/',
        txt_list='./raw/list_cvt/testlist01.txt',
        cached_info_path='./raw/cached_test_video_info.txt',
        sampler=sampler,
        return_item_subpath=True,
        clips_num=3,
        name='test',
        video_transform=transforms.Compose([
            transforms.Resize((128, 171)),
            transforms.CenterCrop((112, 112)),
            transforms.ToTensor(),
        ])
    )
    test_iter = torch.utils.data.DataLoader(dataset=testdata, batch_size=16, num_workers=8)

    sampler = dataloader.RandomSampling(num=16, interval=1)
    traindata = dataloader.VideoIter(
        video_prefix='./raw/data/',
        txt_list='./raw/list_cvt/trainlist01.txt',
        cached_info_path='./raw/cached_train_video_info.txt',
        sampler=sampler,
        return_item_subpath=False,
        clips_num=1,
        name='train',
        video_transform=transforms.Compose([
            transforms.Resize((128, 171)),
            transforms.RandomCrop((112, 112)),
            transforms.ToTensor(),
        ])
    )
    train_iter = torch.utils.data.DataLoader(dataset=traindata, batch_size=16, num_workers=8)

    net = C3D()
    model = mymodel(net, test_loader=test_iter,
                    train_loader=train_iter,
                    epoch_nums=50,
                    checkpoint_path='./record/')
    model.load_checkpoint('./c3d.pickle')
    '''
    model.load_checkpoint('./c3d.pickle')
    fc8weight=model.net.fc8.weight[0:404:4]
    fc8bias = model.net.fc8.bias[0:404:4]
    model.net.fc8 = nn.Linear(4096, 101)
    model.net.fc8.weight.data=fc8weight
    model.net.fc8.bias.data=fc8bias
    '''

    #model.run()
"""
References
----------
[1] Tran, Du, et al. "Learning spatiotemporal features with 3d convolutional networks." 
Proceedings of the IEEE international conference on computer vision. 2015.
"""
