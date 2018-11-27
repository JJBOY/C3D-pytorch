import argparse
import dataloader
from network import *
from torchvision import  transforms
from model import mymodel

parser = argparse.ArgumentParser(description="PyTorch Video Classification Parser")
parser.add_argument('--train_or_test', type=bool, default=False,
                    help="train if True and test is False")

if __name__ == '__main__':
    args = parser.parse_args()
    sampler = dataloader.RandomSampling(num=16, interval=1)
    testdata = dataloader.VideoIter(
        video_prefix='./raw/data/',
        txt_list='./raw/list_cvt/testlist01.txt',
        cached_info_path='./raw/cached_test_video_info.txt',
        sampler=sampler,
        return_item_subpath=True,
        clips_num=1,
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
    net.init_weight()
    model = mymodel(net, test_loader=test_iter,
                    train_loader=train_iter,
                    epoch_nums=40,
                    checkpoint_path='./record/',
                    test_clips=1)
    model.run()