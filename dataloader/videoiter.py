import os
import cv2
import numpy as np
from torchvision import transforms

from sampler import RandomSampling, SequentialSampling
import torch.utils.data as data
import logging
from PIL import Image
import torch

'''
这里本来是为直接读取视频准备的
现在不直接读取视频了，而是把视频先采样成图片,10FPS
class Video(object):
    """basic Video class"""

    def __init__(self, vid_path):
        self.open(vid_path)

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__del__()

    def reset(self):
        self.close()
        self.vid_path = None
        self.frame_count = -1
        self.faulty_frame = None
        return self

    def open(self, vid_path):
        assert os.path.exists(vid_path), "VideoIter:: cannot locate: `{}'".format(vid_path)

        # close previous video & reset variables
        self.reset()

        # try to open video
        cap = cv2.VideoCapture(vid_path)
        if cap.isOpened():
            self.cap = cap
            self.vid_path = vid_path
        else:
            raise IOError("VideoIter:: failed to open video: `{}'".format(vid_path))

        return self

    def count_frames(self, check_validity=False):
        offset = 0
        if self.vid_path.endswith('.flv'):
            offset = -1
        unverified_frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) + offset

        self.frame_count = unverified_frame_count
        assert self.frame_count > 0, \
            "VideoIter:: Video: `{}' has no frames".format(self.vid_path)
        return self.frame_count

    def extract_frames(self, idxs, force_color=True):
        assert self.cap is not None, 'No opened video!'
        if len(idxs) < 1:
            return []

        frames = []
        for idx in idxs:
            assert (self.frame_count < 0) or (idx > self.frame_count), \
                "idxs: {} > total valid frames({})".format(idxs, self.frame_count)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            res, frame = self.cap.read()

            pre_idx = idx
            if not res:
                self.faulty_frame = idx
                return None
            if len(frame.shape) < 3:
                if force_color:
                    # convert gray to BGR
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            else:  # convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        return frames

    def close(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            self.cap = None
        return self
'''
class Video(object):
    """basic Video class"""

    def __init__(self, vid_path):
        self.path=vid_path
        self.frame_count=0
        assert os.path.exists(self.path) is not None, 'No this video!'

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.path=None
        self.frame_count=0

    def count_frames(self):
        self.frame_count=len(os.listdir(self.path))
        assert self.frame_count > 0, \
            "VideoIter:: Video: `{}' has no frames".format(self.vid_path)
        return self.frame_count

    def extract_frames(self, idxs):

        frames = []
        for idx in idxs:
            try:
                img_path=self.path+'/%05d.jpg'%(idx+1)
                #print(img_path)
                frame=Image.open(img_path)
            except Exception as e:
                print('When open the image {} , occur the error {}'.format(img_path,e))
            frames.append(frame)
        return frames




class VideoIter(data.Dataset):

    def __init__(self,
                 video_prefix,
                 txt_list,
                 sampler=None,
                 video_transform=None,
                 name="UCF101",

                 cached_info_path=None,
                 return_item_subpath=False,
                 clips_num=1,
                 shuffle_list_seed=None,
                 ):
        super(VideoIter, self).__init__()
        # load params
        self.sampler = sampler  # 采样方式
        self.video_prefix = video_prefix  # video所在目录
        self.video_transform = video_transform
        self.return_item_subpath = return_item_subpath  # 是否返回video文件名,一般只在测试时使用
        self.clips_num = clips_num  # 每个video截取的clips数，测试时使用

        self.rng = np.random.RandomState(shuffle_list_seed if shuffle_list_seed else 0)
        # load video list
        self.video_list = self._get_video_list(video_prefix=video_prefix,
                                               txt_list=txt_list,
                                               cached_info_path=cached_info_path)

        self.rng.shuffle(self.video_list)
        self.crop_mean=torch.from_numpy(np.load('./tools/crop_mean.npy').transpose(0,3,1,2)/255.0)
        logging.info("VideoIter:: iterator initialized (phase: '{:s}', num: {:d})".format(name, len(self.video_list)))

    def _get_video_list(self,
                        video_prefix,
                        txt_list,
                        cached_info_path=None):

        assert os.path.exists(video_prefix), "VideoIter:: failed to locate: `{}'".format(video_prefix)
        assert os.path.exists(txt_list), "VideoIter:: failed to locate: `{}'".format(txt_list)

        cached_video_info = {}
        if cached_info_path:
            if os.path.exists(cached_info_path):
                f = open(cached_info_path, 'r')
                cached_video_prefix = f.readline().split()[1]
                cached_txt_list = f.readline().split()[1]
                if (cached_video_prefix == video_prefix.replace(" ", "")) and (
                        cached_txt_list == txt_list.replace(" ", "")):
                    print("VideoIter:: loading cached video info from: `{}'".format(cached_info_path))
                    lines = f.readlines()
                    for line in lines:
                        video_subpath, frame_count = line.split()
                        cached_video_info.update({video_subpath: int(frame_count)})
                else:
                    print(">> Cached video list mismatched: " +
                          "(prefix:{}, list:{}) != expected (prefix:{}, list:{})".format( \
                              cached_video_prefix, cached_txt_list, video_prefix, txt_list))
                f.close()

        # building dataset
        video_list = []
        new_video_info = {}
        count_less_16=0
        with open(txt_list) as f:
            lines = f.read().splitlines()
            logging.info("VideoIter:: found {} videos in `{}'".format(len(lines), txt_list))
            for i, line in enumerate(lines):
                v_id, label, video_subpath = line.split()
                video_subpath=video_subpath[:-4]
                video_path = os.path.join(video_prefix, video_subpath)
                if not os.path.exists(video_path):
                    print("VideoIter:: >> cannot locate `{}'".format(video_path))
                    continue

                if video_subpath in cached_video_info:
                    frame_count = cached_video_info[video_subpath]
                elif video_subpath in new_video_info:
                    frame_count = new_video_info[video_subpath]
                else:
                    self.video = Video(video_path)
                    frame_count = self.video.count_frames()
                    if frame_count<16:
                        count_less_16+=1
                        continue
                    new_video_info.update({video_subpath: frame_count})

                info = [int(v_id), int(label), video_subpath, frame_count]
                video_list.append(info)
        print('There are {} video,s frame counts being less 16'.format(count_less_16))
        # caching video list
        if cached_info_path and len(new_video_info) > 0:
            logging.info(
                "VideoIter:: adding {} lines new video info to: {}".format(len(new_video_info), cached_info_path))
            cached_video_info.update(new_video_info)
            with open(cached_info_path, 'w') as f:
                # head
                f.write("video_prefix: {:s}\n".format(video_prefix.replace(" ", "")))
                f.write("txt_list: {:s}\n".format(txt_list.replace(" ", "")))
                # content
                for i, (video_subpath, frame_count) in enumerate(cached_video_info.items()):
                    if i > 0:
                        f.write("\n")
                    f.write("{:s}\t{:d}".format(video_subpath, frame_count))

        return video_list
        # formate:
        # [vid, label, video_subpath, frame_count]

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        #测试时采样多个16帧的片段
        #训练时只采样一个片段
        clips_input = None
        if self.return_item_subpath:
            num = self.clips_num
        else:
            num = 1

        for i in range(num):
            succ = False
            while not succ:
                try:
                    clip_input, label, vid_subpath = self.getitem_from_raw_video(index)
                    if clips_input is None:
                        clips_input = clip_input.unsqueeze(0)
                    else:
                        clips_input = torch.cat((clips_input, clip_input.unsqueeze(0)), 0)
                    succ = True
                except Exception as e:
                    index = self.rng.choice(range(0, self.__len__()))
                    print("VideoIter:: ERROR!! (Force using another index:\n{})\n{}".format(index, e))

        if self.return_item_subpath:
            return clips_input.transpose(1, 2), label, vid_subpath
        else:
            return clip_input.transpose(0, 1), label

    def getitem_from_raw_video(self, index):
        v_id, label, vid_subpath, frame_count = self.video_list[index]
        video_path = os.path.join(self.video_prefix, vid_subpath)
        assert frame_count>16,'Video {} frame count < 16'

        faulty_frames = []
        successful_trial = False
        try:  # try to get clip that not faulty
            with Video(vid_path=video_path) as video:
                for i_trial in range(20):
                    sampled_idxs = self.sampler.sampling(range_max=frame_count, v_id=v_id, prev_failed=(i_trial > 0))
                    if set(list(sampled_idxs)).intersection(faulty_frames):
                        continue
                    sampled_frames = video.extract_frames(idxs=sampled_idxs)
                    if sampled_frames is None:
                        faulty_frames.append(video.faulty_frame)
                    else:
                        successful_trial = True
                        break
        except IOError as e:
            print(">> I/O error({0}): {1}".format(e.errno, e.strerror))

        # if can't get the right clip ,then throw out the video
        assert successful_trial, \
            "VideoIter:: >> frame {} is error & backup is inavailable. [{}]'".\
                format(faulty_frames,video_path)

        # now sampled_frames shape is (frame_nums,W,H,C)
        # need to convert to (C,frame_nums,W,H) for 3D convolution

        clip_input = None

        if self.video_transform is not None:
            # apply video augmentation
            for i in range(len(sampled_frames)):
                img = self.video_transform(sampled_frames[i])
                if clip_input is None:
                    clip_input = img.unsqueeze(0)
                else:
                    clip_input = torch.cat((clip_input, img.unsqueeze(0)), 0)

        return clip_input-self.crop_mean, label, vid_subpath


if __name__ == '__main__':
    interval = 2
    num = 16
    sampler = RandomSampling(num=num, interval=interval)
    traindataloader = VideoIter(
        video_prefix='../raw/data/',
        txt_list='../raw/list_cvt/trainlist01.txt',
        cached_info_path='../raw/cached_train_video_info.txt',
        sampler=sampler,
        return_item_subpath=False,
        video_transform=transforms.Compose([
            transforms.Resize((128, 171)),
            transforms.CenterCrop((112, 112)),
            transforms.ToTensor(),
        ])
    )
    sampler = SequentialSampling(num=num, interval=interval)
    testdataloader = VideoIter(
        video_prefix='../raw/data/',
        txt_list='../raw/list_cvt/testlist01.txt',
        cached_info_path='../raw/cached_test_video_info.txt',
        sampler=sampler,
        return_item_subpath=True,
        clips_num=5,
        video_transform=transforms.Compose([
            transforms.Resize((128, 171)),
            transforms.CenterCrop((112, 112)),
            transforms.ToTensor(),
        ])
    )
    print(traindataloader[0][0].shape)
    print(testdataloader[0][0].shape)
'''
training 01
min frame number 29 from vid 9296
max frame number 1776 from vid 3173
'''
