# C3D  pytorch

This is an impelementaion of C3D using Pytorch on UCF101 dataset.

I try both training from scratch and fine-fune the pre-train model on the sport-1M provided by [C3D-caffe][1].However,the results are not so good,both about 30% top5 accuracy which is faraway from the paper [C3D: Generic Features for Video Analysis][2].I will be appreciate if anybody can give me some advice to improve the accuracy.

## Dataset

I trained the model on [UCF101][3].

For convenience, I sampled pictures from the raw video 10FPS using the ./raw/video2img.sh


[1]: https://github.com/facebook/C3D
[2]: http://vlg.cs.dartmouth.edu/c3d/
[3]: http://crcv.ucf.edu/data/UCF101.php
