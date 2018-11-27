import torch
import torch.nn as nn


class C3D(nn.Module):
    '''
    conv1  in:16*3*112*112   out:16*64*112*112
    pool1  in:16*64*56*56    out:16*64*56*56
    conv2  in:16*64*56*56    out:16*128*56*56
    pool2  in:16*128*56*56   out:8*128*28*28
    conv3a in:8*128*28*28    out:8*256*28*28
    conv3b in:8*256*28*28    out:8*256*28*28
    pool3  in:8*256*28*28    out:4*256*14*14
    conv4a in:4*512*14*14    out:8*512*14*14
    conv4b in:4*512*14*14    out:8*512*14*14
    pool4  in:4*512*14*14    out:2*512*7*7
    conv5a in:2*512*7*7      out:2*512*7*7
    conv5b in:2*512*7*7      out:2*512*7*7
    pool5  in:2*512*7*7      out:1*512*4*4

    '''

    def __init__(self):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 101)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def init_weight(self):
        for name, para in self.named_parameters():
            if name.find('weight') != -1:
                nn.init.xavier_normal_(para.data)
            else:
                nn.init.constant_(para.data, 0)

    def forward(self, x):
        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)

        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
        h = self.pool5(h)

        h = h.view(-1, 8192)

        h = self.relu(self.fc6(h))
        h = self.dropout(h)

        h = self.relu(self.fc7(h))
        h = self.dropout(h)

        logits = self.fc8(h)

        return logits

