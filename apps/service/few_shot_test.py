import pandas as pd
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps
import torch.nn as nn
import torch.nn.functional as F

class Config():
    testing_dir = "./testset"
    train_batch_size = 10
    train_number_epochs = 2
    model_name = "model0403_3.pth"
    query_dir = "./test/queryset2"
    support_dir = "./test/supportset2"

class SiameseNetworkTestDataset(Dataset):
    i = 0
    # 클래스의 인스턴스를 초기화하는 메서드
    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        nownum = SiameseNetworkTestDataset.i
        query_class = nownum // 1000
        if query_class != 0 : nownum = nownum - (query_class * 1000)
        query_img = nownum // 100
        if query_img != 0 : nownum = nownum - (query_img * 100)
        support_class = nownum // 10
        if support_class != 0 : nownum = nownum - (support_class * 10)
        support_img = nownum

        print('/s{0}/img{1}.jpg'.format(query_class + 1, query_img + 1))
        print('/s{0}/img{1}.jpg'.format(support_img + 1, support_class + 1))
        img0_tuple = Config.query_dir + '/s{0}/img{1}.jpg'.format(query_class + 1, query_img + 1)
        img1_tuple = Config.support_dir + '/s{0}/img{1}.jpg'.format(support_img + 1, support_class + 1)
        SiameseNetworkTestDataset.i += 1

        # if SiameseNetworkTestDataset.j > 100 :
        #     SiameseNetworkTestDataset.i += 1
        #     SiameseNetworkTestDataset.j = 1
        #     print(SiameseNetworkTestDataset.i)

        img0 = Image.open(img0_tuple)
        img1 = Image.open(img1_tuple)


        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1

    def __len__(self):
        return 10000

class SiameseNetwork(nn.Module):  # Siamese neural network 모델 클래스
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(  # 3개의 레이어를 가진 은닉층
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 32, kernel_size=3),  # 1개의 input_channel, 4개의 output_channel, 3*3 kernel_size
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, stride=2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 64, kernel_size=3),  # 4개의 input_channel, 8개의 output_channel, 3*3 kernel_size
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, stride=2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 128, kernel_size=3),  # 8개의 input_channel, 8개의 output_channel, 3*3 kernel_size
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, stride=2),
        )

        self.fc1 = nn.Sequential(  # 3개의 레이어를 가지는 완전연결층
            nn.Linear(128 * 16 * 8, 128),  # 500개의 입력을 받아 500개의 출력을 갖는 레이어
            nn.ReLU(inplace=True),

            nn.Linear(128, 64),  # 500개의 입력을 받아 500개의 출력을 갖는 레이어
            nn.ReLU(inplace=True),

            nn.Linear(64, 10))  # 500개의 입력을 받아 5개의 출력을 갖는 레이어

    def forward_once(self, x):  # 한 이미지를 CNN에 입력으로 넣는다
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):  # forward_once 메서드를 두 번 호출하여 두 이미지에 대해 각각 벡터값을 출력하고 반환
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


net = SiameseNetwork()
device = torch.device('cpu')
net.load_state_dict(torch.load(Config.model_name, map_location=device))

folder_dataset_test = dset.ImageFolder(root=Config.testing_dir)
siamese_dataset = SiameseNetworkTestDataset(imageFolderDataset=folder_dataset_test, transform=transforms.Compose(
    [transforms.Resize((128, 64)), transforms.ToTensor()]), should_invert=False)

test_dataloader = DataLoader(siamese_dataset, num_workers=0, batch_size=1, shuffle=True)
dataiter = iter(test_dataloader)

classnum = 0 # 0~9
querynum = 0 # 0~99
supportnum = 0 # 0~9
# distance_data = [[] for _ in range(100)]
distance_data = np.ones((10,100,10) , dtype='f')

for i in range(1000):
    x0, x1 = next(dataiter)
    concatenated = torch.cat((x0, x1), 0)
    output1, output2 = net(Variable(x0), Variable(x1))
    euclidean_distance = F.pairwise_distance(output1, output2)
    print(output1, output2)
    # distance_data[gx].append('{:.2f}'.format(euclidean_distance.item()))
    distance_data[classnum][querynum][supportnum] = '{:.2f}'.format(euclidean_distance.item())
    print('in to distance_data[{0}][{1}][{2}]'.format(classnum, querynum, supportnum))
    print('-------------------------')
    print()
    supportnum += 1
    if supportnum > 9 :
        querynum += 1
        supportnum = 0
        if querynum > 99 :
            classnum += 1
            querynum = 0



