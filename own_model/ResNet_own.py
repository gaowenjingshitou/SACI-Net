import torch
import torch.nn as nn
import torchvision
import numpy as np
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

# __all__ = ['ResNet50', 'ResNet101', 'ResNet152','ResNet153']
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes, out_channels=places, kernel_size=7, stride=stride, padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )


class Bottleneck(nn.Module):
    def __init__(self, in_places, places, stride=1, downsampling=False, expansion=4):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places * self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places * self.expansion, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(places * self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class _PositionAttentionModule(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, **kwargs):
        super(_PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()  # [b,512,14,14]
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x  # [b,512,14,14]

        return out


class _ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(_ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x  #

        return out


class ResNet(nn.Module):
    def __init__(self, blocks, norm_layer=nn.BatchNorm2d, norm_kwargs=None, num_classes=1000, expansion=4, **kwargs):
        super(ResNet, self).__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_planes=3, places=64)

        self.layer1 = self.make_layer(in_places=64, places=64, block=blocks[0], stride=1)

        self.layer1_down = nn.Sequential(nn.Conv2d(256, 1024, 1, stride=2, bias=True),
                                         norm_layer(1024, **({} if norm_kwargs is None else norm_kwargs)),
                                         nn.Conv2d(1024, 2048, 1, stride=2, bias=True),
                                         norm_layer(2048, **({} if norm_kwargs is None else norm_kwargs)))

        self.layer2 = self.make_layer(in_places=256, places=128, block=blocks[1], stride=2)

        self.conv_p1 = nn.Sequential(
            nn.Conv2d(512, 512 // 4, 3, padding=1, bias=False),
            norm_layer(512 // 4, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(512, 512 // 4, 3, padding=1, bias=False),
            norm_layer(512 // 4, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.pam = _PositionAttentionModule(512 // 4, **kwargs)
        self.cam = _ChannelAttentionModule(**kwargs)
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(512 // 4, 512 // 4, 3, padding=1, bias=True),
            norm_layer(512 // 4, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(512 // 4, 512 // 4, 3, padding=1, bias=True),
            norm_layer(512 // 4, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True))

        self.fusion_cp1 = nn.Sequential(nn.Conv2d(512 // 4, 512, 1, bias=True),
                                        nn.BatchNorm2d(512))

        self.down = nn.Sequential(nn.Conv2d(512, 1024, 1, stride=2, bias=True), nn.BatchNorm2d(1024))

        self.layer3 = self.make_layer(in_places=512, places=256, block=blocks[2], stride=2)

        self.conv_p1_layer3 = nn.Sequential(
            nn.Conv2d(1024, 1024 // 4, 3, padding=1, bias=False),
            norm_layer(1024 // 4, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c1_layer3 = nn.Sequential(
            nn.Conv2d(1024, 1024 // 4, 3, padding=1, bias=False),
            norm_layer(1024 // 4, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.pam_layer3 = _PositionAttentionModule(1024 // 4, **kwargs)
        self.cam_layer3 = _ChannelAttentionModule(**kwargs)
        self.conv_p2_layer3 = nn.Sequential(
            nn.Conv2d(1024 // 4, 1024 // 4, 3, padding=1, bias=True),
            norm_layer(1024 // 4, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c2_layer3 = nn.Sequential(
            nn.Conv2d(1024 // 4, 1024 // 4, 3, padding=1, bias=True),
            norm_layer(1024 // 4, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True))

        self.fusion_cp1_layer3 = nn.Sequential(nn.Conv2d(1024 // 4, 1024, 1, bias=True),
                                               nn.BatchNorm2d(1024))

        self.down_layer3 = nn.Sequential(nn.Conv2d(1024, 2048, 1, stride=1, bias=True), nn.BatchNorm2d(2048))

        self.layer4 = self.make_layer(in_places=1024, places=512, block=blocks[3], stride=1)

        self.conv_p1_layer4 = nn.Sequential(
            nn.Conv2d(2048, 2048 // 4, 3, padding=1, bias=False),
            norm_layer(2048 // 4, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c1_layer4 = nn.Sequential(
            nn.Conv2d(2048, 2048 // 4, 3, padding=1, bias=False),
            norm_layer(2048 // 4, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.pam_layer4 = _PositionAttentionModule(2048 // 4, **kwargs)
        self.cam_layer4 = _ChannelAttentionModule(**kwargs)
        self.conv_p2_layer4 = nn.Sequential(
            nn.Conv2d(2048 // 4, 2048 // 4, 3, padding=1, bias=True),
            norm_layer(2048 // 4, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c2_layer4 = nn.Sequential(
            nn.Conv2d(2048 // 4, 2048 // 4, 3, padding=1, bias=True),
            norm_layer(2048 // 4, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True))

        self.fusion_cp1_layer4 = nn.Sequential(nn.Conv2d(2048 // 4, 2048, 1, bias=True),
                                               nn.BatchNorm2d(2048))
        self.avgpool = nn.AvgPool2d(14, stride=1)


        self.fc = nn.Linear(2048,num_classes)





        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places, stride, downsampling=True))
        for i in range(1, block):
            layers.append(Bottleneck(places * self.expansion, places))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)  # [b,256,56,56]
        x_layer1 = self.layer1_down(x)  # torch.Size([1, 2048, 14, 14])
        layer2 = self.layer2(x)  # [b,512,28,28]

        feat_p = self.conv_p1(layer2)  # torch.Size([1, 128, 28, 28])
        feat_p = self.pam(feat_p)  # torch.Size([1, 128, 28, 28])
        feat_p = self.conv_p2(feat_p)  # torch.Size([1, 128, 28, 28])

        feat_c = self.conv_c1(layer2)  # torch.Size([1, 128, 28, 28])

        feat_c = self.cam(feat_c)  # torch.Size([1, 128, 28, 28])

        feat_c = self.conv_c2(feat_c)  # torch.Size([1, 128, 28, 28])

        feat_fusion_layer1 = feat_p + feat_c  # torch.Size([1, 128, 28, 28])
        x = self.fusion_cp1(feat_fusion_layer1)  # torch.Size([1, 512, 28, 28])

        # print("new_F2生成完毕")

        x_downsample = self.down(x)  # torch.Size([1, 1024, 14, 14])

        layer3 = self.layer3(x)  # torch.Size([1, 1024, 14, 14])
        layer3 = x_downsample + layer3
        feat_p_layer3 = self.conv_p1_layer3(layer3)
        feat_p_layer3 = self.pam_layer3(feat_p_layer3)
        feat_p_layer3 = self.conv_p2_layer3(feat_p_layer3)

        feat_c_layer3 = self.conv_c1_layer3(layer3)

        feat_c_layer3 = self.cam_layer3(feat_c_layer3)

        feat_c_layer3 = self.conv_c2_layer3(feat_c_layer3)
        feat_fusion_layer3 = feat_p_layer3 + feat_c_layer3
        x = self.fusion_cp1_layer3(feat_fusion_layer3)  # torch.Size([1, 1024, 14, 14])

        # print("new_F3生成完毕")

        layer4 = self.layer4(x) + self.down_layer3(x)

        feat_p_layer4 = self.conv_p1_layer4(layer4)
        feat_p_layer4 = self.pam_layer4(feat_p_layer4)
        feat_p_layer4 = self.conv_p2_layer4(feat_p_layer4)

        feat_c_layer4 = self.conv_c1_layer4(layer4)

        feat_c_layer4 = self.cam_layer4(feat_c_layer4)

        feat_c_layer4 = self.conv_c2_layer4(feat_c_layer4)
        feat_fusion_layer4 = feat_p_layer4 + feat_c_layer4
        x = self.fusion_cp1_layer4(feat_fusion_layer4)  # torch.Size([1, 2048, 14, 14])
        x = x_layer1 + x
        # print("new_F4生成完毕")

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x


def ResNet50():
    return ResNet([3, 4, 6, 3])


def ResNet101():
    return ResNet([3, 4, 23, 3])

def ResNet153():
    ResNet153 = models.resnet152(pretrained=True)
    return ResNet153

def ResNet152():


    resNet152 =models.resnet152(pretrained=True)
    pretrained_dict = resNet152.state_dict()


    model = ResNet([3, 8, 36, 3])
    model_dict = model.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 更新现有的model_dict
    model_dict.update(pretrained_dict)

    # 加载真正需要的state_dict
    model.load_state_dict(model_dict)
    return model






if __name__ == '__main__':
    # model = torchvision.models.resnet50()
    model = ResNet152()
    #  print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    print(out.shape)
