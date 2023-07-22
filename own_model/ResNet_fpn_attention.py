import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
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
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x

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
        out = self.beta * feat_e + x

        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
class FPN(nn.Module):
    def __init__(self, block, layers,norm_layer=nn.BatchNorm2d,norm_kwargs=None,**kwargs):
        super(FPN, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # Bottom-up layers
        self.layer1 = self._make_layer(block,  64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        
        
        
        
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        
        
        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels
        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_p1_smooth1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            norm_layer(256, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c1_smooth1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            norm_layer(256, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.pam_smooth1 = _PositionAttentionModule(256, **kwargs)
        self.cam_smooth1 = _ChannelAttentionModule(**kwargs)
        self.conv_p2_smooth1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            norm_layer(256, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c2_smooth1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            norm_layer(256, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv_p1_smooth2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            norm_layer(256, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c1_smooth2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            norm_layer(256, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.pam_smooth2 = _PositionAttentionModule(256, **kwargs)
        self.cam_smooth2 = _ChannelAttentionModule(**kwargs)
        self.conv_p2_smooth2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            norm_layer(256, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c2_smooth2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            norm_layer(256, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True))
        
        
        
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv_p1_smooth3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            norm_layer(256, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c1_smooth3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            norm_layer(256, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.pam_smooth3 = _PositionAttentionModule(256, **kwargs)
        self.cam_smooth3 = _ChannelAttentionModule(**kwargs)
        self.conv_p2_smooth2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            norm_layer(256, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c2_smooth3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            norm_layer(256, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True))
        
        
        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

        self.avgpool = nn.AvgPool2d(56, stride=1)

        self.classifier = nn.Linear(256, 3)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)
    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear',align_corners=True) + y
    
    
    
    def forward(self, x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        #print(f'c1:{c1.shape}')
        c2 = self.layer1(c1)#torch.Size([1, 256, 56, 56])
        #print(f'c2:{c2.shape}')
        c3 = self.layer2(c2)#torch.Size([1, 512, 28, 28])
        #print(f'c3:{c3.shape}')
        c4 = self.layer3(c3)#torch.Size([1, 1024, 14, 14])
        #print(f'c4:{c4.shape}')
        c5 = self.layer4(c4)#torch.Size([1, 2048, 7, 7])
        #print(f'c5:{c5.shape}')
        # Top-down
        p5 = self.toplayer(c5)#torch.Size([1, 256, 7, 7])
        #print(f'p5:{p5.shape}')
        #print("p5生成",p5.shape)

       
        
        p4 = self._upsample_add(p5, self.latlayer1(c4))#torch.Size([1, 256, 14, 14])
        
        
        #print(f'latlayer1(c4):{self.latlayer1(c4).shape}, p4:{p4.shape}')
        p3 = self._upsample_add(p4, self.latlayer2(c3))#torch.Size([1, 256, 28, 28])
        #print(f'latlayer1(c3):{self.latlayer2(c3).shape}, p3:{p3.shape}')
        p2 = self._upsample_add(p3, self.latlayer3(c2))#torch.Size([1, 256, 56, 56])
        #print(f'latlayer1(c2):{self.latlayer3(c2).shape}, p2:{p2.shape}')
        # Smooth
        p4 = self.smooth1(p4)#torch.Size([1, 256, 14, 14])
        feat_p_p4 = self.conv_p1_smooth1(p4 )
        feat_p_p4 = self.pam_smooth1(feat_p_p4)  # torch.Size([1, 256, 14, 14])
        feat_p_p4 = self.conv_p2_smooth1(feat_p_p4)  # torch.Size([1, 256, 14, 14])

        feat_c_p4 = self.conv_c1_smooth1(p4 )#torch.Size([1, 256, 14, 14])  # torch.Size([2, 512, 14, 14])
        feat_c_p4 = self.cam_smooth1(feat_c_p4)  #torch.Size([1, 256, 14, 14])
        feat_c_p4 = self.conv_c2_smooth1(feat_c_p4)  # torch.Size([1, 256, 14, 14])

        #feat_fusion_p4 = feat_p_p4 + feat_c_p4 #torch.Size([1, 256, 14, 14])
        feat_fusion_p4 = feat_p_p4 + feat_c_p4
        #print('p4 生成',feat_fusion_p4.shape)
        p3_smooth2 = self.smooth2(p3)#torch.Size([1, 256, 28, 28])
        p4_unsample = F.interpolate(feat_fusion_p4, size=(28, 28), mode='bilinear', align_corners=True) + p3_smooth2 #torch.Size([1, 256, 28, 28])

        feat_p_p3 = self.conv_p1_smooth2(p4_unsample)
        feat_p_p3 = self.pam_smooth2(feat_p_p3)  # torch.Size([1, 256, 14, 14])
        feat_p_p3 = self.conv_p2_smooth2(feat_p_p3)  # torch.Size([1, 256, 14, 14])

        feat_c_p3 = self.conv_c1_smooth2(p4_unsample)  # torch.Size([1, 256, 14, 14])  # torch.Size([2, 512, 14, 14])
        feat_c_p3 = self.cam_smooth2(feat_c_p3)  # torch.Size([1, 256, 14, 14])
        feat_c_p3 = self.conv_c2_smooth2(feat_c_p3)  # torch.Size([1, 256, 14, 14])

        feat_fusion_p3 = feat_p_p3 + feat_c_p3  # torch.Size([1, 256, 28, 28])
        #print('p3 生成',feat_fusion_p3.shape)
        
         #torch.Size([1, 256, 28, 28])

        p2_smooth3 = self.smooth3(p2)#torch.Size([1, 256, 56, 56])
        p3_unsample = F.interpolate(feat_fusion_p3, size=(56, 56), mode='bilinear', align_corners=True) + p2_smooth3

        feat_p_p2 = self.conv_p1_smooth2(p3_unsample)
        feat_p_p2 = self.pam_smooth2(feat_p_p2)  # torch.Size([1, 256, 14, 14])
        feat_p_p2 = self.conv_p2_smooth2(feat_p_p2)  # torch.Size([1, 256, 14, 14])

        feat_c_p2 = self.conv_c1_smooth2(p3_unsample)  # torch.Size([1, 256, 14, 14])  # torch.Size([2, 512, 14, 14])
        feat_c_p2 = self.cam_smooth2(feat_c_p2)  # torch.Size([1, 256, 14, 14])
        feat_c_p2= self.conv_c2_smooth2(feat_c_p2)  # torch.Size([1, 256, 14, 14])

        feat_fusion_p2 = feat_p_p2 + feat_c_p2
        #print('p2 生成',feat_fusion_p2.shape)
        x = self.avgpool(feat_fusion_p2)
        x = x.view(x.size(0), -1)

        x = self.classifier(x)
        return x
        
        

def FPN101():
    # return FPN(Bottleneck, [2,4,23,3])
    #return FPN(Bottleneck, [3,4,6,3])
    return FPN(Bottleneck, [3, 8, 36, 3])

def ResNet152_FPN_attention():


    resNet152 =models.resnet152(pretrained=True)
    pretrained_dict = resNet152.state_dict()


    model = FPN(Bottleneck,[3, 8, 36, 3])
    model_dict = model.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 更新现有的model_dict
    model_dict.update(pretrained_dict)

    # 加载真正需要的state_dict
    model.load_state_dict(model_dict)
    return model


if __name__ == "__main__":
    #net = FPN101()
    net=ResNet152_FPN_attention()
    fms = net(torch.randn(1, 3, 224, 224))
    print(fms.shape)
	
    
	# for fm in fms:
	# 	print(fm.size())