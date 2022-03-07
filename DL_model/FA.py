import torch
import torch.nn as nn
import torch.nn.functional as F

class FA(nn.Module):
    def __init__(self, pretraind_model):
        super(FA, self).__init__()
        # self.input_conv = original_model.conv1
        self.conv1 = nn.Conv2d(3, 64, 7, 1, 3, 1, 1, bias=False)
        self.conv1.weight = pretraind_model.conv1.weight
        self.conv1.bias = pretraind_model.conv1.bias
        self.pool = nn.MaxPool2d(2,2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Sequential(*list(pretraind_model.layer1.children())[:])
        self.conv3 = nn.Sequential(*list(pretraind_model.layer2.children())[:])
        self.conv4 = nn.Sequential(*list(pretraind_model.layer3.children())[:])
        self.conv5 = nn.Sequential(*list(pretraind_model.layer4.children())[:])

        self.sp1 = nn.Sequential(
            nn.Conv2d(64, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        )
        self.sp2 = nn.Sequential(
            nn.Conv2d(64, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        )
        self.sp3 = nn.Sequential(
            nn.Conv2d(128, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        )
        self.sp4 = nn.Sequential(
            nn.Conv2d(256, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        )
        self.sp5 = nn.Sequential(
            nn.Conv2d(512, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        )

        self.output = nn.Sequential(
            nn.Conv2d(3+8*5, 8, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(8, 8, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(8, 8, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(8, 1, 1)
        )


    def forward(self, x):
        c1 = self.conv1(x)
        c1 = self.bn1(c1)
        c1 = self.relu(c1)

        c2 = self.conv2(c1)
        c2_up = F.upsample(c2, size=(x.size(2), x.size(3)), mode='bilinear')
        c3  =self.conv3(c2_up)
        c3_up = F.upsample(c3, size=(x.size(2), x.size(3)), mode='bilinear')
        c4 = self.conv4(c3_up)
        c4_up = F.upsample(c4, size=(x.size(2), x.size(3)), mode='bilinear')
        c5 = self.conv5(c4_up)

        sp1 = F.upsample(self.sp1(c1), size=(x.size(2), x.size(3)), mode='bilinear')
        sp2 = F.upsample(self.sp2(c2), size=(x.size(2), x.size(3)), mode='bilinear')
        sp3 = F.upsample(self.sp3(c3), size=(x.size(2), x.size(3)), mode='bilinear')
        sp4 = F.upsample(self.sp4(c4), size=(x.size(2), x.size(3)), mode='bilinear')
        sp5 = F.upsample(self.sp5(c5), size=(x.size(2), x.size(3)), mode='bilinear')

        cat = torch.cat([x, sp1, sp2, sp3, sp4, sp5], 1)
        out = self.output(cat)

        return F.sigmoid(out)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, gpu, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.gpu = gpu
        self.conv1 = conv3x3(inplanes, planes, stride).to(self.gpu)
        self.bn1 = nn.BatchNorm2d(planes).to(self.gpu)
        self.relu = nn.ReLU(inplace=True).to(self.gpu)
        self.conv2 = conv3x3(planes, planes).to(self.gpu)
        self.bn2 = nn.BatchNorm2d(planes).to(self.gpu)
        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = self.downsample.to(gpu)
        self.stride = stride

    def forward(self, x):
        x = x.to(self.gpu)
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class BasicBlock3(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock3, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class BasicBlock2(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, gpu, stride=1, downsample=None):
        super(BasicBlock2, self).__init__()
        self.gpu = gpu
        self.conv1 = conv3x3(inplanes, planes, stride).to(self.gpu)
        self.bn1 = nn.BatchNorm2d(planes).to(self.gpu)
        self.relu = nn.ReLU(inplace=True).to(self.gpu)
        self.conv2 = conv3x3(planes, planes).to(self.gpu)
        self.bn2 = nn.BatchNorm2d(planes).to(self.gpu)
        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = self.downsample.to(gpu)
        self.stride = stride

    def forward(self, x):
        x = x.to(self.gpu)
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class FA_HRF(nn.Module):
    def __init__(self, pretraind_model):
        super(FA_HRF, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, 2 , 3, bias=False)
        self.conv1.load_state_dict(pretraind_model.conv1.state_dict().copy())

        self.bn1 = nn.BatchNorm2d(64)
        self.bn1.load_state_dict(pretraind_model.bn1.state_dict().copy())
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = self._make_layer(BasicBlock, 64, 3,
                                      gpu=[torch.device('cuda:0'), torch.device('cuda:0'), torch.device('cuda:0')], stride=2)
        # self.conv2.load_state_dict(pretraind_model.layer1.state_dict().copy())
        self.conv3 = self._make_layer(BasicBlock, 128, 4,
                                      gpu=[torch.device('cuda:0'), torch.device('cuda:0'), torch.device('cuda:0'), torch.device('cuda:0')], stride=2)
        self.conv3.load_state_dict(pretraind_model.layer2.state_dict().copy())
        self.conv4 = self._make_layer(BasicBlock, 256, 6,
                                      gpu=[torch.device('cuda:0'), torch.device('cuda:0'), torch.device('cuda:0'),
                                           torch.device('cuda:0'), torch.device('cuda:0'), torch.device('cuda:0')], stride=2)
        self.conv4.load_state_dict(pretraind_model.layer3.state_dict().copy())
        self.conv5 = self._make_layer(BasicBlock, 512, 3,
                                      gpu=[torch.device('cuda:0'), torch.device('cuda:0'), torch.device('cuda:0')], stride=2)
        self.conv5.load_state_dict(pretraind_model.layer4.state_dict().copy())

        self.conv1.to(torch.device('cuda:0'))
        self.bn1.to(torch.device('cuda:0'))
        self.relu.to(torch.device('cuda:0'))
        # self.conv2.to(torch.device('cuda:1'))
        # self.conv3.to(torch.device('cuda:2'))
        # self.conv4.to(torch.device('cuda:2'))
        # self.conv5.to(torch.device('cuda:2'))

        self.sp1 = nn.Sequential(
            nn.Conv2d(64, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        ).to(torch.device('cuda:0'))
        self.sp2 = nn.Sequential(
            nn.Conv2d(64, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        ).to(torch.device('cuda:0'))
        self.sp3 = nn.Sequential(
            nn.Conv2d(128, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        ).to(torch.device('cuda:0'))
        self.sp4 = nn.Sequential(
            nn.Conv2d(256, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        ).to(torch.device('cuda:0'))
        self.sp5 = nn.Sequential(
            nn.Conv2d(512, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        ).to(torch.device('cuda:0'))

        self.output = self.make_infer(2, 5 * 8).to(torch.device('cuda:0'))

    def _make_layer(self, block, planes, blocks, gpu, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, gpu[0], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, gpu[i]))

        return nn.Sequential(*layers)

    def make_infer(self, n_infer, n_in_feat):
        infer_layers = []
        for i in range(n_infer - 1):
            if i == 0:
                conv = nn.Sequential(
                    nn.Conv2d(n_in_feat, 8, 3, 1, 1),
                    nn.ReLU(True)
                )
            else:
                conv = nn.Sequential(
                    nn.Conv2d(8, 8, 3, 1, 1),
                    nn.ReLU(True)
                )
            infer_layers.append(conv)

        if n_infer == 1:
            infer_layers.append(nn.Sequential(nn.Conv2d(n_in_feat, 1, 1)))
        else:
            infer_layers.append(nn.Sequential(nn.Conv2d(8, 1, 1)))

        return nn.Sequential(*infer_layers)

    def forward(self, x):

        up_cnt = 1
        c1 = self.conv1(x)
        c1 = self.bn1(c1)
        c1 = self.relu(c1)
        sp1 = F.upsample(self.sp1(c1.to(torch.device('cuda:0'))), size=(x.size(2), x.size(3)), mode='bilinear')
        c1 = F.upsample(c1.to(torch.device('cuda:0')), size=(x.size(2), x.size(3)), mode='bilinear')

        c2 = self.conv2(c1)
        sp2 = F.upsample(self.sp2(c2.to(torch.device('cuda:0'))), size=(x.size(2), x.size(3)), mode='bilinear')
        # c2 = F.upsample(c2.to(torch.device('cpu')), size=(x.size(2), x.size(3)), mode='bilinear')

        c3 = self.conv3(c2)
        sp3 = F.upsample(self.sp3(c3.to(torch.device('cuda:0'))), size=(x.size(2), x.size(3)), mode='bilinear')
        # c3 = F.upsample(c3, size=(x.size(2), x.size(3)), mode='bilinear')

        c4 = self.conv4(c3)
        sp4 = F.upsample(self.sp4(c4.to(torch.device('cuda:0'))), size=(x.size(2), x.size(3)), mode='bilinear')
        # c4 = F.upsample(c4, size=(x.size(2), x.size(3)), mode='bilinear')

        c5 = self.conv5(c4)
        sp5 = F.upsample(self.sp5(c5.to(torch.device('cuda:0'))), size=(x.size(2), x.size(3)), mode='bilinear')

        cat = torch.cat([sp1, sp2, sp3, sp4, sp5], 1)
        out = self.output(cat)

        # a = self.a(x)
        # out = self.b(a)

        return F.sigmoid(out)

class FAG_segm_Net(nn.Module):
    def __init__(self):
        super(FAG_segm_Net, self).__init__()

        self.feature1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True)
        )
        self.feature2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(True),
        )
        self.feature3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(True),
        )
        self.feature4 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(True),
        )
        self.feature5 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(True)
        )
        self.infer1 = nn.Sequential(
            nn.Conv2d(64, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.infer2 = nn.Sequential(
            nn.Conv2d(128, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.infer3 = nn.Sequential(
            nn.Conv2d(256, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.infer4 = nn.Sequential(
            nn.Conv2d(512, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.infer5 = nn.Sequential(
            nn.Conv2d(512, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(16*5, 1, 1, 1, 0)
        )

    def forward(self, x):
        x1 = self.feature1(x)
        x2 = self.feature2(x1)
        x3 = self.feature3(x2)
        x4 = self.feature4(x3)
        x5 = self.feature5(x4)

        infer1 = F.upsample(self.infer1(x1), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
        infer2 = F.upsample(self.infer2(x2), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
        infer3 = F.upsample(self.infer3(x3), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
        infer4 = F.upsample(self.infer4(x4), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
        infer5 = F.upsample(self.infer5(x5), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)

        cat = torch.cat([infer1, infer2, infer3, infer4, infer5], 1)

        out = self.classifier(cat)
        return F.sigmoid(out)

class SSA(nn.Module):
    def __init__(self, pretraind_model):
        super(SSA, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, 1 , 3, bias=False).to(torch.device('cuda:0'))
        # self.conv1.load_state_dict(pretraind_model.conv1.state_dict().copy())

        self.bn1 = nn.BatchNorm2d(64).to(torch.device('cuda:0'))
        self.bn1.load_state_dict(pretraind_model.bn1.state_dict().copy())
        self.relu = nn.ReLU(inplace=True).to(torch.device('cuda:0'))

        self.conv2 = self._make_layer(BasicBlock, 64, 3,
                                      gpu=[torch.device('cuda:0'), torch.device('cuda:0'), torch.device('cuda:0')], stride=2)
        # self.conv2.load_state_dict(pretraind_model.layer1.state_dict().copy())
        self.conv3 = self._make_layer(BasicBlock, 128, 4,
                                      gpu=[torch.device('cuda:0'), torch.device('cuda:0'), torch.device('cuda:0'), torch.device('cuda:0')], stride=2)
        self.conv3.load_state_dict(pretraind_model.layer2.state_dict().copy())
        self.conv4 = self._make_layer(BasicBlock, 256, 6,
                                      gpu=[torch.device('cuda:0'), torch.device('cuda:0'), torch.device('cuda:0'),
                                           torch.device('cuda:0'), torch.device('cuda:0'), torch.device('cuda:0')], stride=2)
        self.conv4.load_state_dict(pretraind_model.layer3.state_dict().copy())
        self.conv5 = self._make_layer(BasicBlock, 512, 3,
                                      gpu=[torch.device('cuda:0'), torch.device('cuda:0'), torch.device('cuda:0')], stride=2)
        self.conv5.load_state_dict(pretraind_model.layer4.state_dict().copy())

        self.extention_conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 1,),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        ).to(torch.device('cuda:0'))
        self.extention_conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 1,),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        ).to(torch.device('cuda:0'))
        self.extention_conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        ).to(torch.device('cuda:0'))

        self.sp1 = nn.Sequential(
            nn.Conv2d(64, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        ).to(torch.device('cuda:0'))
        self.sp2 = nn.Sequential(
            nn.Conv2d(64, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        ).to(torch.device('cuda:0'))
        self.sp3 = nn.Sequential(
            nn.Conv2d(128, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        ).to(torch.device('cuda:0'))
        self.sp4 = nn.Sequential(
            nn.Conv2d(256, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        ).to(torch.device('cuda:0'))
        self.sp5 = nn.Sequential(
            nn.Conv2d(512, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        ).to(torch.device('cuda:0'))

        self.output = self.make_infer(2, 3 + 5 * 8).to(torch.device('cuda:0'))

    def _make_layer(self, block, planes, blocks, gpu, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, gpu[0], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, gpu[i]))

        return nn.Sequential(*layers)

    def make_infer(self, n_infer, n_in_feat):
        infer_layers = []
        for i in range(n_infer - 1):
            if i == 0:
                conv = nn.Sequential(
                    nn.Conv2d(n_in_feat, 8, 3, 1, 1),
                    nn.BatchNorm2d(8),
                    nn.ReLU(True)
                )
            else:
                conv = nn.Sequential(
                    nn.Conv2d(8, 8, 3, 1, 1),
                    nn.BatchNorm2d(8),
                    nn.ReLU(True)
                )
            infer_layers.append(conv)

        if n_infer == 1:
            infer_layers.append(nn.Sequential(nn.Conv2d(n_in_feat, 1, 1)))
        else:
            infer_layers.append(nn.Sequential(nn.Conv2d(8, 8, 3, 1, 1), nn.ReLU(True)))
            infer_layers.append(nn.Sequential(nn.Conv2d(8, 1, 1)))

        return nn.Sequential(*infer_layers)

    def forward(self, x):
        c1 = self.conv1(x)
        c1 = self.bn1(c1)
        c1 = self.relu(c1)
        sp1 = self.sp1(c1)

        c2 = self.conv2(c1)
        sp2 = F.upsample(self.sp2(c2), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
        # c2 = F.upsample(c2, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)

        c3 = self.conv3(c2)
        sp3 = F.upsample(self.sp3(c3), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
        # c3 = F.upsample(c3, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)

        c4 = self.conv4(c3)
        sp4 = F.upsample(self.sp4(c4), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
        # c4 = F.upsample(c4, size=(x.size(2), x.size(3)), mode='bilinear')

        c5 = self.conv5(c4)
        sp5 = F.upsample(self.sp5(c5), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)

        cat = torch.cat([x, sp1, sp2, sp3, sp4, sp5], 1)
        out = self.output(cat)

        return F.sigmoid(out)

class max_FAG_aggreagation(nn.Module):
    def __init__(self, pretraind_model):
        super(max_FAG_aggreagation, self).__init__()
        self.inplanes = 64
        # self.init_conv = nn.Sequential(
        #     nn.Conv2d(41,3,1,1,0),
        #     nn.BatchNorm2d(3),
        #     nn.ReLU(True)
        # )
        self.conv1 = nn.Conv2d(3, 64, 7, 1 , 3, bias=False).to(torch.device('cuda:2'))
        self.conv1.load_state_dict(pretraind_model.conv1.state_dict().copy())

        self.bn1 = nn.BatchNorm2d(64).to(torch.device('cuda:2'))
        self.bn1.load_state_dict(pretraind_model.bn1.state_dict().copy())
        self.relu = nn.ReLU(inplace=True).to(torch.device('cuda:2'))

        self.conv2 = self._make_layer(BasicBlock, 64, 3,
                                      gpu=[torch.device('cuda:2'), torch.device('cuda:2'), torch.device('cuda:2')], stride=2)
        # self.conv2.load_state_dict(pretraind_model.layer1.state_dict().copy())
        self.conv3 = self._make_layer(BasicBlock, 128, 4,
                                      gpu=[torch.device('cuda:2'), torch.device('cuda:2'), torch.device('cuda:2'), torch.device('cuda:2')], stride=2)
        self.conv3.load_state_dict(pretraind_model.layer2.state_dict().copy())
        self.conv4 = self._make_layer(BasicBlock, 256, 6,
                                      gpu=[torch.device('cuda:2'), torch.device('cuda:2'), torch.device('cuda:2'),
                                           torch.device('cuda:2'), torch.device('cuda:2'), torch.device('cuda:2')], stride=2)
        self.conv4.load_state_dict(pretraind_model.layer3.state_dict().copy())
        self.conv5 = self._make_layer(BasicBlock, 512, 3,
                                      gpu=[torch.device('cuda:2'), torch.device('cuda:2'), torch.device('cuda:2')], stride=2)
        self.conv5.load_state_dict(pretraind_model.layer4.state_dict().copy())

        self.extention_conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 1,),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        ).to(torch.device('cuda:2'))
        self.extention_conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 1,),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        ).to(torch.device('cuda:2'))
        self.extention_conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        ).to(torch.device('cuda:2'))

        self.sp1 = nn.Sequential(
            nn.Conv2d(64, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        ).to(torch.device('cuda:2'))
        self.sp2 = nn.Sequential(
            nn.Conv2d(64, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        ).to(torch.device('cuda:2'))
        self.sp3 = nn.Sequential(
            nn.Conv2d(128, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        ).to(torch.device('cuda:2'))
        self.sp4 = nn.Sequential(
            nn.Conv2d(256, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        ).to(torch.device('cuda:2'))
        self.sp5 = nn.Sequential(
            nn.Conv2d(512, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        ).to(torch.device('cuda:2'))

        self.output = self.make_infer(2, 5 * 8).to(torch.device('cuda:2'))

    def _make_layer(self, block, planes, blocks, gpu, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, gpu[0], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, gpu[i]))

        return nn.Sequential(*layers)

    def make_infer(self, n_infer, n_in_feat):
        infer_layers = []
        for i in range(n_infer - 1):
            if i == 0:
                conv = nn.Sequential(
                    nn.Conv2d(n_in_feat, 8, 3, 1, 1),
                    nn.BatchNorm2d(8),
                    nn.ReLU(True)
                )
            else:
                conv = nn.Sequential(
                    nn.Conv2d(8, 8, 3, 1, 1),
                    nn.BatchNorm2d(8),
                    nn.ReLU(True)
                )
            infer_layers.append(conv)

        if n_infer == 1:
            infer_layers.append(nn.Sequential(nn.Conv2d(n_in_feat, 1, 1)))
        else:
            infer_layers.append(nn.Sequential(nn.Conv2d(8, 8, 3, 1, 1), nn.ReLU(True)))
            infer_layers.append(nn.Sequential(nn.Conv2d(8, 1, 1)))

        return nn.Sequential(*infer_layers)

    def forward(self, x):

        # init_x = self.init_conv(x)
        c1 = self.conv1(x)
        c1 = self.bn1(c1)
        c1 = self.relu(c1)
        sp1 = self.sp1(c1)

        c2 = self.conv2(c1)
        # sp2 = F.upsample(self.sp2(c2), size=(x.size(2), x.size(3)), mode='bilinear')
        sp2 = F.interpolate(self.sp2(c2), size=(x.size(2), x.size(3)), mode='bilinear')
        # c2 = F.upsample(c2, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)

        c3 = self.conv3(c2)
        # sp3 = F.upsample(self.sp3(c3), size=(x.size(2), x.size(3)), mode='bilinear')
        sp3 = F.interpolate(self.sp3(c3), size=(x.size(2), x.size(3)), mode='bilinear')
        # c3 = F.upsample(c3, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)

        c4 = self.conv4(c3)
        # sp4 = F.upsample(self.sp4(c4), size=(x.size(2), x.size(3)), mode='bilinear')
        sp4 = F.interpolate(self.sp4(c4), size=(x.size(2), x.size(3)), mode='bilinear')
        # c4 = F.upsample(c4, size=(x.size(2), x.size(3)), mode='bilinear')

        c5 = self.conv5(c4)
        # sp5 = F.upsample(self.sp5(c5), size=(x.size(2), x.size(3)), mode='bilinear')
        sp5 = F.interpolate(self.sp5(c5), size=(x.size(2), x.size(3)), mode='bilinear')

        cat = torch.cat([sp1, sp2, sp3, sp4, sp5], 1)
        out = self.output(cat)

        return torch.sigmoid(out)

class SSANet3(nn.Module):
    def __init__(self, pretraind_model):
        super(SSANet3, self).__init__()
        self.scale_factor = 2
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, 1 , 3, bias=False)
        self.conv1.load_state_dict(pretraind_model.conv1.state_dict().copy())

        self.bn1 = nn.BatchNorm2d(64)
        self.bn1.load_state_dict(pretraind_model.bn1.state_dict().copy())
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = self._make_layer(BasicBlock3, 64, 3, stride=2)
        # self.conv2.load_state_dict(pretraind_model.layer1.state_dict().copy())
        self.conv3 = self._make_layer(BasicBlock3, 128, 4, stride=2)
        self.conv3.load_state_dict(pretraind_model.layer2.state_dict().copy())
        self.conv4 = self._make_layer(BasicBlock3, 256, 6, stride=2)
        self.conv4.load_state_dict(pretraind_model.layer3.state_dict().copy())
        self.conv5 = self._make_layer(BasicBlock3, 512, 3, stride=2)
        self.conv5.load_state_dict(pretraind_model.layer4.state_dict().copy())

        self.compressed_conif = [64, 64, 128, 256, 512]
        self.compressed = []
        for i in range(5):
            layer = nn.Sequential(
            nn.Conv2d(self.compressed_conif[i], 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True))
            self.compressed.append(layer.cuda())

        self.output = self.make_infer(2, 5 * 8)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def make_infer(self, n_infer, n_in_feat):
        infer_layers = []
        for i in range(n_infer - 1):
            if i == 0:
                conv = nn.Sequential(
                    nn.Conv2d(n_in_feat, 8, 3, 1, 1),
                    nn.BatchNorm2d(8),
                    nn.ReLU(True)
                )
            else:
                conv = nn.Sequential(
                    nn.Conv2d(8, 8, 3, 1, 1),
                    nn.BatchNorm2d(8),
                    nn.ReLU(True)
                )
            infer_layers.append(conv)

        if n_infer == 1:
            infer_layers.append(nn.Sequential(nn.Conv2d(n_in_feat, 1, 1)))
        else:
            infer_layers.append(nn.Sequential(nn.Conv2d(8, 8, 3, 1, 1), nn.ReLU(True)))
            infer_layers.append(nn.Sequential(nn.Conv2d(8, 1, 1)))

        return nn.Sequential(*infer_layers)

    def forward(self, x):
        origin_size = (x.size(2), x.size(3))
        fixed_down_size = (x.size(2)//self.scale_factor, x.size(3)//self.scale_factor)
        c1 = self.conv1(x)
        c1 = self.bn1(c1)
        c1 = self.relu(c1)
        sp1 = self.compressed[0](c1)

        c2 = self.conv2(c1)
        sp2 = F.interpolate(self.compressed[1](c2), size=origin_size, mode='bilinear')
        c2 = F.interpolate(c2, size=fixed_down_size, mode='bilinear')

        c3 = self.conv3(c2)
        sp3 = F.interpolate(self.compressed[2](c3), size=origin_size, mode='bilinear')
        c3 = F.interpolate(c3, size=fixed_down_size, mode='bilinear')

        c4 = self.conv4(c3)
        sp4 = F.interpolate(self.compressed[3](c4), size=origin_size, mode='bilinear')
        c4 = F.interpolate(c4, size=fixed_down_size, mode='bilinear')

        c5 = self.conv5(c4)
        sp5 = F.interpolate(self.compressed[4](c5), size=origin_size, mode='bilinear')

        cat = torch.cat([sp1, sp2, sp3, sp4, sp5], 1)
        out = self.output(cat)

        return F.sigmoid(out)

class FA_SSANet3_version2(nn.Module):
    def __init__(self, pretraind_model):
        super(FA_SSANet3_version2, self).__init__()

        self.net = SSANet3(pretraind_model)

    def forward(self, x):

        out = self.net(x)
        return out