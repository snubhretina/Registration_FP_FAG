import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
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

        self.conv2 = self._make_layer(BasicBlock, 64, 3, stride=2)
        # self.conv2.load_state_dict(pretraind_model.layer1.state_dict().copy())
        self.conv3 = self._make_layer(BasicBlock, 128, 4, stride=2)
        self.conv3.load_state_dict(pretraind_model.layer2.state_dict().copy())
        self.conv4 = self._make_layer(BasicBlock, 256, 6, stride=2)
        self.conv4.load_state_dict(pretraind_model.layer3.state_dict().copy())
        self.conv5 = self._make_layer(BasicBlock, 512, 3, stride=2)
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

# class FA_SSANet3_version2(nn.Module):
#     def __init__(self, pretraind_model):
#         super(FA_SSANet3_version2, self).__init__()
#
#         self.net = SSANet3(pretraind_model)
#
#     def forward(self, x):
#
#         out = self.net(x)
#         return out
class AV_net(nn.Module):
    def __init__(self):
        super(AV_net, self).__init__()

        self.net = SSANet3(models.resnet34(pretrained=True))

    def forward(self, x):

        out = self.net(x)
        return out