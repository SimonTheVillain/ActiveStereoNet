import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import pdb
from .blocks import *

import pdb

class SiameseTower(nn.Module):
    def __init__(self, scale_factor, ch_in=3):
        super(SiameseTower, self).__init__()

        self.conv1 = conv_block(nc_in=ch_in, nc_out=32, k=3, s=1, norm=None, act=None)
        res_blocks = [ResBlock(32, 32, 3, 1, 1)] * 3# simon: are we not sharing weights this way?
        assert res_blocks[0] is not res_blocks[1]
        self.res_blocks = nn.Sequential(*res_blocks)
        convblocks = [conv_block(32, 32, k=3, s=2, norm='bn', act='lrelu')] * int(math.log2(scale_factor))
        self.conv_blocks = nn.Sequential(*convblocks)
        self.conv2 = conv_block(nc_in=32, nc_out=32, k=3, s=1, norm=None, act=None)
    
    def forward(self, x):

        #pdb.set_trace()
        out = self.conv1(x)
        out = self.res_blocks(out)
        out = self.conv_blocks(out)
        out = self.conv2(out)

        return out


class SiameseTower2(nn.Module):
    def __init__(self, scale_factor, ch_in=3):
        super(SiameseTower2, self).__init__()

        self.conv1 = conv_block(nc_in=ch_in, nc_out=32, k=3, s=1, norm=None, act=None)
        res_blocks = [ResBlock(32, 32, 3, 1, 1),
                      ResBlock(32, 32, 3, 1, 1),
                      ResBlock(32, 32, 3, 1, 1)] # in the original paper, i doubt they would be sharing weights as in the implementation by blar...
        self.res_blocks = nn.Sequential(*res_blocks)

        convblocks = []#conv_block(ch_in, 32, k=3, s=2, norm='bn', act='lrelu')]
        for i in range(int(math.log2(scale_factor))):
            convblocks.append(conv_block(32, 32, k=3, s=2, norm='bn', act='lrelu'))
        self.conv_blocks = nn.Sequential(*convblocks)
        self.conv2 = conv_block(nc_in=32, nc_out=32, k=3, s=1, norm=None, act=None)

    def forward(self, x):
        # pdb.set_trace()
        out = self.conv1(x)
        out = self.res_blocks(out)
        out = self.conv_blocks(out)
        out = self.conv2(out)

        return out

class CoarseNet(nn.Module):
    def __init__(self, maxdisp, scale_factor, img_shape, concatenate=False):
        super(CoarseNet, self).__init__()
        self.maxdisp = maxdisp
        self.scale_factor = scale_factor
        self.img_shape = img_shape
        self.concatenate = concatenate

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        if concatenate:
            self.conv3d_1 = conv3d_block(64, 32, 3, 1, norm='bn', act='lrelu')
        else:
            self.conv3d_1 = conv3d_block(32, 32, 3, 1, norm='bn', act='lrelu')
        self.conv3d_2 = conv3d_block(32, 32, 3, 1, norm='bn', act='lrelu')
        self.conv3d_3 = conv3d_block(32, 32, 3, 1, norm='bn', act='lrelu')
        self.conv3d_4 = conv3d_block(32, 32, 3, 1, norm='bn', act='lrelu')

        self.conv3d_5 = conv3d_block(32, 1, 3, 1, norm=None, act=None)
        self.disp_reg = DisparityRegression(self.maxdisp)
        self.disp_reg2 = DisparityRegression(self.maxdisp//scale_factor)

    def costVolume(self, refimg_fea, targetimg_fea, views):
        if self.concatenate:
            #Cost Volume64
            cost = torch.zeros(refimg_fea.size()[0], refimg_fea.size()[1]*2, self.maxdisp//self.scale_factor, refimg_fea.size()[2], refimg_fea.size()[3]).cuda()
            views = views.lower()
            if views == 'left':
                for i in range(self.maxdisp//self.scale_factor):
                    if i > 0:
                        cost[:, :refimg_fea.size()[1], i, :, i:] = refimg_fea[:,:,:,i:] # the left image
                        cost[:, refimg_fea.size()[1]:, i, :, i:] = targetimg_fea[:,:,:,:-i]
                    else:
                        cost[:, :refimg_fea.size()[1], i, :,:] = refimg_fea
                        cost[:, refimg_fea.size()[1]:, i, :,:] = targetimg_fea
            elif views == 'right':
                for i in range(self.maxdisp // self.scale_factor):
                    if i > 0:
                        cost[:, :refimg_fea.size()[1], i, :, :-i] = refimg_fea[:,:,:,i:]
                        cost[:, refimg_fea.size()[1]:, i, :, :-i] = targetimg_fea[:,:,:,:-i]
                    else:
                        cost[:, :refimg_fea.size()[1], i, :,:] = refimg_fea
                        cost[:, refimg_fea.size()[1]:, i, :,:] = targetimg_fea
            return cost
        else:
            #Cost Volume32
            cost = torch.zeros(refimg_fea.size()[0], refimg_fea.size()[1], self.maxdisp//self.scale_factor, refimg_fea.size()[2], refimg_fea.size()[3]).cuda()
            views = views.lower()
            if views == 'left':
                for i in range(self.maxdisp//self.scale_factor):
                    if i > 0:
                        cost[:, :, i, :, i:] = refimg_fea[:,:,:,i:] - targetimg_fea[:,:,:,:-i] # the left image
                    else:
                        cost[:, :, i, :,:] = refimg_fea - targetimg_fea
            elif views == 'right':
                for i in range(self.maxdisp // self.scale_factor):
                    if i > 0:
                        cost[:, :, i, :, :-i] = refimg_fea[:,:,:,i:] - targetimg_fea[:,:,:,:-i]
                    else:
                        cost[:, :, i, :,:] = refimg_fea - targetimg_fea
            return cost

    def Coarsepred(self, cost):
        #pdb.set_trace()
        #cost = self.conv3d_1(cost)
        #cost = self.conv3d_2(cost) + cost
        #cost = self.conv3d_3(cost) + cost
        #cost = self.conv3d_4(cost) + cost

        #this isn't in
        cost = self.conv3d_1(cost)
        cost = self.conv3d_2(cost)
        cost = self.conv3d_3(cost)
        cost = self.conv3d_4(cost)
        
        cost = self.conv3d_5(cost)
        #the old code did the upsampling before classification
        #pdb.set_trace()
        #cost = F.interpolate(cost, size=[self.maxdisp, self.img_shape[1], self.img_shape[0]], mode='trilinear', align_corners=False)
        #pdb.set_trace()
        #debug_presoftmax = cost
        #pred = cost.softmax(dim=2).squeeze(dim=1)
        #pred = self.disp_reg(pred)

        #lets try doing the upsampling after
        presoftmax = cost
        pred = cost.softmax(dim=2).squeeze(dim=1)
        pred = self.disp_reg2(pred) * self.scale_factor
        pred = F.interpolate(pred, size=[self.img_shape[1], self.img_shape[0]], mode='bilinear', align_corners=False)


        return pred, presoftmax
    
    def forward(self, refimg_fea, targetimg_fea):
        '''
        Args:
            refimg_fea: output of SiameseTower for a left image
            targetimg_fea: output of SiameseTower for the right image

        '''
        cost_left = self.costVolume(refimg_fea, targetimg_fea, 'left')
        #cost_right = self.costVolume(refimg_fea, targetimg_fea, 'right')

        pred_left, presoftmax = self.Coarsepred(cost_left)
        #pred_right = self.Coarsepred(cost_right)

        return pred_left, presoftmax#, pred_right
        


        
class RefineNet(nn.Module):
    def __init__(self, ch_in=3):
        super(RefineNet, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # stream_1, left_img
        self.conv1_s1 = conv_block(ch_in, 16, 3, 1)
        self.resblock1_s1 = ResBlock(16, 16, 3, 1, 1)
        self.resblock2_s1 = ResBlock(16, 16, 3, 1, 2)

        # stream_2, upsampled low_resolution disp
        self.conv1_s2 = conv_block(1, 16, 1, 1)
        self.resblock1_s2 = ResBlock(16, 16, 3, 1, 1)
        self.resblock2_s2 = ResBlock(16, 16, 3, 1, 2)

        # cat
        self.resblock3 = ResBlock(32, 32, 3, 1, 4) # todo: find out why padding of 4?
        self.resblock4 = ResBlock(32, 32, 3, 1, 8) # todo: find out why padding of 8?
        self.resblock5 = ResBlock(32, 32, 3, 1, 1)
        self.resblock6 = ResBlock(32, 32, 3, 1, 1)
        self.conv2 = conv_block(32, 1, 3, 1)

    def forward(self, left_img, up_disp):
        
        stream1 = self.conv1_s1(left_img)
        stream1 = self.resblock1_s1(stream1)
        stream1 = self.resblock2_s1(stream1)

        stream2 = self.conv1_s2(up_disp)
        stream2 = self.resblock1_s2(stream2)
        stream2 = self.resblock2_s2(stream2)

        out = torch.cat((stream1, stream2), 1)
        out = self.resblock3(out)
        out = self.resblock4(out)
        out = self.resblock5(out)
        out = self.resblock6(out)
        out = self.conv2(out)

        return out

        
class InvalidationNet(nn.Module):
    def __init__(self):
        super(InvalidationNet, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        #resblocks1 = [ResBlock(64, 64, 3, 1, 1)] * 5 # is this really what we want?
        resblocks1 = []
        for i in range(5):
            resblocks1.append(ResBlock(64, 64, 3, 1, 1))


        self.resblocks1 = nn.Sequential(*resblocks1)
        self.conv1 = conv_block(64, 1, 3, 1, norm=None, act=None)

        self.conv2 = conv_block(5, 32, 3, 1)
        #resblocks2 = [ResBlock(32, 32, 3, 1, 1)] * 4
        resblocks2 = []
        for i in range(4):
            resblocks2.append(ResBlock(32, 32, 3, 1, 1))
        self.resblocks2 = nn.Sequential(*resblocks2)
        self.conv3 = conv_block(32, 1, 3, 1, norm=None, act=None)

    def forward(self, left_tower, right_tower, left_img, freso_disp):

        features = torch.cat((left_tower, right_tower), 1)
        out1 = self.resblocks1(features)
        out1 = self.conv1(out1)
        #todo: some kind of upsampling here?
        input = torch.cat((left_img, out1, freso_disp), 1)
        
        out2 = self.conv2(input)
        out2 = self.resblocks2(out2)
        out2 = self.conv3(out2)

        return out2
        


class ActiveStereoNet(nn.Module):
    def __init__(self, maxdisp, scale_factor, img_shape, ch_in=3):
        super(ActiveStereoNet, self).__init__()
        self.maxdisp = maxdisp
        self.scale_factor = scale_factor
        self.SiameseTower = SiameseTower2(scale_factor, ch_in=ch_in)
        # TODO: the original stereoNet paper is subtracting (so no concatenation)
        self.CoarseNet = CoarseNet(maxdisp, scale_factor, img_shape, concatenate=False)
        self.RefineNet = RefineNet(ch_in=ch_in)
        self.InvalidationNet = InvalidationNet()
        self.two_sided = False
        self.img_shape = img_shape


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        
        
    
    def forward(self, left, right):
        
        #pdb.set_trace()
        left_tower = self.SiameseTower(left)
        right_tower = self.SiameseTower(right)
        #pdb.set_trace()
        coarseup_pred, presoftmax = self.CoarseNet(left_tower, right_tower)
        #print(f"coarse {coarseup_pred.mean()}")
        #print(coarseup_pred.shape)
        res_disp = self.RefineNet(left, coarseup_pred)
        #print(f"refinement {res_disp.mean()}")
        ref_pred = coarseup_pred + res_disp
        #ref_pred = coarseup_pred #debug: get rid of the refine step

        #invalidation = self.InvalidationNet()

        return F.leaky_relu(ref_pred), coarseup_pred, presoftmax#nn.ReLU(False)(ref_pred)

