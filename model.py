import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
		)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        outputs = self.conv2(x)
        return outputs
    
class SCNN(nn.Module):
    def __init__(
            self,
            input_size,
            ms_ks=9,
            pretrained=True
    ):
        """
        Argument
            ms_ks: kernel size in message passing conv
        """
        super(SCNN, self).__init__()
        self.pretrained = pretrained
        self.net_init(input_size, ms_ks)
        if not pretrained:
            self.weight_init()

        self.scale_background = 0.4
        self.scale_seg = 1.0
        self.scale_exist = 0.1

        self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([self.scale_background, 1, 1, 1, 1]))
        self.bce_loss = nn.BCELoss()
    
    
    def forward(self, img, seg_gt=None, exist_gt=None):
        # Complete the forward function for the rest of the encoder

        x1 = self.conv1(img)  #64
        #print(x1.shape)
        x2 = self.max_pool(x1)
        #print(x2.shape)
        x3 = self.conv2(x2)   #128
        #print(x3.shape)
        x4 = self.max_pool(x3)
        #print(x4.shape)
        x5 = self.conv3(x4)   #256 
        #print(x5.shape)
        x6 = self.max_pool(x5)
        #print(x6.shape)
        x7 = self.conv4(x6)   #512 
        #print(x7.shape)
        x8 = self.message_passing_forward(x7) #512
        #print(x8.shape)
        # Complete the forward function for the rest of the decoder

        x = self.detrans1(x8) #256 
        #print(x.shape)
        x = self.deconv1(torch.cat([x,x5],1))
        #print(x.shape)
        x = self.detrans2(x)  #128 
        #print(x.shape)
        x = self.deconv2(torch.cat([x,x3],1))
        #print(x.shape)
        x = self.detrans3(x)  #64 
        #print(x.shape)
        x = self.deconv3(torch.cat([x,x1],1))
        #print(x.shape)
        x = self.layer2(x)
        #print("after layer2", x.shape)
        seg_pred = x
        #seg_pred = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=True)
        #print("seg_pred", seg_pred.shape)
        
        #print("x8", x8.shape) 
        #x = self.layer4(x8)
        #print("layer4", x.shape) 
        x = self.layer3(x)
        #print("layer3", x.shape) 
        #print("x", x.shape)
        x = x.view(-1, self.fc_input_feature)
        #print("x", x.shape)
        exist_pred = self.fc(x)
        #print("exist_pred", exist_pred.shape)
        if seg_gt is not None and exist_gt is not None:
            loss_seg = self.ce_loss(seg_pred, seg_gt)
            loss_exist = self.bce_loss(exist_pred, exist_gt)
            loss = loss_seg * self.scale_seg + loss_exist * self.scale_exist
        else:
            loss_seg = torch.tensor(0, dtype=img.dtype, device=img.device)
            loss_exist = torch.tensor(0, dtype=img.dtype, device=img.device)
            loss = torch.tensor(0, dtype=img.dtype, device=img.device)

        return seg_pred, exist_pred, loss_seg, loss_exist, loss
    
    
    '''
    def forward(self, img, seg_gt=None, exist_gt=None):
        x = self.backbone(img)
        #print(x.shape)
        x = self.layer1(x)
        #print(x.shape)
        x = self.message_passing_forward(x)
        #print(x.shape)
        x = self.layer2(x)
        #print("after layer2", x.shape)
        seg_pred = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=True)
        #print("seg_pred", seg_pred.shape)
        x = self.layer3(x)
        x = x.view(-1, self.fc_input_feature)
        exist_pred = self.fc(x)

        if seg_gt is not None and exist_gt is not None:
            loss_seg = self.ce_loss(seg_pred, seg_gt)
            loss_exist = self.bce_loss(exist_pred, exist_gt)
            loss = loss_seg * self.scale_seg + loss_exist * self.scale_exist
        else:
            loss_seg = torch.tensor(0, dtype=img.dtype, device=img.device)
            loss_exist = torch.tensor(0, dtype=img.dtype, device=img.device)
            loss = torch.tensor(0, dtype=img.dtype, device=img.device)

        return seg_pred, exist_pred, loss_seg, loss_exist, loss
    '''
    def message_passing_forward(self, x):
        Vertical = [True, True, False, False]
        Reverse = [False, True, False, True]
        for ms_conv, v, r in zip(self.message_passing, Vertical, Reverse):
            x = self.message_passing_once(x, ms_conv, v, r)
        return x

    def message_passing_once(self, x, conv, vertical=True, reverse=False):
        """
        Argument:
        ----------
        x: input tensor
        vertical: vertical message passing or horizontal
        reverse: False for up-down or left-right, True for down-up or right-left
        """
        nB, C, H, W = x.shape
        if vertical:
            slices = [x[:, :, i:(i + 1), :] for i in range(H)]
            dim = 2
        else:
            slices = [x[:, :, :, i:(i + 1)] for i in range(W)]
            dim = 3
        if reverse:
            slices = slices[::-1]

        out = [slices[0]]
        for i in range(1, len(slices)):
            out.append(slices[i] + F.relu(conv(out[i - 1])))
        if reverse:
            out = out[::-1]
        return torch.cat(out, dim=dim)

    def net_init(self, input_size, ms_ks):
        input_w, input_h = input_size
        self.fc_input_feature = 5 * int(input_w/2) * int(input_h/2)
        #self.fc_input_feature = 5 * int(input_w/16) * int(input_h/16)
        #self.backbone = models.vgg16_bn(pretrained=self.pretrained).features
        '''
        self.backbone = models.resnet50(pretrained=self.pretrained)
        print(self.backbone)
        modules=list(self.backbone.children())[:-4]
        self.backbone = nn.Sequential(*modules)
        print(self.backbone)
        '''
        # ----------------- process backbone -----------------
        
        '''
        for i in [34, 37, 40]:
            conv = self.backbone._modules[str(i)]
            dilated_conv = nn.Conv2d(
                conv.in_channels, conv.out_channels, conv.kernel_size, stride=conv.stride,
                padding=tuple(p * 2 for p in conv.padding), dilation=2, bias=(conv.bias is not None)
            )
            dilated_conv.load_state_dict(conv.state_dict())
            self.backbone._modules[str(i)] = dilated_conv
        self.backbone._modules.pop('33')
        self.backbone._modules.pop('43')
        
        
        # ----------------- SCNN part -----------------
        self.layer1 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()  # (nB, 128, 36, 100)
        )
        
        
        # ----------------- add message passing -----------------
        
        self.message_passing = nn.ModuleList()
        self.message_passing.add_module('up_down', nn.Conv2d(128, 128, (1, ms_ks), padding=(0, ms_ks // 2), bias=False))
        self.message_passing.add_module('down_up', nn.Conv2d(128, 128, (1, ms_ks), padding=(0, ms_ks // 2), bias=False))
        self.message_passing.add_module('left_right',
                                        nn.Conv2d(128, 128, (ms_ks, 1), padding=(ms_ks // 2, 0), bias=False))
        self.message_passing.add_module('right_left',
                                        nn.Conv2d(128, 128, (ms_ks, 1), padding=(ms_ks // 2, 0), bias=False))
        # (nB, 128, 36, 100)
        
        
        # ----------------- SCNN part -----------------
        self.layer2 = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(128, 5, 1)  # get (nB, 5, 36, 100)
        )
        
        
        # ----------------- SCNN part -----------------
        
        self.layer3 = nn.Sequential(
            nn.Softmax(dim=1),  # (nB, 5, 36, 100)
            nn.AvgPool2d(2, 2),  # (nB, 5, 18, 50)
        )
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_feature, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid()
        )
        '''
        
        # ----------------- encoder part ----------------- 
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = DoubleConv(3, 64)
        self.conv2 = DoubleConv(64, 128)
        self.conv3 = DoubleConv(128, 256)
        self.conv4 = DoubleConv(256, 512)
       
        # ----------------- add message passing -----------------
        
        self.message_passing = nn.ModuleList()
        self.message_passing.add_module('up_down', nn.Conv2d(512, 512, (1, ms_ks), padding=(0, ms_ks // 2), bias=False))
        self.message_passing.add_module('down_up', nn.Conv2d(512, 512, (1, ms_ks), padding=(0, ms_ks // 2), bias=False))
        self.message_passing.add_module('left_right',
                                        nn.Conv2d(512, 512, (ms_ks, 1), padding=(ms_ks // 2, 0), bias=False))
        self.message_passing.add_module('right_left',
                                        nn.Conv2d(512, 512, (ms_ks, 1), padding=(ms_ks // 2, 0), bias=False))
        # (nB, 1024, x, x)
        
        # ----------------- Decoder ------------------- 
        
        self.detrans1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.deconv1  = DoubleConv(512, 256)
        self.detrans2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.deconv2  = DoubleConv(256, 128)
        self.detrans3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.deconv3  = DoubleConv(128, 64)
        
        # ----------------- SCNN part -----------------
        self.layer2 = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(64, 5, 1)  # get (nB, 5, x, x)
        )
        
        # ----------------- SCNN part -----------------
        
        self.layer3 = nn.Sequential(
            nn.Softmax(dim=1),  # (nB, 5, 36, 100)
            nn.AvgPool2d(2, 2),  # (nB, 5, 18, 50)
        )
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_feature, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid()
        )
        
        
    
    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.xavier_uniform(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data[:] = 1.
                m.bias.data.zero_()
    
    
    
    '''
    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data[:] = 1.
                m.bias.data.zero_()
    '''