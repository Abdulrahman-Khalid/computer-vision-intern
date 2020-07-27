import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum()
    
    
    def forward(self, outputs, anchor_idx, pos_idx, neg_idx, size_average=True):
        anchor = outputs[anchor_idx].unsqueeze(0)
        pos = outputs[pos_idx].unsqueeze(0)
        neg = outputs[neg_idx].unsqueeze(0)

        distance_positive = nn.functional.pairwise_distance(anchor, pos)
        distance_negative = nn.functional.pairwise_distance(anchor, neg)
#         print("distance_positive:", distance_positive)
#         print("distance_negative:", distance_negative)
        
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()

class TripletLossMiniBatch(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLossMiniBatch, self).__init__()
        self.margin = margin
        
    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)
    
    def forward(self, outputs, anchors, positives, negatives, size_average=True):
        distance_positive = nn.functional.pairwise_distance(outputs[anchors], outputs[positives])
        distance_negative = nn.functional.pairwise_distance(outputs[anchors], outputs[negatives])        
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()

def get_triplet(labels, foreground, background):
    anchor_label = random.randint(0, 1)
    if anchor_label == 0:
        # choose index from foreground and background
        anchor_idx = random.randrange(0, len(background))
        pos_idx = random.randrange(0, len(background))
        while anchor_idx == pos_idx:
            pos_idx = random.randrange(0, len(background))
        neg_idx = random.randrange(0, len(foreground))

        # get real index from outputs 
        anchor_idx = background[anchor_idx]
        pos_idx = background[pos_idx]
        neg_idx = foreground[neg_idx]

    elif anchor_label == 1:
        # choose index from foreground and background
        anchor_idx = random.randrange(0, len(foreground))
        pos_idx = random.randrange(0, len(foreground))
        while anchor_idx == pos_idx:
            pos_idx = random.randrange(0, len(foreground))
        neg_idx = random.randrange(0, len(background))

        # get real index from outputs
        anchor_idx = foreground[anchor_idx]
        pos_idx = foreground[pos_idx]
        neg_idx = background[neg_idx]
    return anchor_idx, pos_idx, neg_idx

class EmbeddingNet(nn.Module):
    def __init__(self, last_out_channel=8):
        super(EmbeddingNet, self).__init__()
        self.last_out_channel = last_out_channel
        self.convnet = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=4, kernel_size=(3, 3), stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=4, out_channels=self.last_out_channel, kernel_size=(3, 3), stride=1, padding=1),
                                     nn.ReLU())
    def forward(self, x):
        x = self.convnet(x)
        x = torch.transpose(x, 0, 2)
        x = torch.transpose(x, 1, 3)
        x = x.reshape(-1, self.last_out_channel) 
        return x


class EmbeddingNet2(nn.Module):
    def __init__(self, last_out_channel=32):
        super(EmbeddingNet2, self).__init__()
        self.last_out_channel = last_out_channel
        self.convnet = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=4, kernel_size=(3, 3), stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3, 3), stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=16, out_channels=self.last_out_channel, kernel_size=(3, 3), stride=1, padding=1),
                                     nn.ReLU())
    def forward(self, x):
        x = self.convnet(x)
#         print("x.shape:", x.shape)
#         x = x.view(-1, self.last_out_channel)
#         print("x.shape:", x.shape)
        x = torch.transpose(x, 0, 2)
        x = torch.transpose(x, 1, 3)
        x = x.reshape(-1, self.last_out_channel) 
        return x
    
class EmbeddingNet3(nn.Module):
    def __init__(self, last_out_channel=64):
        super(EmbeddingNet3, self).__init__()
        self.last_out_channel = last_out_channel
        self.convnet = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=4, kernel_size=(3, 3), stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3, 3), stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=32, out_channels=self.last_out_channel, kernel_size=(3, 3), stride=1, padding=1),
                                     nn.ReLU())
    def forward(self, x):
        x = self.convnet(x)
        x = torch.transpose(x, 0, 2)
        x = torch.transpose(x, 1, 3)
        x = x.reshape(-1, self.last_out_channel) 
        return x
    
class EmbeddingNet4(nn.Module):
    def __init__(self, last_out_channel=256):
        super(EmbeddingNet4, self).__init__()
        self.last_out_channel = last_out_channel
        self.convnet = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=4, kernel_size=(3, 3), stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3, 3), stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=128, out_channels=self.last_out_channel, kernel_size=(3, 3), stride=1, padding=1),
                                     nn.ReLU())
    def forward(self, x):
        x = self.convnet(x)
        x = torch.transpose(x, 0, 2)
        x = torch.transpose(x, 1, 3)
        x = x.reshape(-1, self.last_out_channel) 
        return x

class EmbeddingNet5(nn.Module):
    def __init__(self, last_out_channel=256):
        super(EmbeddingNet5, self).__init__()
        self.last_out_channel = last_out_channel
        self.convnet = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=4, kernel_size=(3, 3), stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3, 3), stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=128, out_channels=self.last_out_channel, kernel_size=(3, 3), stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=self.last_out_channel, out_channels=self.last_out_channel, kernel_size=(3, 3), stride=1, padding=1),
                                     nn.ReLU())
    def forward(self, x):
        x = self.convnet(x)
        x = torch.transpose(x, 0, 2)
        x = torch.transpose(x, 1, 3)
        x = x.reshape(-1, self.last_out_channel) 
        return x


class EmbeddingNet6(nn.Module):
    def __init__(self, last_out_channel=256):
        super(EmbeddingNet6, self).__init__()
        self.last_out_channel = last_out_channel
        self.convnet = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=4, kernel_size=(3, 3), stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3, 3), stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=128, out_channels=self.last_out_channel, kernel_size=(3, 3), stride=1, padding=1),
                                     nn.BatchNorm2d(self.last_out_channel),
                                     nn.ReLU())
    def forward(self, x):
        x = self.convnet(x)
        x = torch.transpose(x, 0, 2)
        x = torch.transpose(x, 1, 3)
        x = x.reshape(-1, self.last_out_channel) 
        return x

class EmbeddingNet7(nn.Module):
    def __init__(self, last_out_channel=8):
        super(EmbeddingNet7, self).__init__()
        self.last_out_channel = last_out_channel
        self.convnet = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=4, kernel_size=(3, 3), stride=1, padding=1),
                                     nn.BatchNorm2d(4),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=4, out_channels=self.last_out_channel, kernel_size=(3, 3), stride=1, padding=1),
                                     nn.BatchNorm2d(self.last_out_channel),
                                     nn.ReLU())
    def forward(self, x):
        x = self.convnet(x)
        x = torch.transpose(x, 0, 2)
        x = torch.transpose(x, 1, 3)
        x = x.reshape(-1, self.last_out_channel) 
        return x
    
class EmbeddingNet8(nn.Module):
    def __init__(self, last_out_channel=256):
        super(EmbeddingNet8, self).__init__()
        self.last_out_channel = last_out_channel
        self.convnet = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), stride=1, padding=1),
                                     nn.BatchNorm2d(8),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1, padding=1),
                                     nn.BatchNorm2d(16),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=1),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=128, out_channels=self.last_out_channel, kernel_size=(3, 3), stride=1, padding=1),
                                     nn.BatchNorm2d(self.last_out_channel),
                                     nn.ReLU())
    def forward(self, x):
        x = self.convnet(x)
        x = torch.transpose(x, 0, 2)
        x = torch.transpose(x, 1, 3)
        x = x.reshape(-1, self.last_out_channel) 
        return x
    
class EmbeddingNet9(nn.Module):
    def __init__(self, last_out_channel=64):
        super(EmbeddingNet9, self).__init__()
        self.last_out_channel = last_out_channel
        self.convnet = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1, padding=1),
                                     nn.BatchNorm2d(16),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=64, out_channels=self.last_out_channel, kernel_size=(3, 3), stride=1, padding=1),
                                     nn.BatchNorm2d(self.last_out_channel),
                                     nn.ReLU())
    def forward(self, x):
        x = self.convnet(x)
        x = torch.transpose(x, 0, 2)
        x = torch.transpose(x, 1, 3)
        x = x.reshape(-1, self.last_out_channel) 
        return x   

# I didn't use this init function I depend on pytorch default
def init_weights(m):
    if isinstance(m, nn.Conv2d):
#         torch.nn.init.kaiming_normal_(m.weight)
        torch.nn.init.normal_(m.weight, mean=0, std=1.0)