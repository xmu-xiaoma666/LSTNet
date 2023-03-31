
import torch
import numpy as np
from torch._C import dtype
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF


class ImageAugmentation():
    def __init__(self):
        super().__init__()

    """
        PepperSaltNoise
    """
    def addPepperSaltNoise(self,detections,p=0.2,pn=0.05):
        feat=detections
        if(np.random.rand()<p):
            bs,grids,dim=detections.shape
            maxnum=detections.max().item()
            minnum=detections.min().item()
            peper=torch.full((dim,),maxnum)
            salt=torch.full((dim,),minnum)

            #add bs*grids*p Peppers
            for _ in range(int(bs*grids*pn)):
                row=np.random.randint(bs)
                col=np.random.randint(grids)
                feat[row][col]=peper

            #add bs*grids*p Salts
            for _ in range(int(bs*grids*pn)):
                row=np.random.randint(bs)
                col=np.random.randint(grids)
                feat[row][col]=salt

        return feat

    """
       GaussianNoise 
    """
    def addGaussianNoise(self,detections,p=0.2,mean=0,var=0.0001):
        feat=detections
        if(np.random.randn()<p):
            maxnum=detections.max().item()
            normdet=detections/maxnum
            #generate guassian noise
            noise = torch.from_numpy(np.random.normal(mean, var ** 0.5, detections.shape))
            newdet=normdet+noise
            newdet=torch.clamp(newdet,0,1)
            feat=newdet*maxnum
        return feat.to(torch.float32)

    """
       resizePool
    """
    def resizePool(self,detections,p=0.2,poolsize=2,stride=2):
        feat=detections
        if(np.random.randn()<p):
            m = nn.MaxPool2d(poolsize, stride=stride)
            use_feat=detections[:,:49,:]
            # nouse_feat=detections[:,49:,:]
            bs,gs,dim=use_feat.shape
            use_feat=use_feat.view(bs,int(np.sqrt(gs)),int(np.sqrt(gs)),dim)
            #maxpool
            output= m(use_feat.permute(0,3,1,2))
            #upsample
            output= F.interpolate(output, size=[int(np.sqrt(gs)),int(np.sqrt(gs))])
            output=output.permute(0,2,3,1)
            output=output.view(bs,-1,dim)
            feat=output
            # feat=torch.cat((output,nouse_feat),dim=1)
        return feat

    """
       RandomCrop
    """
    def randomCrop(self,detections,p=0.2,cropsize=5):
        feat=detections
        if(np.random.randn()<p):
            use_feat=detections[:,:49,:]
            # nouse_feat=detections[:,49:,:]
            bs,gs,dim=use_feat.shape
            use_feat=use_feat.view(bs,int(np.sqrt(gs)),int(np.sqrt(gs)),dim)
            use_feat=use_feat.permute(0,3,1,2)

            #crop
            startRange=np.sqrt(gs)-cropsize
            startRow=np.random.randint(startRange)
            startCol=np.random.randint(startRange)
            output=use_feat[:,:,startRow:startRow+cropsize,startCol:startCol+cropsize]

            #upsample
            output= F.interpolate(output, size=[int(np.sqrt(gs)),int(np.sqrt(gs))])
            output=output.permute(0,2,3,1)
            output=output.view(bs,-1,dim)
            feat=output
            # feat=torch.cat((output,nouse_feat),dim=1)
        return feat


    """
      RandomHorizontalFlip
    """
    def randomHorizontalFlip(self,detections,p=0.2):
        feat=detections
        if(np.random.randn()<p):
            use_feat=detections[:,:49,:]
            # nouse_feat=detections[:,49:,:]
            bs,gs,dim=use_feat.shape
            #reshape
            use_feat=use_feat.view(bs,int(np.sqrt(gs)),int(np.sqrt(gs)),dim)
            use_feat=use_feat.permute(0,3,1,2)

            #HorizontalFlip
            hflip=transforms.RandomHorizontalFlip(p=1)
            output=hflip(use_feat)

            #reshape
            output=output.permute(0,2,3,1)
            output=output.view(bs,-1,dim)
            feat=output
            # feat=torch.cat((output,nouse_feat),dim=1)
        return feat


    """
      RandomVerticalFlip
    """
    def randomVerticalFlip(self,detections,p=0.2):
        feat=detections
        if(np.random.randn()<p):
            use_feat=detections[:,:49,:]
            # nouse_feat=detections[:,49:,:]
            bs,gs,dim=use_feat.shape
            #reshape
            use_feat=use_feat.view(bs,int(np.sqrt(gs)),int(np.sqrt(gs)),dim)
            use_feat=use_feat.permute(0,3,1,2)

            #VerticalFlip
            vflip=transforms.RandomVerticalFlip(p=1)
            output=vflip(use_feat)

            #reshape
            output=output.permute(0,2,3,1)
            output=output.view(bs,-1,dim)
            feat=output
            # feat=torch.cat((output,nouse_feat),dim=1)
        return feat


    """
      randRotate
    """
    def randRotate(self,detections,p=0.5):
        feat=detections
        if(np.random.randn()<p):
            use_feat=detections[:,:49,:]
            # nouse_feat=detections[:,49:,:]
            bs,gs,dim=use_feat.shape
            #reshape
            use_feat=use_feat.view(bs,int(np.sqrt(gs)),int(np.sqrt(gs)),dim)
            use_feat=use_feat.permute(0,3,1,2)

            #rotate
            degree=np.random.randint(60)-30
            output=TF.rotate(use_feat,degree)

            #reshape
            output=output.permute(0,2,3,1)
            output=output.view(bs,-1,dim)
            feat=output
            # feat=torch.cat((output,nouse_feat),dim=1)
        return feat

    """
        channel shuffle
    """
    def channelShuffle(self,detections,p=0.2):
        feat=detections
        if(np.random.randn()<p):
            use_feat=detections[:,:49,:]
            # nouse_feat=detections[:,49:,:]
            bs,gs,dim=use_feat.shape
            #reshape
            use_feat=use_feat.view(bs,int(np.sqrt(gs)),int(np.sqrt(gs)),dim)
            use_feat=use_feat.permute(0,3,1,2)

            #channel shuffle
            indexs=np.arange(dim)
            np.random.shuffle(indexs)
            output=use_feat[:,indexs,:,:]

            #reshape
            output=output.permute(0,2,3,1)
            output=output.view(bs,-1,dim)
            feat=output
            # feat=torch.cat((output,nouse_feat),dim=1)
        return feat

    """
        randMask
    """
    def randMask(self,detections,p=0.3,pn=0.1):
        feat=detections
        if(np.random.rand()<p):
            bs,grids,dim=detections.shape
            salt=torch.full((dim,),0.0)

            #Mask
            for _ in range(int(bs*grids*pn)):
                row=np.random.randint(bs)
                col=np.random.randint(grids)
                feat[row][col]=salt

        return feat


    def randnChooseOne1(self,detections):
        feat=detections
        augs=['addPepperSaltNoise','addGaussianNoise']
        aug=augs[np.random.randint(len(augs))]
        feat=getattr(self,aug)(feat)
        return feat


    def randnChooseOne(self,detections):
        feat=detections
        augs=['addPepperSaltNoise','resizePool','randomCrop']
        aug=augs[np.random.randint(len(augs))]
        feat=getattr(self,aug)(feat)
        return feat

    def randnChooseOne3(self,detections):
        feat=detections
        augs=['addPepperSaltNoise','resizePool','randomCrop','randomHorizontalFlip','randomVerticalFlip','randRotate']
        aug=augs[np.random.randint(len(augs))]
        feat=getattr(self,aug)(feat)
        return feat

    def randnChooseOne4(self,detections):
        feat=detections
        augs=['addPepperSaltNoise','resizePool','randomCrop','randRotate']
        aug=augs[np.random.randint(len(augs))]
        feat=getattr(self,aug)(feat,p=0.3)
        return feat

    def randnChooseOne45(self,detections):
        feat=detections
        augs=['randomCrop']
        aug=augs[np.random.randint(len(augs))]
        feat=getattr(self,aug)(feat,p=0.3)
        return feat

    def randnChooseOne44(self,detections):
        feat=detections
        augs=['randomCrop','randRotate']
        aug=augs[np.random.randint(len(augs))]
        feat=getattr(self,aug)(feat,p=0.3)
        return feat

    def randnChooseOne5(self,detections):
        feat=detections
        augs=['addPepperSaltNoise','resizePool','randomCrop','randRotate']
        aug=augs[np.random.randint(len(augs))]
        feat=getattr(self,aug)(feat,p=0.5)
        return feat

    def randnChooseOne6(self,detections):
        feat=detections
        augs=['resizePool','randomCrop','randRotate','randMask']
        aug=augs[np.random.randint(len(augs))]
        feat=getattr(self,aug)(feat,p=0.6)
        return feat

    def randnChooseOne7(self,detections):
        feat=detections
        augs=['addPepperSaltNoise','resizePool','randomCrop','randRotate']
        aug=augs[np.random.randint(len(augs))]
        feat=getattr(self,aug)(feat,p=0.7)
        return feat