import pydiffvg
import torch
import torchvision
from PIL import Image
import numpy as np

# Use GPU if available
pydiffvg.set_use_gpu(torch.cuda.is_available())

def inv_exp(a,x,xpow=1):
    return pow(a,pow(1.-x,xpow))

import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F

import visdom

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)

vis=visdom.Visdom(port=8080)

smoothing = GaussianSmoothing(4, 5, 1)

settings=pydiffvg.SvgOptimizationSettings()
settings.global_override(["optimize_color"],False)
settings.global_override(["optimize_alpha"],False)
settings.global_override(["gradients","optimize_color"],False)
settings.global_override(["gradients","optimize_alpha"],False)
settings.global_override(["gradients","optimize_stops"],False)
settings.global_override(["gradients","optimize_location"],False)
settings.global_override(["optimizer"],"Adam")
settings.global_override(["paths","optimize_points"],False)
settings.global_override(["transforms","transform_lr"],1e-2)
settings.undefault("linearGradient3152")
settings.retrieve("linearGradient3152")[0]["transforms"]["optimize_transforms"]=False

#optim=pydiffvg.OptimizableSvg("note_small.svg",settings,verbose=True)
optim=pydiffvg.OptimizableSvg("heart_green.svg",settings,verbose=True)

#img=torchvision.transforms.ToTensor()(Image.open("note_transformed.png")).permute(1,2,0)
img=torchvision.transforms.ToTensor()(Image.open("heart_green_90.png")).permute(1,2,0)

name="heart_green_90"

pydiffvg.imwrite(img.cpu(), 'results/simple_transform_svg/target.png')
target = img.clone().detach().requires_grad_(False)

img=optim.render()
pydiffvg.imwrite(img.cpu(), 'results/simple_transform_svg/init.png')

def smooth(input, kernel):
    input=torch.nn.functional.pad(input.permute(2,0,1).unsqueeze(0), (2, 2, 2, 2), mode='reflect')
    output=kernel(input)
    return output

def printimg(optim):
    img=optim.render()
    comp = img.clone().detach()
    bg = torch.tensor([[[1., 1., 1.]]])
    comprgb = comp[:, :, 0:3]
    compalpha = comp[:, :, 3].unsqueeze(2)
    comp = comprgb * compalpha \
           + bg * (1 - compalpha)
    return comp

def comp_loss_and_grad(img, tgt, it, sz):
    dif=img-tgt

    loss=dif.pow(2).mean()

    dif=dif.detach()

    cdif=dif.clone().abs()
    cdif[:,:,3]=1.

    resdif=torch.nn.functional.interpolate(cdif.permute(2,0,1).unsqueeze(0),sz,mode='bilinear').squeeze().permute(1,2,0).abs()
    pydiffvg.imwrite(resdif[:,:,0:4], 'results/simple_transform_svg/dif_{:04}.png'.format(it))

    dif=dif.numpy()
    padded=np.pad(dif,[(1,1),(1,1),(0,0)],mode='edge')
    #print(padded[:-2,:,:].shape)
    grad_x=(padded[:-2,:,:]-padded[2:,:,:])[:,1:-1,:]
    grad_y=(padded[:,:-2,:]-padded[:,2:,:])[1:-1,:,:]

    resshape=dif.shape
    resshape=(resshape[0],resshape[1],2)
    res=np.zeros(resshape)

    for x in range(resshape[0]):
        for y in range(resshape[1]):
            A=np.concatenate((grad_x[x,y,:][:,np.newaxis],grad_y[x,y,:][:,np.newaxis]),axis=1)
            b=-dif[x,y,:]
            v=np.linalg.lstsq(np.dot(A.T,A),np.dot(A.T,b))
            res[x,y,:]=v[0]

    return loss, res

import colorsys
def print_gradimg(gradimg,it,shape=None):
    out=torch.zeros((gradimg.shape[0],gradimg.shape[1],3),requires_grad=False,dtype=torch.float32)
    for x in range(gradimg.shape[0]):
        for y in range(gradimg.shape[1]):
            h=math.atan2(gradimg[x,y,1],gradimg[x,y,0])
            s=math.tanh(np.linalg.norm(gradimg[x,y,:]))
            v=1.
            vec=(gradimg[x,y,:].clip(min=-1,max=1)/2)+.5
            #out[x,y,:]=torch.tensor(colorsys.hsv_to_rgb(h,s,v),dtype=torch.float32)
            out[x,y,:]=torch.tensor([vec[0],vec[1],0])

    if shape is not None:
        out=torch.nn.functional.interpolate(out.permute(2,0,1).unsqueeze(0),shape,mode='bilinear').squeeze().permute(1,2,0)
    pydiffvg.imwrite(out.cpu(), 'results/simple_transform_svg/grad_{:04}.png'.format(it))

# Run 150 Adam iterations.
for t in range(1000):
    print('iteration:', t)
    optim.zero_grad()
    with open('results/simple_transform_svg/viter_{:04}.svg'.format(t),"w") as f:
        f.write(optim.write_xml())
    scale=inv_exp(1/16,math.pow(t/1000,1),0.5)
    #print(scale)
    #img = optim.render(seed=t+1,scale=scale)
    img = optim.render(seed=t + 1, scale=None)
    vis.line(torch.tensor([img.shape[0]]), X=torch.tensor([t]), win=name + " size", update="append",
             opts={"title": name + " size"})
    #print(img.shape)
    #img = optim.render(seed=t + 1)

    ptgt=target.permute(2,0,1).unsqueeze(0)
    sz=img.shape[0:2]
    restgt=torch.nn.functional.interpolate(ptgt,size=sz,mode='bilinear').squeeze().permute(1,2,0)

    # Compute the loss function. Here it is L2.
    #loss = (smooth(img,smoothing) - smooth(restgt,smoothing)).pow(2).mean()
    #loss = (img - restgt).pow(2).mean()
    #loss=(img-target).pow(2).mean()
    loss,gradimg=comp_loss_and_grad(img, restgt,t,target.shape[0:2])
    print_gradimg(gradimg,t,target.shape[0:2])
    print('loss:', loss.item())
    vis.line(loss.unsqueeze(0), X=torch.tensor([t]), win=name+" loss", update="append",
             opts={"title": name + " loss"})

    # Backpropagate the gradients.
    loss.backward()

    # Take a gradient descent step.
    optim.step()

    # Save the intermediate render.
    comp=printimg(optim)
    pydiffvg.imwrite(comp.cpu(), 'results/simple_transform_svg/iter_{:04}.png'.format(t))


# Render the final result.

img = optim.render()
# Save the images and differences.
pydiffvg.imwrite(img.cpu(), 'results/simple_transform_svg/final.png')
with open('results/simple_transform_svg/final.svg', "w") as f:
    f.write(optim.write_xml())

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/simple_transform_svg/iter_%04d.png", "-vb", "20M",
    "results/simple_transform_svg/out.mp4"])

call(["ffmpeg", "-framerate", "24", "-i",
    "results/simple_transform_svg/grad_%04d.png", "-vb", "20M",
    "results/simple_transform_svg/out_grad.mp4"])

