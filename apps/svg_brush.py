import sys
sys.path.append("../svg")
from geometry import GeometryLoss
import numpy as np
import pygame as pg
import torch
import pydiffvg
import tkinter as tk
from tkinter import filedialog

def box_kernel(val):
    return np.heaviside(-val+1,0)

def cone_kernel(val):
    return np.maximum(0,1-val)

def nptosurf(arr):
    if arr.shape[2]==1:
        #greyscale
        shape=arr.shape
        shape=(shape[0],shape[1],3)
        arr=np.broadcast_to(arr,shape)
    return pg.surfarray.make_surface(arr*255)

def brush_tensor(screen_size,coords,radius,kernel):
    coordarr=np.stack(np.meshgrid(np.linspace(0,screen_size[0]-1,screen_size[0]),np.linspace(0,screen_size[1]-1,screen_size[1]),indexing='ij'),axis=2)
    ctrarr = np.reshape(np.array(coords), [1, 1, 2])
    distarr=np.sqrt(np.sum(np.power(coordarr-ctrarr,2),axis=2))
    valarr=kernel(distarr/radius)
    return torch.tensor(valarr,requires_grad=False,dtype=torch.float32)

def checkerboard(shape, square_size=2):
    xv,yv=np.meshgrid(np.floor(np.linspace(0,shape[1]-1,shape[1])/square_size),np.floor(np.linspace(0,shape[0]-1,shape[0])/square_size))
    bin=np.expand_dims(((xv+yv)%2),axis=2)
    res=bin*np.array([[[1., 1., 1.,]]])+(1-bin)*np.array([[[.75, .75, .75,]]])
    return torch.tensor(res,requires_grad=False,dtype=torch.float32)

def render(optim, viewport):
    scene_args = pydiffvg.RenderFunction.serialize_scene(*optim.build_scene())
    render = pydiffvg.RenderFunction.apply
    img = render(viewport[0],  # width
                 viewport[1],  # height
                 2,  # num_samples_x
                 2,  # num_samples_y
                 0,  # seed
                 None,
                 *scene_args)
    return img

def optimize(optim, viewport, brush_kernel, increase=True, strength=0.1):
    optim.zero_grad()

    geomLoss=torch.tensor(0.)

    for shape, gloss in zip(optim.scene[2],geometryLosses):
        geomLoss+=gloss.compute(shape)

    img=render(optim,viewport)

    imalpha=img[:,:,3]

    multiplied=imalpha*brush_kernel

    loss=((1-multiplied).mean() if increase else multiplied.mean())*strength

    loss+=geomLoss

    loss.backward()

    optim.step()

    return render(optim,viewport)

def get_infile():
    pydiffvg.set_use_gpu(False)
    root = tk.Tk()
    #root.withdraw()

    file_path = filedialog.askopenfilename(initialdir = ".",title = "Select graphic to optimize",filetypes = (("SVG files","*.svg"),("all files","*.*")))

    root.destroy()

    return file_path

def compositebg(img):
    bg=checkerboard(img.shape,2)
    color=img[:,:,0:3]
    alpha=img[:,:,3]
    composite=alpha.unsqueeze(2)*color+(1-alpha).unsqueeze(2)*bg

    return composite

def main():
    infile=get_infile()

    settings=pydiffvg.SvgOptimizationSettings()
    settings.global_override(["optimize_color"],False)
    settings.global_override(["transforms","optimize_transforms"], False)
    settings.global_override(["optimizer"], "SGD")
    settings.global_override(["paths","shape_lr"], 1e-1)

    optim=pydiffvg.OptimizableSvg(infile,settings)

    global geometryLosses
    geometryLosses = []

    for shape in optim.build_scene()[2]:
        geometryLosses.append(GeometryLoss(shape))

    scaling=1
    brush_radius=100
    graphic_size=optim.canvas
    screen_size=(graphic_size[1]*scaling, graphic_size[0]*scaling)

    pg.init()

    screen=pg.display.set_mode(screen_size)
    screen.fill((255,255,255))

    img=render(optim,graphic_size)
    print(img.max())

    npsurf = pg.transform.scale(nptosurf(compositebg(img).detach().permute(1,0,2).numpy()), screen_size)

    screen.blit(npsurf,(0,0))

    pg.display.update()
    clock=pg.time.Clock()

    z=0
    btn=0

    while True:
        clock.tick(60)
        for event in pg.event.get():
            if event.type==pg.QUIT:
                pg.quit()
                sys.exit()

            y, x = pg.mouse.get_pos()
            if event.type == pg.MOUSEBUTTONDOWN:
                if event.button in [1,3]:
                    z=1
                    btn=event.button
                elif event.button == 4:
                    brush_radius*=1.1
                elif event.button == 5:
                    brush_radius/=1.1
                    brush_radius=max(brush_radius,5)
            elif event.type == pg.MOUSEBUTTONUP:
                if event.button in [1,3]:
                    z=0

        if z==1:
            brush=brush_tensor((graphic_size[0],graphic_size[1]), (x/scaling, y/scaling), brush_radius, box_kernel)
            img=optimize(optim,graphic_size,brush,btn==1)
            npsurf = pg.transform.scale(nptosurf(compositebg(img).detach().permute(1,0,2).numpy()), screen_size)


        screen.blit(npsurf,(0,0))
        pg.draw.circle(screen, (255,255,255), (y,x), int(brush_radius*scaling), 1)
        pg.display.update()


if __name__ == '__main__':
    main()

