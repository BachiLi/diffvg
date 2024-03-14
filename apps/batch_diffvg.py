import pydiffvg
import torch
import skimage
import skimage.io
import random
import ttools.modules
import math
from utils import printProgressBar
import time
import datetime
import os
from dotenv import load_dotenv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--n_batch", type=int, default=0)
parser.add_argument("--num_batches", type=int, default=1)
args = parser.parse_args()

load_dotenv()
db_path = os.environ.get("DB_PATH")
out_path = os.environ.get("SVG_PATH")

gamma = 1.0


print("Using CUDA: ", torch.cuda.is_available())
torch.cuda.set_device(args.n_batch)
print(f"on device {torch.cuda.current_device()}")

class PathOptimizer:
    def __init__(self):
        pydiffvg.set_use_gpu(torch.cuda.is_available())
        self.device = pydiffvg.get_device()
        print(self.device)
        self.perception_loss = ttools.modules.LPIPS().to(self.device)
        self.render = pydiffvg.RenderFunction.apply
    
    def load_targets(self, target_paths, imsize=512, gamma=1.):
        targets = []
        for targetpath in target_paths:
            target = skimage.io.imread(targetpath)            
            target = torch.from_numpy(target).to(torch.float32) / 255.0
            target = target.pow(gamma)
            target = target.to(self.device)
            target = target.unsqueeze(0)
            target = target.permute(0, 3, 1, 2)
            target = torch.nn.functional.interpolate(target, size = [imsize, imsize], mode = 'area')
            targets.append(target)
        
        self.targets = torch.cat(targets, 0)


    def initialize(self, num_paths, max_width):

        self.max_width = max_width
        self.canvas_width, self.canvas_height = self.targets.shape[3], self.targets.shape[2]
        self.shapes_list = []
        self.shape_groups_list = []
        self.points_vars = []
        self.stroke_width_vars = []
        self.color_vars = []

        for n in range(self.targets.shape[0]):
            shapes = []
            shape_groups = []
            for i in range(num_paths):
                num_segments = random.randint(1, 3)
                num_control_points = torch.zeros(num_segments, dtype = torch.int32) + 2
                points = []
                p0 = (random.random(), random.random())
                points.append(p0)
                for j in range(num_segments):
                    radius = 0.05
                    p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
                    p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
                    p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
                    points.append(p1)
                    points.append(p2)
                    points.append(p3)
                    p0 = p3
                points = torch.tensor(points)
                points[:, 0] *= self.canvas_width
                points[:, 1] *= self.canvas_height
                path = pydiffvg.Path(num_control_points = num_control_points,
                                        points = points,
                                        stroke_width = torch.tensor(1.0),
                                        is_closed = False)
                shapes.append(path)
                path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes) - 1]),
                                                    fill_color = None,
                                                    stroke_color = torch.tensor([random.random(),
                                                                                random.random(),
                                                                                random.random(),
                                                                                random.random()]))
                shape_groups.append(path_group)
            
            scene_args = pydiffvg.RenderFunction.serialize_scene(\
                self.canvas_width, self.canvas_height, shapes, shape_groups)
            
            
            for path in shapes:
                path.points.requires_grad = True
                self.points_vars.append(path.points)

            for path in shapes:
                path.stroke_width.requires_grad = True
                self.stroke_width_vars.append(path.stroke_width)

            for group in shape_groups:
                group.stroke_color.requires_grad = True
                self.color_vars.append(group.stroke_color)

            self.shapes_list.append(shapes)
            self.shape_groups_list.append(shape_groups)

        self.points_optim = torch.optim.Adam(self.points_vars, lr=1.0)
        self.width_optim = torch.optim.Adam(self.stroke_width_vars, lr=0.1)
        self.color_optim = torch.optim.Adam(self.color_vars, lr=0.01)

    def run_iter(self, t):
        self.points_optim.zero_grad()
        self.width_optim.zero_grad()
        self.color_optim.zero_grad()

        images = []
        for k in range(self.targets.shape[0]):
            scene_args = pydiffvg.RenderFunction.serialize_scene(\
                self.canvas_width, self.canvas_height,
                self.shapes_list[k],
                self.shape_groups_list[k]
                )
            img = self.render(self.canvas_width, self.canvas_height, 2, 2, t, None, *scene_args)
            images.append(img.unsqueeze(0))
        
        images = torch.cat(images,0)
        
        # Compose img with white background
        images = images[:, :, :, 3:4] * images[:, :, :, :3] + torch.ones(images.shape[0], images.shape[1], images.shape[2], 3, device = pydiffvg.get_device()) * (1 - images[:, :, :, 3:4])

        images = images.permute(0, 3, 1, 2) # NHWC -> NCHW

        loss = 0
        loss += 0.4*self.perception_loss(images, self.targets) + (images.mean() - self.targets.mean()).pow(2)
        loss += 0.6*(images - self.targets).pow(2).mean()
    
        # Backpropagate the gradients.
        loss.backward()

        # Take a gradient descent step.
        self.points_optim.step()
        self.width_optim.step()
        self.color_optim.step()

        for k in range(self.targets.shape[0]):
            for path in self.shapes_list[k]:
                path.stroke_width.data.clamp_(1.0, self.max_width)  
            for group in self.shape_groups_list[k]:
                group.stroke_color.data.clamp_(0.0, 1.0)

    def build_images(self, imsize=512):
        images = []
        imscale = imsize/self.targets.shape[2]
        for k in range(self.targets.shape[0]):
            for m in range(len(self.shapes_list[k])):
                self.shapes_list[k][m].points = imscale*self.shapes_list[k][m].points
                self.shapes_list[k][m].stroke_width = imscale*self.shapes_list[k][m].stroke_width
            scene_args = pydiffvg.RenderFunction.serialize_scene(\
                imsize, imsize,
                self.shapes_list[k],
                self.shape_groups_list[k]
                )
            img = self.render(imsize, imsize, 2, 2, t, None, *scene_args)
            images.append(img.unsqueeze(0))
        
        images = torch.cat(images,0)
        
        # Compose img with white background
        images = images[:, :, :, 3:4] * images[:, :, :, :3] + torch.ones(images.shape[0], images.shape[1], images.shape[2], 3, device = pydiffvg.get_device()) * (1 - images[:, :, :, 3:4])
        self.images = images



class TimeCounter:
    def __init__(self, I, N):
        self.start = time.time()
        self.I = I
        self.N = N

    def estimate(self, i, n):
        if i+n>0:
            secs = int(time.time()-global_start)
            completed = (i*self.N+n)/(self.N*self.I)
            estimation = int(secs*(1-completed)/completed)
            print(f"""
                Completed {'{0:.1%}'.format(completed)}
                Elapsed time: {datetime.timedelta(seconds=secs)}.
                Estimated remain: {datetime.timedelta(seconds=estimation)}
            """)


batch_size = 5
num_iters = 400
items = os.listdir(db_path)
items.sort()
items = items[(args.n_batch*len(items))//args.num_batches : min(len(items), ((args.n_batch+1)*len(items))//args.num_batches)]
global_start = time.time()
time_counter = TimeCounter(len(items), 1000//batch_size)

for i, item in enumerate(items):
    folder = f'{db_path}'
    img_names = os.listdir(folder)

    n_batches = math.ceil(len(img_names)/batch_size)
    for n in range(n_batches):

        time_counter.estimate(i, n)

        batch_names = img_names[n*batch_size:min(len(img_names),(n+1)*batch_size)]
        target_paths = [f"{folder}/{name}" for name in batch_names]
    
        path_optimizer = PathOptimizer()
        path_optimizer.load_targets(target_paths, imsize=256)
        path_optimizer.initialize(256, 8)
        
        # Adam iterations.
        for t in range(num_iters):
            path_optimizer.run_iter(t)

        path_optimizer.build_images(imsize=512)
        for b, name in enumerate(batch_names):
            pydiffvg.imwrite(path_optimizer.images[b].cpu(), f'{out_path}/{name}', gamma=gamma)
            pydiffvg.imwrite(path_optimizer.images[b].cpu(), f'{out_path}/{name[:-3]}svg', gamma=gamma)
        
        
