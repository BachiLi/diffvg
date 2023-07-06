from cicada_rendering import main
import argparse
from joblib import Parallel, delayed
import time
from plotly import graph_objects as go

parser = argparse.ArgumentParser()
parser.add_argument("--target", help="target image path")
parser.add_argument("--num_paths", type=int, default=256)
parser.add_argument("--max_width", type=float, default=8.0)
parser.add_argument("--use_lpips_loss", dest='use_lpips_loss', action='store_true')
parser.add_argument("--num_iter", type=int, default=500)
parser.add_argument("--use_blob", dest='use_blob', action='store_true')
parser.add_argument("--imsize", type=int, default=0)
args = parser.parse_args()

args.use_lpips_loss=True
args.imsize = 0
args.num_iter = 50
args.num_path = 256

# for n in range(240):
#     target = f'../../stablediffusion/outputs/txt2img-samples/samples/{n:05d}.png'
#     savepath = f'results/cicada_db/{n:05d}.png'
#     main(args)
    # assert False

def run_iter(n):
    target = f'../../stablediffusion/outputs/txt2img-samples/samples/{n:05d}.png'
    savepath = f'results/cicada_db/{n:05d}.png'
    return main(args, targetpath=target, savepath=savepath)
    


nn = [1, 2, 3, 4]
yy = []
# for n in nn:
n=2
import torch
# Fuck it, I'll parallelize using batches
torch.cuda.empty_cache()
start = time.time()
results = Parallel(n_jobs=n)(delayed(run_iter)(k) for k in range(n))
yy.append(int((time.time()-start)/n))
print(n, yy)

fig = go.Figure(go.Scatter(x=nn ,y=yy))
fig.show()

# start = time.time()
# for n in range(6):
#     run_iter(n)
# print(int(time.time()-start))