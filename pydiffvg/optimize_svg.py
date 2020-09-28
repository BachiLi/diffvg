import json
import copy
import xml.etree.ElementTree as etree
from xml.dom import minidom
import warnings
import torch
import numpy as np
import re
import sys
import pydiffvg
import math
from collections import namedtuple
import cssutils

class SvgOptimizationSettings:

    default_params = {
        "optimize_color": True,
        "color_lr": 2e-3,
        "optimize_alpha": False,
        "alpha_lr": 2e-3,
        "optimizer": "Adam",
        "transforms": {
            "optimize_transforms":True,
            "transform_mode":"rigid",
            "translation_mult":1e-3,
            "transform_lr":2e-3
        },
        "circles": {
            "optimize_center": True,
            "optimize_radius": True,
            "shape_lr": 2e-1
        },
        "paths": {
            "optimize_points": True,
            "shape_lr": 2e-1
        },
        "gradients": {
            "optimize_stops": True,
            "stop_lr": 2e-3,
            "optimize_color": True,
            "color_lr": 2e-3,
            "optimize_alpha": False,
            "alpha_lr": 2e-3,
            "optimize_location": True,
            "location_lr": 2e-1
        }
    }

    optims = {
        "Adam": torch.optim.Adam,
        "SGD": torch.optim.SGD,
        "ASGD": torch.optim.ASGD,
    }

    #region methods
    def __init__(self, f=None):
        self.store = {}
        if f is None:
            self.store["default"] = copy.deepcopy(SvgOptimizationSettings.default_params)
        else:
            self.store = json.load(f)

    # create default alias for root
    def default_name(self, dname):
        self.dname = dname
        if dname not in self.store:
            self.store[dname] = self.store["default"]

    def retrieve(self, node_id):
        if node_id not in self.store:
            return (self.store["default"], False)
        else:
            return (self.store[node_id], True)

    def reset_to_defaults(self, node_id):
        if node_id in self.store:
            del self.store[node_id]

        return self.store["default"]

    def undefault(self, node_id):
        if node_id not in self.store:
            self.store[node_id] = copy.deepcopy(self.store["default"])

        return self.store[node_id]

    def override_optimizer(self, optimizer):
        if optimizer is not None:
            for v in self.store.values():
                v["optimizer"] = optimizer

    def global_override(self, path, value):
        for store in self.store.values():
            d = store
            for key in path[:-1]:
                d = d[key]

            d[path[-1]] = value

    def save(self, file):
        self.store["default"] = self.store[self.dname]
        json.dump(self.store, file, indent="\t")
    #endregion

class OptimizableSvg:

    class TransformTools:
        @staticmethod
        def parse_matrix(vals):
            assert(len(vals)==6)
            return np.array([[vals[0],vals[2],vals[4]],[vals[1], vals[3], vals[5]],[0,0,1]])

        @staticmethod
        def parse_translate(vals):
            assert(len(vals)>=1 and len(vals)<=2)
            mat=np.eye(3)
            mat[0,2]=vals[0]
            if len(vals)>1:
                mat[1,2]=vals[1]
            return mat

        @staticmethod
        def parse_rotate(vals):
            assert (len(vals) == 1 or len(vals) == 3)
            mat = np.eye(3)
            rads=math.radians(vals[0])
            sint=math.sin(rads)
            cost=math.cos(rads)
            mat[0:2, 0:2] = np.array([[cost,-sint],[sint,cost]])
            if len(vals) > 1:
                tr1=parse_translate(vals[1:3])
                tr2=parse_translate([-vals[1],-vals[2]])
                mat=tr1 @ mat @ tr2
            return mat

        @staticmethod
        def parse_scale(vals):
            assert (len(vals) >= 1 and len(vals) <= 2)
            d=np.array([vals[0], vals[1] if len(vals)>1 else vals[0],1])
            return np.diag(d)

        @staticmethod
        def parse_skewx(vals):
            assert(len(vals)==1)
            m=np.eye(3)
            m[0,1]=vals[0]
            return m

        @staticmethod
        def parse_skewy(vals):
            assert (len(vals) == 1)
            m = np.eye(3)
            m[1, 0] = vals[0]
            return m

        @staticmethod
        def transformPoints(pointsTensor, transform):
            assert(transform is not None)
            one=torch.ones((pointsTensor.shape[0],1),device=pointsTensor.device)
            homo_points = torch.cat([pointsTensor, one], dim=1)
            mult = transform.mm(homo_points.permute(1,0)).permute(1,0)
            tfpoints=mult[:, 0:2].contiguous()
            #print(torch.norm(mult[:,2]-one))
            assert(pointsTensor.shape == tfpoints.shape)
            return tfpoints

        @staticmethod
        def promote_numpy(M):
            ret = np.eye(3)
            ret[0:2, 0:2] = M
            return ret

        @staticmethod
        def recompose_numpy(Theta,ScaleXY,ShearX,TXY):
            cost=math.cos(Theta)
            sint=math.sin(Theta)
            Rot=np.array([[cost, -sint],[sint, cost]])
            Scale=np.diag(ScaleXY)
            Shear=np.eye(2)
            Shear[0,1]=ShearX

            Translate=np.eye(3)
            Translate[0:2,2]=TXY
            
            M=OptimizableSvg.TransformTools.promote_numpy(Rot @ Scale @ Shear) @ Translate
            return M

        @staticmethod
        def promote(m):
            M=torch.eye(3).to(m.device)
            M[0:2,0:2]=m
            return M

        @staticmethod
        def make_rot(Theta):
            sint=Theta.sin().squeeze()
            cost=Theta.cos().squeeze()
            #m=torch.tensor([[cost, -sint],[sint, cost]])
            Rot=torch.stack((torch.stack((cost,-sint)),torch.stack((sint,cost))))
            return Rot

        @staticmethod
        def make_scale(ScaleXY):
            if ScaleXY.squeeze().dim()==0:
                ScaleXY=ScaleXY.squeeze()
                #uniform scale
                return torch.diag(torch.stack([ScaleXY,ScaleXY])).to(ScaleXY.device)
            else:
                return torch.diag(ScaleXY).to(ScaleXY.device)

        @staticmethod
        def make_shear(ShearX):
            m=torch.eye(2).to(ShearX.device)
            m[0,1]=ShearX
            return m

        @staticmethod
        def make_translate(TXY):
            m=torch.eye(3).to(TXY.device)
            m[0:2,2]=TXY
            return m

        @staticmethod
        def recompose(Theta,ScaleXY,ShearX,TXY):
            Rot=OptimizableSvg.TransformTools.make_rot(Theta)
            Scale=OptimizableSvg.TransformTools.make_scale(ScaleXY)
            Shear=OptimizableSvg.TransformTools.make_shear(ShearX)
            Translate=OptimizableSvg.TransformTools.make_translate(TXY)

            return OptimizableSvg.TransformTools.promote(Rot.mm(Scale).mm(Shear)).mm(Translate)

        TransformDecomposition=namedtuple("TransformDecomposition","theta scale shear translate")
        TransformProperties=namedtuple("TransformProperties", "has_rotation has_scale has_mirror scale_uniform has_shear has_translation")

        @staticmethod
        def make_named(decomp):
            if not isinstance(decomp,OptimizableSvg.TransformTools.TransformDecomposition):
                decomp=OptimizableSvg.TransformTools.TransformDecomposition(theta=decomp[0],scale=decomp[1],shear=decomp[2],translate=decomp[3])
            return decomp

        @staticmethod
        def analyze_transform(decomp):
            decomp=OptimizableSvg.TransformTools.make_named(decomp)
            epsilon=1e-3
            has_rotation=abs(decomp.theta)>epsilon
            has_scale=abs((abs(decomp.scale)-1)).max()>epsilon
            scale_len=decomp.scale.squeeze().ndim>0 if isinstance(decomp.scale,np.ndarray) else decomp.scale.squeeze().dim() > 0
            has_mirror=scale_len and decomp.scale[0]*decomp.scale[1] < 0
            scale_uniform=not scale_len or abs(abs(decomp.scale[0])-abs(decomp.scale[1]))<epsilon
            has_shear=abs(decomp.shear)>epsilon
            has_translate=max(abs(decomp.translate[0]),abs(decomp.translate[1]))>epsilon

            return OptimizableSvg.TransformTools.TransformProperties(has_rotation=has_rotation,has_scale=has_scale,has_mirror=has_mirror,scale_uniform=scale_uniform,has_shear=has_shear,has_translation=has_translate)

        @staticmethod
        def check_and_decomp(M):
            decomp=OptimizableSvg.TransformTools.decompose(M) if M is not None else OptimizableSvg.TransformTools.TransformDecomposition(theta=0,scale=(1,1),shear=0,translate=(0,0))
            props=OptimizableSvg.TransformTools.analyze_transform(decomp)
            return (decomp, props)

        @staticmethod
        def tf_to_string(M):
            tfstring = "matrix({} {} {} {} {} {})".format(M[0, 0], M[1, 0], M[0, 1], M[1, 1], M[0, 2], M[1, 2])
            return tfstring

        @staticmethod
        def decomp_to_string(decomp):
            decomp = OptimizableSvg.TransformTools.make_named(decomp)
            ret=""
            props=OptimizableSvg.TransformTools.analyze_transform(decomp)
            if props.has_rotation:
                ret+="rotate({}) ".format(math.degrees(decomp.theta.item()))
            if props.has_scale:
                if decomp.scale.dim()==0:
                    ret += "scale({}) ".format(decomp.scale.item())
                else:
                    ret+="scale({} {}) ".format(decomp.scale[0], decomp.scale[1])
            if props.has_shear:
                ret+="skewX({}) ".format(decomp.shear.item())
            if props.has_translation:
                ret+="translate({} {}) ".format(decomp.translate[0],decomp.translate[1])

            return ret

        @staticmethod
        def decompose(M):
            m = M[0:2, 0:2]
            t0=M[0:2, 2]
            #get translation so that we can post-multiply with it
            TXY=np.linalg.solve(m,t0)

            T=np.eye(3)
            T[0:2,2]=TXY

            q, r = np.linalg.qr(m)

            ref = np.array([[1, 0], [0, np.sign(np.linalg.det(q))]])

            Rot = np.dot(q, ref)

            ref2 = np.array([[1, 0], [0, np.sign(np.linalg.det(r))]])

            r2 = np.dot(ref2, r)

            Ref = np.dot(ref, ref2)
            
            sc = np.diag(r2)
            Scale = np.diagflat(sc)

            Shear = np.eye(2)
            Shear[0, 1] = r2[0, 1] / sc[0]
            #the actual shear coefficient
            ShearX=r2[0, 1] / sc[0]

            if np.sum(sc) < 0:
                # both scales are negative, flip this and add a 180 rotation
                Rot = np.dot(Rot, -np.eye(2))
                Scale = -Scale

            Theta = math.atan2(Rot[1, 0], Rot[0, 0])
            ScaleXY = np.array([Scale[0,0],Scale[1,1]*Ref[1,1]])

            return OptimizableSvg.TransformTools.TransformDecomposition(theta=Theta, scale=ScaleXY, shear=ShearX, translate=TXY)

    #region suboptimizers

    #optimizes color, but really any tensor that needs to stay between 0 and 1 per-entry
    class ColorOptimizer:
        def __init__(self,tensor,optim_type,lr):
            self.tensor=tensor
            self.optim=optim_type([tensor],lr=lr)

        def zero_grad(self):
            self.optim.zero_grad()

        def step(self):
            self.optim.step()
            self.tensor.data.clamp_(min=1e-4,max=1.)

    #optimizes gradient stop positions
    class StopOptimizer:
        def __init__(self,stops,optim_type,lr):
            self.stops=stops
            self.optim=optim_type([stops],lr=lr)

        def zero_grad(self):
            self.optim.zero_grad()

        def step(self):
            self.optim.step()
            self.stops.data.clamp_(min=0., max=1.)
            self.stops.data, _ = self.stops.sort()
            self.stops.data[0] = 0.
            self.stops.data[-1]=1.

    #optimizes gradient: stop, positions, colors+opacities, locations
    class GradientOptimizer:
        def __init__(self, begin, end, offsets, stops, optim_params):
            self.begin=begin.clone().detach() if begin is not None else None
            self.end=end.clone().detach() if end is not None else None
            self.offsets=offsets.clone().detach() if offsets is not None else None
            self.stop_colors=stops[:,0:3].clone().detach() if stops is not None else None
            self.stop_alphas=stops[:,3].clone().detach() if stops is not None else None
            self.optimizers=[]

            if optim_params["gradients"]["optimize_stops"] and self.offsets is not None:
                self.offsets.requires_grad_(True)
                self.optimizers.append(OptimizableSvg.StopOptimizer(self.offsets,SvgOptimizationSettings.optims[optim_params["optimizer"]],optim_params["gradients"]["stop_lr"]))
            if optim_params["gradients"]["optimize_color"] and self.stop_colors is not None:
                self.stop_colors.requires_grad_(True)
                self.optimizers.append(OptimizableSvg.ColorOptimizer(self.stop_colors,SvgOptimizationSettings.optims[optim_params["optimizer"]],optim_params["gradients"]["color_lr"]))
            if optim_params["gradients"]["optimize_alpha"] and self.stop_alphas is not None:
                self.stop_alphas.requires_grad_(True)
                self.optimizers.append(OptimizableSvg.ColorOptimizer(self.stop_alphas,SvgOptimizationSettings.optims[optim_params["optimizer"]],optim_params["gradients"]["alpha_lr"]))
            if optim_params["gradients"]["optimize_location"] and self.begin is not None and self.end is not None:
                self.begin.requires_grad_(True)
                self.end.requires_grad_(True)
                self.optimizers.append(SvgOptimizationSettings.optims[optim_params["optimizer"]]([self.begin,self.end],lr=optim_params["gradients"]["location_lr"]))


        def get_vals(self):
            return self.begin, self.end, self.offsets, torch.cat((self.stop_colors,self.stop_alphas.unsqueeze(1)),1) if self.stop_colors is not None and self.stop_alphas is not None else None

        def zero_grad(self):
            for optim in self.optimizers:
                optim.zero_grad()

        def step(self):
            for optim in self.optimizers:
                optim.step()

    class TransformOptimizer:
        def __init__(self,transform,optim_params):
            self.transform=transform
            self.optimizes=optim_params["transforms"]["optimize_transforms"] and transform is not None
            self.params=copy.deepcopy(optim_params)
            self.transform_mode=optim_params["transforms"]["transform_mode"]

            if self.optimizes:
                optimvars=[]
                self.residual=None
                lr=optim_params["transforms"]["transform_lr"]
                tmult=optim_params["transforms"]["translation_mult"]
                decomp,props=OptimizableSvg.TransformTools.check_and_decomp(transform.cpu().numpy())
                if self.transform_mode=="move":
                    #only translation and rotation should be set
                    if props.has_scale or props.has_shear or props.has_mirror:
                        print("Warning: set to optimize move only, but input transform has residual scale or shear")
                        self.residual=self.transform.clone().detach().requires_grad_(False)
                        self.Theta=torch.tensor(0,dtype=torch.float32,requires_grad=True,device=transform.device)
                        self.translation=torch.tensor([0, 0],dtype=torch.float32,requires_grad=True,device=transform.device)
                    else:
                        self.residual=None
                        self.Theta=torch.tensor(decomp.theta,dtype=torch.float32,requires_grad=True,device=transform.device)
                        self.translation=torch.tensor(decomp.translate,dtype=torch.float32,requires_grad=True,device=transform.device)
                    optimvars+=[{'params':x,'lr':lr} for x in [self.Theta]]+[{'params':self.translation,'lr':lr*tmult}]
                elif self.transform_mode=="rigid":
                    #only translation, rotation, and uniform scale should be set
                    if props.has_shear or props.has_mirror or not props.scale_uniform:
                        print("Warning: set to optimize rigid transform only, but input transform has residual shear, mirror or non-uniform scale")
                        self.residual = self.transform.clone().detach().requires_grad_(False)
                        self.Theta = torch.tensor(0, dtype=torch.float32, requires_grad=True,device=transform.device)
                        self.translation = torch.tensor([0, 0], dtype=torch.float32, requires_grad=True,device=transform.device)
                        self.scale=torch.tensor(1, dtype=torch.float32, requires_grad=True,device=transform.device)
                    else:
                        self.residual = None
                        self.Theta = torch.tensor(decomp.theta, dtype=torch.float32, requires_grad=True,device=transform.device)
                        self.translation = torch.tensor(decomp.translate, dtype=torch.float32, requires_grad=True,device=transform.device)
                        self.scale = torch.tensor(decomp.scale[0], dtype=torch.float32, requires_grad=True,device=transform.device)
                    optimvars += [{'params':x,'lr':lr} for x in [self.Theta, self.scale]]+[{'params':self.translation,'lr':lr*tmult}]
                elif self.transform_mode=="similarity":
                    if props.has_shear or not props.scale_uniform:
                        print("Warning: set to optimize rigid transform only, but input transform has residual shear or non-uniform scale")
                        self.residual = self.transform.clone().detach().requires_grad_(False)
                        self.Theta = torch.tensor(0, dtype=torch.float32, requires_grad=True,device=transform.device)
                        self.translation = torch.tensor([0, 0], dtype=torch.float32, requires_grad=True,device=transform.device)
                        self.scale=torch.tensor(1, dtype=torch.float32, requires_grad=True,device=transform.device)
                        self.scale_sign=torch.tensor(1,dtype=torch.float32,requires_grad=False,device=transform.device)
                    else:
                        self.residual = None
                        self.Theta = torch.tensor(decomp.theta, dtype=torch.float32, requires_grad=True,device=transform.device)
                        self.translation = torch.tensor(decomp.translate, dtype=torch.float32, requires_grad=True,device=transform.device)
                        self.scale = torch.tensor(decomp.scale[0], dtype=torch.float32, requires_grad=True,device=transform.device)
                        self.scale_sign = torch.tensor(np.sign(decomp.scale[0]*decomp.scale[1]), dtype=torch.float32, requires_grad=False,device=transform.device)
                    optimvars += [{'params':x,'lr':lr} for x in [self.Theta, self.scale]]+[{'params':self.translation,'lr':lr*tmult}]
                elif self.transform_mode=="affine":
                    self.Theta = torch.tensor(decomp.theta, dtype=torch.float32, requires_grad=True,device=transform.device)
                    self.translation = torch.tensor(decomp.translate, dtype=torch.float32, requires_grad=True,device=transform.device)
                    self.scale = torch.tensor(decomp.scale, dtype=torch.float32, requires_grad=True,device=transform.device)
                    self.shear = torch.tensor(decomp.shear, dtype=torch.float32, requires_grad=True,device=transform.device)
                    optimvars += [{'params':x,'lr':lr} for x in [self.Theta, self.scale, self.shear]]+[{'params':self.translation,'lr':lr*tmult}]
                else:
                    raise ValueError("Unrecognized transform mode '{}'".format(self.transform_mode))
                self.optimizer=SvgOptimizationSettings.optims[optim_params["optimizer"]](optimvars)

        def get_transform(self):
            if not self.optimizes:
                return self.transform
            else:
                if self.transform_mode == "move":
                    composed=OptimizableSvg.TransformTools.recompose(self.Theta,torch.tensor([1.],device=self.Theta.device),torch.tensor(0.,device=self.Theta.device),self.translation)
                    return self.residual.mm(composed) if self.residual is not None else composed
                elif self.transform_mode == "rigid":
                    composed = OptimizableSvg.TransformTools.recompose(self.Theta, self.scale, torch.tensor(0.,device=self.Theta.device),
                                                                       self.translation)
                    return self.residual.mm(composed) if self.residual is not None else composed
                elif self.transform_mode == "similarity":
                    composed=OptimizableSvg.TransformTools.recompose(self.Theta, torch.cat((self.scale,self.scale*self.scale_sign)),torch.tensor(0.,device=self.Theta.device),self.translation)
                    return self.residual.mm(composed) if self.residual is not None else composed
                elif self.transform_mode == "affine":
                    composed = OptimizableSvg.TransformTools.recompose(self.Theta, self.scale, self.shear, self.translation)
                    return composed
                else:
                    raise ValueError("Unrecognized transform mode '{}'".format(self.transform_mode))

        def tfToString(self):
            if self.transform is None:
                return None
            elif not self.optimizes:
                return OptimizableSvg.TransformTools.tf_to_string(self.transform)
            else:
                if self.transform_mode == "move":
                    str=OptimizableSvg.TransformTools.decomp_to_string((self.Theta,torch.tensor([1.]),torch.tensor(0.),self.translation))
                    return (OptimizableSvg.TransformTools.tf_to_string(self.residual) if self.residual is not None else "")+" "+str
                elif self.transform_mode == "rigid":
                    str = OptimizableSvg.TransformTools.decomp_to_string((self.Theta, self.scale, torch.tensor(0.),
                                                                       self.translation))
                    return (OptimizableSvg.TransformTools.tf_to_string(self.residual) if self.residual is not None else "")+" "+str
                elif self.transform_mode == "similarity":
                    str=OptimizableSvg.TransformTools.decomp_to_string((self.Theta, torch.cat((self.scale,self.scale*self.scale_sign)),torch.tensor(0.),self.translation))
                    return (OptimizableSvg.TransformTools.tf_to_string(self.residual) if self.residual is not None else "")+" "+str
                elif self.transform_mode == "affine":
                    str = OptimizableSvg.TransformTools.decomp_to_string((self.Theta, self.scale, self.shear, self.translation))
                    return composed

        def zero_grad(self):
            if self.optimizes:
                self.optimizer.zero_grad()

        def step(self):
            if self.optimizes:
                self.optimizer.step()

    #endregion

    #region Nodes
    class SvgNode:
        def __init__(self,id,transform,appearance,settings):
            self.id=id
            self.children=[]
            self.optimizers=[]
            self.device = settings.device
            self.transform=torch.tensor(transform,dtype=torch.float32,device=self.device) if transform is not None else None
            self.transform_optim=OptimizableSvg.TransformOptimizer(self.transform,settings.retrieve(self.id)[0])
            self.optimizers.append(self.transform_optim)
            self.proc_appearance(appearance,settings.retrieve(self.id)[0])

        def tftostring(self):
            return self.transform_optim.tfToString()

        def appearanceToString(self):
            appstring=""
            for key,value in self.appearance.items():
                if key in ["fill", "stroke"]:
                    #a paint-type value
                    if value[0] == "none":
                        appstring+="{}:none;".format(key)
                    elif value[0] == "solid":
                        appstring += "{}:{};".format(key,OptimizableSvg.rgb_to_string(value[1]))
                    elif value[0] == "url":
                        appstring += "{}:url(#{});".format(key,value[1].id)
                        #appstring += "{}:{};".format(key,"#ff00ff")
                elif key in ["opacity", "fill-opacity", "stroke-opacity", "stroke-width", "fill-rule"]:
                    appstring+="{}:{};".format(key,value)
                else:
                    raise ValueError("Don't know how to write appearance parameter '{}'".format(key))
            return appstring


        def write_xml_common_attrib(self,node,tfname="transform"):
            if self.transform is not None:
                node.set(tfname,self.tftostring())
            if len(self.appearance)>0:
                node.set('style',self.appearanceToString())
            if self.id is not None:
                node.set('id',self.id)


        def proc_appearance(self,appearance,optim_params):
            self.appearance=appearance
            for key, value in appearance.items():
                if key == "fill" or key == "stroke":
                    if optim_params["optimize_color"] and value[0]=="solid":
                        value[1].requires_grad_(True)
                        self.optimizers.append(OptimizableSvg.ColorOptimizer(value[1],SvgOptimizationSettings.optims[optim_params["optimizer"]],optim_params["color_lr"]))
                elif key == "fill-opacity" or key == "stroke-opacity" or key == "opacity":
                    if optim_params["optimize_alpha"]:
                        value[1].requires_grad_(True)
                        self.optimizers.append(OptimizableSvg.ColorOptimizer(value[1], optim_params["optimizer"],
                                                                             optim_params["alpha_lr"]))
                elif key == "fill-rule" or key == "stroke-width":
                    pass
                else:
                    raise RuntimeError("Unrecognized appearance key '{}'".format(key))

        def prop_transform(self,intform):
            return intform.matmul(self.transform_optim.get_transform()) if self.transform is not None else intform

        def prop_appearance(self,inappearance):
            outappearance=copy.copy(inappearance)
            for key,value in self.appearance.items():
                if key == "fill":
                    #gets replaced
                    outappearance[key]=value
                elif key == "fill-opacity":
                    #gets multiplied
                    outappearance[key] = outappearance[key]*value
                elif key == "fill-rule":
                    #gets replaced
                    outappearance[key] = value
                elif key =="opacity":
                    # gets multiplied
                    outappearance[key] = outappearance[key]*value
                elif key == "stroke":
                    # gets replaced
                    outappearance[key] = value
                elif key == "stroke-opacity":
                    # gets multiplied
                    outappearance[key] = outappearance[key]*value
                elif key =="stroke-width":
                    # gets replaced
                    outappearance[key] = value
                else:
                    raise RuntimeError("Unrecognized appearance key '{}'".format(key))
            return outappearance

        def zero_grad(self):
            for optim in self.optimizers:
                optim.zero_grad()
            for child in self.children:
                child.zero_grad()

        def step(self):
            for optim in self.optimizers:
                optim.step()
            for child in self.children:
                child.step()

        def get_type(self):
            return "Generic node"

        def is_shape(self):
            return False

        def build_scene(self,shapes,shape_groups,transform,appearance):
            raise NotImplementedError("Abstract SvgNode cannot recurse")

    class GroupNode(SvgNode):
        def __init__(self, id, transform, appearance,settings):
            super().__init__(id, transform, appearance,settings)

        def get_type(self):
            return "Group node"

        def build_scene(self,shapes,shape_groups,transform,appearance):
            outtf=self.prop_transform(transform)
            outapp=self.prop_appearance(appearance)
            for child in self.children:
                child.build_scene(shapes,shape_groups,outtf,outapp)

        def write_xml(self, parent):
            elm=etree.SubElement(parent,"g")
            self.write_xml_common_attrib(elm)

            for child in self.children:
                child.write_xml(elm)

    class RootNode(SvgNode):
        def __init__(self, id, transform, appearance,settings):
            super().__init__(id, transform, appearance,settings)

        def write_xml(self,document):
            elm=etree.Element('svg')
            self.write_xml_common_attrib(elm)
            elm.set("version","2.0")
            elm.set("width",str(document.canvas[0]))
            elm.set("height", str(document.canvas[1]))
            elm.set("xmlns","http://www.w3.org/2000/svg")
            elm.set("xmlns:xlink","http://www.w3.org/1999/xlink")
            #write definitions before we write any children
            document.write_defs(elm)

            #write the children
            for child in self.children:
                child.write_xml(elm)

            return elm

        def get_type(self):
            return "Root node"

        def build_scene(self,shapes,shape_groups,transform,appearance):
            outtf = self.prop_transform(transform).to(self.device)
            for child in self.children:
                child.build_scene(shapes,shape_groups,outtf,appearance)

        @staticmethod
        def get_default_appearance(device):
            default_appearance = {"fill": ("solid", torch.tensor([0., 0., 0.],device=device)),
                                  "fill-opacity": torch.tensor([1.],device=device),
                                  "fill-rule": "nonzero",
                                  "opacity": torch.tensor([1.],device=device),
                                  "stroke": ("none", None),
                                  "stroke-opacity": torch.tensor([1.],device=device),
                                  "stroke-width": torch.tensor([0.],device=device)}
            return default_appearance

        @staticmethod
        def get_default_transform():
            return torch.eye(3)



    class ShapeNode(SvgNode):
        def __init__(self, id, transform, appearance,settings):
            super().__init__(id, transform, appearance,settings)

        def get_type(self):
            return "Generic shape node"

        def is_shape(self):
            return True

        def construct_paint(self,value,combined_opacity,transform):
            if value[0]   == "none":
                return None
            elif value[0] == "solid":
                return torch.cat([value[1],combined_opacity]).to(self.device)
            elif value[0] == "url":
                #get the gradient object from this node
                return value[1].getGrad(combined_opacity,transform)
            else:
                raise ValueError("Unknown paint value type '{}'".format(value[0]))

        def make_shape_group(self,appearance,transform,num_shapes,num_subobjects):
            fill=self.construct_paint(appearance["fill"],appearance["opacity"]*appearance["fill-opacity"],transform)
            stroke=self.construct_paint(appearance["stroke"],appearance["opacity"]*appearance["stroke-opacity"],transform)
            sg = pydiffvg.ShapeGroup(shape_ids=torch.tensor(range(num_shapes, num_shapes + num_subobjects)),
                                     fill_color=fill,
                                     use_even_odd_rule=appearance["fill-rule"]=="evenodd",
                                     stroke_color=stroke,
                                     shape_to_canvas=transform,
                                     id=self.id)
            return sg

    class PathNode(ShapeNode):
        def __init__(self, id, transform, appearance,settings, paths):
            super().__init__(id, transform, appearance,settings)
            self.proc_paths(paths,settings.retrieve(self.id)[0])

        def proc_paths(self,paths,optim_params):
            self.paths=paths
            if optim_params["paths"]["optimize_points"]:
                ptlist=[]
                for path in paths:
                    ptlist.append(path.points.requires_grad_(True))
                self.optimizers.append(SvgOptimizationSettings.optims[optim_params["optimizer"]](ptlist,lr=optim_params["paths"]["shape_lr"]))

        def get_type(self):
            return "Path node"

        def build_scene(self,shapes,shape_groups,transform,appearance):
            applytf=self.prop_transform(transform)
            applyapp = self.prop_appearance(appearance)
            sg=self.make_shape_group(applyapp,applytf,len(shapes),len(self.paths))
            for path in self.paths:
                disp_path=pydiffvg.Path(path.num_control_points,path.points,path.is_closed,applyapp["stroke-width"],path.id)
                shapes.append(disp_path)
            shape_groups.append(sg)

        def path_to_string(self,path):
            path_string = "M {},{} ".format(path.points[0][0].item(), path.points[0][1].item())
            idx = 1
            numpoints = path.points.shape[0]
            for type in path.num_control_points:
                toproc = type + 1
                if type == 0:
                    # add line
                    path_string += "L "
                elif type == 1:
                    # add quadric
                    path_string += "Q "
                elif type == 2:
                    # add cubic
                    path_string += "C "
                while toproc > 0:
                    path_string += "{},{} ".format(path.points[idx % numpoints][0].item(),
                                                   path.points[idx % numpoints][1].item())
                    idx += 1
                    toproc -= 1
            if path.is_closed:
                path_string += "Z "

            return path_string

        def paths_string(self):
            pstr=""
            for path in self.paths:
                pstr+=self.path_to_string(path)
            return pstr

        def write_xml(self, parent):
            elm = etree.SubElement(parent, "path")
            self.write_xml_common_attrib(elm)
            elm.set("d",self.paths_string())

            for child in self.children:
                child.write_xml(elm)

    class RectNode(ShapeNode):
        def __init__(self, id, transform, appearance,settings, rect):
            super().__init__(id, transform, appearance,settings)
            self.rect=torch.tensor(rect,dtype=torch.float,device=settings.device)
            optim_params=settings.retrieve(self.id)[0]
            #borrowing path settings for this
            if optim_params["paths"]["optimize_points"]:
                self.optimizers.append(SvgOptimizationSettings.optims[optim_params["optimizer"]]([self.rect],lr=optim_params["paths"]["shape_lr"]))

        def get_type(self):
            return "Rect node"

        def build_scene(self,shapes,shape_groups,transform,appearance):
            applytf=self.prop_transform(transform)
            applyapp = self.prop_appearance(appearance)
            sg=self.make_shape_group(applyapp,applytf,len(shapes),1)
            shapes.append(pydiffvg.Rect(self.rect[0:2],self.rect[0:2]+self.rect[2:4],applyapp["stroke-width"],self.id))
            shape_groups.append(sg)

        def write_xml(self, parent):
            elm = etree.SubElement(parent, "rect")
            self.write_xml_common_attrib(elm)
            elm.set("x",str(self.rect[0]))
            elm.set("y", str(self.rect[1]))
            elm.set("width", str(self.rect[2]))
            elm.set("height", str(self.rect[3]))

            for child in self.children:
                child.write_xml(elm)

    class CircleNode(ShapeNode):
        def __init__(self, id, transform, appearance,settings, rect):
            super().__init__(id, transform, appearance,settings)
            self.circle=torch.tensor(rect,dtype=torch.float,device=settings.device)
            optim_params=settings.retrieve(self.id)[0]
            #borrowing path settings for this
            if optim_params["paths"]["optimize_points"]:
                self.optimizers.append(SvgOptimizationSettings.optims[optim_params["optimizer"]]([self.circle],lr=optim_params["paths"]["shape_lr"]))

        def get_type(self):
            return "Circle node"

        def build_scene(self,shapes,shape_groups,transform,appearance):
            applytf=self.prop_transform(transform)
            applyapp = self.prop_appearance(appearance)
            sg=self.make_shape_group(applyapp,applytf,len(shapes),1)
            shapes.append(pydiffvg.Circle(self.circle[2],self.circle[0:2],applyapp["stroke-width"],self.id))
            shape_groups.append(sg)

        def write_xml(self, parent):
            elm = etree.SubElement(parent, "circle")
            self.write_xml_common_attrib(elm)
            elm.set("cx",str(self.circle[0]))
            elm.set("cy", str(self.circle[1]))
            elm.set("r", str(self.circle[2]))

            for child in self.children:
                child.write_xml(elm)


    class EllipseNode(ShapeNode):
        def __init__(self, id, transform, appearance,settings, ellipse):
            super().__init__(id, transform, appearance,settings)
            self.ellipse=torch.tensor(ellipse,dtype=torch.float,device=settings.device)
            optim_params=settings.retrieve(self.id)[0]
            #borrowing path settings for this
            if optim_params["paths"]["optimize_points"]:
                self.optimizers.append(SvgOptimizationSettings.optims[optim_params["optimizer"]]([self.ellipse],lr=optim_params["paths"]["shape_lr"]))

        def get_type(self):
            return "Ellipse node"

        def build_scene(self,shapes,shape_groups,transform,appearance):
            applytf=self.prop_transform(transform)
            applyapp = self.prop_appearance(appearance)
            sg=self.make_shape_group(applyapp,applytf,len(shapes),1)
            shapes.append(pydiffvg.Ellipse(self.ellipse[2:4],self.ellipse[0:2],applyapp["stroke-width"],self.id))
            shape_groups.append(sg)

        def write_xml(self, parent):
            elm = etree.SubElement(parent, "ellipse")
            self.write_xml_common_attrib(elm)
            elm.set("cx", str(self.ellipse[0]))
            elm.set("cy", str(self.ellipse[1]))
            elm.set("rx", str(self.ellipse[2]))
            elm.set("ry", str(self.ellipse[3]))

            for child in self.children:
                child.write_xml(elm)

    class PolygonNode(ShapeNode):
        def __init__(self, id, transform, appearance,settings, points):
            super().__init__(id, transform, appearance,settings)
            self.points=points
            optim_params=settings.retrieve(self.id)[0]
            #borrowing path settings for this
            if optim_params["paths"]["optimize_points"]:
                self.optimizers.append(SvgOptimizationSettings.optims[optim_params["optimizer"]]([self.points],lr=optim_params["paths"]["shape_lr"]))

        def get_type(self):
            return "Polygon node"

        def build_scene(self,shapes,shape_groups,transform,appearance):
            applytf=self.prop_transform(transform)
            applyapp = self.prop_appearance(appearance)
            sg=self.make_shape_group(applyapp,applytf,len(shapes),1)
            shapes.append(pydiffvg.Polygon(self.points,True,applyapp["stroke-width"],self.id))
            shape_groups.append(sg)

        def point_string(self):
            ret=""
            for i in range(self.points.shape[0]):
                pt=self.points[i,:]
                #assert pt.shape == (1,2)
                ret+= str(pt[0])+","+str(pt[1])+" "
            return ret

        def write_xml(self, parent):
            elm = etree.SubElement(parent, "polygon")
            self.write_xml_common_attrib(elm)
            elm.set("points",self.point_string())

            for child in self.children:
                child.write_xml(elm)

    class GradientNode(SvgNode):
        def __init__(self, id, transform,settings,begin,end,offsets,stops,href):
            super().__init__(id, transform, {},settings)
            self.optim=OptimizableSvg.GradientOptimizer(begin, end, offsets, stops, settings.retrieve(id)[0])
            self.optimizers.append(self.optim)
            self.href=href

        def is_ref(self):
            return self.href is not None

        def get_type(self):
            return "Gradient node"

        def get_stops(self):
            _, _, offsets, stops=self.optim.get_vals()
            return offsets, stops

        def get_points(self):
            begin, end, _, _ =self.optim.get_vals()
            return begin, end

        def write_xml(self, parent):
            elm = etree.SubElement(parent, "linearGradient")
            self.write_xml_common_attrib(elm,tfname="gradientTransform")

            begin, end, offsets, stops = self.optim.get_vals()

            if self.href is None:
                #we have stops
                for idx, offset in enumerate(offsets):
                    stop=etree.SubElement(elm,"stop")
                    stop.set("offset",str(offset.item()))
                    stop.set("stop-color",OptimizableSvg.rgb_to_string(stops[idx,0:3]))
                    stop.set("stop-opacity",str(stops[idx,3].item()))
            else:
                elm.set('xlink:href', "#{}".format(self.href.id))

            if begin is not None and end is not None:
                #no stops
                elm.set('x1', str(begin[0].item()))
                elm.set('y1', str(begin[1].item()))
                elm.set('x2', str(end[0].item()))
                elm.set('y2', str(end[1].item()))

                # magic value to make this work
                elm.set("gradientUnits", "userSpaceOnUse")

            for child in self.children:
                child.write_xml(elm)

        def getGrad(self,combined_opacity,transform):
            if self.is_ref():
                offsets, stops=self.href.get_stops()
            else:
                offsets, stops=self.get_stops()

            stops=stops.clone()
            stops[:,3]*=combined_opacity

            begin,end = self.get_points()

            applytf=self.prop_transform(transform)
            begin=OptimizableSvg.TransformTools.transformPoints(begin.unsqueeze(0),applytf).squeeze()
            end = OptimizableSvg.TransformTools.transformPoints(end.unsqueeze(0), applytf).squeeze()

            return pydiffvg.LinearGradient(begin, end, offsets, stops)
    #endregion

    def __init__(self, filename, settings=SvgOptimizationSettings(),optimize_background=False, verbose=False, device=torch.device("cpu")):
        self.settings=settings
        self.verbose=verbose
        self.device=device
        self.settings.device=device

        tree = etree.parse(filename)
        root = tree.getroot()

        #in case we need global optimization
        self.optimizers=[]
        self.background=torch.tensor([1.,1.,1.],dtype=torch.float32,requires_grad=optimize_background,device=self.device)

        if optimize_background:
            p=settings.retrieve("default")[0]
            self.optimizers.append(OptimizableSvg.ColorOptimizer(self.background,SvgOptimizationSettings.optims[p["optimizer"]],p["color_lr"]))

        self.defs={}

        self.depth=0

        self.dirty=True
        self.scene=None

        self.parseRoot(root)

    recognised_shapes=["path","circle","rect","ellipse","polygon"]

    #region core functionality
    def build_scene(self):
        if self.dirty:
            shape_groups=[]
            shapes=[]
            self.root.build_scene(shapes,shape_groups,OptimizableSvg.RootNode.get_default_transform().to(self.device),OptimizableSvg.RootNode.get_default_appearance(self.device))
            self.scene=(self.canvas[0],self.canvas[1],shapes,shape_groups)
            self.dirty=False
        return self.scene

    def zero_grad(self):
        self.root.zero_grad()
        for optim in self.optimizers:
            optim.zero_grad()
        for item in self.defs.values():
            if issubclass(item.__class__,OptimizableSvg.SvgNode):
                item.zero_grad()

    def render(self,scale=None,seed=0):
        #render at native resolution
        scene = self.build_scene()
        scene_args = pydiffvg.RenderFunction.serialize_scene(*scene)
        render = pydiffvg.RenderFunction.apply
        out_size=(scene[0],scene[1]) if scale is None else (int(scene[0]*scale),int(scene[1]*scale))
        img = render(out_size[0],  # width
                     out_size[1],  # height
                     2,  # num_samples_x
                     2,  # num_samples_y
                     seed,  # seed
                     None, # background_image
                     *scene_args)
        return img

    def step(self):
        self.dirty=True
        self.root.step()
        for optim in self.optimizers:
            optim.step()
        for item in self.defs.values():
            if issubclass(item.__class__, OptimizableSvg.SvgNode):
                item.step()
    #endregion

    #region reporting

    def offset_str(self,s):
        return ("\t"*self.depth)+s

    def reportSkippedAttribs(self, node, non_skipped=[]):
        skipped=set([k for k in node.attrib.keys() if not OptimizableSvg.is_namespace(k)])-set(non_skipped)
        if len(skipped)>0:
            tag=OptimizableSvg.remove_namespace(node.tag) if "id" not in node.attrib else "{}#{}".format(OptimizableSvg.remove_namespace(node.tag),node.attrib["id"])
            print(self.offset_str("Warning: Skipping the following attributes of node '{}': {}".format(tag,", ".join(["'{}'".format(atr) for atr in skipped]))))

    def reportSkippedChildren(self,node,skipped):
        skipped_names=["{}#{}".format(elm.tag,elm.attrib["id"]) if "id" in elm.attrib else elm.tag for elm in skipped]
        if len(skipped)>0:
            tag = OptimizableSvg.remove_namespace(node.tag) if "id" not in node.attrib else "{}#{}".format(OptimizableSvg.remove_namespace(node.tag),
                                                                                            node.attrib["id"])
            print(self.offset_str("Warning: Skipping the following children of node '{}': {}".format(tag,", ".join(["'{}'".format(name) for name in skipped_names]))))

    #endregion

    #region parsing
    @staticmethod
    def remove_namespace(s):
        """
            {...} ... -> ...
        """
        return re.sub('{.*}', '', s)

    @staticmethod
    def is_namespace(s):
        return re.match('{.*}', s) is not None

    @staticmethod
    def parseTransform(node):
        if "transform" not in node.attrib and "gradientTransform" not in node.attrib:
            return None

        tf_string=node.attrib["transform"] if "transform" in node.attrib else node.attrib["gradientTransform"]
        tforms=tf_string.split(")")[:-1]
        mat=np.eye(3)
        for tform in tforms:
            type = tform.split("(")[0]
            args = [float(val) for val in re.split("[, ]+",tform.split("(")[1])]
            if type == "matrix":
                mat=mat @ OptimizableSvg.TransformTools.parse_matrix(args)
            elif type == "translate":
                mat = mat @ OptimizableSvg.TransformTools.parse_translate(args)
            elif type == "rotate":
                mat = mat @ OptimizableSvg.TransformTools.parse_rotate(args)
            elif type == "scale":
                mat = mat @ OptimizableSvg.TransformTools.parse_scale(args)
            elif type == "skewX":
                mat = mat @ OptimizableSvg.TransformTools.parse_skewx(args)
            elif type == "skewY":
                mat = mat @ OptimizableSvg.TransformTools.parse_skewy(args)
            else:
                raise ValueError("Unknown transform type '{}'".format(type))
        return mat

    #dictionary that defines what constant do we need to multiply different units to get the value in pixels
    #gleaned from the CSS definition
    unit_dict = {"px":1,
                 "mm":4,
                 "cm":40,
                 "in":25.4*4,
                 "pt":25.4*4/72,
                 "pc":25.4*4/6
                 }

    @staticmethod
    def parseLength(s):
        #length is a number followed possibly by a unit definition
        #we assume that default unit is the pixel (px) equal to 0.25mm
        #last two characters might be unit
        val=None
        for i in range(len(s)):
            try:
                val=float(s[:len(s)-i])
                unit=s[len(s)-i:]
                break
            except ValueError:
                continue
        if len(unit)>0 and unit not in OptimizableSvg.unit_dict:
            raise ValueError("Unknown or unsupported unit '{}' encountered while parsing".format(unit))
        if unit != "":
            val*=OptimizableSvg.unit_dict[unit]
        return val

    @staticmethod
    def parseOpacity(s):
        is_percent=s.endswith("%")
        s=s.rstrip("%")
        val=float(s)
        if is_percent:
            val=val/100
        return np.clip(val,0.,1.)

    @staticmethod
    def parse_color(s):
        """
            Hex to tuple
        """
        if s[0] != '#':
            raise ValueError("Color argument `{}` not supported".format(s))
        s = s.lstrip('#')
        if len(s)==6:
            rgb = tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))
            return torch.tensor([rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0])
        elif len(s)==3:
            rgb = tuple((int(s[i:i + 1], 16)) for i in (0, 1, 2))
            return torch.tensor([rgb[0] / 15.0, rgb[1] / 15.0, rgb[2] / 15.0])
        else:
            raise ValueError("Color argument `{}` not supported".format(s))
        # sRGB to RGB
        # return torch.pow(torch.tensor([rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0]), 2.2)


    @staticmethod
    def rgb_to_string(val):
        byte_rgb=(val.clone().detach()*255).type(torch.int)
        byte_rgb.clamp_(min=0,max=255)
        s="#{:02x}{:02x}{:02x}".format(*byte_rgb)
        return s

    #parses a "paint" string for use in fill and stroke definitions
    @staticmethod
    def parsePaint(paintStr,defs,device):
        paintStr=paintStr.strip()
        if paintStr=="none":
            return ("none", None)
        elif paintStr[0]=="#":
            return ("solid",OptimizableSvg.parse_color(paintStr).to(device))
        elif paintStr.startswith("url"):
            url=paintStr.lstrip("url(").rstrip(")").strip("\'\"").lstrip("#")
            if url not in defs:
                raise ValueError("Paint-type attribute referencing an unknown object with ID '#{}'".format(url))
            return ("url",defs[url])
        else:
            raise ValueError("Unrecognized paint string: '{}'".format(paintStr))

    appearance_keys=["fill","fill-opacity","fill-rule","opacity","stroke","stroke-opacity","stroke-width"]

    @staticmethod
    def parseAppearance(node, defs, device):
        ret={}
        parse_keys = OptimizableSvg.appearance_keys
        local_dict={key:value for key,value in node.attrib.items() if key in parse_keys}
        css_dict={}
        style_dict={}
        appearance_dict={}
        if "class" in node.attrib:
            cls=node.attrib["class"]
            if "."+cls in defs:
                css_string=defs["."+cls]
                css_dict={item.split(":")[0]:item.split(":")[1] for item in css_string.split(";") if len(item)>0 and item.split(":")[0] in parse_keys}
        if "style" in node.attrib:
            style_string=node.attrib["style"]
            style_dict={item.split(":")[0]:item.split(":")[1] for item in style_string.split(";") if len(item)>0 and item.split(":")[0] in parse_keys}
        appearance_dict.update(css_dict)
        appearance_dict.update(style_dict)
        appearance_dict.update(local_dict)
        for key,value in appearance_dict.items():
            if key=="fill":
                ret[key]=OptimizableSvg.parsePaint(value,defs,device)
            elif key == "fill-opacity":
                ret[key]=torch.tensor(OptimizableSvg.parseOpacity(value),device=device)
            elif key == "fill-rule":
                ret[key]=value
            elif key == "opacity":
                ret[key]=torch.tensor(OptimizableSvg.parseOpacity(value),device=device)
            elif key == "stroke":
                ret[key]=OptimizableSvg.parsePaint(value,defs,device)
            elif key == "stroke-opacity":
                ret[key]=torch.tensor(OptimizableSvg.parseOpacity(value),device=device)
            elif key == "stroke-width":
                ret[key]=torch.tensor(OptimizableSvg.parseLength(value),device=device)
            else:
                raise ValueError("Error while parsing appearance attributes: key '{}' should not be here".format(key))

        return ret

    def parseRoot(self,root):
        if self.verbose:
            print(self.offset_str("Parsing root"))
        self.depth += 1

        # get document canvas dimensions
        self.parseViewport(root)
        canvmax=np.max(self.canvas)
        self.settings.global_override(["transforms","translation_mult"],canvmax)
        id=root.attrib["id"] if "id" in root.attrib else None

        transform=OptimizableSvg.parseTransform(root)
        appearance=OptimizableSvg.parseAppearance(root,self.defs,self.device)

        version=root.attrib["version"] if "version" in root.attrib else "<unknown version>"
        if version != "2.0":
            print(self.offset_str("Warning: Version {} is not 2.0, strange things may happen".format(version)))

        self.root=OptimizableSvg.RootNode(id,transform,appearance,self.settings)

        if self.verbose:
            self.reportSkippedAttribs(root, ["width", "height", "id", "transform","version", "style"]+OptimizableSvg.appearance_keys)

        #go through the root children and parse them appropriately
        skipped=[]
        for child in root:
            if OptimizableSvg.remove_namespace(child.tag) in OptimizableSvg.recognised_shapes:
                self.parseShape(child,self.root)
            elif OptimizableSvg.remove_namespace(child.tag) == "defs":
                self.parseDefs(child)
            elif OptimizableSvg.remove_namespace(child.tag) == "style":
                self.parseStyle(child)
            elif OptimizableSvg.remove_namespace(child.tag) == "g":
                self.parseGroup(child,self.root)
            else:
                skipped.append(child)

        if self.verbose:
            self.reportSkippedChildren(root,skipped)

        self.depth-=1

    def parseShape(self,shape,parent):
        tag=OptimizableSvg.remove_namespace(shape.tag)
        if self.verbose:
            print(self.offset_str("Parsing {}#{}".format(tag,shape.attrib["id"] if "id" in shape.attrib else "<No ID>")))

        self.depth+=1
        if tag == "path":
            self.parsePath(shape,parent)
        elif tag == "circle":
            self.parseCircle(shape,parent)
        elif tag == "rect":
            self.parseRect(shape,parent)
        elif tag == "ellipse":
            self.parseEllipse(shape,parent)
        elif tag == "polygon":
            self.parsePolygon(shape,parent)
        else:
            raise ValueError("Encountered unknown shape type '{}'".format(tag))
        self.depth -= 1

    def parsePath(self,shape,parent):
        path_string=shape.attrib['d']
        name = ''
        if 'id' in shape.attrib:
            name = shape.attrib['id']
        paths = pydiffvg.from_svg_path(path_string)
        for idx, path in enumerate(paths):
            path.stroke_width = torch.tensor([0.],device=self.device)
            path.num_control_points=path.num_control_points.to(self.device)
            path.points=path.points.to(self.device)
            path.source_id = name
            path.id = "{}-{}".format(name,idx) if len(paths)>1 else name
        transform = OptimizableSvg.parseTransform(shape)
        appearance = OptimizableSvg.parseAppearance(shape,self.defs,self.device)
        node=OptimizableSvg.PathNode(name,transform,appearance,self.settings,paths)
        parent.children.append(node)

        if self.verbose:
            self.reportSkippedAttribs(shape, ["id","d","transform","style"]+OptimizableSvg.appearance_keys)
            self.reportSkippedChildren(shape,list(shape))

    def parseEllipse(self, shape, parent):
        cx = float(shape.attrib["cx"]) if "cx" in shape.attrib else 0.
        cy = float(shape.attrib["cy"]) if "cy" in shape.attrib else 0.
        rx = float(shape.attrib["rx"])
        ry = float(shape.attrib["ry"])
        name = ''
        if 'id' in shape.attrib:
            name = shape.attrib['id']
        transform = OptimizableSvg.parseTransform(shape)
        appearance = OptimizableSvg.parseAppearance(shape, self.defs, self.device)
        node = OptimizableSvg.EllipseNode(name, transform, appearance, self.settings, (cx, cy, rx, ry))
        parent.children.append(node)

        if self.verbose:
            self.reportSkippedAttribs(shape, ["id", "x", "y", "r", "transform",
                                              "style"] + OptimizableSvg.appearance_keys)
            self.reportSkippedChildren(shape, list(shape))

    def parsePolygon(self, shape, parent):
        points_string = shape.attrib['points']
        name = ''
        points=[]
        for point_string in points_string.split(" "):
            if len(point_string) == 0:
                continue
            coord_strings=point_string.split(",")
            assert len(coord_strings)==2
            points.append([float(coord_strings[0]),float(coord_strings[1])])
        points=torch.tensor(points,dtype=torch.float,device=self.device)
        if 'id' in shape.attrib:
            name = shape.attrib['id']
        transform = OptimizableSvg.parseTransform(shape)
        appearance = OptimizableSvg.parseAppearance(shape, self.defs, self.device)
        node = OptimizableSvg.PolygonNode(name, transform, appearance, self.settings, points)
        parent.children.append(node)

        if self.verbose:
            self.reportSkippedAttribs(shape, ["id", "points", "transform", "style"] + OptimizableSvg.appearance_keys)
            self.reportSkippedChildren(shape, list(shape))

    def parseCircle(self,shape,parent):
        cx = float(shape.attrib["cx"]) if "cx" in shape.attrib else 0.
        cy = float(shape.attrib["cy"]) if "cy" in shape.attrib else 0.
        r = float(shape.attrib["r"])
        name = ''
        if 'id' in shape.attrib:
            name = shape.attrib['id']
        transform = OptimizableSvg.parseTransform(shape)
        appearance = OptimizableSvg.parseAppearance(shape, self.defs, self.device)
        node = OptimizableSvg.CircleNode(name, transform, appearance, self.settings, (cx, cy, r))
        parent.children.append(node)

        if self.verbose:
            self.reportSkippedAttribs(shape, ["id", "x", "y", "r", "transform",
                                              "style"] + OptimizableSvg.appearance_keys)
            self.reportSkippedChildren(shape, list(shape))

    def parseRect(self,shape,parent):
        x =      float(shape.attrib["x"]) if "x" in shape.attrib else 0.
        y =      float(shape.attrib["y"]) if "y" in shape.attrib else 0.
        width =  float(shape.attrib["width"])
        height = float(shape.attrib["height"])
        name = ''
        if 'id' in shape.attrib:
            name = shape.attrib['id']
        transform = OptimizableSvg.parseTransform(shape)
        appearance = OptimizableSvg.parseAppearance(shape, self.defs, self.device)
        node = OptimizableSvg.RectNode(name, transform, appearance, self.settings, (x,y,width,height))
        parent.children.append(node)

        if self.verbose:
            self.reportSkippedAttribs(shape, ["id", "x", "y", "width", "height", "transform", "style"] + OptimizableSvg.appearance_keys)
            self.reportSkippedChildren(shape, list(shape))

    def parseGroup(self,group,parent):
        tag = OptimizableSvg.remove_namespace(group.tag)
        id = group.attrib["id"] if "id" in group.attrib else "<No ID>"
        if self.verbose:
            print(self.offset_str("Parsing {}#{}".format(tag, id)))

        self.depth+=1

        transform=self.parseTransform(group)

        #todo process more attributes
        appearance=OptimizableSvg.parseAppearance(group,self.defs,self.device)
        node=OptimizableSvg.GroupNode(id,transform,appearance,self.settings)
        parent.children.append(node)

        if self.verbose:
            self.reportSkippedAttribs(group,["id","transform","style"]+OptimizableSvg.appearance_keys)

        skipped_children=[]
        for child in group:
            if OptimizableSvg.remove_namespace(child.tag) in OptimizableSvg.recognised_shapes:
                self.parseShape(child,node)
            elif OptimizableSvg.remove_namespace(child.tag) == "defs":
                self.parseDefs(child)
            elif OptimizableSvg.remove_namespace(child.tag) == "style":
                self.parseStyle(child)
            elif OptimizableSvg.remove_namespace(child.tag) == "g":
                self.parseGroup(child,node)
            else:
                skipped_children.append(child)

        if self.verbose:
            self.reportSkippedChildren(group,skipped_children)

        self.depth-=1

    def parseStyle(self,style_node):
        tag = OptimizableSvg.remove_namespace(style_node.tag)
        id = style_node.attrib["id"] if "id" in style_node.attrib else "<No ID>"
        if self.verbose:
            print(self.offset_str("Parsing {}#{}".format(tag, id)))

        if style_node.attrib["type"] != "text/css":
            raise ValueError("Only text/css style recognized, got {}".format(style_node.attrib["type"]))

        self.depth += 1

        # creating only a dummy node
        node = OptimizableSvg.SvgNode(id, None, {}, self.settings)

        if self.verbose:
            self.reportSkippedAttribs(def_node, ["id"])

        if len(style_node)>0:
            raise ValueError("Style node should not have children (has {})".format(len(style_node)))

        # collect CSS classes
        sheet = cssutils.parseString(style_node.text)
        for rule in sheet:
            if hasattr(rule, 'selectorText') and hasattr(rule, 'style'):
                name = rule.selectorText
                if len(name) >= 2 and name[0] == '.':
                    self.defs[name] = rule.style.getCssText().replace("\n","")
                else:
                    raise ValueError("Unrecognized CSS selector {}".format(name))
            else:
                raise ValueError("No style or selector text in CSS rule")

        if self.verbose:
            self.reportSkippedChildren(def_node, skipped_children)

        self.depth -= 1

    def parseDefs(self,def_node):
        #only linear gradients are currently supported
        tag = OptimizableSvg.remove_namespace(def_node.tag)
        id = def_node.attrib["id"] if "id" in def_node.attrib else "<No ID>"
        if self.verbose:
            print(self.offset_str("Parsing {}#{}".format(tag, id)))

        self.depth += 1


        # creating only a dummy node
        node = OptimizableSvg.SvgNode(id, None, {},self.settings)

        if self.verbose:
            self.reportSkippedAttribs(def_node, ["id"])

        skipped_children = []
        for child in def_node:
            if OptimizableSvg.remove_namespace(child.tag) == "linearGradient":
                self.parseGradient(child,node)
            elif OptimizableSvg.remove_namespace(child.tag) in OptimizableSvg.recognised_shapes:
                raise NotImplementedError("Definition/instantiation of shapes not supported")
            elif OptimizableSvg.remove_namespace(child.tag) == "defs":
                raise NotImplementedError("Definition within definition not supported")
            elif OptimizableSvg.remove_namespace(child.tag) == "g":
                raise NotImplementedError("Groups within definition not supported")
            else:
                skipped_children.append(child)

            if len(node.children)>0:
                #take this node out and enter it into defs
                self.defs[node.children[0].id]=node.children[0]
                node.children.pop()


        if self.verbose:
            self.reportSkippedChildren(def_node, skipped_children)

        self.depth -= 1

    def parseGradientStop(self,stop):
        param_dict={key:value for key,value in stop.attrib.items() if key in ["id","offset","stop-color","stop-opacity"]}
        style_dict={}
        if "style" in stop.attrib:
            style_dict={item.split(":")[0]:item.split(":")[1] for item in stop.attrib["style"].split(";") if len(item)>0}
        param_dict.update(style_dict)

        offset=OptimizableSvg.parseOpacity(param_dict["offset"])
        color=OptimizableSvg.parse_color(param_dict["stop-color"])
        opacity=OptimizableSvg.parseOpacity(param_dict["stop-opacity"]) if "stop-opacity" in param_dict else 1.

        return offset, color, opacity

    def parseGradient(self, gradient_node, parent):
        tag = OptimizableSvg.remove_namespace(gradient_node.tag)
        id = gradient_node.attrib["id"] if "id" in gradient_node.attrib else "<No ID>"
        if self.verbose:
            print(self.offset_str("Parsing {}#{}".format(tag, id)))

        self.depth += 1
        if "stop" not in [OptimizableSvg.remove_namespace(child.tag) for child in gradient_node]\
            and "href" not in [OptimizableSvg.remove_namespace(key) for key in gradient_node.attrib.keys()]:
            raise ValueError("Gradient {} has neither stops nor a href link to them".format(id))

        transform=self.parseTransform(gradient_node)
        begin=None
        end = None
        offsets=[]
        stops=[]
        href=None

        if "x1" in gradient_node.attrib or "y1" in gradient_node.attrib:
            begin=np.array([0.,0.])
            if "x1" in gradient_node.attrib:
                begin[0] = float(gradient_node.attrib["x1"])
            if "y1" in gradient_node.attrib:
                begin[1] = float(gradient_node.attrib["y1"])
            begin = torch.tensor(begin.transpose(),dtype=torch.float32)

        if "x2" in gradient_node.attrib or "y2" in gradient_node.attrib:
            end=np.array([0.,0.])
            if "x2" in gradient_node.attrib:
                end[0] = float(gradient_node.attrib["x2"])
            if "y2" in gradient_node.attrib:
                end[1] = float(gradient_node.attrib["y2"])
            end=torch.tensor(end.transpose(),dtype=torch.float32)

        stop_nodes=[node for node in list(gradient_node) if OptimizableSvg.remove_namespace(node.tag)=="stop"]
        if len(stop_nodes)>0:
            stop_nodes=sorted(stop_nodes,key=lambda n: float(n.attrib["offset"]))

            for stop in stop_nodes:
                offset, color, opacity = self.parseGradientStop(stop)
                offsets.append(offset)
                stops.append(np.concatenate((color,np.array([opacity]))))

        hkey=next((value for key,value in gradient_node.attrib.items() if OptimizableSvg.remove_namespace(key)=="href"),None)
        if hkey is not None:
            href=self.defs[hkey.lstrip("#")]

        parent.children.append(OptimizableSvg.GradientNode(id,transform,self.settings,begin.to(self.device) if begin is not None else begin,end.to(self.device) if end is not None else end,torch.tensor(offsets,dtype=torch.float32,device=self.device) if len(offsets)>0 else None,torch.tensor(np.array(stops),dtype=torch.float32,device=self.device) if len(stops)>0 else None,href))

        self.depth -= 1

    def parseViewport(self, root):
        if "width" in root.attrib and "height" in root.attrib:
            self.canvas = np.array([int(math.ceil(float(root.attrib["width"]))), int(math.ceil(float(root.attrib["height"])))])
        elif "viewBox" in root.attrib:
            s=root.attrib["viewBox"].split(" ")
            w=s[2]
            h=s[3]
            self.canvas = np.array(
                [int(math.ceil(float(w))), int(math.ceil(float(h)))])
        else:
            raise ValueError("Size information is missing from document definition")
    #endregion

    #region writing
    def write_xml(self):
        tree=self.root.write_xml(self)
        
        return minidom.parseString(etree.tostring(tree, 'utf-8')).toprettyxml(indent="  ")

    def write_defs(self,root):
        if len(self.defs)==0:
            return

        defnode = etree.SubElement(root, 'defs')
        stylenode = etree.SubElement(root,'style')
        stylenode.set('type','text/css')
        stylenode.text=""

        defcpy=copy.copy(self.defs)
        while len(defcpy)>0:
            torem=[]
            for key,value in defcpy.items():
                if issubclass(value.__class__,OptimizableSvg.SvgNode):
                    if value.href is None or value.href not in defcpy:
                        value.write_xml(defnode)
                        torem.append(key)
                    else:
                        continue
                else:
                    #this is a string, and hence a CSS attribute
                    stylenode.text+=key+" {"+value+"}\n"
                    torem.append(key)

            for key in torem:
                del defcpy[key]
    #endregion


