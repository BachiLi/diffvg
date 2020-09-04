import math
import torch

class GeometryLoss:
    def __init__(self, pathObj, xyalign=True, parallel=True, smooth_node=True):
        self.pathObj=pathObj
        self.pathId=pathObj.id
        self.get_segments(pathObj)
        if xyalign:
            self.make_hor_ver_constraints(pathObj)

        self.xyalign=xyalign
        self.parallel=parallel
        self.smooth_node=smooth_node

        if parallel:
            self.make_parallel_constraints(pathObj)

        if smooth_node:
            self.make_smoothness_constraints(pathObj)

    def make_smoothness_constraints(self,pathObj):
        self.smooth_nodes=[]
        for idx, node in enumerate(self.iterate_nodes()):
            sm, t0, t1=self.node_smoothness(node,pathObj)
            if abs(sm)<1e-2:
                self.smooth_nodes.append((node,((t0.norm()/self.segment_approx_length(node[0],pathObj)).item(),(t1.norm()/self.segment_approx_length(node[1],pathObj)).item())))
                #print("Node {} is smooth (smoothness {})".format(idx,sm))
            else:
                #print("Node {} is not smooth (smoothness {})".format(idx, sm))
                pass

    def node_smoothness(self,node,pathObj):
        t0=self.tangent_out(node[0],pathObj)
        t1=self.tangent_in(node[1],pathObj)
        t1rot=torch.stack((-t1[1],t1[0]))
        smoothness=t0.dot(t1rot)/(t0.norm()*t1.norm())

        return smoothness, t0, t1

    def segment_approx_length(self,segment,pathObj):
        if segment[0]==0:
            #line
            idxs=self.segList[segment[0]][segment[1]]
            #should have a pair of indices now
            length=(pathObj.points[idxs[1],:]-pathObj.points[idxs[0],:]).norm()
            return length
        elif segment[0]==1:
            #quadric
            idxs = self.segList[segment[0]][segment[1]]
            # should have a pair of indices now
            length = (pathObj.points[idxs[1],:] - pathObj.points[idxs[0],:]).norm()+(pathObj.points[idxs[2],:] - pathObj.points[idxs[1],:]).norm()
            return length
        elif segment[0]==2:
            #cubic
            idxs = self.segList[segment[0]][segment[1]]
            # should have a pair of indices now
            length = (pathObj.points[idxs[1],:] - pathObj.points[idxs[0],:]).norm()+(pathObj.points[idxs[2],:] - pathObj.points[idxs[1],:]).norm()+(pathObj.points[idxs[3],:] - pathObj.points[idxs[2],:]).norm()
            return length

    def tangent_in(self, segment,pathObj):
        if segment[0]==0:
            #line
            idxs=self.segList[segment[0]][segment[1]]
            #should have a pair of indices now
            tangent=(pathObj.points[idxs[1],:]-pathObj.points[idxs[0],:])/2
            return tangent
        elif segment[0]==1:
            #quadric
            idxs = self.segList[segment[0]][segment[1]]
            # should have a pair of indices now
            tangent = (pathObj.points[idxs[1],:] - pathObj.points[idxs[0],:])
            return tangent
        elif segment[0]==2:
            #cubic
            idxs = self.segList[segment[0]][segment[1]]
            # should have a pair of indices now
            tangent = (pathObj.points[idxs[1],:] - pathObj.points[idxs[0],:])
            return tangent

        assert(False)

    def tangent_out(self, segment, pathObj):
        if segment[0] == 0:
            # line
            idxs = self.segList[segment[0]][segment[1]]
            # should have a pair of indices now
            tangent = (pathObj.points[idxs[0],:] - pathObj.points[idxs[1],:]) / 2
            return tangent
        elif segment[0] == 1:
            # quadric
            idxs = self.segList[segment[0]][segment[1]]
            # should have a pair of indices now
            tangent = (pathObj.points[idxs[1],:] - pathObj.points[idxs[2],:])
            return tangent
        elif segment[0] == 2:
            # cubic
            idxs = self.segList[segment[0]][segment[1]]
            # should have a pair of indices now
            tangent = (pathObj.points[idxs[2],:] - pathObj.points[idxs[3],:])
            return tangent

        assert (False)

    def get_segments(self, pathObj):
        self.segments=[]
        self.lines = []
        self.quadrics=[]
        self.cubics=[]
        self.segList =(self.lines,self.quadrics,self.cubics)
        idx=0
        total_points=pathObj.points.shape[0]
        for ncp in pathObj.num_control_points.numpy():
            if ncp==0:
                self.segments.append((0,len(self.lines)))
                self.lines.append((idx, (idx + 1) % total_points))
                idx+=1
            elif ncp==1:
                self.segments.append((1, len(self.quadrics)))
                self.quadrics.append((idx, (idx + 1), (idx+2) % total_points))
                idx+=ncp+1
            elif ncp==2:
                self.segments.append((2, len(self.cubics)))
                self.cubics.append((idx, (idx + 1), (idx+2), (idx + 3) % total_points))
                idx += ncp + 1

    def iterate_nodes(self):
        for prev, next in zip([self.segments[-1]]+self.segments[:-1],self.segments):
            yield (prev, next)

    def make_hor_ver_constraints(self, pathObj):
        self.horizontals=[]
        self.verticals=[]
        for idx, line in enumerate(self.lines):
            startPt=pathObj.points[line[0],:]
            endPt=pathObj.points[line[1],:]

            dif=endPt-startPt

            if abs(dif[0])<1e-6:
                #is horizontal
                self.horizontals.append(idx)

            if abs(dif[1])<1e-6:
                #is vertical
                self.verticals.append(idx)

    def make_parallel_constraints(self,pathObj):
        slopes=[]
        for lidx, line in enumerate(self.lines):
            startPt = pathObj.points[line[0], :]
            endPt = pathObj.points[line[1], :]

            dif = endPt - startPt

            slope=math.atan2(dif[1],dif[0])
            if slope<0:
                slope+=math.pi

            minidx=-1
            for idx, s in enumerate(slopes):
                if abs(s[0]-slope)<1e-3:
                    minidx=idx
                    break

            if minidx>=0:
                slopes[minidx][1].append(lidx)
            else:
                slopes.append((slope,[lidx]))

        self.parallel_groups=[sgroup[1] for sgroup in slopes if len(sgroup[1])>1 and (not self.xyalign or (sgroup[0]>1e-3 and abs(sgroup[0]-(math.pi/2))>1e-3))]

    def make_line_diff(self,pathObj,lidx):
        line = self.lines[lidx]
        startPt = pathObj.points[line[0], :]
        endPt = pathObj.points[line[1], :]

        dif = endPt - startPt
        return dif

    def calc_hor_ver_loss(self,loss,pathObj):
        for lidx in self.horizontals:
            dif = self.make_line_diff(pathObj,lidx)
            loss+=dif[0].pow(2)

        for lidx in self.verticals:
            dif = self.make_line_diff(pathObj,lidx)
            loss += dif[1].pow(2)

    def calc_parallel_loss(self,loss,pathObj):
        for group in self.parallel_groups:
            diffs=[self.make_line_diff(pathObj,lidx) for lidx in group]
            difmat=torch.stack(diffs,1)
            lengths=difmat.pow(2).sum(dim=0).sqrt()
            difmat=difmat/lengths
            difmat=torch.cat((difmat,torch.zeros(1,difmat.shape[1])))
            rotmat=difmat[:,list(range(1,difmat.shape[1]))+[0]]
            cross=difmat.cross(rotmat)
            ploss=cross.pow(2).sum()*lengths.sum()*10
            loss+=ploss

    def calc_smoothness_loss(self,loss,pathObj):
        for node, tlengths in self.smooth_nodes:
            sl,t0,t1=self.node_smoothness(node,pathObj)
            #add smoothness loss
            loss+=sl.pow(2)*t0.norm().sqrt()*t1.norm().sqrt()
            tl=((t0.norm()/self.segment_approx_length(node[0],pathObj))-tlengths[0]).pow(2)+((t1.norm()/self.segment_approx_length(node[1],pathObj))-tlengths[1]).pow(2)
            loss+=tl*10

    def compute(self, pathObj):
        if pathObj.id != self.pathId:
            raise ValueError("Path ID {} does not match construction-time ID {}".format(pathObj.id,self.pathId))

        loss=torch.tensor(0.)
        if self.xyalign:
            self.calc_hor_ver_loss(loss,pathObj)

        if self.parallel:
            self.calc_parallel_loss(loss, pathObj)

        if self.smooth_node:
            self.calc_smoothness_loss(loss,pathObj)

        #print(loss.item())

        return loss
