#imports
import numpy as np
import matplotlib.pyplot as plt
import os

from math import floor, ceil
from random import randint

from sklearn.neighbors import KDTree
from skimage.util.shape import view_as_windows
from skimage import io

from PIL import Image, ImageDraw
from IPython.display import clear_output

class patchBasedTextureSynthesis:
    
    def __init__(self, exampleMapPath, in_outputPath, in_outputSize, in_patchSize, in_overlapSize, in_windowStep = 5, in_mirror_hor = True, in_mirror_vert = True, in_shapshots = True):
        self.exampleMap = self.loadExampleMap(exampleMapPath)
        self.snapshots = in_shapshots
        self.outputPath = in_outputPath
        self.outputSize = in_outputSize
        self.patchSize = in_patchSize
        self.overlapSize = in_overlapSize
        self.mirror_hor = in_mirror_hor
        self.mirror_vert = in_mirror_vert
        self.total_patches_count = 0 #excluding mirrored versions
        self.windowStep = 5
        self.iter = 0
        
        self.checkIfDirectoryExists() #check if output directory exists
        self.examplePatches = self.prepareExamplePatches()
        self.canvas, self.filledMap, self.idMap = self.initCanvas()
        self.initFirstPatch() #place random block to start with
        self.kdtree_topOverlap, self.kdtree_leftOverlap, self.kdtree_combined = self.initKDtrees()

        self.PARM_truncation = 0.8
        self.PARM_attenuation = 2

    def checkIfDirectoryExists(self):
        if not os.path.exists(self.outputPath):
            os.makedirs(self.outputPath)
        
    def resolveAll(self):
        self.saveParams()
        #resolve all unresolved patches
        for i in range(np.sum(1-self.filledMap).astype(int)):
            self.resolveNext()
            
        if not self.snapshots:
            img = Image.fromarray(np.uint8(self.canvas*255))
            img = img.resize((self.outputSize[0], self.outputSize[1]), resample=0, box=None)
            img.save(self.outputPath + 'out.jpg')
        # else:
        #     self.visualize([0,0], [], [], showCandidates=False)
        return img
    def saveParams(self):
        #write
        text_file = open(self.outputPath + 'params.txt', "w")
        text_file.write("PatchSize: %d \nOverlapSize: %d \nMirror Vert: %d \nMirror Hor: %d" % (self.patchSize, self.overlapSize, self.mirror_vert, self.mirror_hor))
        text_file.close()
        
    def resolveNext(self):
        #coordinate of the next one to resolve
        coord = self.idCoordTo2DCoord(np.sum(self.filledMap), np.shape(self.filledMap)) #get 2D coordinate of next to resolve patch
        #get overlap areas of the patch we want to resolve
        overlapArea_Top = self.getOverlapAreaTop(coord)
        overlapArea_Left = self.getOverlapAreaLeft(coord)
        #find most similar patch from the examples
        dist, ind = self.findMostSimilarPatches(overlapArea_Top, overlapArea_Left, coord)
        
        if self.mirror_hor or self.mirror_vert:
            #check that top and left neighbours are not mirrors
            dist, ind = self.checkForMirrors(dist, ind, coord)

        #choose random valid patch
        probabilities = self.distances2probability(dist, self.PARM_truncation, self.PARM_attenuation)
        chosenPatchId = np.random.choice(ind, 1, p=probabilities)
        
        #update canvas
        blend_top = (overlapArea_Top is not None)
        blend_left = (overlapArea_Left is not None)
        self.updateCanvas(chosenPatchId, coord[0], coord[1], blend_top, blend_left)
        
        #update filledMap and id map ;)
        self.filledMap[coord[0], coord[1]] = 1
        self.idMap[coord[0], coord[1]] = chosenPatchId
        
        #visualize
        # self.visualize(coord, chosenPatchId, ind)
        
        self.iter += 1
        
    def visualize(self, coord, chosenPatchId, nonchosenPatchId, showCandidates = True):
        #full visualization includes both example and generated img
        canvasSize = np.shape(self.canvas)
        #insert generated image
        vis = np.zeros((canvasSize[0], canvasSize[1] * 2, 3)) + 0.2
        vis[:, 0:canvasSize[1]] = self.canvas
        #insert example
        exampleHighlited = np.copy(self.exampleMap)
        if showCandidates:
            exampleHighlited = self.hightlightPatchCandidates(chosenPatchId, nonchosenPatchId)
        h = floor(canvasSize[0] / 2)
        w = floor(canvasSize[1] / 2)
        exampleResized = self.resize(exampleHighlited, [h, w])
        offset_h = floor((canvasSize[0] - h) / 2) 
        offset_w = floor((canvasSize[1] - w) / 2)
        
        vis[offset_h:offset_h+h, canvasSize[1]+offset_w:canvasSize[1]+offset_w+w] = exampleResized
        
        #show live update
        plt.imshow(vis)
        clear_output(wait=True)
        display(plt.show())
        
        if self.snapshots:
            img = Image.fromarray(np.uint8(vis*255))
            img = img.resize((self.outputSize[0]*2, self.outputSize[1]), resample=0, box=None)
            img.save(self.outputPath + 'out' + str(self.iter) + '.jpg')
        
    def hightlightPatchCandidates(self, chosenPatchId, nonchosenPatchId):
        
        result = np.copy(self.exampleMap)
        
        #mod patch ID
        chosenPatchId = chosenPatchId[0] % self.total_patches_count
        if len(nonchosenPatchId)>0:
            nonchosenPatchId = nonchosenPatchId % self.total_patches_count
            #exlcude chosen from nonchosen
            nonchosenPatchId = np.delete(nonchosenPatchId, np.where(nonchosenPatchId == chosenPatchId))
            #highlight non chosen candidates
            c = [0.25, 0.9 ,0.45]
            self.highlightPatches(result, nonchosenPatchId, color=c, highlight_width = 4, alpha = 0.5)
        
        #hightlight chosen
        c = [1.0, 0.25, 0.15]
        self.highlightPatches(result, [chosenPatchId], color=c, highlight_width = 4, alpha = 1)
        
        return result
    
    def highlightPatches(self, writeResult, patchesIDs, color, highlight_width = 2, solid = False, alpha = 0.1):
        
        searchWindow = self.patchSize + 2*self.overlapSize
        
        #number of possible steps
        row_steps = floor((np.shape(writeResult)[0] - searchWindow) / self.windowStep) + 1
        col_steps = floor((np.shape(writeResult)[1] - searchWindow) / self.windowStep) + 1 
        
        for i in range(len(patchesIDs)):
            
            chosenPatchId = patchesIDs[i]
            
            #patch Id to step
            patch_row = floor(chosenPatchId / col_steps)
            patch_col = chosenPatchId - patch_row * col_steps
            
            #highlight chosen patch (below are boundaries of the example patch)
            row_start = self.windowStep* patch_row
            row_end = self.windowStep * patch_row + searchWindow
            col_start = self.windowStep * patch_col
            col_end = self.windowStep * patch_col + searchWindow
            
            if not solid:
                w = highlight_width
                overlap = np.copy(writeResult[row_start:row_start+w, col_start:col_end])
                writeResult[row_start:row_start+w, col_start:col_end] = overlap * (1-alpha) + (np.zeros(np.shape(overlap))+color) * alpha #top
                overlap = np.copy(writeResult[row_end-w:row_end, col_start:col_end])
                writeResult[row_end-w:row_end, col_start:col_end] = overlap * (1-alpha) + (np.zeros(np.shape(overlap))+color) * alpha #bot
                overlap = np.copy( writeResult[row_start:row_end, col_start:col_start+w])
                writeResult[row_start:row_end, col_start:col_start+w] = overlap * (1-alpha) + (np.zeros(np.shape(overlap))+color) * alpha #left
                overlap = np.copy(writeResult[row_start:row_end, col_end-w:col_end])
                writeResult[row_start:row_end, col_end-w:col_end] = overlap * (1-alpha) + (np.zeros(np.shape(overlap))+color) * alpha #end
            else:
                a = alpha
                writeResult[row_start:row_end, col_start:col_end] =  writeResult[row_start:row_end, col_start:col_end] * (1-a) + (np.zeros(np.shape(writeResult[row_start:row_end, col_start:col_end]))+color) * a
        
        
    def resize(self, imgArray, targetSize):
        img = Image.fromarray(np.uint8(imgArray*255))
        img = img.resize((targetSize[0], targetSize[1]), resample=0, box=None)
        return np.array(img)/255
        
    def findMostSimilarPatches(self, overlapArea_Top, overlapArea_Left, coord, in_k=5):
        
        #check which KD tree we need to use
        if (overlapArea_Top is not None) and (overlapArea_Left is not None):
            combined = self.getCombinedOverlap(overlapArea_Top.reshape(-1), overlapArea_Left.reshape(-1))
            dist, ind = self.kdtree_combined.query([combined], k=in_k)
        elif overlapArea_Top is not None:
            dist, ind = self.kdtree_topOverlap.query([overlapArea_Top.reshape(-1)], k=in_k)
        elif overlapArea_Left is not None:
            dist, ind = self.kdtree_leftOverlap.query([overlapArea_Left.reshape(-1)], k=in_k)
        else:
            raise Exception("ERROR: no valid overlap area is passed to -findMostSimilarPatch-")
        dist = dist[0]
        ind = ind[0]
        
        return dist, ind
     
    #disallow visually similar blocks to be placed next to each other
    def checkForMirrors(self, dist, ind, coord, thres = 3):
        remove_i = []
        #do I have a top or left neighbour
        if coord[0]-1>-1:
            top_neigh = int(self.idMap[coord[0]-1, coord[1]])
            for i in range(len(ind)): 
                if (abs(ind[i]%self.total_patches_count - top_neigh%self.total_patches_count) < thres):
                    remove_i.append(i)     
        if  coord[1]-1>-1:
            left_neigh = int(self.idMap[coord[0], coord[1]-1])
            for i in range(len(ind)):
                if (abs(ind[i]%self.total_patches_count - left_neigh%self.total_patches_count) < thres):
                    remove_i.append(i)  
        
        dist = np.delete(dist, remove_i)
        ind = np.delete(ind, remove_i)
        
        return dist, ind

        
    def distances2probability(self, distances, PARM_truncation, PARM_attenuation):

        probabilities = 1 - distances / np.max(distances)  
        probabilities *= (probabilities > PARM_truncation)
        probabilities = pow(probabilities, PARM_attenuation) #attenuate the values
        #check if we didn't truncate everything!
        if np.sum(probabilities) == 0:
            #then just revert it
            probabilities = 1 - distances / np.max(distances) 
            probabilities *= (probabilities > PARM_truncation*np.max(probabilities)) # truncate the values (we want top truncate%)
            probabilities = pow(probabilities, PARM_attenuation)
        probabilities /= np.sum(probabilities) #normalize so they add up to one  

        return probabilities
        
    def getOverlapAreaTop(self, coord):
        #do I have a top neighbour
        if coord[0]-1>-1:
            canvasPatch = self.patchCoord2canvasPatch(coord)
            return canvasPatch[0:self.overlapSize, :, :]
        else:
            return None
        
    def getOverlapAreaLeft(self, coord):
        #do I have a left neighbour
        if coord[1]-1>-1:
            canvasPatch = self.patchCoord2canvasPatch(coord)
            return canvasPatch[:, 0:self.overlapSize, :]    
        else:
            return None 
 
    def initKDtrees(self):
        #prepate overlap patches
        topOverlap = self.examplePatches[:, 0:self.overlapSize, :, :]
        leftOverlap = self.examplePatches[:, :, 0:self.overlapSize, :]
        shape_top = np.shape(topOverlap)
        shape_left = np.shape(leftOverlap)
                                   
        flatten_top = topOverlap.reshape(shape_top[0], -1)
        flatten_left = leftOverlap.reshape(shape_left[0], -1)
        flatten_combined = self.getCombinedOverlap(flatten_top, flatten_left) 
        
        tree_top = KDTree(flatten_top)
        tree_left = KDTree(flatten_left)
        tree_combined = KDTree(flatten_combined)
       
        return tree_top, tree_left, tree_combined
    
    #the corner of 2 overlaps is counted double
    def getCombinedOverlap(self, top, left):
        shape = np.shape(top)
        if len(shape) > 1:
            combined = np.zeros((shape[0], shape[1]*2))
            combined[0:shape[0], 0:shape[1]] = top
            combined[0:shape[0], shape[1]:shape[1]*2] = left
        else:
            combined = np.zeros((shape[0]*2))
            combined[0:shape[0]] = top
            combined[shape[0]:shape[0]*2] = left
        return combined

    def initFirstPatch(self):
        #grab a random block 
        patchId = randint(0, np.shape(self.examplePatches)[0])
        #mark out fill map
        self.filledMap[0, 0] = 1
        self.idMap[0, 0] = patchId % self.total_patches_count
        #update canvas
        self.updateCanvas(patchId, 0, 0, False, False)
        #visualize
        # self.visualize([0,0], [patchId], [])

        
    def prepareExamplePatches(self):
        
        searchKernelSize = self.patchSize + 2 * self.overlapSize
        
        result = view_as_windows(self.exampleMap, [searchKernelSize, searchKernelSize, 3] , self.windowStep)
        shape = np.shape(result)
        result = result.reshape(shape[0]*shape[1], searchKernelSize, searchKernelSize, 3)
        
        self.total_patches_count = shape[0]*shape[1]
        
        if self.mirror_hor:
            #flip along horizonal axis
            hor_result = np.zeros(np.shape(result))
            
            for i in range(self.total_patches_count):
                hor_result[i] = result[i][::-1, :, :]
            
            result = np.concatenate((result, hor_result))
        if self.mirror_vert:
            vert_result = np.zeros((shape[0]*shape[1], searchKernelSize, searchKernelSize, 3))
            
            for i in range(self.total_patches_count):
                vert_result[i] = result[i][:, ::-1, :]
            
            result = np.concatenate((result, vert_result))
        
        return result

    def initCanvas(self):
        
        #check whether the outputSize adheres to patch+overlap size
        num_patches_X = ceil((self.outputSize[0]-self.overlapSize)/(self.patchSize+self.overlapSize))
        num_patches_Y = ceil((self.outputSize[1]-self.overlapSize)/(self.patchSize+self.overlapSize))
        #calc needed output image size
        required_size_X = num_patches_X*self.patchSize + (num_patches_X+1)*self.overlapSize
        required_size_Y = num_patches_Y*self.patchSize + (num_patches_X+1)*self.overlapSize
        
        #create empty canvas
        canvas = np.zeros((required_size_X, required_size_Y, 3))
        filledMap = np.zeros((num_patches_X, num_patches_Y)) #map showing which patches have been resolved
        idMap = np.zeros((num_patches_X, num_patches_Y)) - 1 #stores patches id
        
        print("modified output size: ", np.shape(canvas))
        print("number of patches: ", np.shape(filledMap)[0])

        return canvas, filledMap, idMap

    def idCoordTo2DCoord(self, idCoord, imgSize):
        row = int(floor(idCoord / imgSize[0]))
        col = int(idCoord - row * imgSize[1])
        return [row, col]

    def updateCanvas(self, inputPatchId, coord_X, coord_Y, blendTop = False, blendLeft = False):
        #translate Patch coordinate into Canvas coordinate
        x_range = self.patchCoord2canvasCoord(coord_X)
        y_range = self.patchCoord2canvasCoord(coord_Y)
        examplePatch = self.examplePatches[inputPatchId]
        if blendLeft:
            canvasOverlap = self.canvas[x_range[0]:x_range[1], y_range[0]:y_range[0]+self.overlapSize]
            examplePatchOverlap = np.copy(examplePatch[0][:, 0:self.overlapSize])
            examplePatch[0][:, 0:self.overlapSize] = self.linearBlendOverlaps(canvasOverlap, examplePatchOverlap, 'left')
        if blendTop:
            canvasOverlap = self.canvas[x_range[0]:x_range[0]+self.overlapSize, y_range[0]:y_range[1]]
            examplePatchOverlap = np.copy(examplePatch[0][0:self.overlapSize, :])
            examplePatch[0][0:self.overlapSize, :] = self.linearBlendOverlaps(canvasOverlap, examplePatchOverlap, 'top')
        self.canvas[x_range[0]:x_range[1], y_range[0]:y_range[1]] = examplePatch
        
    def linearBlendOverlaps(self, canvasOverlap, examplePatchOverlap, mode):
        if mode == 'left':
            mask = np.repeat(np.arange(self.overlapSize)[np.newaxis, :], np.shape(canvasOverlap)[0], axis=0) / self.overlapSize
        elif mode == 'top':
            mask = np.repeat(np.arange(self.overlapSize)[:, np.newaxis], np.shape(canvasOverlap)[1], axis=1) / self.overlapSize
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2) #cast to 3d array
        return canvasOverlap * (1 - mask) + examplePatchOverlap * mask
    
    #def minimumBoundaryError(self, canvasOverlap, examplePatchOverlap, mode)
    
    def patchCoord2canvasCoord(self, coord):
        return [(self.patchSize+self.overlapSize)*coord, (self.patchSize+self.overlapSize)*(coord+1) + self.overlapSize]
    
    def patchCoord2canvasPatch(self, coord):
        x_range = self.patchCoord2canvasCoord(coord[0])
        y_range = self.patchCoord2canvasCoord(coord[1])
        return np.copy(self.canvas[x_range[0]:x_range[1], y_range[0]:y_range[1]])
    
    def loadExampleMap(self, exampleMapPath):
        exampleMap = io.imread(exampleMapPath) #returns an MxNx3 array
        exampleMap = exampleMap / 255.0 #normalize
        #make sure it is 3channel RGB
        if (np.shape(exampleMap)[-1] > 3): 
            exampleMap = exampleMap[:,:,:3] #remove Alpha Channel
        elif (len(np.shape(exampleMap)) == 2):
            exampleMap = np.repeat(exampleMap[np.newaxis, :, :], 3, axis=0) #convert from Grayscale to RGB
        return exampleMap
