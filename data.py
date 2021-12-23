import os
import numpy as np
import pandas as pd
import csv
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib
import random
from PIL import Image
import glob
import SimpleITK as sitk
import matplotlib.pylab as plt
from scrollview import ScrollView
import scipy.ndimage as ndimage

def world_2_voxel(world_coordinates, origin, spacing):
    '''
    This function is used to convert the world coordinates to voxel coordinates using 
    the origin and spacing of the ct_scan
    '''
    stretched_voxel_coordinates = np.absolute(world_coordinates - origin)
    voxel_coordinates = (stretched_voxel_coordinates / spacing).astype(int)
    return voxel_coordinates

def voxel_2_world(voxel_coordinates, origin, spacing):
    '''
    This function is used to convert the voxel coordinates to world coordinates using 
    the origin and spacing of the ct_scan.
    '''
    stretched_voxel_coordinates = voxel_coordinates * spacing
    world_coordinates = stretched_voxel_coordinates + origin
    return world_coordinates

class luna16_dataset(Dataset):
    def __init__(self, mode='train', dataset_path='data/', annotations_path='annotations.csv'):
        super().__init__()
        self.dataset_path = dataset_path
        self.mode = mode
        self.data = sorted(glob.glob(os.path.join(dataset_path, mode, "*.mhd")))
        self.annotations = pd.read_csv(os.path.join(dataset_path, annotations_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        seriesuid = self.data[index].lstrip(os.path.join(self.dataset_path, self.mode)).rstrip(".mhd")
        annotations = self.annotations.loc[self.annotations["seriesuid"]==seriesuid]
        coords = np.array(annotations.filter(items=["coordX", "coordY", "coordZ"]))
        diameters = np.array(annotations.filter(items=["diameter_mm"]))
        
        itkimage = sitk.ReadImage(self.data[index], sitk.sitkFloat32)
        data = sitk.GetArrayFromImage(itkimage) # z,y,x
        data = np.transpose(data, (2,1,0)) # z,y,x -> x,y,z
        origin = np.array(list(itkimage.GetOrigin())) # x,y,z
        spacing = np.array(list(itkimage.GetSpacing())) # x,y,z
        
        # Create mask
        mask = np.zeros_like(data) # x,y,z
        for coord, diameter in zip(coords, diameters):
            min_coord = coord - diameter/2
            max_coord = coord + diameter/2
            min_voxel_coord = world_2_voxel(min_coord, origin, spacing) # x,y,z
            max_voxel_coord = world_2_voxel(max_coord, origin, spacing) # x,y,z
            mask[min_voxel_coord[0]:max_voxel_coord[0], min_voxel_coord[1]:max_voxel_coord[1], min_voxel_coord[2]:max_voxel_coord[2]] = 1
            
        # Resize voxel data and mask
        # Input resize to: 32x32x32
        x_dim, y_dim, z_dim = data.shape
        data = torch.Tensor(ndimage.zoom(data, [64/x_dim, 64/y_dim, 64/z_dim])).unsqueeze(0)
        mask = torch.Tensor(ndimage.zoom(mask, [64/x_dim, 64/y_dim, 64/z_dim]))
        
        """
        # Visualize resized data
        fig, ax = plt.subplots()
        ScrollView(np.transpose(data, (2,1,0))).plot(ax, cmap='gray')
        plt.show()
        """
        
        return data, mask
        
if __name__ == '__main__':
    dataset = luna16_dataset("train")
    print(len(dataset))
    loader = DataLoader(dataset, shuffle=True, batch_size=4)
    print(len(loader))
    dataset[0]