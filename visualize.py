import numpy as np
import SimpleITK as sitk
import matplotlib.pylab as plt
from scrollview import ScrollView

DATA_PATH = "data/subset1/1.3.6.1.4.1.14519.5.2.1.6279.6001.104562737760173137525888934217.mhd"

itkimage = sitk.ReadImage(DATA_PATH, sitk.sitkFloat32)
ct_scan = sitk.GetArrayFromImage(itkimage)

# Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
origin = np.array(list(itkimage.GetOrigin()))
# Read the spacing along each dimension
spacing = np.array(list(itkimage.GetSpacing()))

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

# Using Coords from annotations.csv
label = world_2_voxel(np.array([-124.8342624, 127.2471546, -473.0644785]), origin, spacing)

# Show all images
"""plt.figure(figsize=(40,40))
plt.gray()
plt.subplots_adjust(0,0,1,1,0.01,0.01)
for i in range(ct_scan.shape[0]):
    plt.subplot(11,11,i+1), plt.imshow(ct_scan[i]), plt.axis('off')"""

# Show images using ScrollView
fig, ax = plt.subplots()
ScrollView(ct_scan, slice=label[2]).plot(ax, cmap='gray')
plt.show()