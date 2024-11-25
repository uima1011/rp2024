import h5py
import os
import numpy as np
from loguru import logger


def store_orthographic_data(scene_file, heightmap, colormap, grasp_pose, workspace_bounds, target_directory):
    """Stores processed data in HDF5 format."""
    os.makedirs(target_directory, exist_ok=True)
    file_name = os.path.join(target_directory, scene_file)
    
    logger.info(f"Storing processed data to {file_name}")
    with h5py.File(file_name, 'w') as f:
        # Store heightmap and colormap with compression
        f.create_dataset('heightmap', 
                        data=heightmap,
                        compression='gzip')
        
        f.create_dataset('colormap', 
                        data=colormap,
                        chunks=(colormap.shape[0], colormap.shape[1], 3),
                        compression='gzip')
        
        # Store grasp pose and workspace bounds
        f.create_dataset('grasp_pose', data=grasp_pose)
        f.create_dataset('workspace_bounds', data=np.array(workspace_bounds))
