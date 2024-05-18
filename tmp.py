
# import numpy as np

# sample = np.load('data_test/chair_000002.npz')

# vertices = sample['vertices']
# faces = sample['faces'] 

# for i in range(100):
#     tmp = {}
#     tmp['vertices'] = vertices
#     tmp['faces'] = faces
#     tgt_file_path = 'data_test/chair_'+ str(i).zfill(6)+'.npz'
#     np.savez(tgt_file_path, **tmp)
    
    
import os
import numpy as np
from einops import repeat

from meshgpt_pytorch.misc import find_files_with_extension

def remove_duplicate_vertices_and_faces(mesh:dict, return_indices=False):
    points, faces = mesh['vertices'], mesh['faces']
    
    # 移除重复的点并创建索引映射    
    unique_points, inverse_indices = np.unique(points, axis=0, return_inverse=True)

    updated_faces = inverse_indices[faces] # 更新线段索引，但是线段的顺序还是不变
    
    # 删除重复的线段，确保每条线段的小索引在前，大索引在后
    updated_faces = np.sort(updated_faces, axis=1)
    unique_faces, indices = np.unique(updated_faces, axis=0, return_index=True) # 这里 unique 之后，faces 的顺序会被打乱
    
    sorted_indices = np.argsort(indices)
    unique_faces = unique_faces[sorted_indices] # 因此这里对 line 的顺序进行了重新排序，恢复原有的顺序，这是有必要的
    
    mesh['vertices'] = unique_points
    mesh['faces'] = unique_faces
    
    if return_indices:
        return mesh, indices, sorted_indices
    else:
        return mesh

def scale_vertices(vertices, target_radius=1.0, center=None):

    # 计算几何中心
    if center == None:
        # center = np.mean(vertices, axis=0)
        
        # 每个维度的最小值
        min_values = np.min(vertices, axis=0)
        # 每个维度的最大值
        max_values = np.max(vertices, axis=0)
        
        center = (min_values + max_values) / 2  

    # 计算每个顶点到中心的距离，并找到最大距离
    distances = np.linalg.norm(vertices - center, axis=1)
    max_distance = np.max(distances)

    # 计算缩放因子
    scale_factor = target_radius / max_distance

    # 应用缩放
    scaled_vertices = center + (vertices - center) * scale_factor

    return scaled_vertices

def sort_mesh_vertices_and_faces(mesh:dict, return_sorted_indices=False):
    points, faces = mesh['vertices'], mesh['faces']
    
    # Example tolerance value
    tolerance = 0.005  # Adjust this as per your requirement

    # Adjust points based on tolerance
    adjusted_points = np.round(points / tolerance) * tolerance
    # adjusted_points = points

    if points.shape[-1] == 3:
        # 根据 Z-Y-X 规则对顶点排序
        sorted_indices = np.lexsort((adjusted_points[:, 0], adjusted_points[:, 1], adjusted_points[:, 2]))
    elif points.shape[-1] == 2:
        # 根据 Y-X 规则对顶点排序
        sorted_indices = np.lexsort((adjusted_points[:, 0], adjusted_points[:, 1]))
    else:
        raise NotImplementedError
    
    sorted_points = points[sorted_indices]

    # 计算 inverse indices
    inverse_indices = np.argsort(sorted_indices)

    updated_faces = inverse_indices[faces] # 更新线段索引，但是线段的顺序还是不变

    # outer sort
    sorted_indices = np.lexsort((updated_faces[:, 2], updated_faces[:, 1], updated_faces[:, 0]))
    updated_faces = updated_faces[sorted_indices]

    # 更新 faceset 的线段
    mesh['vertices'] = sorted_points
    mesh['faces'] = updated_faces

    if return_sorted_indices:
        return mesh, sorted_indices
    
    return mesh, None

# 定义一个函数来创建Z轴的旋转矩阵
def rotation_matrix_z(angle):
    radians = np.radians(angle)
    cos_theta, sin_theta = np.cos(radians), np.sin(radians)
    return np.array([[cos_theta, -sin_theta, 0], 
                     [sin_theta, cos_theta, 0], 
                     [0, 0, 1]])

def get_rotaion_matrix_3d():
    rot_matrix_all = np.zeros((4, 3, 3))
    
    angles = [0, 90, 180, 270]
    for i in range(4):
        # 创建旋转矩阵
        angle = angles[i]
        rot_matrix = rotation_matrix_z(angle)
        rot_matrix_all[i] = rot_matrix
    
    return rot_matrix_all

def rotate_and_flip(points):
    if points.shape[-1] == 3:
        rotation_all = get_rotaion_matrix_3d()
    # elif points.shape[-1] == 2:
    #     rotation_all = get_rotation_matrix_2d()
    else:
        raise NotImplementedError("Only 2D and 3D points are supported.")

    new_points = []

    for rotation in rotation_all:
        # 旋转
        rotated_vertices  = np.dot(points, rotation.T)
        
        new_points.append(rotated_vertices)
        
        # flip
        flipped_vertices = np.copy(rotated_vertices)  # 创建副本以避免就地修改
        flipped_vertices[..., 0] = -flipped_vertices[..., 0]
        
        new_points.append(flipped_vertices)

    return new_points # 1 -> 8

###### Mesh operations ######
def prune_unused_vertices(vert, face):
    # Identify all unique vertex indices used in face
    unique_vert_ind = np.unique(face)
    mapper = np.zeros(vert.shape[0], dtype=int)
    mapper[unique_vert_ind] = np.arange(unique_vert_ind.shape[0])
    new_face = mapper[face]
    # Create the new vertices array using only the used vertices
    new_vert = vert[unique_vert_ind]

    return new_vert, new_face, unique_vert_ind
    # # Example usage:
    # verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 2, 2]])  # 5 vertices
    # faces = np.array([[0, 2, 3], [0, 3, 4]])  # 2 faces, vertex 4 is unused
    # new_verts, new_faces = prune_unused_vertices(verts, faces)
    # print("New Vertices:\n", new_verts)
    # print("New Faces:\n", new_faces)

data_path = '/studio/datasets/abo/decimated_mesh_for_meshgpt_400_welded/'
tgt_data_path = '/studio/datasets/abo/meshgpt_400_aug/'
data_path = find_files_with_extension(data_path, 'npz')
i = 0

for i, file_path in enumerate(data_path):
    loaded = np.load(file_path, allow_pickle=True)['arr_0'].item()
    # vert, face = loaded["vert"], 
    sample = {'vertices': loaded["vert"], 'faces':loaded["face"]}
    
    # if len(sample['faces'])  400:
        # print(tensor_path)
    # new_sample = remove_duplicate_vertices_and_faces(sample)

    vertices = sample['vertices']
    faces = sample['faces']
    
    vertices = scale_vertices(vertices)
    # sample['vertices'] = vertices
    
    new_vertices = rotate_and_flip(vertices)
    new_faces = repeat(faces, 'n nlv -> b n nlv', b=8)

    tgt_file_path = tgt_data_path + str(i).zfill(5) + "_0.npz"

    for j in range(8):
        mesh = {}
        mesh['vertices'] = new_vertices[j]
        mesh['faces'] = new_faces[j]        
        # mesh = sort_faces(line_set=mesh)    
        mesh, _ = sort_mesh_vertices_and_faces(mesh)
        
        # np.savez(tgt_file_path, vertices=vertices, faces=faces)

        np.savez(
            tgt_file_path.replace('_0', f'_{j}'), 
            vertices = mesh['vertices'], 
            faces = mesh['faces']
        )
    
    # new_vert, new_face, unique_vert_ind = prune_unused_vertices(vert=vertices,face=faces)
    
    # if len(new_vert) > 600:
    #     print(tensor_path, len(new_vert))
    
    # if len(new_sample['vertices']) < len(sample['vertices']):
    #     print(tensor_path)

    if len(vertices) > 400:
        try:
            os.remove(file_path)
            print(f"File {file_path} has been deleted successfully.")
        except FileNotFoundError:
            print(f"File {file_path} not found.")
        except PermissionError:
            print(f"Permission denied to delete {file_path}.")
        except Exception as e:
            print(f"Error occurred while deleting file {file_path}: {e}")  
        pass      

    else:
        i += 1
        

print(i)
