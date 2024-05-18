import numpy as np
import random
import torch

seed = 1234
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

def recon():
    import torch
    from meshgpt_pytorch import MeshAutoencoder, Dataset
    from utils.utils import load_ae

    from meshgpt_pytorch.config import VERTEX_DIMENSION
    from meshgpt_pytorch.data import custom_collate
    
    from argparse import ArgumentParser
    from utils.config import NestedDictToClass, load_config
    
    program_parser = ArgumentParser(description='Train a line autoencoder model.')
    program_parser.add_argument('--config', type=str, default='', help='Path to config file.')
    program_args = program_parser.parse_args()

    cfg = load_config(program_args.config)
    args = NestedDictToClass(cfg)
    
    #==============================
    # load model
    #==============================
    autoencoder = MeshAutoencoder(
        # use_linear_attn = args.model.use_linear_attn,
        use_residual_lfq = args.model.use_residual_lfq,
        encoder_dims_through_depth = args.model.encoder_dims_through_depth,
        decoder_dims_through_depth = args.model.decoder_dims_through_depth,
        codebook_size = args.model.codebook_size,
        attn_encoder_depth = args.model.attn_encoder_depth,
        attn_decoder_depth = args.model.attn_decoder_depth,
    )

    autoencoder = load_ae(autoencoder, args.model.ae_ckpt_path)
    autoencoder = autoencoder.to('cuda')
    autoencoder.eval()

    data_from_dataset = True
    if data_from_dataset:
        val_dataset = Dataset(
            is_train=True,
            is_single = args.data.is_single,
            dataset_file_path = args.data.test_set_file_path,
        )

        dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size = 64,
            shuffle = True,
            num_workers = 8,
            pin_memory = True,
            drop_last = False,
            collate_fn = custom_collate
        )
        
        data = next(iter(dataloader))
        vertices = data['vertices']
        # vertice_indices = np.arange(vertices.shape[1])
        # np.random.shuffle(vertice_indices)
        # vertices = vertices[:,vertice_indices,:]
        
        faces = data['faces']
        # line_indices = np.arange(faces.shape[1])
        # np.random.shuffle(line_indices)
        # faces = faces[:,line_indices,:]
        
        # vertices_np = vertices.detach().cpu().numpy()
        # faces_np = faces.detach().cpu().numpy()
        
        # np.savez('recon_faces/npz/vertices_faces_1.npz', vertices=vertices_np[0], faces=faces_np[1])
        
        vertices = vertices.to('cuda')
        faces = faces.to('cuda')
    
    # file_path = 'input_data/right_wf_1112.npz'
    # data = np.load(file_path)
    # vertices = torch.tensor(data['vertices']).to('cuda').unsqueeze(0)
    # faces = torch.tensor(data['faces']).to('cuda').unsqueeze(0).long()
    
    # codes = autoencoder.tokenize(vertices, faces)
    # codes_np = codes.detach().cpu().numpy()
    # np.save('recon_faces/codes_1.npy', codes_np)
    
    # return
    
    face_coords = autoencoder(
        vertices = vertices,
        faces = faces,
        only_return_recon_faces = True
    )

    print(face_coords.shape)


    face_coords = face_coords.detach().cpu().numpy()
            
    for i, sample in enumerate(face_coords):

        # 找出 NaN 值
        nan_mask = np.isnan(sample)

        # 使用 ~nan_mask 选择非 NaN 值
        sample = sample[~nan_mask]

        # 提取顶点
        vertices = sample.reshape(-1, VERTEX_DIMENSION)  # 这会把顶点扁平化为一个 Nx3 的数组
        
        # 创建线索引
        faces = [[i, i + 1, i + 2] for i in range(0, len(vertices), 3)]

        tgt_file_path = f'recon_mesh/npz/mesh_coords_1_{str(i).zfill(5)}.npz'
        np.savez(tgt_file_path, vertices=vertices, faces=faces)
        
        

    
# def save_recon_house():
#     from utils.vis_utils import plot_sample
#     from wfgpt_2d.misc  import get_file_list
    
#     dir_path = 'recon_faces/2d/'
#     file_path_list = get_file_list(dir_path)
#     for file_path in file_path_list:
#         sample = np.load(file_path)
#         vertices = sample['vertices']
#         line_coords = vertices.reshape(-1, 2, 2)
#         file_path = 'recon_faces/imgs/' + file_path.split('/')[-1].split('.')[0] + '.png'
#         plot_sample(line_coords, file_path=file_path)
            
if __name__ == '__main__':
    # save_recon_house()
    recon()