import numpy as np

sample = np.load('data_test/chair_000002.npz')

vertices = sample['vertices']
faces = sample['faces'] 

for i in range(100):
    tmp = {}
    tmp['vertices'] = vertices
    tmp['faces'] = faces
    tgt_file_path = 'data_test/chair_'+ str(i).zfill(6)+'.npz'
    np.savez(tgt_file_path, **tmp)
    
