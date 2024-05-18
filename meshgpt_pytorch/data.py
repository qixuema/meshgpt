from __future__ import annotations

import os
import pickle
from pathlib import Path
from functools import partial
import torch
from torch import Tensor
from torch import is_tensor
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import numpy as np
from numpy.lib.format import open_memmap

from einops import rearrange, reduce

from beartype import beartype
from beartype.typing import Tuple, List, Callable, Dict, Callable

from torchtyping import TensorType

from pytorch_custom_utils.utils import pad_or_slice_to

from collections import defaultdict

from meshgpt_pytorch.misc import find_files_with_extension

# helper fn

def exists(v):
    return v is not None

def identity(t):
    return t

def get_file_list(dir_path):
    file_path_list = [os.path.join(dir_path, i) for i in os.listdir(dir_path)]
    file_path_list.sort()
    return file_path_list


# constants

Vertices = TensorType['nv', 3, float]   # 3 coordinates
Faces = TensorType['nf', 3, int]        # 3 vertices

# decorator for auto-caching texts -> text embeds

# you would decorate your Dataset class with this
# and then change your `data_kwargs = ["text_embeds", "vertices", "faces"]`

@beartype
def cache_text_embeds_for_dataset(
    embed_texts_fn: Callable[[List[str]], Tensor],
    max_text_len: int,
    cache_path: str = './text_embed_cache'
):
    # create path to cache folder

    path = Path(cache_path)
    path.mkdir(exist_ok = True, parents = True)
    assert path.is_dir()

    # global memmap handles

    text_embed_cache = None
    is_cached = None

    # cache function

    def get_maybe_cached_text_embed(
        idx: int,
        dataset_len: int,
        text: str,
        memmap_file_mode = 'w+'
    ):
        nonlocal text_embed_cache
        nonlocal is_cached

        # init cache on first call

        if not exists(text_embed_cache):
            test_embed = embed_texts_fn(['test'])
            feat_dim = test_embed.shape[-1]
            shape = (dataset_len, max_text_len, feat_dim)

            text_embed_cache = open_memmap(str(path / 'cache.text_embed.memmap.npy'), mode = memmap_file_mode, dtype = 'float32', shape = shape)
            is_cached = open_memmap(str(path / 'cache.is_cached.memmap.npy'), mode = memmap_file_mode, dtype = 'bool', shape = (dataset_len,))

        # determine whether to fetch from cache
        # or call text model

        if is_cached[idx]:
            text_embed = torch.from_numpy(text_embed_cache[idx])
        else:
            # cache

            text_embed = get_text_embed(text)
            text_embed = pad_or_slice_to(text_embed, max_text_len, dim = 0, pad_value = 0.)

            is_cached[idx] = True
            text_embed_cache[idx] = text_embed.cpu().numpy()

        mask = ~reduce(text_embed == 0, 'n d -> n', 'all')
        return text_embed[mask]

    # get text embedding

    def get_text_embed(text: str):
        text_embeds = embed_texts_fn([text])
        return text_embeds[0]

    # inner function

    def inner(dataset_klass):
        assert issubclass(dataset_klass, Dataset)

        orig_init = dataset_klass.__init__
        orig_get_item = dataset_klass.__getitem__

        def __init__(
            self,
            *args,
            cache_memmap_file_mode = 'w+',
            **kwargs
        ):
            orig_init(self, *args, **kwargs)

            self._cache_memmap_file_mode = cache_memmap_file_mode

            if hasattr(self, 'data_kwargs'):
                self.data_kwargs = [('text_embeds' if data_kwarg == 'texts' else data_kwarg) for data_kwarg in self.data_kwargs]

        def __getitem__(self, idx):
            items = orig_get_item(self, idx)

            get_text_embed_ = partial(get_maybe_cached_text_embed, idx, len(self), memmap_file_mode = self._cache_memmap_file_mode)

            if isinstance(items, dict):
                if 'texts' in items:
                    text_embed = get_text_embed_(items['texts'])
                    items['text_embeds'] = text_embed
                    del items['texts']

            elif isinstance(items, tuple):
                new_items = []

                for maybe_text in items:
                    if not isinstance(maybe_text, str):
                        new_items.append(maybe_text)
                        continue

                    new_items.append(get_text_embed_(maybe_text))

                items = tuple(new_items)

            return items

        dataset_klass.__init__ = __init__
        dataset_klass.__getitem__ = __getitem__

        return dataset_klass

    return inner

# decorator for auto-caching face edges

# you would decorate your Dataset class with this function
# and then change your `data_kwargs = ["vertices", "faces", "face_edges"]`

@beartype
def cache_face_edges_for_dataset(
    max_edges_len: int,
    cache_path: str = './face_edges_cache',
    assert_edge_len_lt_max: bool = True,
    pad_id = -1
):
    # create path to cache folder

    path = Path(cache_path)
    path.mkdir(exist_ok = True, parents = True)
    assert path.is_dir()

    # global memmap handles

    face_edges_cache = None
    is_cached = None

    # cache function

    def get_maybe_cached_face_edges(
        idx: int,
        dataset_len: int,
        faces: Tensor,
        memmap_file_mode = 'w+'
    ):
        nonlocal face_edges_cache
        nonlocal is_cached

        if not exists(face_edges_cache):
            # init cache on first call

            shape = (dataset_len, max_edges_len, 2)
            face_edges_cache = open_memmap(str(path / 'cache.face_edges_embed.memmap.npy'), mode = memmap_file_mode, dtype = 'float32', shape = shape)
            is_cached = open_memmap(str(path / 'cache.is_cached.memmap.npy'), mode = memmap_file_mode, dtype = 'bool', shape = (dataset_len,))

        # determine whether to fetch from cache
        # or call derive face edges function

        if is_cached[idx]:
            face_edges = torch.from_numpy(face_edges_cache[idx])
        else:
            # cache

            face_edges = derive_face_edges_from_faces(faces, pad_id = pad_id)

            edge_len = face_edges.shape[0]
            assert not assert_edge_len_lt_max or (edge_len <= max_edges_len), f'mesh #{idx} has {edge_len} which exceeds the cache length of {max_edges_len}'

            face_edges = pad_or_slice_to(face_edges, max_edges_len, dim = 0, pad_value = pad_id)

            is_cached[idx] = True
            face_edges_cache[idx] = face_edges.cpu().numpy()

        mask = reduce(face_edges != pad_id, 'n d -> n', 'all')
        return face_edges[mask]

    # inner function

    def inner(dataset_klass):
        assert issubclass(dataset_klass, Dataset)

        orig_init = dataset_klass.__init__
        orig_get_item = dataset_klass.__getitem__

        def __init__(
            self,
            *args,
            cache_memmap_file_mode = 'w+',
            **kwargs
        ):
            orig_init(self, *args, **kwargs)

            self._cache_memmap_file_mode = cache_memmap_file_mode

            if hasattr(self, 'data_kwargs'):
                self.data_kwargs.append('face_edges')

        def __getitem__(self, idx):
            items = orig_get_item(self, idx)

            get_face_edges_ = partial(get_maybe_cached_face_edges, idx, len(self), memmap_file_mode = self._cache_memmap_file_mode)

            if isinstance(items, dict):
                face_edges = get_face_edges_(items['faces'])
                items['face_edges'] = face_edges

            elif isinstance(items, tuple):
                _, faces, *_ = items
                face_edges = get_face_edges_(faces)
                items = (*items, face_edges)

            return items

        dataset_klass.__init__ = __init__
        dataset_klass.__getitem__ = __getitem__

        return dataset_klass

    return inner

# dataset

class DatasetFromTransforms(Dataset):
    @beartype
    def __init__(
        self,
        folder: str,
        transforms: Dict[str, Callable[[Path], Tuple[Vertices, Faces]]],
        data_kwargs: List[str] | None = None,
        augment_fn: Callable = identity
    ):
        folder = Path(folder)
        assert folder.exists and folder.is_dir()
        self.folder = folder

        exts = transforms.keys()
        self.paths = [p for ext in exts for p in folder.glob(f'**/*.{ext}')]

        print(f'{len(self.paths)} training samples found at {folder}')
        assert len(self.paths) > 0

        self.transforms = transforms
        self.data_kwargs = data_kwargs
        self.augment_fn = augment_fn

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        ext = path.suffix[1:]
        fn = self.transforms[ext]

        out = fn(path)
        return self.augment_fn(out)

def scale_and_jitter(vertices, interval=(0.9, 1.1), jitter=0.1):
    assert isinstance(interval, tuple)
    
    scale = torch.rand(1,3) * (interval[1] - interval[0]) + interval[0]
    vertices = vertices * scale
    
    jitter = (torch.rand(1,3) * 2 - 1) * jitter
    vertices = vertices + jitter

    vertices = torch.clamp(vertices, min=-0.99, max=0.99)
    
    return vertices

# @cache_face_edges_for_dataset(max_edges_len=2048)
class Dataset(Dataset):
    @beartype
    def __init__(
        self,
        # transforms: Dict[str, Callable[[Path], Tuple[Vertices, Faces]]],
        dataset_folder = 'data',
        dataset_file_path = '',
        is_train = True,
        transform = scale_and_jitter,
        condition_on_text = False,
        add_room_codes = False,
        with_room_codes = False,
        replica=10,
        is_single=True,
        # data_kwargs = ["vertices", "faces", "face_edges"],

    ):

        self.dataset_folder = dataset_folder
        self.data_path = dataset_file_path
        self.is_train = is_train
        self.transform = transform
        self.condition_on_text = condition_on_text
        self.add_room_codes = add_room_codes
        self.with_room_codes = with_room_codes
        
        self.replica = replica

        self.is_single = is_single
        if self.is_single:
            self.data = self._load_data()
        else:
            # self.data = get_file_list(self.data_path)
            self.data = find_files_with_extension(self.data_path, 'npz')
            
        if self.is_train:
            self.data = self.data[:int(len(self.data)*0.9)]
        else:
            self.data = self.data[int(len(self.data)*0.9):]
        
        print(f'load {len(self.data)} data')
        
        # self.face_edges_all = {}
        
    def _load_data(self):
        data = []
            
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
        
        # if self.is_train == True:
        # data = data[:960]
        
        # print(f'load {len(data)} data')
        
        return data
    
    def __len__(self):
        if self.is_train != True:
            return len(self.data)
        else:
            return len(self.data) * self.replica
        
    def __getitem__(self, idx):
        idx = idx % len(self.data)
        if self.is_single:
            sample = self.data[idx]
        else:
            # tensor_path = self.data[idx]
            sample_file_path = self.data[idx]
            sample = np.load(sample_file_path)
        
            # loaded = np.load(tensor_path, allow_pickle=True)['arr_0'].item()
            # vert, face = loaded["vert"], 
            # sample = {'vertices': loaded["vert"], 'faces':loaded["face"]}
        
        vertices = sample['vertices']
        
        faces = sample['faces'].astype(np.int32)
        
        vertices = torch.from_numpy(vertices).float()
        faces = torch.from_numpy(faces).long()

        # if str(idx) not in self.face_edges_all:
        #     self.face_edges_all[str(idx)] = face_edges
        # else:
        
        # face_edges = self.face_edges_all[idx]

        # face_edges = derive_face_edges_from_faces(faces)


        # transform
        
        if self.transform:
            vertices = self.transform(vertices) 

        data = defaultdict(list)

        data['vertices'] = vertices
        data['faces'] = faces
        # data['face_edges'] = face_edges

        return data
    
    def shuffle(self):
        # num_sample = len(self.data)
        # # 生成打乱的索引
        # shuffled_indices = np.random.permutation(num_sample).tolist()
        # # 使用这些索引来重新排序数组
        # self.data = self.data[shuffled_indices]
        np.random.shuffle(self.data)

# tensor helper functions
@torch.no_grad()
def derive_face_edges_from_faces(
    faces: TensorType['b', 'nf', 3, int],
    pad_id = -1,
    neighbor_if_share_one_vertex = False,
    include_self = True
) -> TensorType['b', 'e', 2, int]:

    is_one_face, device = faces.ndim == 2, faces.device

    if is_one_face:
        faces = rearrange(faces, 'nf c -> 1 nf c')

    max_num_faces = faces.shape[1]
    face_edges_vertices_threshold = 1 if neighbor_if_share_one_vertex else 2

    all_edges = torch.stack(torch.meshgrid(
        torch.arange(max_num_faces, device = device),
        torch.arange(max_num_faces, device = device),
    indexing = 'ij'), dim = -1)

    face_masks = reduce(faces != pad_id, 'b nf c -> b nf', 'all')
    face_edges_masks = rearrange(face_masks, 'b i -> b i 1') & rearrange(face_masks, 'b j -> b 1 j')

    face_edges = []

    for face, face_edge_mask in zip(faces, face_edges_masks):

        shared_vertices = rearrange(face, 'i c -> i 1 c 1') == rearrange(face, 'j c -> 1 j 1 c')
        num_shared_vertices = shared_vertices.any(dim = -1).sum(dim = -1)

        is_neighbor_face = (num_shared_vertices >= face_edges_vertices_threshold) & face_edge_mask

        if not include_self:
            is_neighbor_face &= num_shared_vertices != 3

        face_edge = all_edges[is_neighbor_face]
        face_edges.append(face_edge)

    face_edges = pad_sequence(face_edges, padding_value = pad_id, batch_first = True)

    if is_one_face:
        face_edges = rearrange(face_edges, '1 e ij -> e ij')

    # print(face_edges.shape)

    return face_edges

# custom collater

def first(it):
    return it[0]

def custom_collate(data, pad_id = -1):
    is_dict = isinstance(first(data), dict)

    if is_dict:
        keys = first(data).keys()
        data = [d.values() for d in data]

    output = []

    for datum in zip(*data):
        if is_tensor(first(datum)):
            datum = pad_sequence(datum, batch_first = True, padding_value = pad_id)
        else:
            datum = list(datum)

        output.append(datum)

    output = tuple(output)

    if is_dict:
        output = dict(zip(keys, output))

    return output
