import os
import torch
import torch.nn.functional as F
import numpy as np

import utils.general as utils
from utils import rend_util
from glob import glob
import cv2
import random
import json
import pyquaternion as pyquat

def get_key(my_dict, val):
    for key, value in my_dict.items():
        if np.array_equal(value, val):
            return key
    return None

# Dataset with monocular depth, normal and segmentation mask
class SceneDatasetDN_segs(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir,
                 img_res,
                 scan_id=0,
                 use_mask=False,
                 num_views=-1,
                 if_load_ins_sem=False,
                 if_sem=False,
                 if_confidence=False
                 ):
        self.instance_dir = os.path.join('../data', data_dir, '{0}'.format(scan_id))
        print(self.instance_dir)

        self.if_load_ins_sem = if_load_ins_sem

        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res
        self.num_views = num_views
        assert num_views in [-1, 3, 6, 9]

        self.key_frame_idxs = {}
        self.empty_frame_idxs = []
        self.non_empty_frame_idxs = []
        self.if_sem = if_sem
        self.if_confidence = if_confidence
        
        assert os.path.exists(self.instance_dir), "Data directory is empty"

        self.sampling_idx = None
        
        def glob_data(data_dir):
            data_paths = []
            data_paths.extend(glob(data_dir))
            data_paths = sorted(data_paths)
            return data_paths

        image_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "color", "*.png"))
        depth_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "depth", "*_depth.npy"))
        normal_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "normal", "*_normal.npy"))

        # semantic_paths
        if if_load_ins_sem:
            instance_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "m2f_instance", "*.png"))
            semantic_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "m2f_semantics", "*.png"))
        elif if_sem:
            semantic_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "m2f_semantics", "*.png"))
        else:
            semantic_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "m2f_instance", "*.png"))
    
        if if_confidence:
            confidence_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "m2f_confidence", "*.npz"))

        self.n_images = len(image_paths)
        print('[INFO]: Dataset Size ', self.n_images)

        self.sampling_mode = 'all'
        
        # load camera intrinsics and extrinsics
        self.cam_file = '{0}/cameras.npz'.format(self.instance_dir)
        camera_dict = np.load(self.cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics_all = []
        self.pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)     

            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())


        self.rgb_images = []
        for path in image_paths:
            rgb = rend_util.load_rgb(path)[:3, :, :]
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(torch.from_numpy(rgb).float())
            
        self.depth_images = []
        self.normal_images = []

        for dpath, npath in zip(depth_paths, normal_paths):
            depth = np.load(dpath)
            self.depth_images.append(torch.from_numpy(depth.reshape(-1, 1)).float())
        
            normal = np.load(npath)
            normal = normal.reshape(3, -1).transpose(1, 0)
            # important as the output of omnidata is normalized
            normal = normal * 2. - 1.
            self.normal_images.append(torch.from_numpy(normal).float())

        
        # load semantic
        self.semantic_images = []   
        if if_load_ins_sem:
            self.instance_images = []
        if if_load_ins_sem:
            idx = 0
            for ipath, spath in zip(instance_paths, semantic_paths):
                instance_ori = cv2.imread(ipath, cv2.IMREAD_UNCHANGED).astype(np.int32)
                instance = np.zeros(instance_ori.shape, dtype=np.int32)
                unique_colors = np.unique(instance_ori)
                
                self.key_frame_idxs[idx] = len(unique_colors)
                if len(unique_colors) == 1:
                    self.empty_frame_idxs.append(idx)
                else:
                    self.non_empty_frame_idxs.append(idx)
                    
                used_index = []
                for color in unique_colors:
                    if color == 0:
                        instance[instance_ori == color] = 0
                    else:
                        random_index = random.randint(1, 49)
                        while random_index in used_index:
                            random_index = random.randint(1, 49)
                        used_index.append(random_index)
                        instance[instance_ori == color] = random_index
                self.instance_images.append(torch.from_numpy(instance.reshape(-1, 1)).float())

                semantic = cv2.imread(spath, cv2.IMREAD_UNCHANGED).astype(np.int32)
                semantic[instance_ori == 0] = 0
                self.semantic_images.append(torch.from_numpy(semantic.reshape(-1, 1)).float())

                idx += 1
            self.key_frame_idxs = sorted(self.key_frame_idxs, key=self.key_frame_idxs.get, reverse=True)
        elif not if_sem:
            idx = 0
            for spath in semantic_paths:
                semantic_ori = cv2.imread(spath, cv2.IMREAD_UNCHANGED).astype(np.int32)
                semantic = np.zeros(semantic_ori.shape, dtype=np.int32)
                unique_colors = np.unique(semantic_ori)
                
                self.key_frame_idxs[idx] = len(unique_colors)

                if len(unique_colors) == 1:
                    self.empty_frame_idxs.append(idx)

                used_index = []
                for color in unique_colors:
                    if color == 0:
                        semantic[semantic_ori == color] = 0
                    else:
                        random_index = random.randint(1, 49)
                        while random_index in used_index:
                            random_index = random.randint(1, 49)
                        used_index.append(random_index)
                        semantic[semantic_ori == color] = random_index
                self.semantic_images.append(torch.from_numpy(semantic.reshape(-1, 1)).float())
                idx += 1
            self.key_frame_idxs = sorted(self.key_frame_idxs, key=self.key_frame_idxs.get, reverse=True)
        else:
            for spath in semantic_paths:
                semantic = cv2.imread(spath, cv2.IMREAD_UNCHANGED).astype(np.int32)
                semantic[semantic > 21] = 21
                self.semantic_images.append(torch.from_numpy(semantic.reshape(-1, 1)).float())
            

        if if_confidence:
            self.confidence_images = []
            for cpath in confidence_paths:
                confidence = np.load(cpath)
                confidence = confidence["confidence"]
                confidence[confidence == 0] = 0.1
                self.confidence_images.append(torch.from_numpy(confidence.reshape(-1, 1)).float())
            

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        if not self.if_sem:
            if idx in self.empty_frame_idxs and self.sampling_idx is not None:
                # choose in key frame
                idx = random.choice(self.key_frame_idxs[:50])
            elif self.sampling_idx is not None:
                # 0.2 probability to choose in key frame
                if random.random() < 0.2:
                    idx = random.choice(self.key_frame_idxs[:50])

        
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
            "pose": self.pose_all[idx]
        }
        
        
        ground_truth = {
                "rgb": self.rgb_images[idx],
                "depth": self.depth_images[idx],
                "normal": self.normal_images[idx]
            }
        
        if self.if_load_ins_sem:
            ground_truth["segs"] = self.instance_images[idx]
            ground_truth["segs_sem"] = self.semantic_images[idx]
        elif self.if_confidence:
            ground_truth["segs"] = self.semantic_images[idx]
            ground_truth["confidence"] = self.confidence_images[idx]
        else:
            ground_truth["segs"] = self.semantic_images[idx]

       
        # sample all unqiue classes module
        if self.sampling_idx is not None:
            num_samples = 512
            semantic = ground_truth["segs"]
            unique_semantic, count_semantic = torch.unique(semantic, return_counts=True)

            proportions = count_semantic.float() / self.total_pixels
            samples_per_class = (proportions * num_samples).ceil().int()
            
            add_sampling_idx = []
            for class_id, num_class_samples in zip(unique_semantic, samples_per_class):
                class_indices = torch.where(semantic == class_id)[0]
                sampled_indices = torch.randperm(len(class_indices))[:num_class_samples]
                sampled_class_indices = class_indices[sampled_indices]
                add_sampling_idx.append(sampled_class_indices)

            # Concatenate indices from all classes
            add_sampling_idx = torch.cat(add_sampling_idx, dim=0)
            add_sampling_idx = torch.unique(add_sampling_idx)

            current_sample_count = add_sampling_idx.shape[0]
            if current_sample_count < num_samples:
                remaining_samples = num_samples - current_sample_count

                sampled_flags = torch.zeros(semantic.numel(), dtype=torch.bool)
                sampled_flags[add_sampling_idx] = True
                unsampled_indices = torch.arange(semantic.numel())[~sampled_flags]

                additional_samples = unsampled_indices[torch.randperm(len(unsampled_indices))[:remaining_samples]]
                add_sampling_idx = torch.cat([add_sampling_idx, additional_samples], dim=0)

            self.sampling_idx = torch.cat([self.sampling_idx, add_sampling_idx], dim=0)

        if self.sampling_idx is not None:
            if (self.random_image_for_path is None) or (idx not in self.random_image_for_path):
                ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
                ground_truth["full_rgb"] = self.rgb_images[idx]
                ground_truth["normal"] = self.normal_images[idx][self.sampling_idx, :]
                ground_truth["depth"] = self.depth_images[idx][self.sampling_idx, :]
                ground_truth["full_depth"] = self.depth_images[idx]
                if self.if_load_ins_sem:
                    ground_truth["segs"] = self.instance_images[idx][self.sampling_idx, :]
                    ground_truth["full_segs"] = self.instance_images[idx]
                    ground_truth["segs_sem"] = self.semantic_images[idx][self.sampling_idx, :] 
                else:
                    ground_truth["segs"] = self.semantic_images[idx][self.sampling_idx, :]
                    ground_truth["full_segs"] = self.semantic_images[idx]
                if self.if_confidence:
                    ground_truth["confidence"] = self.confidence_images[idx][self.sampling_idx, :]
            
                sample["uv"] = uv[self.sampling_idx, :]
                sample["is_patch"] = torch.tensor([False])
                self.sampling_idx = self.sampling_idx_epoch
            else:
                # sampling a patch from the image, this could be used for training with depth total variational loss
                # a fix patch sampling, which require the sampling_size should be a H*H continuous path
                patch_size = np.floor(np.sqrt(len(self.sampling_idx))).astype(np.int32)
                start = np.random.randint(self.img_res[1]-patch_size +1)*self.img_res[0] + np.random.randint(self.img_res[1]-patch_size +1) # the start coordinate
                idx_row = torch.arange(start, start + patch_size)
                patch_sampling_idx = torch.cat([idx_row + self.img_res[1]*m for m in range(patch_size)])
                ground_truth["rgb"] = self.rgb_images[idx][patch_sampling_idx, :]
                ground_truth["full_rgb"] = self.rgb_images[idx]
                ground_truth["normal"] = self.normal_images[idx][patch_sampling_idx, :]
                ground_truth["depth"] = self.depth_images[idx][patch_sampling_idx, :]
                ground_truth["full_depth"] = self.depth_images[idx]
                ground_truth["segs"] = self.semantic_images[idx][patch_sampling_idx, :]
            
                sample["uv"] = uv[patch_sampling_idx, :]
                sample["is_patch"] = torch.tensor([True])

        # sample rays from multiple camera views
        if self.if_load_ins_sem and self.sampling_idx is not None:
            num_samples = 64
            num_cameras = 4 # sampling 64 rays from 4 cameras

            random_cameras_indices = torch.randperm(len(self.non_empty_frame_idxs))[:num_cameras].tolist()
            random_cameras_indices = torch.tensor([self.non_empty_frame_idxs[i] for i in random_cameras_indices])
            
            sample['extra_pose'] = torch.stack([self.pose_all[i] for i in random_cameras_indices])
            sample['extra_intrinsics'] = torch.stack([self.intrinsics_all[i] for i in random_cameras_indices])

            sample['extra_uv'] = []
            segs_sem_extra = []
            for camera_idx in random_cameras_indices:
                # sample 16 rays from each camera
                instance = self.instance_images[camera_idx]
                non_bg_area = torch.where(instance > 0)[0]
                if non_bg_area.shape[0] < num_samples:
                    bg_area = torch.where(instance == 0)[0]
                    bg_indices = torch.randperm(len(bg_area))[:num_samples - non_bg_area.shape[0]]
                    non_bg_indices = torch.randperm(len(non_bg_area))[:non_bg_area.shape[0]]
                    random_indices = torch.cat([non_bg_area[non_bg_indices], bg_area[bg_indices]])
                else:
                    random_indices = non_bg_area[torch.randperm(len(non_bg_area))[:num_samples]]
                sample['extra_uv'].append(uv[random_indices, :])
                segs_sem_extra.append(self.semantic_images[camera_idx][random_indices, :])
            
            sample['extra_uv'] = torch.stack(sample['extra_uv'], dim=0)
            ground_truth['extra_segs_sem'] = torch.stack(segs_sem_extra, dim=0)

        return idx, sample, ground_truth


    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    # print(k)
                    ret[k] = torch.stack([obj[k] for obj in entry])
                    # print(ret[k].shape)
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size, sampling_pattern='random', patch_size=6):
        
        if sampling_size == -1:
            self.sampling_idx = None
            self.random_image_for_path = None
        
        else:
            # sampling_pattern = 'square'
            if sampling_pattern == 'random':
                self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]
                self.random_image_for_path = None
                self.sampling_idx_epoch = self.sampling_idx.clone()
            elif sampling_pattern == 'patch':
                self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]
                self.random_image_for_path = torch.randperm(self.n_images, )[:int(self.n_images/10)]
            else:
                raise NotImplementedError('the sampling pattern is not implemented.')

    def get_scale_mat(self):
        return np.load(self.cam_file)['scale_mat_0']

    def read_cameras(self, meta, H, W):
        poses = []
        intrinsics_all = []
        frames = meta["frames"]
        for frame in frames:
            pose = np.array(frame['transform_matrix'])
            fl_x = frame['fl_x']
            fl_y = frame['fl_y']
            cx = frame['cx']
            cy = frame['cy']

            intrinsics = np.array([
                [fl_x, 0, cx, 0],
                [0, fl_y, cy, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

            intrinsics_all.append(torch.from_numpy(intrinsics).float())
            poses.append(torch.from_numpy(pose).float())

        return intrinsics_all, poses