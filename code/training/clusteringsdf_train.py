import imp
import os
from datetime import datetime
from pyhocon import ConfigFactory
import sys
import torch
from tqdm import tqdm
import numpy as np

import utils.general as utils
import utils.plots as plt
from utils import rend_util
from utils.general import get_time
from torch.utils.tensorboard import SummaryWriter
from model.loss import compute_scale_and_shift
from utils.general import BackprojectDepth
import matplotlib.pyplot as matplt
import torch.nn.functional as F
import json
from datetime import datetime


class ClusteringSDFTrainRunner():
    def __init__(self,**kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.batch_size = kwargs['batch_size']
        self.nepochs = kwargs['nepochs']
        self.sem_epochs = kwargs['sem_epochs']
        self.exps_folder_name = kwargs['exps_folder_name']
        self.GPU_INDEX = kwargs['gpu_index']

        self.is_ref = kwargs['is_ref']
        self.if_sem = kwargs['if_sem']
        self.ref_outdir = kwargs['ref_outdir']

        self.expname = self.conf.get_string('train.expname') + kwargs['expname']
        scan_id = kwargs['scan_id'] if kwargs['scan_id'] != -1 else self.conf.get_int('dataset.scan_id', default=-1)
        if scan_id != -1:
            self.expname = self.expname + '_{0}'.format(scan_id)

        self.finetune_folder = kwargs['ft_folder'] if kwargs['ft_folder'] is not None else None
        if kwargs['is_continue'] and kwargs['timestamp'] == 'latest':
            if os.path.exists(os.path.join('../',kwargs['exps_folder_name'],self.expname)):
                timestamps = os.listdir(os.path.join('../',kwargs['exps_folder_name'],self.expname))
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    timestamp = sorted(timestamps)[-1]
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
        else:
            timestamp = kwargs['timestamp']
            is_continue = kwargs['is_continue']
            print(timestamp)
            print("-------------------------------------------------------------")

        if self.GPU_INDEX == 0 and not self.is_ref:
            utils.mkdir_ifnotexists(os.path.join('../',self.exps_folder_name))
            self.expdir = os.path.join('../', self.exps_folder_name, self.expname)
            utils.mkdir_ifnotexists(self.expdir)
            self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
            utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))

            self.plots_dir = os.path.join(self.expdir, self.timestamp, 'plots')
            utils.mkdir_ifnotexists(self.plots_dir)

            # create checkpoints dirs
            self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
            utils.mkdir_ifnotexists(self.checkpoints_path)
            self.model_params_subdir = "ModelParameters"
            self.optimizer_params_subdir = "OptimizerParameters"
            self.scheduler_params_subdir = "SchedulerParameters"

            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.scheduler_params_subdir))

            os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.expdir, self.timestamp, 'runconf.conf')))
        else:
            self.model_params_subdir = "ModelParameters"
            self.optimizer_params_subdir = "OptimizerParameters"
            self.scheduler_params_subdir = "SchedulerParameters"


        print('[INFO]: shell command : {0}'.format(' '.join(sys.argv)))

        print('[INFO]: Loading data ...')

        dataset_conf = self.conf.get_config('dataset')
        if kwargs['scan_id'] != -1:
            dataset_conf['scan_id'] = kwargs['scan_id']

        self.all_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(**dataset_conf)
        
        scan_id = kwargs['scan_id']
        data_dir = self.conf.get_string('dataset.data_dir')
        split = json.load(open('../data/' + data_dir + '/' + str(scan_id) + '/splits_new.json'))
        self.all_dataset.i_split = [split['train'], split['test']]

            
        self.train_dataset = torch.utils.data.Subset(self.all_dataset, self.all_dataset.i_split[0])
        self.test_dataset = torch.utils.data.Subset(self.all_dataset, self.all_dataset.i_split[1])
        
        self.ds_len = len(self.train_dataset)
        print('[INFO]: Finish loading data. Data-set size: {0}'.format(self.ds_len))

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            collate_fn=self.all_dataset.collate_fn,
                                                            num_workers=8,
                                                            pin_memory=True)
        self.plot_dataloader = torch.utils.data.DataLoader(self.test_dataset, 
                                                           batch_size=self.conf.get_int('plot.plot_nimgs'),
                                                           shuffle=True,
                                                           collate_fn=self.all_dataset.collate_fn
                                                           )

        conf_model = self.conf.get_config('model')
        self.model = utils.get_class(self.conf.get_string('train.model_class'))(conf=conf_model)

        self.Grid_MLP = self.model.Grid_MLP
        if torch.cuda.is_available():
            self.model.cuda()

        self.loss = utils.get_class(self.conf.get_string('train.loss_class'))(**self.conf.get_config('loss'))

        # The MLP and hash grid should have different learning rates
        self.lr = self.conf.get_float('train.learning_rate')
        self.lr_factor_for_grid = self.conf.get_float('train.lr_factor_for_grid', default=1.0)
        
        if self.Grid_MLP:
            self.optimizer = torch.optim.Adam([
                {'name': 'encoding', 'params': list(self.model.implicit_network.grid_parameters()), 
                    'lr': self.lr * self.lr_factor_for_grid},
                {'name': 'net', 'params': list(self.model.implicit_network.mlp_parameters()) +\
                    list(self.model.rendering_network.parameters()),
                    'lr': self.lr},
                {'name': 'density', 'params': list(self.model.density.parameters()),
                    'lr': self.lr},
            ], betas=(0.9, 0.99), eps=1e-15)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Exponential learning rate scheduler
        decay_rate = self.conf.get_float('train.sched_decay_rate', default=0.1)
        decay_steps = self.nepochs * len(self.train_dataset)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, decay_rate ** (1./decay_steps))

        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.GPU_INDEX], broadcast_buffers=False, find_unused_parameters=True)
        
        self.do_vis = kwargs['do_vis']

        self.start_epoch = 0
        # Loading a pretrained model for finetuning, the model path can be provided by self.finetune_folder
        if is_continue or self.finetune_folder is not None:
            old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints') if self.finetune_folder is None\
             else os.path.join(self.finetune_folder, 'checkpoints')

            print('[INFO]: Loading pretrained model from {}'.format(old_checkpnts_dir))
            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
            self.model.load_state_dict(saved_model_state["model_state_dict"])
            self.start_epoch = saved_model_state['epoch']

            data = torch.load(
                os.path.join(old_checkpnts_dir, 'OptimizerParameters', str(kwargs['checkpoint']) + ".pth"))
            self.optimizer.load_state_dict(data["optimizer_state_dict"])

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.scheduler.load_state_dict(data["scheduler_state_dict"])

        self.num_pixels = self.conf.get_int('train.num_pixels')
        self.total_pixels = self.all_dataset.total_pixels
        self.img_res = self.all_dataset.img_res
        self.n_batches = len(self.train_dataloader)
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.checkpoint_freq = self.conf.get_int('train.checkpoint_freq', default=100)
        self.split_n_pixels = self.conf.get_int('train.split_n_pixels', default=10000)
        self.plot_conf = self.conf.get_config('plot')
        self.backproject = BackprojectDepth(1, self.img_res[0], self.img_res[1]).cuda()
        
        self.add_objectvio_iter = self.conf.get_int('train.add_objectvio_iter', default=0)
        self.add_entropy_iter = self.conf.get_int('train.add_entropy_iter', default=0)


        self.clustering_weight = self.conf.get_float('cluster.clustering_weight')
        self.if_bg_loss = self.conf.get_int('cluster.if_bg_loss')
        self.if_onehot = self.conf.get_int('cluster.if_onehot')
        self.if_bg_weight = self.conf.get_float('cluster.if_bg_weight')
        self.reg_weight = self.conf.get_float('cluster.reg_weight')
        self.diff_cluster_weight = self.conf.get_float('cluster.diff_cluster_weight')
        self.onehot_weight = self.conf.get_float('cluster.onehot_weight')
        self.non_bg_weight = self.conf.get_float('cluster.non_bg_weight')
        self.if_sem = self.conf.get_int('cluster.if_sem')
        self.sem_weight = self.conf.get_float('cluster.sem_weight')
        self.if_multi_camera_view = self.conf.get_int('cluster.if_multi_camera_view')
        self.cross_view_weight = self.conf.get_float('cluster.cross_view_weight')

        

        self.loss_list = []
        self.color_loss_list = []
        self.eikonal_loss_list = []
        self.smooth_loss_list = []
        self.depth_loss_list = []
        self.normal_l1_list = []
        self.normal_cos_list = []
        self.sem_loss_list = []
        self.onehot_loss_list = []
        self.psnr_list = []
        self.reg_loss_list = []
        self.diff_val_loss_list = []
        self.collision_reg_loss_list = []
        self.bg_loss_list = []
        self.non_bg_loss_list = []
        self.bg_loss_total_list = []
        self.cross_view_list = []

    def save_checkpoints(self, epoch):
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, "latest.pth"))

    def plot_loss(self, loss, loss_output, epoch, psnr, if_sam=False):
        self.loss_list.append(loss)
        self.color_loss_list.append(loss_output['rgb_loss'].item())
        self.eikonal_loss_list.append(loss_output['eikonal_loss'].item())
        self.smooth_loss_list.append(loss_output['smooth_loss'].item())
        self.depth_loss_list.append(loss_output['depth_loss'].item())
        self.normal_l1_list.append(loss_output['normal_l1'].item())
        self.normal_cos_list.append(loss_output['normal_cos'].item())
        self.sem_loss_list.append(loss_output['sem_loss'].item())
        self.onehot_loss_list.append(loss_output['onehot_loss'].item())
        self.psnr_list.append(psnr)
        self.reg_loss_list.append(loss_output['reg_loss'].item()*self.clustering_weight)
        self.diff_val_loss_list.append(loss_output['diff_val_loss'].item()*self.clustering_weight)
        self.collision_reg_loss_list.append(loss_output['collision_reg_loss'].item())
        self.bg_loss_list.append(loss_output['bg_loss'].item())
        self.non_bg_loss_list.append(loss_output['non_bg_loss'].item())
        self.bg_loss_total_list.append(loss_output['bg_loss'].item() + loss_output['non_bg_loss'].item())
        self.cross_view_list.append(loss_output['cross_view_loss'].item())
        
        # plot loss
        fig, axs = matplt.subplots(4, 4, figsize=(10, 12))

        axs[0, 0].plot(self.loss_list, label='total loss', color='r')
        axs[0, 0].set_title('total loss')
        axs[0, 0].set_ylabel('Loss Value')

        axs[0, 1].plot(self.color_loss_list, label='color loss', color='r')
        axs[0, 1].set_title('color loss')
        axs[0, 1].set_ylabel('Loss Value')

        axs[0, 2].plot(self.psnr_list, label='psnr', color='r')
        axs[0, 2].set_title('psnr')
        axs[0, 2].set_ylabel('PSNR Value')

        axs[0, 3].plot(self.smooth_loss_list, label='smooth loss', color='r')
        axs[0, 3].set_title('smooth loss')
        axs[0, 3].set_ylabel('Loss Value')

        axs[1, 0].plot(self.depth_loss_list, label='depth loss', color='r')
        axs[1, 0].set_title('depth loss')
        axs[1, 0].set_ylabel('Loss Value')

        axs[1, 1].plot(self.normal_l1_list, label='normal l1 loss', color='r')
        axs[1, 1].set_title('normal l1 loss')
        axs[1, 1].set_ylabel('Loss Value')

        axs[1, 2].plot(self.normal_cos_list, label='normal cos loss', color='r')
        axs[1, 2].set_title('normal cos loss')
        axs[1, 2].set_ylabel('Loss Value')

        
        axs[1, 3].plot(self.sem_loss_list, label='sem loss', color='r')
        axs[1, 3].set_title('sem loss')
        axs[1, 3].set_ylabel('Loss Value')

        axs[2, 0].plot(self.onehot_loss_list, label='onehot loss', color='r')
        axs[2, 0].set_title('onehot loss')
        axs[2, 0].set_ylabel('Loss Value')

        axs[2, 1].plot(self.reg_loss_list, label='reg loss', color='r')
        axs[2, 1].set_title('reg loss')
        axs[2, 1].set_ylabel('Loss Value')

        axs[2, 2].plot(self.diff_val_loss_list, label='diff val loss', color='r')
        axs[2, 2].set_title('diff val loss')
        axs[2, 2].set_ylabel('Loss Value')

        axs[2, 3].plot(self.collision_reg_loss_list, label='collision reg loss', color='r')
        axs[2, 3].set_title('collision reg loss')
        axs[2, 3].set_ylabel('Loss Value')

        axs[3, 0].plot(self.bg_loss_list, label='background loss', color='r')
        axs[3, 0].set_title('background loss')
        axs[3, 0].set_ylabel('Loss Value')
        
        axs[3, 1].plot(self.non_bg_loss_list, label='non_bg loss', color='r')
        axs[3, 1].set_title('non_bg loss')
        axs[3, 1].set_ylabel('Loss Value')
        
        axs[3, 2].plot(self.bg_loss_total_list, label='bg_total loss', color='r')
        axs[3, 2].set_title('bg_total loss')
        axs[3, 2].set_ylabel('Loss Value')

        axs[3, 3].plot(self.cross_view_list, label='cross_view loss', color='r')
        axs[3, 3].set_title('cross_view loss')
        axs[3, 3].set_ylabel('Loss Value')



        # save loss plot
        matplt.savefig(os.path.join(self.plots_dir, 'loss.png'))

        

    def run(self):
        print("training...")
        if self.GPU_INDEX == 0 :
            self.writer = SummaryWriter(log_dir=os.path.join(self.plots_dir, 'logs'))



        self.iter_step = 0
        for epoch in range(self.start_epoch, self.nepochs + 1):

            if self.GPU_INDEX == 0 and epoch % self.checkpoint_freq == 0 and epoch != 0:
                self.save_checkpoints(epoch)

            if self.GPU_INDEX == 0 and self.do_vis and epoch % self.plot_freq == 0:
                self.model.eval()

                self.all_dataset.change_sampling_idx(-1)
                self.all_dataset.sampling_mode = 'all'

                indices, model_input, ground_truth = next(iter(self.plot_dataloader))
                model_input["intrinsics"] = model_input["intrinsics"].cuda()
                model_input["uv"] = model_input["uv"].cuda()
                model_input['pose'] = model_input['pose'].cuda()
                
                split = utils.split_input(model_input, self.total_pixels, n_pixels=self.split_n_pixels)
                res = []
                for s in tqdm(split):
                    out = self.model(s, indices)
                    d = {'rgb_values': out['rgb_values'].detach(),
                        'normal_map': out['normal_map'].detach(),
                        'depth_values': out['depth_values'].detach()}
                    if 'semantic_values' in out:
                        if self.if_sem: # ignore the background channel
                            out['object_opacity'][:, 0] = 0.0
                        d['semantic_values'] = torch.argmax(out['object_opacity'].detach(),dim=1)
                        d['opacity_values'] = torch.argmax(out['object_opacity'].detach(),dim=1)
                        d['real_opacity_values'] = out['object_opacity'].detach()
                    res.append(d)

                batch_size = ground_truth['rgb'].shape[0]
                model_outputs = utils.merge_output(res, self.total_pixels, batch_size)
                model_outputs['sum_hotmap'] = self.cal_sem_smooth_loss(model_outputs['real_opacity_values'])
                if self.if_multi_camera_view:
                    plot_data = self.get_plot_data(model_input, model_outputs, model_input['pose'], ground_truth['rgb'], ground_truth['normal'], ground_truth['depth'], ground_truth['segs'], if_hotmap=True, seg_sem_gt=ground_truth['segs_sem'])
                else:
                    plot_data = self.get_plot_data(model_input, model_outputs, model_input['pose'], ground_truth['rgb'], ground_truth['normal'], ground_truth['depth'], ground_truth['segs'], if_hotmap=True)

                plt.plot(self.model.module.implicit_network,
                        indices,
                        plot_data,
                        self.plots_dir,
                        epoch,
                        self.img_res,
                        **self.plot_conf
                        )

                self.model.train()

            self.all_dataset.change_sampling_idx(self.num_pixels)
            self.all_dataset.sampling_mode = 'random'

            for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):
                # print(ground_truth['segs'].shape)
                model_input["intrinsics"] = model_input["intrinsics"].cuda()
                model_input["uv"] = model_input["uv"].cuda()
                model_input['pose'] = model_input['pose'].cuda()
                
                self.optimizer.zero_grad()
                
                model_outputs = self.model(model_input, indices)
                
                if epoch >= self.add_objectvio_iter:
                    loss_output = self.loss(model_outputs, ground_truth, call_reg=True, clustering_weight=self.clustering_weight, \
                            if_bg_loss=self.if_bg_loss, if_bg_weight=self.if_bg_weight, non_bg_weight=self.non_bg_weight, \
                            if_sem=self.if_sem, sem_weight=self.sem_weight, \
                            onehot_weight=self.onehot_weight, if_onehot=self.if_onehot, \
                            reg_weight=self.reg_weight, diff_cluster_weight=self.diff_cluster_weight, \
                            if_multi_camera_view=self.if_multi_camera_view, cross_view_weight=self.cross_view_weight)
                else:
                    loss_output = self.loss(model_outputs, ground_truth, call_reg=False, clustering_weight=self.clustering_weight, \
                            if_bg_loss=self.if_bg_loss, if_bg_weight=self.if_bg_weight, non_bg_weight=self.non_bg_weight,\
                            if_sem=self.if_sem, sem_weight=self.sem_weight, \
                            onehot_weight=self.onehot_weight, if_onehot=self.if_onehot, \
                            reg_weight=self.reg_weight, diff_cluster_weight=self.diff_cluster_weight, \
                            if_multi_camera_view=self.if_multi_camera_view, cross_view_weight=self.cross_view_weight)
                
                loss = loss_output['loss']
                loss.backward()
                self.optimizer.step()
                
                psnr = rend_util.get_psnr(model_outputs['rgb_values'],
                                          ground_truth['rgb'].cuda().reshape(-1,3))
                
                self.iter_step += 1                
                
                if self.GPU_INDEX == 0 and data_index % 20 == 0:
                    print(
                        '{0}_{1} [{2}] ({3}/{4}): loss = {5}, rgb_loss = {6}, eikonal_loss = {7}, psnr = {8}, bete={9}, alpha={10}, collision_reg_loss = {11}, onehot_loss = {12}, reg_loss = {13}, diff_val_loss = {14}, bg_loss = {15}, depth_loss = {16}'
                            .format(self.expname, self.timestamp, epoch, data_index, self.n_batches, loss.item(),
                                    loss_output['rgb_loss'].item(),
                                    loss_output['eikonal_loss'].item(),
                                    psnr.item(),
                                    self.model.module.density.get_beta().item(),
                                    1. / self.model.module.density.get_beta().item(),
                                    loss_output['collision_reg_loss'].item(),
                                    loss_output['onehot_loss'].item(),
                                    loss_output['reg_loss'].item(),
                                    loss_output['diff_val_loss'].item(),
                                    loss_output['bg_loss'].item(),
                                    loss_output['depth_loss'].item()))
                    if not (epoch == 0 and data_index == 0):
                        self.plot_loss(loss.item(), loss_output, epoch, psnr.item())
                
                self.all_dataset.change_sampling_idx(self.num_pixels)
                self.scheduler.step()


        if self.GPU_INDEX == 0:
            self.save_checkpoints(epoch)

    def detect_semantic_edges(self, semantic_img, num_classes=92):
        # Convert semantic image to one-hot encoding
        semantic_img = semantic_img.resize(self.img_res[0], self.img_res[1]).unsqueeze(0)
        one_hot = F.one_hot(semantic_img.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        # Define Sobel filters
        sobel_x = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).unsqueeze(0).unsqueeze(0)
        
        if semantic_img.is_cuda:
            sobel_x = sobel_x.cuda()
            sobel_y = sobel_y.cuda()
        
        edges = torch.zeros_like(semantic_img).float()
        
        for i in range(num_classes):
            channel = one_hot[:, i, :, :]
            edge_x = F.conv2d(channel.unsqueeze(1), sobel_x, padding=1)
            edge_y = F.conv2d(channel.unsqueeze(1), sobel_y, padding=1)
            magnitude = torch.sqrt(edge_x**2 + edge_y**2).squeeze(1)
            edges = torch.max(edges, magnitude)
        
        return edges.reshape(-1, 1).squeeze()

        
    def get_plot_data(self, model_input, model_outputs, pose, rgb_gt, normal_gt, depth_gt, seg_gt, edge_gt=None, if_confidence=False, if_hotmap=False, seg_sem_gt=None):
        batch_size, num_samples, _ = rgb_gt.shape

        rgb_eval = model_outputs['rgb_values'].reshape(batch_size, num_samples, 3)
        normal_map = model_outputs['normal_map'].reshape(batch_size, num_samples, 3)
        normal_map = (normal_map + 1.) / 2.
      
        depth_map = model_outputs['depth_values'].reshape(batch_size, num_samples)
        depth_gt = depth_gt.to(depth_map.device)
        scale, shift = compute_scale_and_shift(depth_map[..., None], depth_gt, depth_gt > 0.)
        depth_map = depth_map * scale + shift
        
        seg_map = model_outputs['semantic_values'].reshape(batch_size, num_samples)
        seg_gt = seg_gt.to(seg_map.device)
        
        if self.if_multi_camera_view and seg_sem_gt is not None:
            seg_sem_gt = seg_sem_gt.to(seg_map.device)

        # save point cloud
        depth = depth_map.reshape(1, 1, self.img_res[0], self.img_res[1])
        pred_points = self.get_point_cloud(depth, model_input, model_outputs)

        gt_depth = depth_gt.reshape(1, 1, self.img_res[0], self.img_res[1])
        gt_points = self.get_point_cloud(gt_depth, model_input, model_outputs)

        if if_hotmap:
            confidence_map = abs(model_outputs['sum_hotmap'].reshape(batch_size, num_samples))
            confidence_gt = confidence_map
            plot_data = {
                'rgb_gt': rgb_gt,
                'normal_gt': (normal_gt + 1.)/ 2.,
                'depth_gt': depth_gt,
                'seg_gt': seg_gt,
                'pose': pose,
                'rgb_eval': rgb_eval,
                'normal_map': normal_map,
                'depth_map': depth_map,
                'seg_map': seg_map,
                "pred_points": pred_points,
                "gt_points": gt_points,
                "confidence_map": confidence_map,
                "confidence_gt": confidence_gt
            }
            if self.if_multi_camera_view:
                plot_data['seg_sem_gt'] = seg_sem_gt
        else:
            plot_data = {
                'rgb_gt': rgb_gt,
                'normal_gt': (normal_gt + 1.)/ 2.,
                'depth_gt': depth_gt,
                'seg_gt': seg_gt,
                'pose': pose,
                'rgb_eval': rgb_eval,
                'normal_map': normal_map,
                'depth_map': depth_map,
                'seg_map': seg_map,
                "pred_points": pred_points,
                "gt_points": gt_points
            }

        return plot_data
    
    def get_point_cloud(self, depth, model_input, model_outputs):
        color = model_outputs["rgb_values"].reshape(-1, 3)
        
        K_inv = torch.inverse(model_input["intrinsics"][0])[None]
        points = self.backproject(depth, K_inv)[0, :3, :].permute(1, 0)
        points = torch.cat([points, color], dim=-1)
        return points.detach().cpu().numpy()

    def cal_sem_smooth_loss(self, pred, if_patch=True):
        # if if_patch:
        pred = pred.view(self.img_res[0], self.img_res[1], pred.shape[1])
        diff_tensor = torch.zeros((self.img_res[0], self.img_res[1])).cuda()

        pred = F.softmax(pred, dim=2)
        # neighbor
        diff_tensor[1:, :] += ((pred[1:, :, :] - pred[:-1, :, :]) ** 2).sum(dim=2)
        diff_tensor[:-1, :] += ((pred[1:, :, :] - pred[:-1, :, :]) ** 2).sum(dim=2)
        diff_tensor[:, 1:] += ((pred[:, 1:, :] - pred[:, :-1, :]) ** 2).sum(dim=2)
        diff_tensor[:, :-1] += ((pred[:, 1:, :] - pred[:, :-1, :]) ** 2).sum(dim=2)

        # diagonal
        diff_tensor[1:, 1:] += ((pred[1:, 1:, :] - pred[:-1, :-1, :]) ** 2).sum(dim=2)
        diff_tensor[:-1, :-1] += ((pred[1:, 1:, :] - pred[:-1, :-1, :]) ** 2).sum(dim=2)
        diff_tensor[1:, :-1] += ((pred[1:, :-1, :] - pred[:-1, 1:, :]) ** 2).sum(dim=2)
        diff_tensor[:-1, 1:] += ((pred[1:, :-1, :] - pred[:-1, 1:, :]) ** 2).sum(dim=2)

        return diff_tensor.reshape(-1, 1).squeeze().cuda()
        



    def reference(self, output_dir):
        self.model.eval()

        self.all_dataset.change_sampling_idx(-1)
        
        self.plot_dataloader = torch.utils.data.DataLoader(self.test_dataset, 
                                                           batch_size=self.conf.get_int('plot.plot_nimgs'),
                                                           shuffle=False,
                                                           collate_fn=self.all_dataset.collate_fn
                                                           )

        for i, data in enumerate(self.plot_dataloader):
            indices, model_input, ground_truth = data
            model_input["intrinsics"] = model_input["intrinsics"].cuda()
            model_input["uv"] = model_input["uv"].cuda()
            model_input['pose'] = model_input['pose'].cuda()
            
            
            
            split = utils.split_input(model_input, self.total_pixels, n_pixels=self.split_n_pixels)
            res = []
            for s in tqdm(split):
                out = self.model(s, indices)
                d = {'rgb_values': out['rgb_values'].detach(),
                        'normal_map': out['normal_map'].detach(),
                        'depth_values': out['depth_values'].detach()}
                if 'rgb_un_values' in out:
                    d['rgb_un_values'] = out['rgb_un_values'].detach()
                if 'semantic_values' in out:
                    d['opacity_values'] = torch.sum(out['object_opacity'].detach(),dim=1)
                    if self.if_sem: # ignore the background channel
                        out['object_opacity'][:, 0] = 0.0
                    semantic_values = torch.argmax(out['object_opacity'].detach(),dim=1)
                    if self.if_sem:
                        semantic_values[semantic_values > 21] = 21
                    d['semantic_values'] = semantic_values
                if 'edge_values' in out and out['edge_values'] is not None:
                    d['edge_values'] = out['edge_values'].detach()
                if 'confidence_values' in out and out['confidence_values'] is not None:
                    d['confidence_values'] = abs(out['confidence_values'].detach())
                res.append(d)

            batch_size = ground_truth['rgb'].shape[0]
            model_outputs = utils.merge_output(res, self.total_pixels, batch_size)
            
            plot_data = self.get_plot_data(model_input, model_outputs, model_input['pose'], ground_truth['rgb'], ground_truth['normal'], ground_truth['depth'], ground_truth['segs'], )
            plt.plot(self.model.module.implicit_network,
                    indices,
                    plot_data,
                    None,
                    0,
                    self.img_res,
                    **self.plot_conf,
                    if_ref=True,
                    output_dir=output_dir
                    )