import sys

sys.path.append('../code_official')
import argparse
import torch

import os
from training.clusteringsdf_train import ClusteringSDFTrainRunner
import datetime

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--nepoch', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--sem_epoch', type=int, default=0, help='number of epochs to train for')
    parser.add_argument('--conf', type=str, default='./confs/dtu.conf')
    parser.add_argument('--expname', type=str, default='')
    parser.add_argument("--exps_folder", type=str, default="exps")
    parser.add_argument('--is_continue', default=False, action="store_true",
                        help='If set, indicates continuing from a previous run.')
    parser.add_argument('--timestamp', default='latest', type=str,
                        help='The timestamp of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--checkpoint', default='latest', type=str,
                        help='The checkpoint epoch of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--scan_id', type=str, default="office_3", help='If set, taken to be the scan id.')
    parser.add_argument('--cancel_vis', default=False, action="store_true",
                        help='If set, cancel visualization in intermediate epochs.')
    parser.add_argument("--ft_folder", type=str, default=None, help='If set, finetune model from the given folder path')
    parser.add_argument("--is_ref", default=False, action="store_true", help='If set, only reference')
    parser.add_argument("--if_sem", default=True, action="store_true", help='semantic mode or instance mode')
    parser.add_argument("--dataset", type=str, default="replica", help='Which dataset to use')
    parser.add_argument("--ref_outdir", type=str, default=None, help='reference output dir')

    opt = parser.parse_args()

    '''
    # if using GPUtil
    if opt.gpu == "auto":
        deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False,
                                        excludeID=[], excludeUUID=[])
        gpu = deviceIDs[0]
    else:
        gpu = opt.gpu
    '''
    # gpu = opt.local_rank

    # set distributed training
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
        local_rank = -1

    # print(opt.local_rank)
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank, timeout=datetime.timedelta(1, 1800))
    torch.distributed.barrier()


    trainrunner = ClusteringSDFTrainRunner(conf=opt.conf,
                                    batch_size=opt.batch_size,
                                    nepochs=opt.nepoch,
                                    sem_epochs=opt.sem_epoch,
                                    expname=opt.expname,
                                    gpu_index=local_rank,
                                    exps_folder_name=opt.exps_folder,
                                    is_continue=opt.is_continue,
                                    timestamp=opt.timestamp,
                                    checkpoint=opt.checkpoint,
                                    scan_id=opt.scan_id,
                                    do_vis=not opt.cancel_vis,
                                    ft_folder = opt.ft_folder,
                                    is_ref = opt.is_ref,
                                    if_sem = opt.if_sem,
                                    ref_outdir = opt.ref_outdir,
                                    )

    if opt.is_ref:
        trainrunner.reference(opt.ref_outdir)
    else:
        trainrunner.run()
