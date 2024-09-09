torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 training/exp_runner.py \
--conf confs/instance.conf \
--scan_id office_3