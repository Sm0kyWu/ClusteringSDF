dir=""

mkdir -p $dir/output_merge_all_pose
mkdir -p $dir/output_merge_all_pose/rendering
mkdir -p $dir/output_merge_all_pose/merge
mkdir -p $dir/output_merge_all_pose/segs
mkdir -p $dir/output_merge_all_pose/sem
mkdir -p $dir/output_merge_all_pose/depth
mkdir -p $dir/output_merge_all_pose/normal

torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 training/exp_runner.py \
--conf $dir/runconf.conf --scan_id 8 \
--is_continue --ft_folder $dir \
--is_ref --ref_outdir $dir/output_merge_all_pose \
--timestamp 100 --checkpoint 100
