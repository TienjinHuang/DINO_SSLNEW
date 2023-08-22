
#!/user/bin

python -m torch.distributed.launch --nproc_per_node=4 main_dino.py --arch mobilenet --kernel_size 7 --data_path /scratch-shared/boqian/imagenet  --output_dir ./

python -m torch.distributed.launch --nproc_per_node=4 main_dino.py --arch mobilenet --kernel_size 9 --data_path /scratch-shared/boqian/imagenet  --output_dir ./

python -m torch.distributed.launch --nproc_per_node=4 main_dino.py --arch mobilenet --kernel_size 11 --data_path /scratch-shared/boqian/imagenet  --output_dir ./
