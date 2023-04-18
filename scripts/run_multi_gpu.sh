CUDA_VISIBLE_DEVICES=0 taskset --cpu-list 00-09 python main.py --batch-size=4 --save-attention --end 3000 &
CUDA_VISIBLE_DEVICES=1 taskset --cpu-list 10-19 python main.py --batch-size=4 --save-attention --start 3000 --end 6000 &
CUDA_VISIBLE_DEVICES=2 taskset --cpu-list 20-29 python main.py --batch-size=4 --save-attention --start 6000 --end 9000 &
CUDA_VISIBLE_DEVICES=3 taskset --cpu-list 30-39 python main.py --batch-size=4 --save-attention --start 9000