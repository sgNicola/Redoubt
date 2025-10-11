export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=4,5
python cflow/train_ablition.py --data_dir /home/sgwang/PlanScope/planscope_dataset --model_name scope