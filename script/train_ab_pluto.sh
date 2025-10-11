export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=0,1
python cflow/train_ablition.py --data_dir /home/sgwang/PlanScope/pluto_dataset --model_name pluto