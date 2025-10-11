export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=2,3
python cflow/train_ablition.py --data_dir /home/sgwang/PlanScope/plantf_dataset --model_name plantf