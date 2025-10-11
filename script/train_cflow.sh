export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=2,3,5,7
python cflow/train_cflow.py --data_dir /home/sgwang/PlanScope/plantf_dataset --model_name plantf
