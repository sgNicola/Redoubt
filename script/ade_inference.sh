export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=2,3,5,7
python ade/inference_ade.py --data_dir pluto_dataset --model_name pluto