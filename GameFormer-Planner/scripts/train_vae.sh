export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=2
python vae.py --data_dir "$(pwd)/gameformer_dataset" --model_name gameformer_cnf