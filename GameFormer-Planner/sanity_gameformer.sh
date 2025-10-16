export CUDA_VISIBLE_DEVICES=4,5,6,7
python train_predictor.py \
--train_set $NUPLAN_EXP_ROOT/gameInD/sanity \
--valid_set $NUPLAN_EXP_ROOT/gameInD/val > train_gameformer.log 2>&1