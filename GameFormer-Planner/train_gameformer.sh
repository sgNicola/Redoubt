export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
python train_predictor.py \
--train_set $NUPLAN_EXP_ROOT/gameInD/new_train \
--valid_set $NUPLAN_EXP_ROOT/gameInD/new_val > train_gameformer.log 2>&1