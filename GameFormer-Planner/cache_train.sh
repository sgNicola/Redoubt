cwd=$(pwd)

python cache_process.py \
--data_path $NUPLAN_DATA_ROOT/nuplan-v1.1/trainval \
--map_path $NUPLAN_DATA_ROOT/maps \
--config_path $cwd/config/scenario_filter/train_InD.yaml \
--save_path $NUPLAN_EXP_ROOT/gameInD/new_train > cache.log 2>&1
