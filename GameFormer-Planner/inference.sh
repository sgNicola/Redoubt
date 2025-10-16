cwd=$(pwd)
CKPT_ROOT="$cwd/checkpoints"
export GAMEFORMER="$(pwd)/gameformer_dataset"

python inference_predictor.py \
    --train_set $NUPLAN_EXP_ROOT/gameInD/new_val \
    --model_path "$CKPT_ROOT/game0406.pth" \
    --stage val