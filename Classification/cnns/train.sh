rm -rf core.*
rm -rf ./output/snapshots/*

if [ -n "$1" ]; then
    NUM_EPOCH=$1
else
    NUM_EPOCH=50
fi
echo NUM_EPOCH=$NUM_EPOCH

# training with imagenet
if [ -n "$2" ]; then
    DATA_ROOT=$2
else
    DATA_ROOT=/data/imagenet/ofrecord
fi
echo DATA_ROOT=$DATA_ROOT

LOG_FOLDER=../logs
mkdir -p $LOG_FOLDER
# LOGFILE=$LOG_FOLDER/resnet_training.log
LOGFILE=$LOG_FOLDER/mbnv1_training.log

python3 of_cnn_train_val.py \
     --train_data_dir=$DATA_ROOT/train \
     --num_examples=50 \
     --train_data_part_num=1 \
     --val_data_dir=$DATA_ROOT/validation \
     --num_val_examples=50 \
     --val_data_part_num=1 \
     --num_nodes=1 \
     --gpu_num_per_node=1 \
     --momentum=0.875 \
     --learning_rate=0.001 \
     --loss_print_every_n_iter=1 \
     --batch_size_per_device=1 \
     --val_batch_size_per_device=1 \
     --num_epoch=$NUM_EPOCH \
     --model="mobilenetv1"
     # 2>&1 | tee ${LOGFILE}

echo "Writting log to ${LOGFILE}"
