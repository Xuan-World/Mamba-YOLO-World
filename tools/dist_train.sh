#!/usr/bin/env bash
#export NODE_RANK=1

#export NNODES=2
#export MASTER_ADDR=9.206.42.111

#export NCCL_SOCKET_IFNAME=eth1
#export NCCL_IB_GID_INDEX=3
#nccl_ib_hca=$(bash show_gids |  grep $(hostname -I) |  grep v2 | awk '{print $1 ":" $2}' )
#export NCCL_IB_HCA="=$nccl_ib_hca"
#export NCCL_IB_SL=3
#export NCCL_CHECKS_DISABLE=1
#export NCCL_P2P_DISABLE=0
#export NCCL_LL_THRESHOLD=16384
#export NCCL_IB_CUDA_SUPPORT=1

export TORCH_DISTRIBUTED_DEBUG=DETAIL

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${MASTER_PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    --config $CONFIG \
    --launcher pytorch ${@:3}
