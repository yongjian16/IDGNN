#!/bin/bash
# $1 = 56/57/58
lr=$1
seed=$2


(
    CUDA_VISIBLE_DEVICES=3 bash examples/task-train-reddit4/reddit4-none/Reddit4~all~trans~16~softplus~1.0e-5~value~-1.sh "DCRNNx2" $lr $seed &
    CUDA_VISIBLE_DEVICES=4 bash examples/task-train-reddit4/reddit4-none/Reddit4~all~trans~16~softplus~1.0e-5~value~-1.sh "DySATx2" $lr $seed 
) &
(
    CUDA_VISIBLE_DEVICES=5 bash examples/task-train-reddit4/reddit4-none/Reddit4~all~trans~16~softplus~1.0e-5~value~-1.sh "EvoGCNHx2" $lr $seed &
    CUDA_VISIBLE_DEVICES=6 bash examples/task-train-reddit4/reddit4-none/Reddit4~all~trans~16~softplus~1.0e-5~value~-1.sh "EvoGCNOx2" $lr $seed 
) &&
(
    CUDA_VISIBLE_DEVICES=3 bash examples/task-train-reddit4/reddit4-none/Reddit4~all~trans~16~softplus~1.0e-5~value~-1.sh "GCNx2oGRU" $lr $seed &
    CUDA_VISIBLE_DEVICES=4 bash examples/task-train-reddit4/reddit4-none/Reddit4~all~trans~16~softplus~1.0e-5~value~-1.sh "GCRNM2x2" $lr $seed 
) &
(
    CUDA_VISIBLE_DEVICES=5 bash examples/task-train-reddit4/reddit4-none/Reddit4~all~trans~16~softplus~1.0e-5~value~-1.sh "TGATx2" $lr $seed &
    CUDA_VISIBLE_DEVICES=6 bash examples/task-train-reddit4/reddit4-none/Reddit4~all~trans~16~softplus~1.0e-5~value~-1.sh "TGNOptimLx2" $lr $seed 
) &&
(
    CUDA_VISIBLE_DEVICES=5 bash examples/task-train-reddit4/reddit4-dense/Reddit4~all~trans~dense_GRUoGCN2x2~16~softplus~1.0e-5~value~-1.sh $lr $seed &
    CUDA_VISIBLE_DEVICES=5 bash examples/task-train-reddit4/reddit4-none/IMP~Reddit4~all~trans~none~16~softplus~1.0e-5~value~-1.sh $lr $seed
)
echo "DONE"
