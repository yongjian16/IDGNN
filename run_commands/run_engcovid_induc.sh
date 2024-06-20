#!/bin/bash
# $1 = inductive/inducductive
# $2 = 56/57/58

lr=$1
seed=$2

(
    CUDA_VISIBLE_DEVICES=3 bash examples/task-train-engcovid/engcovid-none/EngCOVID~all~induc~none~16~softplus~1.0e-5~value~-1.sh "DCRNNx2" $lr $seed &
    CUDA_VISIBLE_DEVICES=4 bash examples/task-train-engcovid/engcovid-none/EngCOVID~all~induc~none~16~softplus~1.0e-5~value~-1.sh "DySATx2" $lr $seed 
) &
(
    CUDA_VISIBLE_DEVICES=5 bash examples/task-train-engcovid/engcovid-none/EngCOVID~all~induc~none~16~softplus~1.0e-5~value~-1.sh "EvoGCNHx2" $lr $seed &
    CUDA_VISIBLE_DEVICES=6 bash examples/task-train-engcovid/engcovid-none/EngCOVID~all~induc~none~16~softplus~1.0e-5~value~-1.sh "EvoGCNOx2" $lr $seed 
) &&
(
    CUDA_VISIBLE_DEVICES=3 bash examples/task-train-engcovid/engcovid-none/EngCOVID~all~induc~none~16~softplus~1.0e-5~value~-1.sh "GCNx2oGRU" $lr $seed &
    CUDA_VISIBLE_DEVICES=4 bash examples/task-train-engcovid/engcovid-none/EngCOVID~all~induc~none~16~softplus~1.0e-5~value~-1.sh "GCRNM2x2" $lr $seed 
) &
(
    CUDA_VISIBLE_DEVICES=5 bash examples/task-train-engcovid/engcovid-none/EngCOVID~all~induc~none~16~softplus~1.0e-5~value~-1.sh "TGATx2" $lr $seed &
    CUDA_VISIBLE_DEVICES=6 bash examples/task-train-engcovid/engcovid-none/EngCOVID~all~induc~none~16~softplus~1.0e-5~value~-1.sh "TGNOptimLx2" $lr $seed 
) &&
(
    CUDA_VISIBLE_DEVICES=0 bash examples/task-train-engcovid/engcovid-dense/EngCOVID~all~induc~dense_GRUoGCN2x2~16~softplus~1.0e-5~value~-1.sh $lr $seed &
    CUDA_VISIBLE_DEVICES=1 bash examples/task-train-engcovid/engcovid-none/IMP~EngCOVID~all~none~16~softplus~1.0e-5~value~-1.sh inductive $lr $seed
)
echo "DONE"


