model=$1
lr=$2
seed=$3

/usr/bin/time -f "Max CPU Memory: %M KB\nElapsed: %e sec" \
python -u engcovid.py \
--model $model --win-aggr none --source EngCOVID --target all \
--framework inductive --hidden 16 --activate softplus --epoch 1 \
--lr $lr --weight-decay 1e-5 --clipper value --patience -1 --seed $seed \
--device cuda
