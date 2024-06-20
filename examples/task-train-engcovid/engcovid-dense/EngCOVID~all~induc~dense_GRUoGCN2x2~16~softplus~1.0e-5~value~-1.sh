lr=$1
seed=$2

/usr/bin/time -f "Max CPU Memory: %M KB\nElapsed: %e sec" \
python -u engcovid.py \
--model GRUoGCN2x2 --win-aggr dense --source EngCOVID --target all \
--framework inductive --hidden 16 --activate softplus --epoch 1 \
--lr $lr --weight-decay 1e-5 --clipper value --patience -1 --seed $seed \
--device cuda
