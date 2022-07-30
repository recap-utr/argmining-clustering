#!/usr/bin/env sh

alias app="poetry run python -m argmining_clustering --no-progress"

for glob in "microtexts/*.json" "essays/*.ann" "kialo-small/*.txt"; do
    app "$glob" &
    app "$glob" --predict-mc &
done
wait
