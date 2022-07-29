#!/usr/bin/env sh

alias app="poetry run python -m argmining_clustering"

for glob in "microtexts/*.json" "essays/*.ann" "kialo-small/*.txt"; do
    app "$glob" &
    app "$glob" --predict-mc &
done
