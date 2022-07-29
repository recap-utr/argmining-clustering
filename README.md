# Argument Mining using Clustering

The easiest way to run the evaluation is using Docker:

```sh
# Show help
docker-compose run --rm app poetry run python -m argmining_clustering --help
# Run the evaluation on all json files of the folder `./data/input` and write the graphs to `./data/output/
docker-compose run --rm app poetry run python -m argmining_clustering "**/*.json" --input-folder ./data/input --output-folder ./data/output --model en_core_web_lg
# Predict the major claims instead of using the gold standard
docker-compose run --rm app poetry run python -m argmining_clustering "**/*.json" --input-folder ./data/input --predict-mc
```
