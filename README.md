# Unsupervised Algorithms to Construct Argument Graphs based on Clustering

The code in this repository has been used to conduct the experiments for our paper.
In the `README`, you will find instructions on how to use the software.

## Obtaining the Data

Due to copyright reasons, we do not distribute the data used for our evaluation as part of the source code.
However, you may obtain the resources yourself from the following resources:

- Microtexts: <http://corpora.aifdb.org/Microtext> (download as `zip` or `tar.gz`)
- Essays: <https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2422> (download as `zip`)
- Kialo: Only available upon request

The data should be stored in a subfolder called `data/input` in the project itself. Please also create the folder `data/output` that is used for storing evaluation data or exported graphs.

## Running the Software

The easiest way to run the evaluation is using Docker:

```sh
# Show help
docker-compose run --rm app poetry run python -m argmining_clustering --help
# Run the evaluation on all json files of the folder `./data/input` and write the graphs to `./data/output/
docker-compose run --rm app poetry run python -m argmining_clustering "**/*.json" --save-json --save-pdf
# Do not predict the major claim but use gold standard
docker-compose run --rm app poetry run python -m argmining_clustering "**/*.json" --save-json --save-pdf --preset-mc
```

## Performing the Evaluation

```sh
docker-compose run --rm app ./evaluate.sh
```
