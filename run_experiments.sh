#!/bin/bash

output_dir="results"

python evaluation_metric_experiment.py

for dataset in "small" "go" "xml"
do
    for i in {1..10}; do
        python cv_comparison_experiment.py $dataset $i $output_dir --create &
    done
    wait

    echo "evaluating results"
    for i in {1..10}; do
        echo "file $i"
        python cv_comparison_experiment.py $dataset $i $output_dir --evaluation &
    done
    wait

    python mean.py $dataset $output_dir

done

