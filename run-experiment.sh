!#/bin/bash

data_directory='./data/'
figure_directory='./figures/'
log_directory='./logs/'
output_directory='./outputs/'

datasets=("sir")
methods=("vi" "mcmc" "map")

# Valid options are "process", "time+process", "space+time+process"
model_types=("process" "time+process" "space+time+process")

# Loop over varieties of surrogate (process, temporal/process, spatial/temporal/process)
# for model fitting
for dataset in "${datasets[@]}"
    do
    for model_type in "${model_types[@]}"
        do
        for fit_method in "${methods[@]}"
            do
                log_path=${log_directory}${dataset}_${model_type}_${fit_method}.log
                out_path=${output_directory}${dataset}_${model_type}_${fit_method}.json
                rm -f $log_path
                nohup python code/cli.py fit_model ${data_directory}${dataset}.json $out_path  $model_type $fit_method > $log_path &
            done
        done
    done