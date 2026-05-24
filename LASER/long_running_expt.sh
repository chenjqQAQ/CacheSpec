#!/bin/bash

sudo snap install yq

num_examples=$1
use_agent=$2

feedback=True
scale="large"
dataset_type="param-w-synonym"

gamma=0.5
nu_list=(4)

for nu in "${nu_list[@]}"; do

    expt_type="${dataset_type}_${scale}_nu=${nu}_gamma=${gamma}"

    echo $expt_type $nu $gamma

    BASE_DIR="output_dir_${expt_type}"
    CONFIG_FILE="/home/ubuntu/GenCache/LASER/config.yaml"

    yq eval "del(.data.global_cache_path)" $CONFIG_FILE -i
    yq eval ".data.global_cache_path = \"\${data.data_path}/global_cache_${expt_type}\"" $CONFIG_FILE -i

    yq eval "del(.data.database_path)" $CONFIG_FILE -i
    yq eval ".data.database_path = \"\${data.data_path}/database_${expt_type}.pkl\"" $CONFIG_FILE -i

    yq eval "del(.data.results_path)" $CONFIG_FILE -i
    yq eval ".data.results_path = \"\${general.project_path}/output_dir_${expt_type}\"" $CONFIG_FILE -i


    failures=0
    i=0
    for elem in $( seq $i $num_examples )
    do
        echo "\n====================================="
        echo "Running Prompt: ${i}"

        # replace example number in the config file
        yq eval "del(.example_num)" $CONFIG_FILE -i
        yq eval ".example_num = ${elem}" $CONFIG_FILE -i

        mkdir -p $BASE_DIR/$elem
        if [ $use_agent == "no" ]; then
            nohup python caching_wo_agent.py --start $elem --num_examples 1 --nu $nu --gamma $gamma --dataset_type $dataset_type --dir_name $expt_type --scale $scale --feedback $feedback > $BASE_DIR/$elem/run_log.ans &
        else
            nohup python laser_agent.py --start $elem --num_examples 1 > $BASE_DIR/$elem/run_log.ans &
        fi
        
        pid=$!
        wait $pid
        exit_code=$?

        if [ $exit_code == 1 ]; then
            ((failures++))
        fi

        # Update the results.json file
        agent_failures=$failures
        cat <<< $(jq --arg a "$agent_failures" '.agent_failures=$a' $BASE_DIR/results.json) > $BASE_DIR/results.json

        cp $BASE_DIR/results.json $BASE_DIR/$elem/results$i.json
        ((i++))

        echo "\n=====================================\n\n"

    done

done