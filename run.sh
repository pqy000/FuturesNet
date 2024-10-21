#!/bin/bash

# 检查是否有 log 目录，如果没有则创建
if [ ! -d "log" ]; then
  mkdir log
  echo "Created log directory."
fi

# 从命令行参数获取 year, id, model 列表 和 GPU 数量
years=($1)
ids=($2)
models=($3)
gpus=($4)  # 传入可用的 GPU 列表，例如 "0 1 2 3"

# 变量初始化
gpu_count=${#gpus[@]}  # 获取 GPU 数量
gpu_index=0  # 用于分配 GPU 的索引

# 循环遍历所有 id, year 和 model 组合
for year in "${years[@]}"
do
  for id in "${ids[@]}"
  do
    for model in "${models[@]}"
    do
      # 当前 GPU
      gpu=${gpus[$gpu_index]}

      log_file="log/output_${id}_${year}_${model}_gpu${gpu}.log"
      echo "Running id=$id, year=$year, model=$model on GPU $gpu"
      nohup python3 main.py --id $id --year $year --model $model --gpu $gpu > $log_file 2>&1 &
      echo "Started process for id=$id, year=$year, model=$model on GPU $gpu with PID $!"
      gpu_index=$(( (gpu_index + 1) % gpu_count ))
    done
  done
done

echo "All processes completed."

# bash run.sh "2020" "50" "CNN GRU Attention preTCN FuturesNet" "1 2 3 4 5"
# bash run.sh "2020" "300" "CNN GRU Attention preTCN FuturesNet" "1 2 3 4 5"
# bash run.sh "2020" "500" "CNN GRU Attention preTCN FuturesNet" "1 2 3 4 5"

# bash run.sh "2021" "300" "CNN GRU Attention preTCN FuturesNet" "1 2 3 4 5"
# bash run.sh "2021" "50" "CNN GRU Attention preTCN FuturesNet" "1 2 3 4 5"
# bash run.sh "2021" "500" "CNN GRU Attention preTCN FuturesNet" "1 2 3 4 5"

# bash run.sh "2022" "50" "CNN GRU Attention preTCN FuturesNet"  "0 1 2 3 4"
# bash run.sh "2022" "300" "CNN GRU Attention preTCN FuturesNet" "0 1 2 3 4"
# bash run.sh "2022" "500" "CNN GRU Attention preTCN FuturesNet" "0 1 2 3 4"

# bash run.sh "2020 2021 2022" "50 300" "FuturesNet" "0 1 2 3 4 5"
# bash run.sh "2020 2021 2022" "500" "FuturesNet" "0 1 2"
# bash run.sh "2020 2021 2022" "500" "LSTNet" "3 4 5"

# bash run.sh "2020 2021 2022" "50 300" "LSTNet" "0 1 2 3 4 5"
# bash run.sh "2020 2021 2022" "50 300" "Transformer" "0 1 2 3 4 5"
# bash run.sh "2020 2021 2022" "500" "Transformer" "0 1 2"
# bash run.sh "2020" "50" "CNN" "0"
# bash run.sh "2020 2021 2022" "500" "Transformer" "0 1 2"
# bash run.sh "2020 2021 2022" "500" "Transformer" "0 1 2"
# bash run.sh "2020 2021 2022" "50 300" "CNN" "0 1 2 3 4 5"
# bash run.sh "2020 2021 2022" "500" "CNN" "0 1 2 3 4 5"
# bash run.sh "2020 2021 2022" "300" "CNN GRU" "0 1 2 3 4 5"
# bash run.sh "2020 2021 2022" "500" "CNN GRU" "0 1 2 3 4 5"
# bash run.sh "2020 2021 2022" "50 300" "FuturesNet" "0 1 2 3 4 5"
# bash run.sh "2020 2021 2022" "500" "FuturesNet" "0 1 2"


# bash run.sh "2020 2021 2022" "50 300" "GRU" "0 1 2 3 4 5"
# bash run.sh "2020 2021 2022" "500" "GRU" "0 1 2"

# bash run.sh "2020 2021 2022" "50 300" "LSTNet" "0 1 2 3 4 5"
# bash run.sh "2020 2021 2022" "500" "LSTNet" "0 1 2"
# bash run.sh "2020 2021 2022" "50" "LSTNet" "0 1 2 3 4 5"






