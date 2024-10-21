#!/bin/bash

# 并行执行批量任务的shell脚本
# year: 2020 or 2021
# id: 50, 300, 500
# ./process.sh 2020 "50 300 500"
# 传入的参数
YEAR=$1
IDS=$2  # ID 列表，可以是以空格分隔的多个 ID 值

if [ ! -d "log" ]; then
  mkdir log
  echo "Created log directory."
fi

for ID in $IDS
do
  echo "Running year $YEAR with id $ID"

  nohup python3 process_data.py $YEAR $ID > log/output_${YEAR}_${ID}.log 2>&1 &

  # 打印后台运行的进程号
  echo "Started process for year $YEAR and id $ID in background with PID $!"
done

wait
echo "All processes completed."


