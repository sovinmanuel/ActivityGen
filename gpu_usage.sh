#!/bin/bash

# Header for CSV
echo "Timestamp,CPU Usage (%),RAM Usage (MB),VRAM Usage (MB)" > usage_data.csv

while true; do
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    cpu_usage=$(mpstat 1 1 | awk '/Average:/ {print 100 - $12}')
    ram_usage=$(free -m | awk '/Mem:/ {print $3}')
    vram_usage=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | cut -d ' ' -f1)
    echo "$timestamp,$cpu_usage,$ram_usage,$vram_usage" >> usage_data.csv
    # sleep 1
done
