while true; do
  echo "==== $(date) ====" >> gpu_usage.log
  rocm-smi >> gpu_usage.log
  sleep 5
done &
MONITOR_PID=$!