#!/bin/bash
Help() {
  # Display Help
  echo "ml query optimization solution."
  echo
  echo "Syntax: scriptTemplate [-w|t|i|a|s|tno|tne]"
  echo "example: sh run.sh -w 2 -t 1 -i '1' -a 0.9 -s 3 -o 2 -e 1"
  echo "options:"
  echo "w   int: experiment type in [1, 2, 3]; default: 2."
  echo "t   int: query type in [0, 1, 2]; default: 1."
  echo "i   int: query index; default: 1."
  echo "a   float: target accuracy; default: 0.90."
  echo "s   int: scheme in ['RAW', 'NS', 'PP', 'CORE', 'COREa', 'COREh', 'REORDER'] -> [0, 1, 2, 3, 4, 5, 6]; default: 'CORE'."
  echo "o   int: number of threads used in optimization phase; default: 2."
  echo "e   int: number of threads used in execution phase; default: 1."
}
#sh run_copy.sh -w 2 -t "0, 1, 2" -i "0, 1, 2" -a "0.90, 0.92, 0.94" -s "NS, PP, CORE" -o "1, 2, 3" -e "1, 2, 3"

while getopts ":h:w:t:i:a:s:o:e:" option; do
  case ${option} in

  h)
    Help
    exit;;
  w) EXP_TYPES=${OPTARG} ;;
  t) WORKFLOW_TYPES=${OPTARG} ;;
  i) WORKFLOW_INDEXES=${OPTARG} ;;
  a) TARGET_ACCURACIES=${OPTARG} ;;
  s) SCHEMES=${OPTARG} ;;
  o) THREAD_NUM_OPTS=${OPTARG} ;;
  e) THREAD_NUM_EXES=${OPTARG} ;;
  esac
done

IFS=', ' # space is set as delimiter
read -ra EXPTYPE <<<"$EXP_TYPES"
read -ra WORKTYPE <<<"$WORKFLOW_TYPES"
read -ra WORKINDEX <<<"$WORKFLOW_INDEXES" # str is read into an array as tokens separated by IFS
read -ra TARACC <<<"$TARGET_ACCURACIES"
read -ra SCHE <<<"$SCHEMES"
read -ra TNOPT <<<"$THREAD_NUM_OPTS"
read -ra TNEXE <<<"$THREAD_NUM_EXES"

for exp_type in "${EXPTYPE[@]}"; do
  for workflow_type in "${WORKTYPE[@]}"; do
    for workflow_index in "${WORKINDEX[@]}"; do # access each element of array
      for target_accuracy in "${TARACC[@]}"; do
        for scheme in "${SCHE[@]}"; do
          for thread_num_opt in "${TNOPT[@]}"; do
            for thread_num_exe in "${TNEXE[@]}"; do
              /home/anaconda3/envs/condaenv/bin/python3 -u /home/CorrProxies/main.py "$exp_type" "$workflow_type" "$workflow_index" "$target_accuracy" "$scheme" "$thread_num_opt" "$thread_num_exe" | tee /home/CorrProxies/output/workflow_"$exp_type"_"$workflow_type"_"$workflow_index"_"$target_accuracy"_"$scheme"_"$thread_num_opt"_"$thread_num_exe".out
            done
          done
        done
      done
    done
  done
done
