# This script runs the grader a number of times and redirects the output into grader.txt

n=$1 && shift

if [[ -z $n ]]; then
  echo "Number of iterations not given. Exiting."
  echo "Usage: run_evlatuion.sh <n_iterations>"
  exit 1
fi

if [[ $n -lt 1 ]]; then
  echo "Number of iterations needs to be greater than 1. Exiting."
  echo "Usage: run_evlatuion.sh <n_iterations>"
  exit 1
fi

# Remove grader.txt if it already exists
if [[ -f grader.txt ]]; then
  rm grader.txt
fi

# Remove any stats files that existed before
if [[ -f stat.csv ]] || [[ -f stats.csv ]]; then
  rm stat.txt
  rm stats.txt
fi

# Redirect grader outputs into grader.txt
for (( i=1; i<=$n; i++  ))
do
  echo "starting run of grader $i"
  python -m grader state_agent -v >> grader.txt
done