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

# don't delete the file, add token to show start of a run
echo " << NEW TESTING RUN STARTED >> " >> grader.txt

# Redirect grader outputs into grader.txt
for (( i=1; i<=$n; i++  ))
do
  echo "starting run of grader $i"
  env GRADER_TESTING=1 python -m grader state_agent -v >> grader.txt
done