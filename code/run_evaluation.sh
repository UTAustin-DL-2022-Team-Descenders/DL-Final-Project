export EVALUATION="True"

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

for (( i=1; i<=$n; i++  ))
do
  for opponent in jurgen_agent geoffrey_agent yann_agent yoshua_agent
  do
    # state_agent is on team2 if i is even
    if [[ $(( i%2 )) -eq 0 ]]; then
      team1=$opponent
      team2="state_agent"
    else # state_agent is on team1 if i is odd
      team1="state_agent"
      team2=$opponent
    fi
    # construct run command
    run_cmd="python -m tournament.runner $team1 $team2"

    # run and get the output
    output=$($run_cmd)

    # echo the match-up and the results
    echo "$team1 (Team1) v $team2 (Team2): $output"
  done
done

unset EVALUATION