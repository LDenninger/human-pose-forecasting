run_script_path="$(pwd)/run.py"
exp_path="$(pwd)/experiments"

# Bash Aliases to run scripts mroe easy
alias iexp='python $run_script_path --init_exp'
alias irun='python $run_script_path --init_run'
alias cpconf='python $run_script_path --copy_conf'
alias cllog='python $run_script_path --clear_logs'
alias train='python $run_script_path --train'
alias evaluate='python $run_script_path --evaluate'

alias tboard='tensorboard --logdir $exp_path --port 6060'


# Set environment variables with the experiment/run name for easier access
function setexp() {
    export CURRENT_EXP="$1"
}
function setrun() {
    export CURRENT_RUN="$1"
}
function setuser() {
    export CURRENT_USER="$1"
}
function setmachine() {
    export CURRENT_MACHINE="$1"
}

function setup() {
    echo "------ Experiment Environment Setup ------"
    echo "  Current experiment ---> $CURRENT_EXP"
    echo "  Current run        ---> $CURRENT_RUN"
    echo "  Current user       ---> $CURRENT_USER"
    echo "  Current machine    ---> $CURRENT_MACHINE"
}

function sync_exp() {
    user="$1"
    machine="$2"
    dest="$exp_path/$CURRENT_EXP/$CURRENT_RUN/checkpoints"
    src="$user@$machine.informatik.uni-bonn.de:/home/user/denninge/human-pose-forecasting/experiments/$CURRENT_EXP/$CURRENT_RUN/checkpoints/*"
    scp -r "$src" "$dest"
}

function sync_exp_all(){
    user="$1"
    machine="$2"
    
}

function get_checkpoints() {
  local checkpoint="${1:-final}"
  local directory="experiments"
  local experiments=()
  

  # Call list_subdirectories and capture the result
  readarray -d '' experiments < <(list_subdirectories "$directory")

  # Now you can use the experiments as needed in this function
  for experiment in "${experiments[@]}"; do
    # Process each subdirectory here
    echo "Processing experiment: $experiment"

    # List all runs in the experiment
    local runs=()
    readarray -d '' runs < <(list_subdirectories "$experiment")

    for run in "${runs[@]}"; do
      # Check if run name is empty
      if [ -z "$run" ]; then
        continue
      fi      

      # Process each run here
      echo "Processing run: $run"

      # Load final checkpoint from remote server
      dest="$(pwd)/$experiment/$run/checkpoints"
      src="$CURRENT_USER@$CURRENT_MACHINE.informatik.uni-bonn.de:/home/user/denninge/human-pose-forecasting/experiments/$run/checkpoints/*$checkpoint.pth"

      # Check if the file exists
      if ssh "$CURRENT_USER@$CURRENT_MACHINE.informatik.uni-bonn.de" "[ -f $src ]"; then
          echo "Copying checkpoint from $src to $dest"
          scp -r "$src" "$dest"
      else
          echo "$src"
          echo "Checkpoint $checkpoint not found for $experiment/$run"
      fi
    done
  done
}


function list_subdirectories() {
  local directory="$1"
  local subdirectories=()

  # Check if the provided directory exists
  if [ ! -d "$directory" ]; then
    return 1  # Return an error code if the directory doesn't exist
  fi

  # Use find to list all subdirectories and add them to the array
  while IFS= read -r -d '' subdir; do
    subdirectories+=("$subdir")
  done < <(find "$directory" -mindepth 1 -maxdepth 1 -type d -print0)

  # Return the array of subdirectories
  printf '%s\0' "${subdirectories[@]}"
}
