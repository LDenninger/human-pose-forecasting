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

function setup() {
    echo "------ Experiment Environment Setup ------"
    echo "  Current experiment ---> $CURRENT_EXP"
    echo "  Current run        ---> $CURRENT_RUN"
}

function sync_exp() {
    user="$1"
    machine="$2"
    dest="$exp_path/$CURRENT_EXP/$CURRENT_RUN/checkpoints"
    src="$user@$machine.informatik.uni-bonn.de:/home/user/denninge/human-pose-forecasting/experiments/$CURRENT_EXP/$CURRENT_RUN/checkpoints/*"
    scp -r "$src" "$dest"
}


