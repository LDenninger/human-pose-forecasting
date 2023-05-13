run_script_path="$(pwd)/run.py"

echo "$management_path"

# Bash Aliases to run scripts mroe easy
alias iexp='python $run_script_path --init_exp'
alias irun='python $run_script_path --init_run'
alias cpconf='python $run_script_path --copy_conf'
alias train='python $run_script_path --train'
alias evaluate='python $run_script_path --evaluate'


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