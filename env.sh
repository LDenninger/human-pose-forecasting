script_dir=$(dirname "$0")
management_path=$(realpath "$script_dir/utils/management.py")

# Bash Aliases to run scripts mroe easy
alias iexp='python $management_path --init_exp'
alias irun='python $management_path --init_run'
alias cpconf='python $management_path --copy_conf'
alias train='python $management_path --train'
alias evaluate='python $management_path --evaluate'


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
    echo "  Current run ---> $CURRENT_RUN"
}