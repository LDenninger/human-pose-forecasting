# Function to create .gitkeep file in empty directories
create_gitkeep() {
  # Iterate through each directory
  for dir in "$1"/*; do
    if [ -d "$dir" ]; then
        # Create .gitkeep in every directory to keep the experiment structure
        touch "$dir/.gitkeep"
        # Recursively call the function for subdirectories
        create_gitkeep "$dir"
    fi
  done
}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
experiments_dir="$SCRIPT_DIR/experiments"

# Call the function to create .gitkeep files
create_gitkeep "$experiments_dir"