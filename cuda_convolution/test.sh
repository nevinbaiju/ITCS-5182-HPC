sbatch --partition=GPU --chdir=`pwd` --time=04:00:00 --ntasks=1 --cpus-per-task=16 --gpus-per-task=1 --job-name=pci_l --mem=10G ./cuda_convolution 4194304 768 3