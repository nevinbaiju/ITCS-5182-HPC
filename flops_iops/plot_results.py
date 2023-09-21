import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_results(filename):
    with open(filename, 'r') as file:
        results = np.array(file.read().split('\n')[:-1]).astype(float)

    plt.figure(figsize=(10, 6))
    sns.histplot(results)
    plt.xlabel('GFlops')
    plt.ylabel('Sample Count')
    optimization = filename.split('_')[1]
    plt.title(f'GFlops achieved for -{optimization} optimization')
    print(f"Flops for {optimization}")
    print(f"Average Flops {results.mean()}")
    print(f"Min Flops {results.min()}")
    print(f"Max Flops {results.max()}\n")

plot_results('results_O0')
plot_results('results_O1')
plot_results('results_O2')
plot_results('results_O3')

# plot_results('results_O0int')
# plot_results('results_O1int')
# plot_results('results_O2int')
# plot_results('results_O3int')