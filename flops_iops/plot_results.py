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
    plt.savefig(f'{optimization}.png')

plot_results('results_O0')
plot_results('results_O1')
plot_results('results_O2')
plot_results('results_O3')