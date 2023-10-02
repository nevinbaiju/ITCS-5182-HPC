import numpy as np
import matplotlib.pyplot as plt

def plot(experiment):
    with open(f'results/results_{experiment}.txt', 'r') as file:
        results = file.read()
    y = np.array(results.split('\n')[:-1]).astype('float')
    x = [1] + [x for x in range(16, 513, 16)] + [x for x in range(512, 204801, 512)]
    
    if 'latency' in experiment:
        formatted_exp_name = "-".join(experiment.split('_')).title()
        title = f'{formatted_exp_name} for various sizes of data'
        x_label = f'Size of data (KB)'
        y_label = f'Latency (ns)'
    else:
        formatted_exp_name = "-".join(experiment.split('_')).title()
        title = f'{formatted_exp_name} speed for various sizes of data'
        x_label = f'Size of data (KB)'
        y_label = f'Speed (GB/s)'
        
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.axvline(3072, label='L3', color='red')
    plt.legend()
    
    plt.savefig(f'plots/{experiment}.png')
    plt.clf()

    plt.plot(x[:40], y[:40])
    plt.axvline(64, label='L1', color='green')
    plt.axvline(512, label='L2', color='blue')
    plt.axvline(3072, label='L3', color='orange')
    plt.title(title + " Cache effects")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    
    plt.savefig(f'plots/{experiment}_cache.png')
    plt.clf()

plot('read')
# plot('write')
# plot('read_write')
# plot('latency')