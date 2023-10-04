import numpy as np
import matplotlib.pyplot as plt

def plot(experiment):
    with open(f'results/results_{experiment}.txt', 'r') as file:
        results = file.read()
    y = np.array(results.split('\n')[:-1]).astype('float')
    x = [1] + [x for x in range(16, 20481, 16)] + [x for x in range(20480, 204801, 512)]
    
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
        
    plt.figure(figsize=(18, 12))        
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.axvline(20480, label='L3', color='red')
    plt.legend()
    
    plt.savefig(f'plots/{experiment}.png')
    plt.clf()

    plt.figure(figsize=(18, 12))
    plt.plot(x[:1285], y[:1285])
    plt.axvline(32, label='L1', color='green', linestyle='dotted', linewidth=1.5)
    plt.axvline(256, label='L2', color='orange', linestyle='dotted', linewidth=1.5)
    plt.axvline(20480, label='L3', color='red', linestyle='dotted', linewidth=1.5)
    plt.title(title + " Cache effects")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    
    plt.savefig(f'plots/{experiment}_cache.png')
    plt.clf()

plot('read')
plot('write')
plot('read_write')
plot('latency')