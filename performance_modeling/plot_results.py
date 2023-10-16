import numpy as np
import matplotlib.pyplot as plt

def plot(experiment):
    with open(f'results/results_{experiment}.txt', 'r') as file:
        results = file.read()
    time = np.array(results.split('\n')[:-1]).astype('float')
    x = np.array([x for x in range(3, 100, 2)])
    y = 1024*768/(time)
    
    peak_flops = 1843*1e9
    flops_time = ((x**2 + x**2-1)*1024*768)/peak_flops
    flops_bound = (1024*768)/flops_time

    peak_bandwidth = 76.8
    memory_time = ((x**2)*(1024 + x//2 + 1)*(768 + x//2 + 1) + 1024*768)/(76.8*1e9)
    memory_bound = (1024*768*4)/memory_time
    
    formatted_exp_name = "-".join(experiment.split('_')).title()
    title = f'Throughput for various filter sizes'
    x_label = f'Filter Size'
    y_label = f'Pixels per second'
        
    plt.figure(figsize=(18, 12))        
    plt.plot(x, y, label='measured', color='blue')
    plt.plot(x, flops_bound, label='flops bound', color='red')
    plt.plot(x, memory_bound, label='memory bound', color='green')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim(0, 1e9)
    plt.legend()
    
    plt.savefig(f'plots/{experiment}.png')
    plt.clf()

plot('performance_modeling')