import numpy as np
import matplotlib.pyplot as plt
import os

def plot(experiment):
    for file_name in os.listdir('results'):
        with open(os.path.join('results', file_name), 'r') as file:
            results = file.read()
            width, height = file_name.split('_')[1:3]
            width = int(width)
            height = int(height)
        time = np.array(results.split('\n')[:-1]).astype('float')
        x = np.array([x for x in range(3, 14, 2)])
        y = width*height/(time)
        
        peak_flops = 1843*1e9
        flops_time = ((x**2 + x**2-1)*width*height)/peak_flops
        flops_bound = (width*height)/flops_time

        peak_bandwidth = 76.8
        memory_time = ((x**2)*(width + x//2 + 1)*(height + x//2 + 1) + width*height)/(76.8*1e9)
        memory_bound = (width*height*4)/memory_time
        
        formatted_exp_name = "-".join(experiment.split('_')).title()
        title = f'Throughput for various filter sizes'
        x_label = f'Filter Size'
        y_label = f'Pixels per second'
            
        plt.figure(figsize=(18, 12))        
        plt.scatter(x, y, label='measured', color='blue', marker='x')
        # plt.plot(x, flops_bound, label='flops bound', color='red')
        # plt.plot(x, memory_bound, label='memory bound', color='green')
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        # plt.ylim(0, 1e9)
        plt.legend()
        
        plt.savefig(f'plots/{experiment}_{width}_{height}.png')
        plt.clf()

plot('performance_modeling')