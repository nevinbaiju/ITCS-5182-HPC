import numpy as np
import matplotlib.pyplot as plt
import os

def plot():
    for file_name in os.listdir('results'):
        with open(os.path.join('results', file_name), 'r') as file:
            results = file.read()
            width, height = file_name.split('_')[1:3]
            experiment = "_".join(file_name.split('_')[5:])[:-4]
            width = int(width)
            height = int(height)
        time = np.array(results.split('\n')[:-1]).astype('float')
        x = np.array([x for x in range(3, 16, 2)])
        y = (width*height)/(time)
        
        peak_flops = 1843*1e9
        flops_time = ((x**2 + x**2-1)*width*height)/peak_flops
        flops_bound = (width*height)/flops_time

        peak_bandwidth = 76.8
        memory_time = (4*((x**2)+(width + x//2 + 1)*(height + x//2 + 1) + width*height))/(76.8*1e9)
        memory_bound = (width*height)/memory_time
        
        formatted_exp_name = "-".join(experiment.split('_')).title()
        title = f'Throughput for various filter sizes'

        # Create a figure and axis
        plt.plot(figsize=(8, 6))

        # Create a second y-axis on the right
        plt.plot(x, y)
        plt.title(f'Performance measure for {formatted_exp_name}')
        plt.xlabel('Filter Size')
        plt.ylabel('Pixels/s')
        # ax2.set_ylim(0, 3e7)

        plt.title(f"{title} Img size: {width}x{height}")
        
        plt.savefig(f'plots/{experiment}_{width}_{height}.png')
        plt.clf()

plot()