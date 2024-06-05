import matplotlib.pyplot as plt
import os

data_folder = 'data'

procs = []
times = []
l2_norms = []

for filename in os.listdir(data_folder):
    if filename.startswith('result_') and filename.endswith('.txt'):
        filepath = os.path.join(data_folder, filename)
        with open(filepath, 'r') as f:
            lines = f.readlines()
            grid_size = int(lines[0].strip().split(':')[1])
            num_procs = int(lines[1].strip().split(':')[1])
            l2_norm = float(lines[2].strip().split(':')[1])
            exec_time_str = lines[3].strip().split(':')[1].strip()
            exec_time = float(exec_time_str.split()[0])
            procs.append(num_procs)
            times.append(exec_time)
            l2_norms.append(l2_norm)

procs, times, l2_norms = zip(*sorted(zip(procs, times, l2_norms)))

plt.figure()
plt.plot(procs, times, marker='o')
plt.xlabel('Number of processors')
plt.ylabel('Execution time (microseconds)')
plt.title(f'Scalability Test - Execution Time (Grid size: {grid_size}x{grid_size})')
plt.grid(True)
plt.savefig('scalability_test_time.png')

plt.figure()
plt.plot(procs, l2_norms, marker='o')
plt.xlabel('Number of processors')
plt.ylabel('L2 norm of the error')
plt.title(f'Scalability Test - L2 Norm (Grid size: {grid_size}x{grid_size})')
plt.grid(True)
plt.savefig('scalability_test_l2_norm.png')

plt.show()