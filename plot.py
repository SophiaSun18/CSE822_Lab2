import matplotlib.pyplot as plt
import sys

warps = []
flops = []

for line in sys.stdin:
    if line.startswith('Warps'):
        continue
    parts = line.strip().split(', ')
    if len(parts) == 2:
        warps.append(int(parts[0]))
        flops.append(float(parts[1]) / 1e12)

plt.figure(figsize=(12, 6))
plt.grid(True, color='lightgray', linestyle='-', alpha=0.7)
plt.plot(warps, flops, label='Achieved Throughput', linewidth=2)
plt.axhline(y=26.73, color='r', linestyle='--', label='Peak Throughput (26.73 TFLOPS)', linewidth=2)
plt.xlabel('#Warps', fontsize=16)
plt.ylabel('TFLOPS', fontsize=16)
plt.xticks(range(0, max(warps) + 1, max(1, max(warps) // 10)), fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.show()