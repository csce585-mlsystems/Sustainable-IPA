import matplotlib.pyplot as plt

"""
file syntax:
weights:yolov7-e6e.pt,
batches:5,32,50
...
"""


with open("./out/run-20241128.1946.10.out", "r") as f:
    result = f.read()

result_by_line = result.split("\n")

weights = result_by_line[0].split(":")[1].split()
batches = result_by_line[1].split(":")[1].split()
search_space = []
for i in range(len(weights)):
    for j in range(len(batches)):
        search_space.append(f"{weights[i]}, {batches[j]}")

# TODO: search for line start with EXP for weights and batch size
latency_lines = []
energy_consumption_lines = []

for i in range(2, len(result_by_line)):
    if result_by_line[i].startswith("Speed:"):
        latency_lines.append(result_by_line[i])
    elif result_by_line[i].startswith(" Performance counter"):
        energy_consumption_lines.append(result_by_line[i+2])

latency = []
for line in latency_lines:
    latency.append(float(line.split()[1].split("/")[0]))

energy_consumption = []
for line in energy_consumption_lines:
    energy_consumption.append(float(line.split()[0]))

# figure 3
print("...plotting latency (per input) vs energy consumption")
plt.figure(figsize=(10, 10))
plt.plot(latency, energy_consumption, marker='.')
plt.title("Energy Consumption vs. Latency")
plt.xlabel("Latency per Input (ms)")
plt.ylabel("Energy Consumption (J)")
plt.show()

# energy consumption per batch
for i in range(len(batches)):
    energy_consumption[i] = energy_consumption[i]/float(batches[i])

# figure 4
print("...plotting latency (per input) vs energy consumption (per input)")
plt.figure(figsize=(10, 10))
plt.plot(latency, energy_consumption, marker='.')
plt.title("Energy Consumption vs. Latency")
plt.xlabel("Latency per Input (ms)")
plt.ylabel("Energy Consumption per Input (J)")
plt.show()
