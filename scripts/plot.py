import matplotlib.pyplot as plt

"""
example file syntax:
weights:yolov7-e6e.pt,
batches:5,32,50
Speed:0/
0/ Joules
latency:, , ,
energy_consumption: , , ,
"""

with open("./out/many_batches.out", "r") as f:
    result = f.read()
result_by_line = result.split("\n")

#for r in range(len(result_by_line)):
#    print(r, result_by_line[r])

weights = result_by_line[0].split(":")[1].split()
batches = result_by_line[1].split(":")[1].split()
search_space = []
for i in range(len(weights)):
    for j in range(len(batches)):
        search_space.append(f"{weights[i]}, {batches[j]}")

latency = []
for i in range(3, len(result_by_line), 3):
    latency.append(float(result_by_line[i].split(":")[1].split("/")[0]))

energy_consumption = []
for i in range(4, len(result_by_line), 3):
    energy_consumption.append(float(result_by_line[i].split()[0]))

for i in range(len(batches)):
    energy_consumption[i] = energy_consumption[i]/float(batches[i])

#print(latency, energy_consumption)
#latency = result_by_line[2].split(":")[1].split(",")
#energy_consumption = result_by_line[3].split(":")[1].split(",")

print("...plotting latency vs energy consumption")
print(batches)
plt.figure(figsize=(10, 10))
plt.plot(latency, energy_consumption, marker='.')
plt.title("Energy Consumption vs. Latency")
plt.xlabel("Latency per Input (ms)")
plt.ylabel("Energy Consumption per Input (J)")
plt.show()
