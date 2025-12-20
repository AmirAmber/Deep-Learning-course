import matplotlib.pyplot as plt

# Your data from the output
num_facts = [10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250]
accuracy = [0.98, 0.95, 0.82, 0.68, 0.46, 0.30, 0.30, 0.17, 0.12, 0.10, 0.05, 0.05, 0.04]

plt.figure(figsize=(10, 6))
plt.plot(num_facts, accuracy, marker='o', linestyle='-', color='b', label='Mamba2-370m')

plt.title('Associative Recall Accuracy vs. Number of Facts')
plt.xlabel('Number of Facts')
plt.ylabel('Accuracy')
plt.grid(True)
plt.xticks(num_facts)
plt.ylim(0, 1.05)
plt.legend()

# Save the graph
output_file = 'ar_results_graph.png'
plt.savefig(output_file)
print(f"Graph saved to {output_file}")