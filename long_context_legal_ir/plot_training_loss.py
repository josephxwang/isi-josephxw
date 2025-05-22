import re
import json
import matplotlib.pyplot as plt

losses = []
epochs = []

# File path to your log
log_path = "tevatron/slurm-529982.out"

# Regex to detect dictionary-like lines
dict_line_pattern = re.compile(r"\{.*'loss': .*?\}")

with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        match = dict_line_pattern.search(line)
        if match:
            try:
                # Convert single quotes to double for valid JSON parsing
                log_dict = json.loads(match.group().replace("'", "\""))
                losses.append(log_dict["loss"])
                epochs.append(log_dict["epoch"])
            except json.JSONDecodeError:
                continue

# Plotting
plt.figure(figsize=(10, 6))

# plt.plot(epochs, losses, label="Loss", marker='o', linewidth=1)
plt.plot(epochs, losses, label="Loss", linestyle='-', color='blue', linewidth=2)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("LongFormer 10 epochs on CLERC, 8 negatives/positive")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
