import pandas as pd
import matplotlib.pyplot as plt

data = {
    "Attack": [
        "00-WebDDoS", "01-LDAP", "02-Portmap", "03-DNS", "04-UDPLag", "05-NTP", "06-SNMP", "07-SSDP",
        "08-Syn", "09-TFTP", "10-UDP", "11-NetBIOS", "12-MSSQL"
    ],
    "Baseline_Accuracy": [0.8780, 0.9733, 0.9857, 0.9167, 0.9987, 0.9883, 0.9917, 0.9933, 0.9968, 0.9990, 0.9989, 0.9990, 0.9989],
    "Manipulated_Accuracy": [0.6494, 0.7742, 0.9477, 0.7453, 0.9627, 0.7878, 0.9397, 0.9735, 0.9736, 0.9689, 0.9915, 0.9960, 0.9977],
    "Baseline_F1": [0.8980, 0.9731, 0.9875, 0.9068, 0.9986, 0.9886, 0.9913, 0.9933, 0.9968, 0.9990, 0.9989, 0.9990, 0.9989],
    "Manipulated_F1": [0.4255, 0.8228, 0.9487, 0.7614, 0.9657, 0.8266, 0.9420, 0.9747, 0.9744, 0.9693, 0.9915, 0.9960, 0.9977],
    "Baseline_TPR": [0.8148, 0.9577, 1.0000, 0.8294, 1.0000, 0.9854, 0.9976, 0.9998, 0.9997, 0.9998, 1.0000, 0.9999, 1.0000],
    "Manipulated_TPR": [0.2703, 0.9914, 1.0000, 0.8314, 0.9951, 0.9925, 0.9986, 0.9995, 0.9994, 0.997, 0.9995, 0.9999, 0.9998],
    "Baseline_TNR": [1.0000, 0.9892, 0.9672, 1.0000, 0.9974, 0.9914, 0.9863, 0.9871, 0.9939, 0.9982, 0.9979, 0.9981, 0.9979],
    "Manipulated_TNR": [1.0000, 0.5305, 0.8987, 0.6630, 0.9266, 0.5750, 0.8829, 0.9462, 0.9475, 0.9390, 0.9835, 0.9921, 0.9957],
    "Baseline_FPR": [0.0000, 0.0108, 0.0328, 0.0000, 0.0026, 0.0086, 0.0137, 0.0129, 0.0061, 0.0018, 0.0021, 0.0019, 0.0021],
    "Manipulated_FPR": [0.0000, 0.4695, 0.1013, 0.3370, 0.0734, 0.4250, 0.1171, 0.0538, 0.0525, 0.0610, 0.0165, 0.0079, 0.0043],
    "Baseline_FNR": [0.1852, 0.0423, 0.0000, 0.1706, 0.0000, 0.0146, 0.0024, 0.0002, 0.0003, 0.0002, 0.0000, 0.0001, 0.0000],
    "Manipulated_FNR": [0.7297, 0.0086, 0.0000, 0.1686, 0.0049, 0.0075, 0.0014, 0.0005, 0.0006, 0.0003, 0.0005, 0.0001, 0.0002]   
}

df = pd.DataFrame(data)

# # Print table in terminal
# print("\n=== Baseline vs Manipulated DOS2019 Metrics ===\n")
# print(df.to_string(index=False))

# # Function to plot metric comparisons
# def plot_metric(metric_name):
#     plt.figure(figsize=(12, 6))
#     plt.plot(df["Attack"], df[f"Baseline_{metric_name}"], marker='o', label=f"Baseline {metric_name}")
#     plt.plot(df["Attack"], df[f"Manipulated_{metric_name}"], marker='o', label=f"Manipulated {metric_name}")
#     plt.xticks(rotation=45, ha='right')
#     plt.ylabel(metric_name)
#     plt.title(f"{metric_name} Comparison: Baseline vs Manipulated")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

# # Plot all metrics
# for metric in ["Accuracy", "F1", "TPR", "TNR", "FPR", "FNR"]:
#     plot_metric(metric)



# Metrics to plot
metrics = ["Accuracy", "F1", "TPR", "TNR", "FPR", "FNR"]

# Create subplots
fig, axes = plt.subplots(nrows=len(metrics), ncols=1, figsize=(12, 18))
fig.subplots_adjust(hspace=0.4)

for ax, metric in zip(axes, metrics):
    ax.plot(df["Attack"], df[f"Baseline_{metric}"], marker='o', label=f"Baseline {metric}")
    ax.plot(df["Attack"], df[f"Manipulated_{metric}"], marker='o', label=f"Manipulated {metric}")
    ax.set_title(f"{metric} Comparison", fontsize=12)
    ax.set_ylabel(metric)
    ax.set_xticklabels(df["Attack"], rotation=45, ha='right')
    ax.grid(True)
    ax.legend()

# Save PNG
plt.tight_layout()
plt.savefig("metrics_comparison.png", dpi=300)
print("✅ Saved: metrics_comparison.png")