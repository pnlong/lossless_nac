import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("/home/pnlong/lnac/flac_eval_results.csv")

# Filter for is_native_bit_depth == True
df_filtered = df[df["is_native_bit_depth"] == True]

# Create the line plot
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df_filtered,
    x="flac_compression_level",
    y="overall_compression_rate",
    hue="dataset",
    marker="o"
)

# Set labels and title
plt.xlabel("FLAC Compression Level")
plt.ylabel("Compression Rate")
plt.title("Comparing FLAC Compression Levels")

# Add legend
plt.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc="upper left")

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show the plot
plt.show()

