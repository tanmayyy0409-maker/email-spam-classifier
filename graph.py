import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")

# Count Spam and Ham emails
counts = df["label"].value_counts()

# Create Bar Chart
plt.figure(figsize=(6,4))
counts.plot(kind="bar")

plt.title("Spam vs Ham Emails")
plt.xlabel("Email Type")
plt.ylabel("Number of Emails")

# Save the graph
plt.savefig("spam_ham_chart.png")

plt.show()