# Week 8 Assignment - Data Analysis and Visualization
# Dataset: Iris Dataset
#

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore Dataset

try:
    # Load dataset directly from sklearn
    iris = load_iris(as_frame=True)
    df = iris.frame
    df['species'] = iris.target_names[iris.target]

    print("✅ Dataset loaded successfully!\n")
except FileNotFoundError:
    print("❌ File not found. Please check your dataset path.")
    exit()

# Display first few rows
print("🔹 First 5 rows of the dataset:")
print(df.head(), "\n")

# Check structure and missing values
print("🔹 Dataset info:")
print(df.info(), "\n")

print("🔹 Missing values check:")
print(df.isnull().sum(), "\n")

# Clean dataset (no missing values in iris, but demo included)
df = df.dropna()


# Task 2: Basic Data Analysis
print("🔹 Descriptive statistics:")
print(df.describe(), "\n")

# Grouping example: mean of numerical columns per species
print("🔹 Mean measurements by species:")
grouped = df.groupby('species').mean()
print(grouped, "\n")

# Interesting finding example
print("💡 Observation: Iris-virginica generally has the largest petal and sepal sizes.\n")

# Task 3: Data Visualization

# 1. Line chart - simulate trend (using sepal length as "time series")
plt.figure(figsize=(8, 5))
plt.plot(df.index, df['sepal length (cm)'], label="Sepal Length")
plt.title("Line Chart: Sepal Length Across Samples")
plt.xlabel("Sample Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.grid(True)
plt.show()

# 2. Bar chart - average petal length per species
plt.figure(figsize=(8, 5))
sns.barplot(x="species", y="petal length (cm)", data=df, ci=None, palette="viridis")
plt.title("Bar Chart: Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.show()

# 3. Histogram - distribution of sepal width
plt.figure(figsize=(8, 5))
plt.hist(df['sepal width (cm)'], bins=20, color="skyblue", edgecolor="black")
plt.title("Histogram: Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter plot - Sepal length vs. Petal length
plt.figure(figsize=(8, 5))
sns.scatterplot(
    x="sepal length (cm)",
    y="petal length (cm)",
    hue="species",
    data=df,
    palette="deep"
)
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()

print(" All tasks completed successfully!")
