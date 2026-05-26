import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
df1 = pd.read_csv("Unemployment in India.csv")
df2 = pd.read_csv("Unemployment_Rate_upto_11_2020.csv")

# Show first 5 rows
print(df1.head())
print(df2.head())

# Dataset information
print("Dataset 1 Info:")
print(df1.info())

print("\nDataset 2 Info:")
print(df2.info())

# Check missing values
print("\nMissing Values in Dataset 1:")
print(df1.isnull().sum())

print("\nMissing Values in Dataset 2:")
print(df2.isnull().sum())
# Remove extra spaces from column names
df1.columns = df1.columns.str.strip()
df2.columns = df2.columns.str.strip()

# Remove missing values from dataset 1
df1 = df1.dropna()

# Check if cleaning worked
print("\nMissing Values After Cleaning:")
print(df1.isnull().sum())

# Show cleaned column names
print("\nDataset 1 Columns:")
print(df1.columns)

print("\nDataset 2 Columns:")
print(df2.columns)
# Statistical summary of unemployment rate
print("\nDataset 1 Statistical Summary:")
print(df1["Estimated Unemployment Rate (%)"].describe())

print("\nDataset 2 Statistical Summary:")
print(df2["Estimated Unemployment Rate (%)"].describe())
# Convert Date column into proper datetime format
df1["Date"] = pd.to_datetime(df1["Date"], dayfirst=True)
df2["Date"] = pd.to_datetime(df2["Date"], dayfirst=True)

# Average unemployment by date
trend1 = df1.groupby("Date")["Estimated Unemployment Rate (%)"].mean()

# Average unemployment by date
trend1 = df1.groupby("Date")["Estimated Unemployment Rate (%)"].mean()

# Professional graph
plt.figure(figsize=(12,6))

plt.plot(
    trend1.index,
    trend1.values,
    marker="o",
    linewidth=2
)

plt.title("Unemployment Trend in India", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Average Unemployment Rate (%)", fontsize=12)

plt.grid(True)
plt.xticks(rotation=45)

plt.show()
# Create Covid period category
df2["Covid Period"] = df2["Date"].apply(
    lambda x: "Before Covid" if x < pd.Timestamp("2020-03-01") else "During Covid"
)

# Average unemployment during each period
covid_impact = df2.groupby("Covid Period")[
    "Estimated Unemployment Rate (%)"
].mean()

# Print values
print("\nCovid Impact on Unemployment:")
print(covid_impact)

# Bar graph
plt.figure(figsize=(8,5))

plt.bar(
    covid_impact.index,
    covid_impact.values
)

plt.title("Covid-19 Impact on Unemployment in India")
plt.xlabel("Period")
plt.ylabel("Average Unemployment Rate (%)")

plt.show()
# Average unemployment by state
state_unemployment = df1.groupby("Region")[
    "Estimated Unemployment Rate (%)"
].mean()

# Top 10 states
top_states = state_unemployment.sort_values(
    ascending=False
).head(10)

print("\nTop 10 States with Highest Unemployment:")
print(top_states)

# Bar graph
plt.figure(figsize=(12,6))

plt.bar(
    top_states.index,
    top_states.values
)

plt.title("Top 10 States with Highest Unemployment")
plt.xlabel("State")
plt.ylabel("Average Unemployment Rate (%)")

plt.xticks(rotation=45)

plt.show()
# Extract month from date
df1["Month"] = df1["Date"].dt.month_name()

# Monthly unemployment average
monthly_trend = df1.groupby("Month")[
    "Estimated Unemployment Rate (%)"
].mean()

print("\nMonthly Unemployment Trend:")
print(monthly_trend)

# Plot graph
plt.figure(figsize=(12,6))

plt.plot(
    monthly_trend.index,
    monthly_trend.values,
    marker="o"
)

plt.title("Monthly Unemployment Trend")
plt.xlabel("Month")
plt.ylabel("Average Unemployment Rate (%)")

plt.xticks(rotation=45)
plt.grid(True)

plt.show()
# Correlation Heatmap
plt.figure(figsize=(8,6))

sns.heatmap(
    df1.select_dtypes(include=["float64", "int64"]).corr(),
    annot=True
)

plt.title("Correlation Heatmap")
plt.show()