import pandas as pd
import matplotlib.pyplot as plt

# Create the data
data = {
    'category': ['Highly Profitable', 'Moderately Profitable', 'Break-even', 'Unprofitable'],
    'count': [150, 400, 250, 200]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Create the bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(df['category'], df['count'], color='#8884d8')

# Customize the chart
plt.title('Customer Profitability Distribution')
plt.xlabel('Category')
plt.ylabel('Count')

# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height}',
             ha='center', va='bottom')

# Add a grid
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Adjust layout and display the chart
plt.tight_layout()
plt.show()

# Optionally, save the chart as an image file
# plt.savefig('customer_profitability_distribution.png')
