import pandas as pd
import matplotlib.pyplot as plt

# Load the data
ages = pd.read_csv('frame_age.csv', header=None).squeeze()

# Calculate 99.9th percentile
upper = ages.quantile(0.999)

plt.figure(figsize=(16, 4), dpi=100)
plt.scatter(range(len(ages)), ages, s=1)  # s=1 for 1-pixel points
plt.xlabel('Index')
plt.ylabel('Frame Age')
plt.title('Frame Age Scatterplot')
plt.ylim(0, upper)
plt.tight_layout()
plt.savefig('frame_age_scatter.png', dpi=100)
plt.show() 