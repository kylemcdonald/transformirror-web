import pandas as pd
import matplotlib.pyplot as plt

# Load the data with ID and frame age columns
data = pd.read_csv('frame_age.csv', header=None, names=['ID', 'FrameAge'])

plt.figure(figsize=(16, 4), dpi=100)

# Create scatter plot with colors based on ID
colors = ['red' if id_val == 1 else 'blue' for id_val in data['ID']]
scatter = plt.scatter(range(len(data)), data['FrameAge'], c=colors, s=1)  # s=1 for 1-pixel points

# Add legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=6, label='ID=1'),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=6, label='ID=2')]
plt.legend(handles=legend_elements)

plt.xlabel('Index')
plt.ylabel('Frame Age')
plt.title('Frame Age Scatterplot (Red: ID=1, Blue: ID=2)')
plt.ylim(0, 1000)
plt.tight_layout()
plt.savefig('frame_age_scatter.png', dpi=100)
plt.show() 