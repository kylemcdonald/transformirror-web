import pandas as pd
import matplotlib.pyplot as plt

# Load the data
ages = pd.read_csv('frame_age.csv', header=None).squeeze()

plt.figure(figsize=(16, 4), dpi=100)
plt.scatter(range(len(ages)), ages, s=1)  # s=1 for 1-pixel points
plt.xlabel('Index')
plt.ylabel('Frame Age')
plt.title('Frame Age Scatterplot')
plt.ylim(0, 1000)
plt.tight_layout()
plt.savefig('frame_age_scatter.png', dpi=100)
plt.show() 

# same as the above but for processed_age.csv
ages = pd.read_csv('processed_age.csv', header=None).squeeze()

plt.figure(figsize=(16, 4), dpi=100)
plt.plot(range(len(ages))[:], ages[:], lw=0.5)  # s=1 for 1-pixel points
plt.xlabel('Index')
plt.ylabel('Frame Age')
plt.title('Processed Frame Age Scatterplot')
plt.ylim(0, 1000)
plt.tight_layout()
plt.savefig('processed_frame_age_scatter.png', dpi=100)
plt.show() 

# output_timing.csv has two columns: current_time and processed_timestamp
# plot them as a line plot
timing = pd.read_csv('output_timing.csv', header=None)

plt.figure(figsize=(16, 16), dpi=100)
plt.plot(timing.iloc[1000:1200, 0], timing.iloc[1000:1200, 1], lw=0.5)  # s=1 for 1-pixel points
plt.xlabel('output time')
plt.ylabel('input time')
# plt.ylim(0, 1000)
plt.tight_layout() 
plt.savefig('timing.png', dpi=100)
plt.show() 