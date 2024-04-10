import os

directory = '/scratch/f006dg0/mcas-gmra/LANL_analysis/plots/'

for filename in os.listdir(directory):
    if filename.endswith(".txt_plt.png"):
        new_filename = filename.replace(".txt_plt.png", "_plt.png")
        os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
