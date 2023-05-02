# %%
import os
import pandas as pd
import matplotlib.pyplot as plt

# %%
my_dir = "/Users/stefan/Dropbox/MZE Noten"

# %% [markdown]
# # First Ideas
# Iterate through the whole directory and save the filename and the first and second directory-instrumentname

# %%
df_metadata = []

for dirpath, dirnames, filenames in os.walk(my_dir):
    for filename in filenames:
        if filename.endswith('.pdf'):
            cur_file = dict()
            cur_file['fn'] = filename
            cur_file['instrument_group'] = dirpath.split(os.path.sep)[5]

            try:
                # if it has two levels of instruments
                cur_file['instrument'] = dirpath.split(os.path.sep)[6]
            except:
                cur_file['instrument'] = ''

            cur_file['path'] = os.path.join(cur_file['instrument_group'], cur_file['instrument'], cur_file['fn'])

            df_metadata.append(cur_file)

df_metadata = pd.DataFrame(df_metadata)
print(df_metadata.head(10))

# %%
# make list to pandas dataframe and extract the expected name of the instrument from the filename

def extract_instrument(filename):
    if " - " in filename:
        return filename[(filename.rfind(' - ')) + 3:-4]
    else:
        return ""
    

def extract_title(filename):
    if " - " in filename:
        return filename[:(filename.rfind(' - '))]
    else:
        return ""

df_metadata['instrument_from_fn'] = df_metadata['fn'].apply(extract_instrument)
df_metadata['title'] = df_metadata['fn'].apply(extract_title)
df_metadata

# %%
# Data Cleanup
df_metadata[df_metadata['instrument_group'] == 'Partitur']

# partitur is not considered

# %%
# First Statistics

print('number of files:', df_metadata.shape)

print('number of pieces:', df_metadata.groupby('title').ngroups)

print('number of files per piece')
df_metadata.groupby('title').size().sort_values(ascending=True).plot.barh(figsize=(10, 30), fontsize=4)
plt.show()

# number of files per instrument group
df_metadata.groupby('instrument_group').size().sort_values(ascending=True).plot.barh()
plt.show()

# number of files per instrument
df_metadata.groupby('instrument_from_fn').size().sort_values(ascending=True).plot.barh(figsize=(10, 30), fontsize=4)

# %%
