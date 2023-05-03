# %%
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pypdf import PdfReader

# %%
path_data = Path('data') 
# %%
# Gather dataset
# Iterate through the whole directory and save the filename and the first and second directory-instrumentname

df_metadata = []

for cur_path in path_data.rglob('*.pdf'):
    cur_file = dict()
    cur_file['path'] = str(cur_path)
    cur_file['fn'] = cur_path.name
    cur_path_parts = cur_path.parts
    cur_file['instrument_group'] = cur_path_parts[1]

    if len(cur_path_parts) >= 5:
        cur_file['instrument_voice'] = cur_path_parts[2]
    else:
        cur_file['instrument_voice'] = ''

    # open pdf file
    reader = PdfReader(cur_path)
    cur_file['number_of_pages'] = len(reader.pages)

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

# %%
# Data Cleanup

df_metadata = df_metadata.drop(index=df_metadata[df_metadata['instrument_group'] == 'Partitur'].index)
df_metadata = df_metadata[~df_metadata['path'].str.contains('Marsch CD Walter Behrens')]
df_metadata = df_metadata[~df_metadata['path'].str.contains('Bläserbuch zum Gotteslob')]
df_metadata = df_metadata.drop_duplicates(subset=['title', 'fn'])

# %%
# Dataset Statistics

print('number of files:', df_metadata.shape)

print('number of pieces:', df_metadata.groupby('title').ngroups)

print('number of files per piece')
df_metadata.groupby('title').size().sort_values(ascending=True).plot.barh(figsize=(10, 30), fontsize=4)
plt.show()

# number of files per instrument group
df_metadata.groupby('instrument_group').size().sort_values(ascending=True).plot.barh()
plt.show()

# number of titles per instrument group
stats_titles = (~df_metadata.groupby(['instrument_group', 'title']).size().unstack(0).isna()).sum()
stats_titles.sort_values(ascending=True).plot.barh()
plt.show()

# number of files per instrument
df_metadata.groupby('instrument_from_fn').size().sort_values(ascending=True).plot.barh(figsize=(10, 30), fontsize=4)
plt.show()

# number of pages
df_metadata.groupby('number_of_pages').size().plot.barh()
plt.show()

df_metadata.boxplot(column='number_of_pages', by='instrument_group', rot=90)
plt.show()


# %%
df_metadata.to_csv('metadata.csv')

