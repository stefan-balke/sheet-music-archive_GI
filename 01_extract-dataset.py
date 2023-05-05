# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from pypdf import PdfReader
sns.set_theme(style='whitegrid')

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

# %% save
df_metadata.to_csv('metadata.csv')

# %%
# Dataset Statistics
df_metadata = pd.read_csv('metadata.csv', header=0)

print('number of files:', df_metadata.shape)
print('number of pages:', df_metadata['number_of_pages'].sum())
print('number of pieces:', df_metadata.groupby('title').ngroups)

df_metadata.groupby('title').size().sort_values(ascending=True).plot.barh(figsize=(10, 30), fontsize=4)
plt.title('Number of Files per Piece')
plt.show()

df_metadata.groupby('title').size().sort_values(ascending=True).tail(20).plot.barh(figsize=(10, 10))
plt.title('Number of Files per Piece (Top 20)')
plt.savefig('number_of_files_per_piece_top-20.pdf')
plt.show()

df_plot = df_metadata.groupby('title').size().sort_values(ascending=True).tail(20)
sns.set_color_codes('muted')
sns.barplot(y=df_plot.index, x=df_plot.values, color='b')
plt.xlabel('#Files per Musical Work')
plt.ylabel('Musical Work')
sns.despine(left=True, bottom=True)
plt.savefig('number_of_files_per_piece_top-20.pdf', bbox_inches='tight')
plt.show()


# number of files per instrument group
df_metadata.groupby('instrument_group').size().sort_values(ascending=True).plot.barh()
plt.title('Number of Files per Instrument Group')
plt.show()

# number of titles per instrument group
df_plot = (~df_metadata.groupby(['instrument_group', 'title']).size().unstack(0).isna()).sum().sort_values(ascending=True)
sns.set_color_codes('muted')
sns.barplot(y=df_plot.index, x=df_plot.values, color='b')
plt.xlabel('#Musical Works')
plt.ylabel('Instrument Group')
sns.despine(left=True, bottom=True)
plt.savefig('number_of_files_per_instrument_group.pdf', bbox_inches='tight')
plt.show()

# number of files per instrument
df_metadata.groupby('instrument_from_fn').size().sort_values(ascending=True).plot.barh(figsize=(10, 30), fontsize=4)
plt.show()

# number of files per instrument (Top 20)
df_plot = df_metadata.groupby('instrument_from_fn').size().sort_values(ascending=True).tail(20)
sns.set_color_codes('muted')
sns.barplot(y=df_plot.index, x=df_plot.values, color='b')
plt.xlabel('#Files')
plt.ylabel('Instrument')
sns.despine(left=True, bottom=True)
plt.savefig('number_of_files_per_instrument_top-20.pdf', bbox_inches='tight')
plt.show()

# number of pages
df_metadata.groupby('number_of_pages').size().plot.barh()
plt.show()

df_metadata.boxplot(column='number_of_pages', by='instrument_group', rot=90)
plt.show()


# %%
