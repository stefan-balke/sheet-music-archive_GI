# %%
import os
import pandas as pd

# %%
my_dir = "/Users/stefan/Dropbox/MZE Noten"

# %% [markdown]
# # First Ideas
# Iterate through the whole directory and save the filename and the first and second directory-instrumentname

# %%
pdf_files = [("erste Instrumentenebene", "zweite Instrumentenebene", "Dateiname")]
for dirpath, dirnames, filenames in os.walk(my_dir):
    for filename in filenames:
        if filename.endswith('.pdf'):
            print(dirpath)
            try:
                pdf_files.append((dirpath.split(os.path.sep)[5], dirpath.split(os.path.sep)[6], filename))
            except:
                pdf_files.append((dirpath.split(os.path.sep)[5], None, filename))

pdf_files

# %% [markdown]
# make list to pandas dataframe and extract the expected name of the instrument from the filename

# %%
def extract_title(filename):
    if " - " in filename:
        return filename[filename.rfind(" - ") + 3:]
    else:
        return ""

pdf_files_pd = pd.DataFrame(pdf_files[1:], columns=pdf_files[0])
pdf_files_pd["Dateiname_Instrument"] = pdf_files_pd["Dateiname"].apply(extract_title)
pdf_files_pd

# %% [markdown]
# iterate over the results

# %%
for ix, row in pdf_files_pd.iterrows():
    print(row["Dateiname_Instrument"], "|||", row["erste Instrumentenebene"], " ----- ", row["zweite Instrumentenebene"])

# %% [markdown]
# ## get unique instrument names
# 
# bit of work to do still....

# %%
pd.set_option("display.max_rows", None)
print(pdf_files_pd["Dateiname_Instrument"].value_counts(dropna=False))
pd.set_option("display.max_rows", 10)



# %%
