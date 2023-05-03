# %%
import pandas as pd
import easyocr
import cv2
import numpy as np
from pypdf import PdfReader

# %%
df_metadata = pd.read_csv('metadata.csv', header=0)

# %%
# initialize easyOCR model
ocr_reader = easyocr.Reader(['de', 'en'])

# %%
def process_image(path):
    # Read first page from PDF
    reader = PdfReader(path)
    scan = reader.pages[0].images[0].data
    scan = np.frombuffer(scan, np.uint8)
    scan = cv2.imdecode(scan, cv2.IMREAD_COLOR)
    scan = cv2.cvtColor(scan, cv2.COLOR_BGR2RGB)
    scan = cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY)

    # import matplotlib.pyplot as plt
    # plt.imshow(scan)
    # plt.show()

    # extract text
    # result = ocr_reader.readtext(scan)
    result = 0
    return result

# %%
results = dict()

for _, cur_row in df_metadata.head(10).iterrows():
    cur_path = cur_row['path']
    

    try:
        results[cur_path] = process_image(cur_path)
    except:
        print('Skipped {}'.format(cur_path))

# %%
