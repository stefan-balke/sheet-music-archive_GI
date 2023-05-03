# %%
import pandas as pd
import easyocr
import cv2
import numpy as np
from pypdf import PdfReader
import fitz

# %%
df_metadata = pd.read_csv('metadata.csv', header=0).sample(100)

# %%
# initialize easyOCR model
ocr_reader = easyocr.Reader(['de', 'en'])

# %%
def pix2np(pix):
    # Source: https://stackoverflow.com/a/53066662
    im = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    im = np.ascontiguousarray(im[..., [2, 1, 0]])  # rgb to bgr
    return im


def process_image(path):
    # Read first page from PDF
    # reader = PdfReader(path)
    # scan = reader.pages[0].images[0].data
    scan = fitz.open(path)
    scan = scan[0].get_pixmap(dpi=300)
    scan = pix2np(scan)
    # scan = cv2.imdecode(scan, cv2.IMREAD_COLOR)
    scan = cv2.cvtColor(scan, cv2.COLOR_BGR2RGB)
    scan = cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY)

    # import matplotlib.pyplot as plt
    # plt.imshow(scan)
    # plt.show()

    # extract text
    result = ocr_reader.readtext(scan)

    return result

# %%
results = dict()

for _, cur_row in df_metadata.iterrows():
    cur_path = cur_row['path']  

    try:
        results[cur_path] = process_image(cur_path)
    except:
        print('Skipped {}'.format(cur_path))


# %%
