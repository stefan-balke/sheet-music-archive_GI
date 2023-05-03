# %%
import numpy as np
import pandas as pd
import fitz
import matplotlib.pyplot as plt

# %%
# Load data
df_metadata = pd.read_csv('metadata.csv', header=0)
ocr_results = np.load('ocr_results.npz', allow_pickle=True)['results'].flatten()[0]
sample_pieces = list(ocr_results.keys())
df_metadata = df_metadata[df_metadata['path'].isin(sample_pieces)]

# %%
def pix2np(pix):
    # Source: https://stackoverflow.com/a/53066662
    im = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    im = np.ascontiguousarray(im[..., [2, 1, 0]])  # rgb to bgr
    return im


def load_image(path):
    scan = fitz.open(path)
    scan = scan[0].get_pixmap(dpi=300)
    scan = pix2np(scan)

    return scan

# %%
# 1st title detector:
# Take biggest bounding box from OCR detector
predictions = []

for cur_piece in sample_pieces:
    cur_ocr_result = ocr_results[cur_piece]

    box_sizes = []

    for cur_box in cur_ocr_result:
        coords = cur_box[0]
        cur_box_size = (coords[1][0] - coords[0][0]) * (coords[3][1] - coords[0][1])
        box_sizes.append(cur_box_size)

    cur_prediction = dict()
    cur_prediction['path'] = cur_piece
    cur_prediction['pred_1st'] = cur_ocr_result[np.argmax(box_sizes)][-2]
    predictions.append(cur_prediction)

predictions = pd.DataFrame(predictions)

# %%
# 2nd title detector:
# Same as first but include gemoetric information
predictions_2 = []

for cur_piece in sample_pieces:
    cur_ocr_result = ocr_results[cur_piece]
    cur_img = load_image(cur_piece)

    box_sizes = []

    for cur_box in cur_ocr_result:
        coords = cur_box[0]
        cur_box_size = (coords[1][0] - coords[0][0]) * (coords[3][1] - coords[0][1])

        # check if box is within the first 10% of the document (from top)
        max_box_pos = np.max(np.array(coords)[:, 0] / cur_img.shape[0])
        if max_box_pos <= 0.2:
            box_sizes.append(cur_box_size)

    if len(box_sizes) > 0:
        cur_prediction = dict()
        cur_prediction['path'] = cur_piece
        cur_prediction['pred_2nd'] = cur_ocr_result[np.argmax(box_sizes)][-2]
        predictions_2.append(cur_prediction)

predictions_2 = pd.DataFrame(predictions_2)

# %%
# Join all predictions
predictions = predictions.merge(predictions_2, on='path')
predictions = predictions.merge(df_metadata, on='path')
predictions[['path', 'pred_1st', 'pred_2nd', 'title']].to_csv('pred.csv')
# %%
