# %%
import pandas as pd
import editdistance as edit
# %%
predictions = pd.read_csv("pred.csv", index_col=0, header=0)
predictions
# %%
for i, row in predictions.iterrows():
    print(row[["pred_1st", "pred_2nd", "title"]])
    print(row[["pred_1st", "pred_2nd"]].values.tolist())
    break
# %%
def evaluate(ground_truth:str, predictions:list, fuzzy:bool=False):
    results = {}

    #preprocessing:
    ground_truth = ground_truth.lower()

    predictions = [item.lower() for item in predictions]

    # ggf noch zahlen vor AM I und II Titeln entfernen

    for i, pred in enumerate(predictions):
        if fuzzy:
            results[i] = edit.eval(ground_truth, pred)
        else:
            results[i] = pred == ground_truth
    
    return results
# %%
predictions["pred_1st_result"] = None
predictions["pred_1st_result_fuzzy"] = None
predictions["pred_2nd_result"] = None
predictions["pred_2nd_result_fuzzy"] = None

for i, row in predictions.iterrows():
    result_fuzzy = evaluate(row["title"], row[["pred_1st", "pred_2nd"]].values.tolist(), fuzzy=True)
    result_non_fuzzy = evaluate(row["title"], row[["pred_1st", "pred_2nd"]].values.tolist())
    
    predictions.loc[i, "pred_1st_result"] = result_non_fuzzy[0]
    predictions.loc[i, "pred_1st_result_fuzzy"] = result_fuzzy[0]
    predictions.loc[i, "pred_2nd_result"] = result_non_fuzzy[1]
    predictions.loc[i, "pred_2nd_result_fuzzy"] = result_fuzzy[1]

predictions
# %%
predictions.to_csv("pred_with_acc.csv")
# %%
