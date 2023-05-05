# %%
import pandas as pd
import editdistance as edit
import seaborn as sns
import matplotlib.pyplot as plt

# %%
predictions = pd.read_csv('pred.csv', index_col=0, header=0)
predictions

# %%
def evaluate(ground_truth:str, predictions:list, fuzzy:bool=False):
    results = {}

    # preprocessing: lowercase all strings
    ground_truth = ground_truth.lower().replace(' ', '')
    predictions = [item.lower().replace(' ', '') for item in predictions]

    # ggf noch zahlen vor AM I und II Titeln entfernen
    for i, pred in enumerate(predictions):
        if fuzzy:
            results[i] = edit.eval(ground_truth, pred)
        else:
            results[i] = pred == ground_truth
    
    return results

# %%
predictions['pred_1st_result'] = None
predictions['pred_1st_result_fuzzy'] = None
predictions['pred_2nd_result'] = None
predictions['pred_2nd_result_fuzzy'] = None
predictions['pred_3rd_result'] = None
predictions['pred_3rd_result_fuzzy'] = None

for i, row in predictions.iterrows():
    result_fuzzy = evaluate(row['title'], row[['pred_1st', 'pred_2nd', 'pred_3rd']].values.tolist(), fuzzy=True)
    result_non_fuzzy = evaluate(row['title'], row[['pred_1st', 'pred_2nd', 'pred_3rd']].values.tolist())
    
    predictions.loc[i, 'pred_1st_result'] = result_non_fuzzy[0]
    predictions.loc[i, 'pred_1st_result_fuzzy'] = result_fuzzy[0]
    predictions.loc[i, 'pred_2nd_result'] = result_non_fuzzy[1]
    predictions.loc[i, 'pred_2nd_result_fuzzy'] = result_fuzzy[1]
    predictions.loc[i, 'pred_3rd_result'] = result_non_fuzzy[2]
    predictions.loc[i, 'pred_3rd_result_fuzzy'] = result_fuzzy[2]

predictions
# %%
predictions.to_csv('pred_with_acc.csv')

# %%
print('Accuracy 1st Prediction', predictions['pred_1st_result'].sum() / predictions.shape[0])
print('Accuracy 2nd Prediction', predictions['pred_2nd_result'].sum() / predictions.shape[0])
print('Accuracy 3rd Prediction', predictions['pred_3rd_result'].sum() / predictions.shape[0])

# %%
sns.violinplot(predictions[['pred_1st_result_fuzzy', 'pred_2nd_result_fuzzy', 'pred_3rd_result_fuzzy']], cut=0)
plt.xticks([0, 1, 2], ['1', '2', '3'])
plt.xlabel('Approach')
plt.ylabel('Edit Distance')
plt.savefig('edit_distance.pdf', bbox_inches='tight')
# %%
