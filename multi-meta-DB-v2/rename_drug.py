import numpy as np
import pandas as pd
unseen = np.load("570unseen.npy", allow_pickle=True)
seen = np.load("570seen.npy", allow_pickle=True)
drug_raw = pd.read_csv("df_570.csv")
seen_list = []
for cv in seen:
    print(len(cv))
    seen_list_cv = []
    for i in cv:
        row_id=list(drug_raw[drug_raw['name']==i]["index"])[0]
        seen_list_cv.append(row_id)
    seen_list.append(seen_list_cv)
unseen_list = []
for cv in unseen:
    unseen_list_cv = []
    for i in cv:
        row_id=list(drug_raw[drug_raw['name']==i]["index"])[0]
        unseen_list_cv.append(row_id)
    unseen_list.append(unseen_list_cv)
np.save("570seen_rename.npy", seen_list)
np.save("570unseen_rename.npy", unseen_list)