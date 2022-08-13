import pickle
import pandas as pd

# # load df
# with open('dataseds_dict.pickle', 'rb') as handle:
#     dataseds_dict = pickle.load(handle)
# df = pd.DataFrame.from_dict(dataseds_dict, orient="index")
#
# # filter by accuracy
# df = df[df["accuracy test"] >= 0.75]
#
# # convert source path to column
# df = df.reset_index()
# df.rename(columns = {'index':'path'}, inplace = True)
#
# # save to csv
# df.to_csv('data/all_datasets.csv')

df1 = pd.read_csv('data/all_datasets.csv', index_col=0)
columns = [col for col in df1]

for index, row in df1.iterrows():
    if index > 3:
        break
    for col in columns:
        print(f"value in {col} = {row[col]}")
