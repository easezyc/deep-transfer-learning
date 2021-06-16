import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from keras.preprocessing.sequence import pad_sequences
import seaborn as sns
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import pickle
import datetime
root_path = './'
random.seed(2021)


def save_pkl(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)

ad_feature = pd.read_csv('{}adFeature.csv'.format(root_path)).fillna('-1')
if os.path.exists('{}userFeature.csv'.format(root_path)):
    user_feature=pd.read_csv('{}userFeature.csv'.format(root_path))
else:
    userFeature_data = []
    with open('{}userFeature.data'.format(root_path), 'r') as f:
        for i, line in enumerate(f):
            line = line.strip().split('|')
            userFeature_dict = {}
            for each in line:
                each_list = each.split(' ')
                userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
            userFeature_data.append(userFeature_dict)
            if i % 100000 == 0:
                print(i)
        user_feature = pd.DataFrame(userFeature_data)
        user_feature.to_csv('{}userFeature.csv'.format(root_path), index=False)
        del userFeature_data
        
user_feature.uid = user_feature.uid.astype('float64')
data_seed = pd.read_csv('{}train.csv'.format(root_path))
data_nonseed = pd.read_csv('{}test1_truth.csv'.format(root_path), header = None, names = ['aid', 'uid', 'label'])
data_seed.loc[data_seed['label']==-1,'label'] = 0
data_nonseed.loc[data_nonseed['label']==-1,'label'] = 0
user_feature = user_feature.fillna('-1')

ID_col = 'aid'
item_col = ['advertiserId', 'campaignId', 'creativeSize', 
            'adCategoryId', 'productId', 'productType']
static_context_col = ['carrier', 'consumptionAbility', 'LBS', 'age',
               'education', 'gender', 'house']
dynamic_context_col = ['ct', 'interest1', 'interest2', 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']
dict_dynamic_length ={
    'ct': 4, 
    'interest1': 10, 
    'interest2': 10, 
    'interest3': 10, 
    'interest4': 10, 
    'interest5': 10, 
    'kw1': 5, 
    'kw2': 5, 
    'kw3': 5, 
    'topic1': 5, 
    'topic2': 5, 
    'topic3': 5
}
columns = [ID_col, 'label'] + item_col + static_context_col + dynamic_context_col

start_time = datetime.datetime.now()
dict_dynamic_set ={
    'ct': {-1}, 
    'interest1': {-1}, 
    'interest2': {-1}, 
    'interest3': {-1}, 
    'interest4': {-1}, 
    'interest5': {-1}, 
    'kw1': {-1}, 
    'kw2': {-1}, 
    'kw3': {-1}, 
    'topic1': {-1}, 
    'topic2': {-1}, 
    'topic3': {-1}
}
num = 0
for fea_name in dynamic_context_col:
    user_feature[fea_name + '_length'] = 0
    #user_feature[fea_name] = user_feature[fea_name].fillna('-1')
    
def extract_dynamic_feature(x):
    global num
    for fea_name in dynamic_context_col:
        mark = False
        if x[fea_name] == '-1':
            mark = True
        tmp = x[fea_name].split(' ')[:10]
        if mark == True:
            x[fea_name + '_length'] = 0
        else:
            x[fea_name + '_length'] = len(tmp)
        tmp = np.array(tmp).astype('int')
        x[fea_name] = tmp
    num += 1
    if num % 10000 == 0:
        print('processed ID: {}'.format(num))
        end_time = datetime.datetime.now()
        print('time:', (end_time - start_time).seconds)
    return x

user_feature = user_feature.apply(func=extract_dynamic_feature, axis=1)

save_pkl(user_feature, './processed_data/user_feature1.pkl')
print('feature1')

for fea_name in dynamic_context_col:
    tmp = pad_sequences(user_feature[fea_name], maxlen=dict_dynamic_length[fea_name], padding='post', truncating='post', value=-1)
    user_feature[fea_name] = tmp.tolist()
    print(fea_name + ' padding finished')
print(user_feature)

save_pkl(user_feature, './processed_data/user_feature2.pkl')
print('feature2')

for fea_name in dynamic_context_col:
    tmp_list = []
    for item in user_feature[fea_name]:
        tmp_list.extend(item)
    dict_dynamic_set[fea_name] = set(tmp_list)
    print(fea_name + ' set finished')
    
dict_map = {}
for fea_name in dynamic_context_col:
    dict_map[fea_name] = dict(zip(dict_dynamic_set[fea_name], range(len(dict_dynamic_set[fea_name]))))
    print('{}: {},'.format(fea_name, len(dict_dynamic_set[fea_name]) + 1))
    
def map_dict_func(x):
    for fea_name in dynamic_context_col:
        x[fea_name] = np.array([dict_map[fea_name][i] for i in x[fea_name]])
    return x
user_feature = user_feature.apply(func=map_dict_func, axis=1)

del dict_dynamic_set

save_pkl(user_feature, './processed_data/user_feature3.pkl')
print('feature3')

data_seed = pd.merge(data_seed, ad_feature, on='aid', how='left')
data_seed = pd.merge(data_seed, user_feature, on='uid', how='left')

data_nonseed = pd.merge(data_nonseed, ad_feature, on='aid', how='left')
data_nonseed = pd.merge(data_nonseed, user_feature, on='uid', how='left')
del user_feature
data_seed = data_seed[columns]
data_nonseed = data_nonseed[columns]

aid_seed_counts = data_seed[data_seed['label'] == 1].aid.value_counts()
print("总 aid 个数: {}".format(len(aid_seed_counts)))

counts = [0]
for i in range(1, 8):
    counts.append( sum(aid_seed_counts <= i * 2000) / len(aid_seed_counts) )
counts.append(1)
x = range(0,9)
plot_data = pd.DataFrame()
plot_data['x'] = x
plot_data['y'] = counts
plot_data['style'] = 1

plt.rc('font',family='Times New Roman')
fig, ax = plt.subplots()
sns.lineplot(data=plot_data, x="x", y="y", style = 'style', markers=True, legend=False)
plt.xticks([0, 1, 2, 3, 8], [0, 2, 4, 6, '>14'], fontsize=12)
plt.yticks([0, 0.6, 0.8, 0.9, 1], [0, '60%', '80%', '90%', '100%'], fontsize=12)
plt.xlabel('Number of seed users (x10$^3$)', fontsize=14)
plt.ylabel('Proportion of ads', fontsize=14)
plt.xlim(0, 8)
plt.ylim(0, 1)
ax.yaxis.grid(True) # Hide the horizontal gridlines
ax.xaxis.grid(True) # Show the vertical gridlines

seed_length = data_seed.shape[0]
columns = [ID_col] + item_col + static_context_col
print(seed_length)
all_data = pd.concat([data_seed, data_nonseed]).fillna('-1')
for feature in columns:
#     tmp_set = set(all_data[feature])
#     tmp_dict = dict(zip(tmp_set, range(len(tmp_set))))
#     all_data[feature] = all_data[feature].map(tmp_dict)
    try:
        all_data[feature] = LabelEncoder().fit_transform(all_data[feature].apply(int))
    except:
        all_data[feature] = LabelEncoder().fit_transform(all_data[feature])

for feature in columns:
    print('{}: {},'.format(feature, max(all_data[feature])))

aid_set = set(all_data.aid)
train_aid = set(random.sample(aid_set, int(len(aid_set) / 2)))
test_aid = aid_set - train_aid
print(test_aid)

aid_seed_counts = all_data.loc[(all_data['label'] == 1) & (all_data['aid'].isin(test_aid))].aid.value_counts()
test_cold_aid = set(aid_seed_counts[aid_seed_counts <= 2000].index)
test_hot_aid = test_aid - test_cold_aid

aid_seed_counts = data_seed.loc[(data_seed['label'] == 1) & (data_seed['aid'].isin(test_aid))].aid.value_counts()
print("总 aid 个数: {}".format(len(aid_seed_counts)))

counts = [0]
for i in range(1, 8):
    counts.append( sum(aid_seed_counts <= i * 2000) / len(aid_seed_counts) )
counts.append(1)
x = range(0,9)
plot_data = pd.DataFrame()
plot_data['x'] = x
plot_data['y'] = counts
plot_data['style'] = 1

plt.rc('font',family='Times New Roman')
fig, ax = plt.subplots()
sns.lineplot(data=plot_data, x="x", y="y", style = 'style', markers=True, legend=False)
plt.xticks([0, 1, 2, 3, 8], [0, 2, 4, 6, '>14'], fontsize=12)
plt.yticks([0, 0.6, 0.8, 0.9, 1], [0, '60%', '80%', '90%', '100%'], fontsize=12)
plt.xlabel('Number of seed users (x10$^3$)', fontsize=14)
plt.ylabel('Proportion of ads', fontsize=14)
plt.xlim(0, 8)
plt.ylim(0, 1)
ax.yaxis.grid(True) # Hide the horizontal gridlines
ax.xaxis.grid(True) # Show the vertical gridlines

data_stage1 = all_data.iloc[: seed_length]
data_stage2 = all_data.iloc[seed_length:]

train_data_stage1 =  data_stage1.loc[(data_stage1['aid'].isin(train_aid))]
train_data_stage2 =  data_stage2.loc[(data_stage2['aid'].isin(train_aid))]

test_hot_data_stage1 =  data_stage1.loc[(data_stage1['aid'].isin(test_hot_aid))]
test_hot_data_stage2 =  data_stage2.loc[(data_stage2['aid'].isin(test_hot_aid))]

test_cold_data_stage1 =  data_stage1.loc[(data_stage1['aid'].isin(test_cold_aid))]
test_cold_data_stage2 =  data_stage2.loc[(data_stage2['aid'].isin(test_cold_aid))]


save_pkl(train_data_stage1, './processed_data/train_stage1.pkl')
save_pkl(train_data_stage2, './processed_data/train_stage2.pkl')

save_pkl(test_hot_data_stage1, './processed_data/test_hot_stage1.pkl')
save_pkl(test_hot_data_stage2, './processed_data/test_hot_stage2.pkl')

save_pkl(test_cold_data_stage1, './processed_data/test_cold_stage1.pkl')
save_pkl(test_cold_data_stage2, './processed_data/test_cold_stage2.pkl')

