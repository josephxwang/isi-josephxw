import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats

# df1 = pd.read_csv('wq-bert-base-uncased-1977-d.csv')
# df2 = pd.read_csv('wq-bert-base-uncased-1977-r.csv')

df1 = pd.read_csv('eval_bert_base_uncased_raw.csv')
df2 = pd.read_csv('eval_bert_base_uncased_news.csv')

# Merge on index
df = pd.merge(df1, df2, how='left', left_on='Unnamed: 0', right_on='Unnamed: 0')

# print(df.columns)
# print(len(df1), len(df2), len(df))

t_statistics = []

# t statistic for a p value of 0.05 for a one-sided paired t-test
t = 1.645

for _ in range(1000):
    # Randomly sample 1000 scores with replacement, bootstrapping
    sample = df.sample(n=1000, replace=True) # Sampling same sentences
    data_raw = sample['score_x']
    data_news = sample['score_y']

    # Perform paired t-test
    t_statistic, p_value = stats.ttest_rel(data_raw, data_news)
    
    t_statistics.append(t_statistic)

    # print(f't: {t_statistic}, p: {p_value}')
    
plt.hist(t_statistics, bins=30)
plt.axvline(x=t, color='red', linestyle='--')

# plt.title('Distribution of t statistics from paired t-test,\n1000 samples of n=1000 from WQ 1977-R (78.87) and 1977-D (83.24)')
plt.title('Distribution of t statistics from paired t-test,\n1000 samples of n=1000 from BERT-base-unc WQ Baseline (74.49) and WQ-News (45.71)')
plt.xlabel('t statistic')
plt.ylabel('Count')

plt.text(2, 3, 't=1.645 for p-value of 0.05', fontsize=10, color='red')

plt.show()