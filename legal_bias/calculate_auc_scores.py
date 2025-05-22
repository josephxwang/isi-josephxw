import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import simps
from scipy.stats import linregress
# import statistics

bias_types = ['race-color', 'gender', 'socioeconomic', 'nationality', 'religion', 'age', 'sexual-orientation', 'physical-appearance', 'disability']

dfs = []

for bias_type in bias_types:
    dfs.append(pd.read_csv(f'scores/crows_pairs/crows-bert-base-uncased-{bias_type}.csv'))

xs = []
ys = []

for _ in range(len(dfs)):
    xs.append([])
    ys.append([])

# for epsilon in np.arange(0, 50, 0.1):

# Calculate base WinoQueer score
wq_scores = []
for i, df in enumerate(dfs):
    epsilon = 0
    score = 100
    while abs(score - 50) > 0.1:
        # How to calculate base score, where likelihood of more stereotypical sentence is higher than likelihood of less stereotypical sentence
        # stereo_count = len(df[df['sent_more_score'] - df['sent_less_score'] > epsilon])
        stereo_count = len(df[df['sent_more_score'] - df['sent_less_score'] > epsilon]) + 0.5 * len(df[abs(df['sent_more_score'] - df['sent_less_score']) < epsilon])
                
        # Option 2, use total number of sentence pairs in denominator, seems more robust (factors in "neutral" sentence pairs)
        score = stereo_count / len(df) * 100
        
        if epsilon == 0:
            wq_scores.append(score)
        
        xs[i].append(epsilon)
        ys[i].append(score)
        
        epsilon += 0.1

# Calculate area under curve, and geometric mean score
aucs = []
for i, df in enumerate(dfs):    
    # Calculate area under curve
    # Translate curve to have horizontal asymptote at 0 instead of 50
    y = [n - 50 for n in ys[i]]
    auc = simps(y, xs[i])
    aucs.append(auc)
    
    plt.scatter(wq_scores[i], auc, label=bias_types[i])

# m, b = np.polyfit(wq_scores, mean_scores, 1)

slope, intercept, r_value, p_value, std_err = linregress(wq_scores, aucs)
# line = slope * np.array(wq_scores) + intercept

plt.xlabel('CrowS-Pairs score')
plt.ylabel('Area under curve (AUC)')
plt.title('CrowS-Pairs score vs area under curve (AUC) for its 9 bias categories')
print(f'{r_value=:.2f}')

plt.legend()

plt.show()
