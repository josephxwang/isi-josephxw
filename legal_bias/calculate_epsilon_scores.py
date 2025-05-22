import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import simps
from scipy.stats import linregress
import statistics

models = {
    'BERT': ['BERT-base-uncased', 'BERT-base-cased', 'BERT-large-uncased', 'BERT-large-cased'],
    'RoBERTa': ['RoBERTa-base', 'RoBERTa-large'],
    'ALBERT': ['ALBERT-base', 'ALBERT-large', 'ALBERT-xxlarge'],
    'BART': ['BART-base', 'BART-large'],
    'gpt2': ['gpt2', 'gpt2-medium', 'gpt2-xl'],
    'BLOOM': ['BLOOM-560m', 'BLOOM-3b', 'BLOOM-7b1'],
    'OPT': ['OPT-350m', 'OPT-2b7', 'OPT-6b7']
}

model_families = ['BERT', 'RoBERTa', 'ALBERT', 'BART', 'gpt2', 'BLOOM', 'OPT']

all_masked_models = ['BERT-base-uncased', 'BERT-base-cased', 'BERT-large-uncased', 'BERT-large-cased', 'RoBERTa-base', 'RoBERTa-large', 'ALBERT-base', 'ALBERT-large', 'ALBERT-xxlarge', 'BART-base', 'BART-large']
all_autoregressive_models = ['gpt2', 'gpt2-medium', 'gpt2-xl', 'BLOOM-560m', 'BLOOM-3b', 'BLOOM-7b1', 'OPT-350m', 'OPT-2b7', 'OPT-6b7']

# base_models = ['BERT-base-uncased', 'RoBERTa-base', 'ALBERT-base', 'BART-base', 'gpt2', 'BLOOM-560m', 'OPT-350m']
# masked_models = ['BERT-base-uncased', 'RoBERTa-base', 'ALBERT-base', 'BART-base']

all_models = all_masked_models + ['BART-base', 'BART-large'] + all_autoregressive_models

# masked_dfs = []
# autoregressive_dfs = []

dfs = []

# df = pd.read_csv('scores/winoqueer/wq-bert-base-uncased.csv')

for i, model_family in enumerate(model_families):
    dfs.append([])
    for model in models[model_family]:
    # if model in masked_models:
        model = model.lower().replace('-', '_')
        dfs[i].append(pd.read_csv(f'scores/winoqueer/raw/eval_{model}_raw.csv'))
    # else:
    #     dfs.append(pd.read_csv(f'scores/winoqueer/wq-{model.lower()}.csv'))

# stereo_count = len(df[df['sent_more_score'] > df['sent_less_score']])
# wq_score = stereo_count / len(df) * 100

for index in range(len(dfs)):
    xs = []
    ys = []

    for _ in range(len(dfs[index])):
        xs.append([])
        ys.append([])

    # for epsilon in np.arange(0, 50, 0.1):
    
    # Calculate base WinoQueer score,
    wq_scores = []
    for i, df in enumerate(dfs[index]):
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

    # for i in range(len(xs)):
    #     plt.plot(xs[i], ys[i], label=models[i])
        
    # plt.xlabel('Epsilon')
    # plt.ylabel('Bias score')
    # plt.legend()

    # Calculate area under curve, and geometric mean score
    aucs = []
    mean_scores = []
    for i, df in enumerate(dfs[index]):    
        # Calculate area under curve
        # Translate curve to have horizontal asymptote at 0 instead of 50
        y = [n - 50 for n in ys[i]]
        auc = simps(y, xs[i])
        aucs.append(auc)

        # Calculate geometric mean
        # scores = []
        # epsilon = 0
        # dx = 4
        # for j in range(4):
        #     stereo_count = len(df[df['sent_more_score'] - df['sent_less_score'] > epsilon]) + 0.5 * len(df[abs(df['sent_more_score'] - df['sent_less_score']) < epsilon])
                    
        #     # Option 2, use total number of sentence pairs in denominator, seems more robust (factors in "neutral" sentence pairs)
        #     score = stereo_count / len(df) * 100
            
        #     if j == 0:
        #         wq_score = score
        #         wq_scores.append(wq_score)
            
        #     scores.append(score)    
        #     epsilon += dx
        # mean_score = statistics.geometric_mean(scores)
        # mean_scores.append(mean_score)

        # print(f'{models[i]}: {wq_score=:.2f}, {auc=:.2f}, {mean_score=:.2f}')

    # m, b = np.polyfit(wq_scores, mean_scores, 1)

    # slope, intercept, r_value, p_value, std_err = linregress(mean_scores, aucs)
    # line = slope * np.array(wq_scores) + intercept

    plt.scatter(wq_scores, aucs, label=model_families[index])
    # plt.plot(wq_scores, line, linestyle='dotted', color='red')

plt.xlabel('WinoQueer score')
plt.ylabel('Area under curve (AUC)')
plt.title('WinoQueer score vs area under curve (AUC) for 20 models from 7 different families')
# print(f'{r_value=:.2f}')

plt.legend()

plt.show()
