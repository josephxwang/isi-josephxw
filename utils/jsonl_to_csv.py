import jsonlines
import pandas as pd
from tqdm import tqdm

infile = 'food_comments'
outfile = 'food_comments.csv'

texts = []

with jsonlines.open(infile) as f: # had issues when I formatted jsonl file
    for object in tqdm(f):
        if object['body'] != "":
            texts.append(object['body'])
    
df = pd.DataFrame({'body': texts})
df.to_csv(outfile, index=False)