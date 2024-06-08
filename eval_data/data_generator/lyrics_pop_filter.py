from datasets import load_dataset, load_from_disk
import numpy as np
import seaborn as sns
from rouge_score import rouge_scorer, scoring
import evaluate
import pandas as pd
import re
from dataportraits import code_proc


def remove_prefix_suffix(text_list):
    pattern = r".*Lyrics\s*(\[[\w\s]+\])?"
    end_pattern = r"\d+Embed$"
    processed_texts = [re.sub(pattern, "", text, flags=re.DOTALL) for text in text_list]
    return [re.sub(end_pattern, "", text).rstrip() for text in processed_texts]

dataset = pd.read_csv('eval_data/lyrics_popular.csv')
num_samples = 200
dataset = dataset[:num_samples]
context_len = 100
completion_len = 200

lyrics = remove_prefix_suffix(dataset['song_lyrics'])
singers = dataset['singer']
titles = dataset['song_title']
prompts_nohint = []
prompts_hint = []
prompts_autocomplete = []
gts_autocomplete = []

for i in range(num_samples):
    text = lyrics[i]
    words = text.split(' ')
    prompt_autocomplete = ' '.join(words[:context_len])
    gt_autocomplete = ' '.join(words[context_len:context_len+completion_len]) 
    prompt_nohint = f"What are the lyrics of the song titled '{titles[i]}' produced by {singers[i]}?"
    prompt_hint = f"What are the lyrics of the song titled '{titles[i]}' produced by {singers[i]}? Hint: it starts with '{text[:100]}'"
    
    prompts_nohint.append(prompt_nohint)
    prompts_hint.append(prompt_hint)
    prompts_autocomplete.append(prompt_autocomplete)
    gts_autocomplete.append(gt_autocomplete)
    

# create a new csv file
df = pd.DataFrame({'title': titles, 'Singer': singers, 'prompt_nohint': prompts_nohint, 'prompt_hint': prompts_hint, 'prompt_autocomplete': prompts_autocomplete, 'gt_autocomplete': gts_autocomplete, 'lyrics': lyrics})
df.to_csv('eval_data/lyrics_popular_filtered.csv', index=False)
        

        

