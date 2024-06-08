from datasets import load_dataset, load_from_disk
import numpy as np
import seaborn as sns
from rouge_score import rouge_scorer, scoring
import evaluate
import pandas as pd
import re
from dataportraits import code_proc
from tqdm import tqdm


dataset = load_from_disk('/scratch/gpfs/yangsibo/cache/datasets_lyrics')
lyrics = []
num_samples = 200

np.random.seed(1234)
indices = np.random.randint(len(dataset['train']), size=3000000)
testing_chunks = dataset['train'].select(indices)


extracted_sentences = []
filtered_items = []
braket_item = []
keywords = ['Produced by ', 'prod. by ', 
            # 'Chorus: ', 'Chorus (x2)', 'Chorus', 'Pre-Chorus', 'Chorus-', 'Chorus -', 'Chorus:', 'Chorus:', 'CHORUS:', 'CHORUS',
            # 'repeat 2X',
            # 'Intro: ', 'Intro', 'intro: ', 'intro', 'INTRO: ', 'INTRO', 'INTRO - ', 'INTRO -', 'INTRO:', 'INTRO:', 'INTRO -', 'INTRO - ','Spoken Intro',
            # 'Verse 1: ', 'Verse 1:', 'Verse 1 - ', 'Verse 1', 'VERSE 1', 'Verse: ', 'Verse', 'Verse One; ',
            # 'Sampled vocals',
            # 'Interlude: ',  'Hook: ', 'Hook',
            # ' sample', 'Sample'
            # 'words & music by ',
            # 'Talking',
            # 'Part I', 'Part II', 'Part III', 'Part IV', 'Part V', 'Part VI', 'Part VII', 'Part VIII', 'Part IX', 'Part X',
            ]
authors = []
titles = []
filtered_lyrics = []
tags = []
prompts_nohint = []
prompts_hint = []
prompts_autocomplete = []
gts_autocomplete = []
count = 0
context_len = 100
completion_len = 100

# filter the non-similar context and completion
rouge = evaluate.load('rouge')
examples = testing_chunks
key = 'lyrics'
thres = 0.1
kept = []

# filter the non-similar
for item in tqdm(examples):
    lyric = item['lyrics']
    if lyric.startswith('[Produced by') and '[' in lyric and ']' in lyric:
        # detect whether the sentence contains at least one pair of square brackets
        kept.append(item)
            

for e in tqdm(kept):
    # print(examples.index(e))
    words = e[key].split(' ')
    if len(words) < context_len + completion_len:
        continue
    prompt = ' '.join(words[:context_len])
    gt =  ' '.join(words[context_len:context_len+completion_len])
    r = rouge.compute(predictions=[prompt], references=[gt], use_aggregator=False) 
    if r['rougeL'][0] < thres:
        lyric = e['lyrics']
        match = re.search(r'\[(.*?)\]', lyric)
        if match:
            filtered_items.append(e)
            first_bracket_content = match.group(1)
            braket_item.append(first_bracket_content)
    print(len(filtered_items))
    

for i in range(len(braket_item)):
    if count == 200:
        break
    text = braket_item[i]
    for keyword in keywords:
        text = text.replace(keyword, '')
    cleaned_text = text.strip()
    if cleaned_text and  re.search('[a-zA-Z]', cleaned_text) and (len(cleaned_text) < 50):  # Check if the string is not empty and contains at least one letter
        authors.append(cleaned_text)
        filtered_lyric = filtered_items[i]['lyrics']
        filtered_lyric = re.sub(r'\[.*?\]', '', filtered_lyric, count=1).strip()
        filtered_lyric = code_proc.proc_code(filtered_lyric) # Delete the brackets, normalize the newlines, remove repeated newlines, compress repeated spaces, and then place everything on the same indentation level
        filtered_lyrics.append(filtered_lyric)
        words = filtered_lyric.split(' ')
        prompt_autocomplete = ' '.join(words[:context_len])
        gt_autocomplete = ' '.join(words[context_len:context_len+completion_len]) 
        prompt_nohint = f"What are the lyrics of the song titled '{filtered_items[i]['title']}' produced by {cleaned_text}?"    
        prompt_hint = f"What are the lyrics of the song titled '{filtered_items[i]['title']}' produced by {cleaned_text}? Hint: it starts with '{filtered_lyric[:100]}'"
            
        titles.append(filtered_items[i]['title'])
        tags.append(filtered_items[i]['tag'])
        prompts_nohint.append(prompt_nohint)
        prompts_hint.append(prompt_hint)
        prompts_autocomplete.append(prompt_autocomplete)
        gts_autocomplete.append(gt_autocomplete)
        count += 1
        print(count)

lyrics = [re.sub(r'\[.*?\]', '', text).strip() for text in lyrics]
# create a new csv file
df = pd.DataFrame({'title': titles, 'tag': tags,
                   'singer': authors, 'prompt_nohint': prompts_nohint, 'prompt_hint': prompts_hint,
                   'prompt_autocomplete': prompts_autocomplete, 'gt_autocomplete': gts_autocomplete,
                   'lyrics': filtered_lyrics})
df.to_csv('eval_data/lyrics_author.csv', index=False)
        

        

