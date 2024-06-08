import string
from datasketch import MinHash
import pandas as pd
import Levenshtein
import os
from tqdm import tqdm
import argparse

def process_sentence(sentence):
    # Convert to lower case
    sentence = sentence.lower()
    words = sentence.split()
    translator = str.maketrans('', '', string.punctuation)
    words = [word.translate(translator) for word in words]
    return words


def find_common_sequences(sentences1, sentence2, min_tokens=3):
    # Split the sentences into tokens
    max_lengths = []
    total_lengths = []
    common_sequences_all = []
    for sentence1 in sentences1:
        tokens1 = process_sentence(sentence1)
        tokens2 = process_sentence(sentence2)

        # Create a matrix to store the lengths of common subsequences
        dp = [[0] * (len(tokens2) + 1) for _ in range(len(tokens1) + 1)]

        # Fill the matrix
        for i in range(1, len(tokens1) + 1):
            for j in range(1, len(tokens2) + 1):
                if tokens1[i - 1] == tokens2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = 0  # Change to 0 to reset the count for non-continuous matches

        # Find all common sequences with their start and end indices
        sequences_with_indices = []
        for i in range(1, len(tokens1) + 1):
            for j in range(1, len(tokens2) + 1):
                if dp[i][j] >= min_tokens:
                    sequence = ' '.join(tokens1[i - dp[i][j]:i])
                    sequences_with_indices.append(((i - dp[i][j], i - 1), sequence))

        # Filter out overlapping sequences to keep only the longest ones
        longest_sequences = []
        sequences_with_indices.sort(key=lambda x: (x[0][0], -x[0][1]))  # Sort by start index and then by reverse end index
        last_end_index = -1
        longest_sequence_length = 0
        total_length = 0
        for indices, sequence in sequences_with_indices:
            if indices[0] > last_end_index:
                longest_sequences.append(sequence)
                last_end_index = indices[1]
                sequence_length = len(sequence.split())
                longest_sequence_length = max(longest_sequence_length, sequence_length)
                total_length += sequence_length

        max_lengths.append(longest_sequence_length)
        total_lengths.append(total_length)
        common_sequences_all.append(longest_sequences)
        
    top_overlapping_id = total_lengths.index(max(total_lengths))

    return top_overlapping_id, common_sequences_all[top_overlapping_id], max_lengths[top_overlapping_id], total_lengths[top_overlapping_id]


def compute_min_hash_similarity(sentence1, sentence2):
    # Convert sentences to sets of 3-grams
    def shingles(sentence, k=3):
        return {sentence[i:i+k] for i in range(len(sentence) - k + 1)}
    
    shingles1 = shingles(sentence1)
    shingles2 = shingles(sentence2)

    # Initialize MinHash objects
    m1, m2 = MinHash(), MinHash()

    # Update MinHash objects with shingles
    for shingle in shingles1:
        m1.update(shingle.encode('utf8'))

    for shingle in shingles2:
        m2.update(shingle.encode('utf8'))

    # Compute Jaccard similarity
    similarity = m1.jaccard(m2)
    return similarity


def add_metrics(res):
    columns_to_remove = ['best_rouge1', 'best_rougeL', 'matching_sequence', 'best_rouge1_ids', 'best_rougeL_ids', 'best_verbatim_matching_ids', 'inference_time', 'total_length']
    res = res.drop(columns=columns_to_remove)
    res = res.rename(columns={'max_length': 'LCS(character)'})
    res = res.rename(columns={'best_verbatim_matching': 'output'})
    columns = list(res.columns)
    columns[3], columns[4], columns[5], columns[6] = columns[6], columns[3], columns[4], columns[5]
    res = res[columns]

    # Compute word level LCS and ACS, Levenstein distance and Minhash
    lcs_word, acs_word, levenshtein_distance, min_hash_similarity = [], [], [], []
    gts = res['gt']
    outputs = res['output']
    for j in range(len(gts)):
        output = outputs[j]
        gt = gts[j]
        _, _, max_length, total_length = find_common_sequences([output], gt)
        levenshtein_dist = Levenshtein.distance(output, gt)
        min_hash_sim = compute_min_hash_similarity(output, gt)
        lcs_word.append(max_length)
        acs_word.append(total_length)
        levenshtein_distance.append(levenshtein_dist)
        min_hash_similarity.append(min_hash_sim)
        
    res['LCS(word)'] = lcs_word
    res['ACS(word)'] = acs_word
    res['Levenshtein Distance'] = levenshtein_distance
    res['Minhash Similarity'] = min_hash_similarity
    
    return res


def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    file_name = args.file_name

    if file_name.endswith(".csv"):
        input_file_path = os.path.join(input_dir, file_name)
        output_file_path = os.path.join(output_dir, file_name.replace(".csv", "_processed.csv"))
        
        # Read the CSV file
        res = pd.read_csv(input_file_path)
        # Add metrics to the DataFrame
        res_processed = add_metrics(res)
        
        # Save the processed DataFrame to a new CSV file
        res_processed.to_csv(output_file_path, index=False)

    print("Processing completed.")
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="res/output_res", help="Directory containing the input CSV files")
    parser.add_argument('--output_dir', type=str, default="res/output_res_processed", help="Directory to save the processed CSV files")
    parser.add_argument('--file_name', type=str, default="newsqa_low_ppl_comp_llama2-7b-chat-hf_context_len_200_completion_len_200_intervention_none_no_context_False.csv", help="Comma-separated list of file names to process")
    args = parser.parse_args()
    main(args)
    

