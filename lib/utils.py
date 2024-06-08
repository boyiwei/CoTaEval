import dataportraits
import string

def process_sentence(sentence):
    # Convert to lower case
    lower_case_sentence = sentence.lower()
    # Split the sentence into words and concatenate them
    concatenated_sentence = ''.join(lower_case_sentence.split())
    # Remove all punctuations
    final_sentence = ''.join(char for char in concatenated_sentence if char not in string.punctuation)
    return final_sentence


def count_elements(lst):
    """
    count the number of the elements for multi-level list
    """
    count = 0
    for element in lst:
        if isinstance(element, list):
            count += count_elements(element)
        else:
            count += 1
    return count


def QUIP(num_matchings, input_sequence, n=50):
    """
    n: int, n-gram size
    """
    n_grams = []
    for i in range(len(input_sequence)):
        n_grams.append(len(input_sequence[i]) - n + 1)
    result = [num / ngram for num, ngram in zip(num_matchings, n_grams)]
    return result


def verbatim_matching_dataportrait(input_sequence, bloom_filter='lyrics.50-50.bf'):
    """
    verbatim_matching using data portraits
    
    input_sequence: list of strings
    dp_dataset: str, the name of the dataset used in the data portrait
    
    output: list of str, matched sequence
    """
    portrait = dataportraits.RedisBFSketch('localhost', 8899, bloom_filter, 50)
    report = portrait.contains_from_text(input_sequence)
    matching_sequences = [item['chains'] for item in report]
    matching_idxs = [item['chain_idxs'] for item in report]
    num_matchings = [count_elements(idxs) for idxs in matching_idxs]
    
    max_len_sublist = []
    for i in range(len(matching_idxs)):
        len_subsublist = [len(sublist) for sublist in matching_idxs[i] if isinstance(sublist, list)]
        max_len_subsublist = max(len_subsublist if len(len_subsublist) > 0 else [0])
        max_len_sublist.append(max_len_subsublist)

    QUIP_score = QUIP(num_matchings, input_sequence)
    # top_overlapping_id = num_matchings.index(max(num_matchings))
    # top_overlapping_id = max_len_sublist.index(max(max_len_sublist))
    if max(max_len_sublist) > 1:
        top_overlapping_id = max_len_sublist.index(max(max_len_sublist))
    else:
        top_overlapping_id = num_matchings.index(max(num_matchings))
    return top_overlapping_id, QUIP_score[top_overlapping_id], matching_sequences[top_overlapping_id]


def find_common_sequences(sentences1, sentence2, min_tokens=1):
    # Split the sentences into tokens
    max_lengths = []
    total_lengths = []
    common_sequences_all = []
    normalized_sentence2 = process_sentence(sentence2) 
    for sentence1 in sentences1:
        normalized_sentence1 = process_sentence(sentence1)
        # tokens1 = sentence1.split()
        # tokens2 = sentence2.split()
        tokens1 = list(normalized_sentence1)
        tokens2 = list(normalized_sentence2)

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



