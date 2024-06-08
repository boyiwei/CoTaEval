import pandas as pd
import numpy as np


def compute_win_rate(df, metric):
    values = df[metric]
    win_rates = []
    for value in values:
        if 'Levenshtein Distance' in metric:
            win_rate = sum(value > other_value for other_value in values) / (len(values) - 1)
        else:
            win_rate = sum(value < other_value for other_value in values) / (len(values) - 1)
        win_rates.append(win_rate)
    return win_rates

def compute_min_indicator(df, metric):
    if 'Levenshtein Distance' in metric:
        max_value = df[metric].max()
        return [1 if value == max_value else 0 for value in df[metric]] 
    else:
        min_value = df[metric].min()
        return [1 if value == min_value else 0 for value in df[metric]] 
    

# NewsQA RAG setting in 7b model
file_list_llama2_7b_chat_news_rag = {"vanilla": "res/output_res_processed/newsqa_low_ppl_comp_llama2-7b-chat-hf_context_len_200_completion_len_200_intervention_none_no_context_False_processed.csv",
             "sys_prompt": "res/output_res_processed/newsqa_low_ppl_comp_llama2-7b-chat-hf_context_len_200_completion_len_200_intervention_sys_prompt-bing_no_context_False_processed.csv",
             "top_k": "res/output_res_processed/newsqa_low_ppl_comp_llama2-7b-chat-hf_context_len_200_completion_len_200_intervention_top_k_3.0_no_context_False_processed.csv",
             "mem_free": "res/output_res_processed/newsqa_low_ppl_comp_llama2-7b-chat-hf_context_len_200_completion_len_200_intervention_mem_free_tokenized-consecutive_6_no_context_False_processed.csv"
}

file_list_llama2_70b_chat_news_rag = {"vanilla": "res/output_res_processed/newsqa_low_ppl_comp_llama2-70b-chat-hf_context_len_200_completion_len_200_intervention_none_no_context_False_processed.csv",
             "sys_prompt": "res/output_res_processed/newsqa_low_ppl_comp_llama2-70b-chat-hf_context_len_200_completion_len_200_intervention_sys_prompt-bing_no_context_False_processed.csv",
             "top_k": "res/output_res_processed/newsqa_low_ppl_comp_llama2-70b-chat-hf_context_len_200_completion_len_200_intervention_top_k_3.0_no_context_False_processed.csv",
             "mem_free": "res/output_res_processed/newsqa_low_ppl_comp_llama2-70b-chat-hf_context_len_200_completion_len_200_intervention_mem_free_tokenized-consecutive_6_no_context_False_processed.csv",
}

file_list_llama2_7b_chat_books_rag = {"vanilla": "res/output_res_processed/booksum_low_ppl_comp_llama2-7b-chat-hf_context_len_200_completion_len_200_intervention_none_no_context_False_processed.csv",
             "sys_prompt": "res/output_res_processed/booksum_low_ppl_comp_llama2-7b-chat-hf_context_len_200_completion_len_200_intervention_sys_prompt-bing_no_context_False_processed.csv",
             "top_k": "res/output_res_processed/booksum_low_ppl_comp_llama2-7b-chat-hf_context_len_200_completion_len_200_intervention_top_k_3.0_no_context_False_processed.csv",
             "mem_free": "res/output_res_processed/booksum_low_ppl_comp_llama2-7b-chat-hf_context_len_200_completion_len_200_intervention_mem_free_tokenized-consecutive_6_no_context_False_processed.csv"
}

file_list_llama2_70b_chat_books_rag = {"vanilla": "res/output_res_processed/booksum_low_ppl_comp_llama2-70b-chat-hf_context_len_200_completion_len_200_intervention_none_no_context_False_processed.csv",
             "sys_prompt": "res/output_res_processed/booksum_low_ppl_comp_llama2-70b-chat-hf_context_len_200_completion_len_200_intervention_sys_prompt-bing_no_context_False_processed.csv",
             "top_k": "res/output_res_processed/booksum_low_ppl_comp_llama2-70b-chat-hf_context_len_200_completion_len_200_intervention_top_k_3.0_no_context_False_processed.csv",
             "mem_free": "res/output_res_processed/booksum_low_ppl_comp_llama2-70b-chat-hf_context_len_200_completion_len_200_intervention_mem_free_tokenized-consecutive_6_no_context_False_processed.csv"
}

file_list_llama2_7b_chat_news_memorization = {"vanilla": "res/output_res_processed/newsqa_low_ppl_comp_llama2-7b-chat-hf_newsqa_STEP2000_context_len_200_completion_len_200_intervention_none_no_context_True_processed.csv",
                    "sys_prompt": "res/output_res_processed/newsqa_low_ppl_comp_llama2-7b-chat-hf_newsqa_STEP2000_context_len_200_completion_len_200_intervention_sys_prompt-bing_no_context_True_processed.csv",
                    "top_k": "res/output_res_processed/newsqa_low_ppl_comp_llama2-7b-chat-hf_newsqa_STEP2000_context_len_200_completion_len_200_intervention_top_k_3.0_no_context_True_processed.csv",
                    "mem_free": "res/output_res_processed/newsqa_low_ppl_comp_llama2-7b-chat-hf_newsqa_STEP2000_context_len_200_completion_len_200_intervention_mem_free_tokenized-consecutive_6_no_context_True_processed.csv",
                    "r_cad": "res/output_res_processed/newsqa_low_ppl_comp_llama2-7b-chat-hf_newsqa_STEP2000_context_len_200_completion_len_200_intervention_cad_1.0_no_context_True_processed.csv",
                    "grad_ascent": "res/output_res_processed/newsqa_low_ppl_comp_llama2-7b-chat-hf_newsqa_STEP2000_grad_ascent1000_1.5e-06_62_context_len_200_completion_len_200_intervention_none_no_context_True_processed.csv",
                    "grad_diff": "res/output_res_processed/newsqa_low_ppl_comp_llama2-7b-chat-hf_newsqa_STEP2000_grad_diff1000_3e-06_62_context_len_200_completion_len_200_intervention_none_no_context_True_processed.csv",
                    "KL": "res/output_res_processed/newsqa_low_ppl_comp_llama2-7b-chat-hf_newsqa_STEP2000_KL1000_2e-06_62_context_len_200_completion_len_200_intervention_none_no_context_True_processed.csv",
                    "idk": "res/output_res_processed/newsqa_low_ppl_comp_llama2-7b-chat-hf_newsqa_STEP2000_idk1000_5e-05_248_context_len_200_completion_len_200_intervention_none_no_context_True_processed.csv"}


def win_rate_rag(file_list):
    res_vanilla = pd.read_csv(file_list['vanilla'])
    res_sys_prompt = pd.read_csv(file_list['sys_prompt'])
    res_top_k = pd.read_csv(file_list['top_k'])
    res_mem_free = pd.read_csv(file_list['mem_free'])
    metrics = ['rouge1', 'rougeL', 'semantic_sim', 'LCS(character)', 'LCS(word)', 'ACS(word)', 'Levenshtein Distance', 'Minhash Similarity']

    a = [0, 0, 0, 0, 0, 0, 0, 0]

    vanilla_win_rate_count, sys_prompt_win_rate_count, top_k_win_rate_count, mem_free_win_rate_count = a, a, a, a
    for i in range(len(res_vanilla)):
        row_vanilla = res_vanilla.iloc[i]
        row_sys_prompt = res_sys_prompt.iloc[i]
        row_top_k = res_top_k.iloc[i]
        row_mem_free = res_mem_free.iloc[i]
        methods = ['vanilla', 'sys_prompt', 'top_k', 'mem_free']
        concatenated_df = pd.DataFrame([row_vanilla, row_sys_prompt, row_top_k, row_mem_free])
        concatenated_df['methods'] = methods
        min_indicator_df = pd.DataFrame()
        win_rate_df = pd.DataFrame()
        min_indicator_df['methods'] = concatenated_df['methods']
        for metric in metrics:
            min_indicator_df[metric + '_min_indicator'] = compute_min_indicator(concatenated_df, metric)
            win_rate_df[metric + '_win_rate'] = compute_win_rate(concatenated_df, metric)
        
        win_rate_count_rows_to_select = [metric + '_win_rate' for metric in metrics]
        win_rate_df = win_rate_df.reset_index(drop=True)
        
        win_rate_count_vanilla = win_rate_df.loc[0, win_rate_count_rows_to_select].tolist()
        vanilla_win_rate_count =  [x + y for x, y in zip(vanilla_win_rate_count, win_rate_count_vanilla)]
        
        win_rate_count_sys_prompt = win_rate_df.loc[1, win_rate_count_rows_to_select].tolist()
        sys_prompt_win_rate_count =  [x + y for x, y in zip(sys_prompt_win_rate_count, win_rate_count_sys_prompt)]
        
        win_rate_count_top_k = win_rate_df.loc[2, win_rate_count_rows_to_select].tolist()
        top_k_win_rate_count =  [x + y for x, y in zip(top_k_win_rate_count, win_rate_count_top_k)]
        
        win_rate_count_mem_free = win_rate_df.loc[3, win_rate_count_rows_to_select].tolist()
        mem_free_win_rate_count =  [x + y for x, y in zip(mem_free_win_rate_count, win_rate_count_mem_free)]

    total_numbers = len(metrics) * len(res_vanilla)

    data = {
        "methods": ["vanilla", "sys_prompt", "top_k", "mem_free"],
        "rouge1": [vanilla_win_rate_count[0]/len(res_vanilla), sys_prompt_win_rate_count[0]/len(res_sys_prompt), top_k_win_rate_count[0]/len(res_top_k), mem_free_win_rate_count[0]/len(res_mem_free)],
        "rougeL": [vanilla_win_rate_count[1]/len(res_vanilla), sys_prompt_win_rate_count[1]/len(res_sys_prompt), top_k_win_rate_count[1]/len(res_top_k), mem_free_win_rate_count[1]/len(res_mem_free)],
        "semantic_sim": [vanilla_win_rate_count[2]/len(res_vanilla), sys_prompt_win_rate_count[2]/len(res_sys_prompt), top_k_win_rate_count[2]/len(res_top_k), mem_free_win_rate_count[2]/len(res_mem_free)],
        "LCS(character)": [vanilla_win_rate_count[3]/len(res_vanilla), sys_prompt_win_rate_count[3]/len(res_sys_prompt), top_k_win_rate_count[3]/len(res_top_k), mem_free_win_rate_count[3]/len(res_mem_free)],
        "LCS(word)": [vanilla_win_rate_count[4]/len(res_vanilla), sys_prompt_win_rate_count[4]/len(res_sys_prompt), top_k_win_rate_count[4]/len(res_top_k), mem_free_win_rate_count[4]/len(res_mem_free)],
        "ACS(word)": [vanilla_win_rate_count[5]/len(res_vanilla), sys_prompt_win_rate_count[5]/len(res_sys_prompt), top_k_win_rate_count[5]/len(res_top_k), mem_free_win_rate_count[5]/len(res_mem_free)],
        "Levenshtein Distance": [vanilla_win_rate_count[6]/len(res_vanilla), sys_prompt_win_rate_count[6]/len(res_sys_prompt), top_k_win_rate_count[6]/len(res_top_k), mem_free_win_rate_count[6]/len(res_mem_free)],
        "Minhash Similarity": [vanilla_win_rate_count[7]/len(res_vanilla), sys_prompt_win_rate_count[7]/len(res_sys_prompt), top_k_win_rate_count[7]/len(res_top_k), mem_free_win_rate_count[7]/len(res_mem_free)],
        "average": [np.sum(vanilla_win_rate_count) / total_numbers, np.sum(sys_prompt_win_rate_count) / total_numbers, np.sum(top_k_win_rate_count) / total_numbers, np.sum(mem_free_win_rate_count) / total_numbers]
    }

    # Create DataFrames
    df = pd.DataFrame(data)
    df.to_csv('win_rate_rag.csv', index=False)


def win_rate_memorization(file_list):
    res_vanilla = pd.read_csv(file_list['vanilla'])
    res_sys_prompt = pd.read_csv(file_list['sys_prompt'])
    res_top_k = pd.read_csv(file_list['top_k'])
    res_mem_free = pd.read_csv(file_list['mem_free'])
    res_r_cad = pd.read_csv(file_list['r_cad'])
    res_grad_ascent = pd.read_csv(file_list['grad_ascent'])
    res_grad_diff = pd.read_csv(file_list['grad_diff'])
    res_KL = pd.read_csv(file_list['KL'])
    res_idk = pd.read_csv(file_list['idk'])

    metrics = ['rouge1', 'rougeL', 'semantic_sim', 'LCS(character)', 'LCS(word)', 'ACS(word)', 'Levenshtein Distance', 'Minhash Similarity']

    a = [0, 0, 0, 0, 0, 0, 0, 0]

    vanilla_win_rate_count, sys_prompt_win_rate_count, top_k_win_rate_count, mem_free_win_rate_count, r_cad_win_rate_count, grad_ascent_win_rate_count, grad_diff_win_rate_count, KL_win_rate_count, idk_win_rate_count= a, a, a, a, a, a, a, a, a

    for i in range(len(res_vanilla)):
        row_vanilla = res_vanilla.iloc[i]
        row_sys_prompt = res_sys_prompt.iloc[i]
        row_top_k = res_top_k.iloc[i]
        row_mem_free = res_mem_free.iloc[i]
        row_r_cad = res_r_cad.iloc[i]
        row_grad_ascent = res_grad_ascent.iloc[i]
        row_grad_diff = res_grad_diff.iloc[i]
        row_KL = res_KL.iloc[i]
        row_idk = res_idk.iloc[i]
        
        methods = ['vanilla', 'sys_prompt', 'top_k', 'mem_free', 'r_cad', 'grad_ascent', 'grad_diff', 'KL', 'idk']
        concatenated_df = pd.DataFrame([row_vanilla, row_sys_prompt, row_top_k, row_mem_free, row_r_cad, row_grad_ascent, row_grad_diff, row_KL, row_idk])
        concatenated_df['methods'] = methods
        min_indicator_df = pd.DataFrame()
        win_rate_df = pd.DataFrame()
        min_indicator_df['methods'] = concatenated_df['methods']
        for metric in metrics:
            min_indicator_df[metric + '_min_indicator'] = compute_min_indicator(concatenated_df, metric)
            win_rate_df[metric + '_win_rate'] = compute_win_rate(concatenated_df, metric)
        
        win_rate_count_rows_to_select = [metric + '_win_rate' for metric in metrics]
        win_rate_df = win_rate_df.reset_index(drop=True)
        
        win_rate_count_vanilla = win_rate_df.loc[0, win_rate_count_rows_to_select].tolist()
        vanilla_win_rate_count =  [x + y for x, y in zip(vanilla_win_rate_count, win_rate_count_vanilla)]
        
        win_rate_count_sys_prompt = win_rate_df.loc[1, win_rate_count_rows_to_select].tolist()
        sys_prompt_win_rate_count =  [x + y for x, y in zip(sys_prompt_win_rate_count, win_rate_count_sys_prompt)]
        
        win_rate_count_top_k = win_rate_df.loc[2, win_rate_count_rows_to_select].tolist()
        top_k_win_rate_count =  [x + y for x, y in zip(top_k_win_rate_count, win_rate_count_top_k)]
        
        win_rate_count_mem_free = win_rate_df.loc[3, win_rate_count_rows_to_select].tolist()
        mem_free_win_rate_count =  [x + y for x, y in zip(mem_free_win_rate_count, win_rate_count_mem_free)]
        
        win_rate_count_r_cad = win_rate_df.loc[4, win_rate_count_rows_to_select].tolist()
        r_cad_win_rate_count =  [x + y for x, y in zip(r_cad_win_rate_count, win_rate_count_r_cad)]
        
        win_rate_count_grad_ascent = win_rate_df.loc[5, win_rate_count_rows_to_select].tolist()
        grad_ascent_win_rate_count =  [x + y for x, y in zip(grad_ascent_win_rate_count, win_rate_count_grad_ascent)]
        
        win_rate_count_grad_diff = win_rate_df.loc[6, win_rate_count_rows_to_select].tolist()
        grad_diff_win_rate_count =  [x + y for x, y in zip(grad_diff_win_rate_count, win_rate_count_grad_diff)]
        
        win_rate_count_KL = win_rate_df.loc[7, win_rate_count_rows_to_select].tolist()
        KL_win_rate_count =  [x + y for x, y in zip(KL_win_rate_count, win_rate_count_KL)]
        
        win_rate_count_idk = win_rate_df.loc[8, win_rate_count_rows_to_select].tolist()
        idk_win_rate_count =  [x + y for x, y in zip(idk_win_rate_count, win_rate_count_idk)]

    total_numbers = len(metrics) * len(res_vanilla)

    data = {
        "methods": ["vanilla", "sys_prompt", "top_k", "mem_free", 'r_cad', 'grad_ascent', 'grad_diff', 'KL', 'idk'],
        "rouge1": [vanilla_win_rate_count[0]/len(res_vanilla), sys_prompt_win_rate_count[0]/len(res_sys_prompt), top_k_win_rate_count[0]/len(res_top_k), mem_free_win_rate_count[0]/len(res_mem_free), r_cad_win_rate_count[0]/len(res_r_cad), grad_ascent_win_rate_count[0]/len(res_grad_ascent), grad_diff_win_rate_count[0]/len(res_grad_diff), KL_win_rate_count[0]/len(res_KL), idk_win_rate_count[0]/len(res_idk)],
        "rougeL": [vanilla_win_rate_count[1]/len(res_vanilla), sys_prompt_win_rate_count[1]/len(res_sys_prompt), top_k_win_rate_count[1]/len(res_top_k), mem_free_win_rate_count[1]/len(res_mem_free), r_cad_win_rate_count[1]/len(res_r_cad), grad_ascent_win_rate_count[1]/len(res_grad_ascent), grad_diff_win_rate_count[1]/len(res_grad_diff), KL_win_rate_count[1]/len(res_KL), idk_win_rate_count[1]/len(res_idk)],
        "semantic_sim": [vanilla_win_rate_count[2]/len(res_vanilla), sys_prompt_win_rate_count[2]/len(res_sys_prompt), top_k_win_rate_count[2]/len(res_top_k), mem_free_win_rate_count[2]/len(res_mem_free), r_cad_win_rate_count[2]/len(res_r_cad), grad_ascent_win_rate_count[2]/len(res_grad_ascent), grad_diff_win_rate_count[2]/len(res_grad_diff), KL_win_rate_count[2]/len(res_KL), idk_win_rate_count[2]/len(res_idk)],
        "LCS(character)": [vanilla_win_rate_count[3]/len(res_vanilla), sys_prompt_win_rate_count[3]/len(res_sys_prompt), top_k_win_rate_count[3]/len(res_top_k), mem_free_win_rate_count[3]/len(res_mem_free), r_cad_win_rate_count[3]/len(res_r_cad), grad_ascent_win_rate_count[3]/len(res_grad_ascent), grad_diff_win_rate_count[3]/len(res_grad_diff), KL_win_rate_count[3]/len(res_KL), idk_win_rate_count[3]/len(res_idk)],
        "LCS(word)": [vanilla_win_rate_count[4]/len(res_vanilla), sys_prompt_win_rate_count[4]/len(res_sys_prompt), top_k_win_rate_count[4]/len(res_top_k), mem_free_win_rate_count[4]/len(res_mem_free), r_cad_win_rate_count[4]/len(res_r_cad), grad_ascent_win_rate_count[4]/len(res_grad_ascent), grad_diff_win_rate_count[4]/len(res_grad_diff), KL_win_rate_count[4]/len(res_KL), idk_win_rate_count[4]/len(res_idk)],
        "ACS(word)": [vanilla_win_rate_count[5]/len(res_vanilla), sys_prompt_win_rate_count[5]/len(res_sys_prompt), top_k_win_rate_count[5]/len(res_top_k), mem_free_win_rate_count[5]/len(res_mem_free), r_cad_win_rate_count[5]/len(res_r_cad), grad_ascent_win_rate_count[5]/len(res_grad_ascent), grad_diff_win_rate_count[5]/len(res_grad_diff), KL_win_rate_count[5]/len(res_KL), idk_win_rate_count[5]/len(res_idk)],
        "Levenshtein Distance": [vanilla_win_rate_count[6]/len(res_vanilla), sys_prompt_win_rate_count[6]/len(res_sys_prompt), top_k_win_rate_count[6]/len(res_top_k), mem_free_win_rate_count[6]/len(res_mem_free), r_cad_win_rate_count[6]/len(res_r_cad), grad_ascent_win_rate_count[6]/len(res_grad_ascent), grad_diff_win_rate_count[6]/len(res_grad_diff), KL_win_rate_count[6]/len(res_KL), idk_win_rate_count[6]/len(res_idk)],
        "Minhash Similarity": [vanilla_win_rate_count[7]/len(res_vanilla), sys_prompt_win_rate_count[7]/len(res_sys_prompt), top_k_win_rate_count[7]/len(res_top_k), mem_free_win_rate_count[7]/len(res_mem_free), r_cad_win_rate_count[7]/len(res_r_cad), grad_ascent_win_rate_count[7]/len(res_grad_ascent), grad_diff_win_rate_count[7]/len(res_grad_diff), KL_win_rate_count[7]/len(res_KL), idk_win_rate_count[7]/len(res_idk)],
        "average": [np.sum(vanilla_win_rate_count) / total_numbers, np.sum(sys_prompt_win_rate_count) / total_numbers, np.sum(top_k_win_rate_count) / total_numbers, np.sum(mem_free_win_rate_count) / total_numbers, np.sum(r_cad_win_rate_count) / total_numbers, np.sum(grad_ascent_win_rate_count) / total_numbers, np.sum(grad_diff_win_rate_count) / total_numbers, np.sum(KL_win_rate_count) / total_numbers, np.sum(idk_win_rate_count) / total_numbers]
    }
    # Create DataFrames
    df = pd.DataFrame(data)

    df.to_csv('win_rate_memorization.csv', index=False)
    

def main(args):
    data_type = args.data_type
    model_name = args.model_name
    scenario = args.scenario
    file_list = f"file_list_{model_name}_{data_type}_{scenario}"
    if args.scenario == "rag":
        win_rate_rag(file_list)
    elif args.scenario == "memorization":
        win_rate_memorization(file_list)
    else:
        print("Invalid scenario. Please choose either 'rag' or 'memorization'.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Compute win rate for different interventions')
    parser.add_argument('--data_type', type=str, help='Data type (e.g., news, books)')
    parser.add_argument('--model_name', type=str, help='Model name (e.g., llama2_7b_chat, llama2_70b_chat)')
    parser.add_argument('--scenario', type=str, help='Scenario (e.g., rag, memorization)')
    args = parser.parse_args()
    main(args)
        