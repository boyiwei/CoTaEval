import os
from tqdm import tqdm
from collections import OrderedDict
import numpy as np
from ipdb import set_trace as bp
import torch

class Evaluator:
    def __init__(self, metrics=None):
        if not metrics:
            metrics = ["rouge", "sacre_bleu", "bertscore", "factkb"]
        self.metrics = metrics
    
    def evaluate(self, predictions, references, documents, metrics=["rouge", "bertscore", "factkb"]):
        result_dict = OrderedDict()
        if "rouge" in metrics:
            rouge_dict = self.calculate_rouge(predictions, references)
            for k, v in rouge_dict.items():
                result_dict[k] = v
        if "sacre_bleu" in metrics:
            sacre_bleu_dict = self.calculate_sacrebleu(predictions, references)
            for k, v in sacre_bleu_dict.items():
                result_dict[k] = v
        if "bertscore" in metrics:
            bertscore_dict = self.calculate_bertscore(predictions, references)
            for k, v in bertscore_dict.items():
                result_dict[k] = v
        # if "factkb" in metrics:
            # result_dict["factkb"] = self.calculate_factkb(predictions, documents)

        for k, v in result_dict.items():
            print(f"{k} -> {v*100:.2f}")
        return result_dict

    def calculate_rouge(self, predictions, references):
        from torchmetrics.functional.text.rouge import rouge_score
        rouge_dict = rouge_score(preds=predictions, target=references)
        return {k: v.item() for k, v in rouge_dict.items()}

    def calculate_sacrebleu(self, predictions, references):
        from torchmetrics.functional.text import sacre_bleu_score
        score = sacre_bleu_score(preds=predictions, target=[[i] for i in references])
        return {"sacre_bleu": score.item()}

    def calculate_bertscore(self, predictions, references):
        import evaluate
        bertscore = evaluate.load("bertscore")
        bertscore_dict = bertscore.compute(predictions=predictions, references=references, model_type="roberta-large-mnli")
        res = {"bertscore_precision": np.mean(bertscore_dict["precision"]), "bertscore_recall": np.mean(bertscore_dict["recall"]), "bertscore_f1": np.mean(bertscore_dict["f1"])}
        return {k: v.item() for k, v in res.items()}

    def calculate_factkb(self, predictions, documents):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        from transformers import AutoTokenizer
        from transformers import AutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained("bunsenfeng/factkb")
        model = AutoModelForSequenceClassification.from_pretrained("bunsenfeng/factkb", torch_dtype=torch.float16)
        model = model.to(device)
        res = []
        for i in range(len(predictions)):
            input_pretokenized = f"{predictions[i]} {tokenizer.sep_token} {documents[i]}"
            tokenized_input = tokenizer(input_pretokenized, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                output = model(input_ids=tokenized_input.input_ids.to(device))
            logits = torch.softmax(output.logits, dim=1)  # (bz, 2)
            res.append(logits.squeeze()[-1].item())
        return np.mean(res)

def configure_model_loading(args):
    # TODO: add AWQ and GPTQ models

    device_name = torch.cuda.get_device_name()
    from transformers import AutoModelForCausalLM
    if "a100" in device_name or "a6000" in device_name:
        device_allow_flash_attention = True
    
    if args.loading_mode == "nf4":
        from transformers import BitsAndBytesConfig
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.float16,
            device_map="balanced",
            quantization_config=nf4_config,
            use_flash_attention_2=device_allow_flash_attention,
            trust_remote_code=True
        )
    elif args.loading_mode == "fp16":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.float16,
            device_map="balanced",
            trust_remote_code=True
        )
    return model


def calculate_rouge(predictions, references):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    result_dict = {"rouge1-pre": 0., "rouge1-rec": 0., "rouge1-f1": 0., "rouge2-pre": 0., "rouge2-rec": 0., "rouge2-f1": 0., "rougeL-pre": 0., "rougeL-rec": 0., "rougeL-f1": 0., }
    for idx in range(len(predictions)):
        scores = scorer.score(predictions[idx], references[idx])
        result_dict["rouge1-pre"] += scores["rouge1"][0]
        result_dict["rouge1-rec"] += scores["rouge1"][1]
        result_dict["rouge1-f1"] += scores["rouge1"][2]
        result_dict["rouge2-pre"] += scores["rouge2"][0]
        result_dict["rouge2-rec"] += scores["rouge2"][1]
        result_dict["rouge2-f1"] += scores["rouge2"][2]
        result_dict["rougeL-pre"] += scores["rougeL"][0]
        result_dict["rougeL-rec"] += scores["rougeL"][1]
        result_dict["rougeL-f1"] += scores["rougeL"][2]
    for k, v in result_dict.items():
        print(f"{k} -> {v/len(predictions)*100:.2f}")
    return result_dict


def load_dataset_qfs(dataset):
    input_dir = f"/uusoc/exports/scratch/brutusxu/decoding/datasets/{dataset}/"
    train, validation, test = [], [], []
    with open(os.path.join(input_dir, "train_triples.tsv"), "r") as fin:
        for i, line in enumerate(fin):
            query, document, summary = line.strip().split("\t")
            train.append([document, query, summary])
    with open(os.path.join(input_dir, "valid_triples.tsv"), "r") as fin:
        for i, line in enumerate(fin):
            query, document, summary = line.strip().split("\t")
            validation.append([document, query, summary])
    with open(os.path.join(input_dir, "test_triples.tsv"), "r") as fin:
        for i, line in enumerate(fin):
            query, document, summary = line.strip().split("\t")
            test.append([document, query, summary])
    return train, validation, test


def template_input_decoder(row, dataset):
    if dataset == "xsum":
        return f"News article: {row[0]}. Summary of the above news article:"
    if dataset == "multi_news":
        return f"News article: {row[0]}. Summary of the above news article:"
    if dataset == "cnn_dailymail":
        return f"News article: {row[0]}. Summary of the above news article:"
    if dataset == "dbpedia_processed":
        return f"Question: {row[1]}. Document: {row[0]}. According to the Document, the one sentence answer to the Question is:"
    if dataset == "pubmedqa_processed":
        return f"Question: {row[1]}. Document: {row[0]}. According to the Document, the detailed answer to the Question is:"
    # else:
        

def get_null_input_decoder(row, dataset):
    if dataset == "xsum":
        return f"News article: . Summary of the above news article:"
    if dataset == "multi_news":
        return f"News article: . Summary of the above news article:"
    if dataset == "cnn_dailymail":
        return f"News article: . Summary of the above news article:"
    if dataset == "dbpedia_processed":
        return f"Question: {row[1]}. Document: . According to the Document, the one sentence answer to the Question is:"
    if dataset == "pubmedqa_processed":
        return f"Question: {row[1]}. Document: . According to the Document, the detailed answer to the Question is:"
    if dataset == "book":
        book = row[0]
        prompt = row[1]
        return f"{book} {prompt}"
    else:
        raise NotImplementedError

def template_input_encoder_decoder(row, dataset):
    if dataset == "xsum":
        return f"Summarize this following article in one or two sentences: {row[0]}"
    if dataset == "multi_news":
        return f"Summarize this following article in one or two sentences: {row[0]}"
    if dataset == "cnn_dailymail":
        return f"Summarize this following article in a few sentences: {row[0]}"
    if dataset == "dbpedia_processed":
        return f"Question: {row[1]}. Document: {row[0]}. According to the Document, the one sentence answer to the Question is:"
    if dataset == "pubmedqa_processed":
        return f"Question: {row[1]}. Document: {row[0]}. According to the Document, the detailed answer to the Question is:"

def get_null_input_encoder_decoder(row, dataset):
    if dataset == "xsum":
        return f"Summarize this following article in one or two sentences: "
    if dataset == "multi_news":
        return f"Summarize this following article in one or two sentences: "
    if dataset == "cnn_dailymail":
        return f"Summarize this following article in a few sentences: "
    if dataset == "dbpedia_processed":
        return f"Question: {row[1]}. Document: . According to the Document, the one sentence answer to the Question is:"
    if dataset == "pubmedqa_processed":
        return f"Question: {row[1]}. Document: . According to the Document, the detailed answer to the Question is:"

def pretokenize(dataset, tokenizer, max_input_length):
    res = []
    for i, row in tqdm(enumerate(dataset), desc="truncating documents..."):
        # print("row[0]: ", row[0])
        try:
            truncated_document = tokenizer.batch_decode(tokenizer(row[0], return_tensors="pt", max_length=max_input_length, truncation=True).input_ids, skip_special_tokens=True)[0]
        except:
            bp()
        res_ = row[1:]
        res_.insert(0, truncated_document)
        res.append(res_)
    return res