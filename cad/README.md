#### Required packages
```
sklearn
torchmetrics
transformers
datasets
evaluate
```

#### Install the cutomized huggingface library with the change of generation scripts
```
cd transformers_cad
pip install -e .
```


#### Add context-aware decoding
replace `transformers/src/transformers/generation/utils.py` with `generation/utils.py`


### To run
Please see sample script from `src/run.sh`