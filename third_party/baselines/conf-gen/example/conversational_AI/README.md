- `conformal_generation.ipynb`: the main notebook to run and analyze the conformal generation experiments on the conversational LLM.

- `preprocess_data.py`: the pre-requisite script to run for obtaining the preprocessed data, which is stored in the `processed_data/` folder, to be consumed by the main experiments.

Notes:
We have included the preprocessed data in the `processed_data/` folder. Experiments in the `conformal_generation.ipynb` may be ran directly. 

If running from scratch based on the original dataset (i.e. ClariQ) is desired, before running `preprocess_data.py`, place the original dataset from ClariQ into the `original_data/` folder. 
The dataset filename is `multi_turn_human_generated_data.tsv`. The ClariQ dataset can be obtained via their official repository: https://github.com/aliannejadi/ClariQ.

