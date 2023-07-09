# Folder structure

    - NER
        - data
            - conll2003_test.pkl
            - conll2003_train.pkl
            - conll2003_val.pkl
            - conll2003word2index_dict.pkl
        - models
            - rnn_model_depp_conll.pt
        - architecture_report.md
        - main.py
        - model.py
        - README.md
        - requirements.txt
        - test.py
        - utils.py

# Usage

1. Run `[python -m] pip install -r requirements.txt` to install all relevant libraries. You may create a virtual environment (e. g. with [conda](https://www.anaconda.com "home website of the Anaconda distribution where you can download the installer to install it")) firstly before running this command.
2. Run `python ./utils.py` to create the word2index_dict-pickle if it does not exist so far.
3. Run `python ./main.py` to train a new model.
4. Run `python ./test.py` to test the trained model.
