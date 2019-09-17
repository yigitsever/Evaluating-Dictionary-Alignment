# Evaluating cross-lingual textual similarity on dictionary alignment

This repository contains the scripts to prepare the resources for the study as well as open source implementations of the methods.

## Requirements
- python3
- nltk
    ```python
    import nltk
    nltk.download('wordnet')
    ```

## Acquiring The Data

```bash
git clone https://github.com/yigitsever/Evaluating-Dictionary-Alignment.git && cd Evaluating-Dictionary-Alignment
./get_data.sh
```

This will create two directories; `dictionaries` and `wordnets`.
Linewise aligned definition files are in `wordnets/ready`.

