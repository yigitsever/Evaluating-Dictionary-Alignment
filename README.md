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

## Acquiring The Embeddings

We use [VecMap](https://github.com/artetxem/vecmap) on [fastText](https://fasttext.cc/) embeddings.
You can skip this step if you are providing your own polylingual embeddings.
Otherwise;

* initialize and update the VecMap submodule;

```bash
git submodule init && git submodule update
```

* make sure `./get_data` is already run and `dictionaries` folder is present.

* run;

```bash
./get_embeddings.sh
```

Bear in mind that this will require around 30 GB free space.

