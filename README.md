# Evaluating cross-lingual textual similarity on dictionary alignment

This repository contains the scripts to prepare the resources for the study as well as open source implementations of the methods.

## Requirements
- Python 3
- nltk
    ```python
    import nltk
    nltk.download('wordnet')
    ```
- [lapjv](https://pypi.org/project/lapjv/)
- [POT](https://pypi.org/project/POT/)
- [mosestokenizer](https://pypi.org/project/mosestokenizer/)
- (Optional) If using VecMap
    * NumPy
    * SciPy

<details><summary>We recommend using a virtual environment</summary>
<p>

In order to create a [virtual environment](https://docs.python.org/3/library/venv.html#venv-def) that resides in a directory `.env` under home;

```bash
cd ~
mkdir -p .env && cd .env
python -m venv evaluating
source ~/.env/evaluating/bin/activate
```

Inside the virtual environment, the python interpreter and the installed packages are isolated.
In order to install all dependencies automatically;

```bash
pip install -r requirements.txt
```

After done with the environment run;

```bash
deactivate
```

</p>


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

* make sure `./get_data` is already run and `dictionaries` directory is present.

* run;

```bash
./get_embeddings.sh
```

Bear in mind that this will require around 50 GB free space.

