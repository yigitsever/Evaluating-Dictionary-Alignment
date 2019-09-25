# Evaluating cross-lingual textual similarity on dictionary alignment

This repository contains the scripts to prepare the resources as well as open source implementations of the methods. Word Mover's Distance and Sinkhorn implementations are extended from [Cross-lingual retrieval with Wasserstein distance](https://github.com/balikasg/WassersteinRetrieval) and supervised implementation is extended from https://github.com/fionn-mac/Manhattan-LSTM.

```bash
git clone https://github.com/yigitsever/Evaluating-Dictionary-Alignment.git
cd Evaluating-Dictionary-Alignment
```

## Requirements

```bash
pip install -r pre_requirements.txt
pip install -r requirements.txt
```

- Python 3
- [nltk](http://www.nltk.org/)
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

After the virtual environment is activated, the python interpreter and the installed packages are isolated. In order for our code to work, the correct environment has to be sourced/activated.
In order to install all dependencies automatically use the [pip](https://pypi.org/project/pip/) package installer using `requirements.txt`, which resides under the repository directory.

```bash
# under Evaluating-Dictionary-Alignment
pip install -r requirements.txt
```

Rest of this README assumes that you are in the repository root directory.

</p>
</details>

## Acquiring The Data

nltk is required for this stage;

```python
import nltk
nltk.download('wordnet')
```

Then;

```bash
./get_data.sh
```

This will create two directories; `dictionaries` and `wordnets`.
Linewise aligned definition files are in `wordnets/ready`.

<details><summary>Language pairs and number of available aligned glosses</summary>
<p>

Source Language | Target Language | # of Pairs
--- | ---  | ---:
English | Bulgarian | 4959
English | Greek | 18136
English | Italian | 12688
English | Romaian | 58754
English | Slovenian | 3144
English | Albanian | 4681
Bulgarian | Greek | 2817
Bulgarian | Italian | 2115
Bulgarian | Romaian | 4701
Greek | Italian | 4801
Greek | Romaian | 2144
Greek | Albanian | 4681
Italian | Romaian | 10353
Romaian | Slovenian | 2085
Romaian | Albanian | 4646

</p>
</details>

## Acquiring The Embeddings

We use [VecMap](https://github.com/artetxem/vecmap) on [fastText](https://fasttext.cc/) embeddings. You can skip this step if you are providing your own polylingual embeddings.

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

## Quick Demo

`demo.sh` is included, downloads data for 2 languages.

```bash
./demo.sh
```
