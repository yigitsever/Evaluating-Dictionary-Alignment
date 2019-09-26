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
- NumPy
- SciPy

<details><summary>We recommend using a virtual environment</summary>
<p>

In order to create a [virtual environment](https://docs.python.org/3/library/venv.html#venv-def) that resides in a directory `.env` under your home directory;

```bash
cd ~
mkdir -p .env && cd .env
python -m venv evaluating
source ~/.env/evaluating/bin/activate
```

After the virtual environment is activated, the python interpreter and the installed packages are isolated within. In order for our code to work, the correct environment has to be sourced/activated.
In order to install all dependencies automatically use the [pip](https://pypi.org/project/pip/) package installer. `pre_requirements.text` includes requirements that packages in `requirements.txt` depend on. Both files come with the repository, so first navigate to the repository and then;

```bash
# under Evaluating-Dictionary-Alignment
pip install -r pre_requirements.txt
pip install -r requirements.txt
```

Rest of this README assumes that you are in the repository root directory.

</p>
</details>

## Acquiring The Data

`nltk` is required for this stage;

```python
import nltk
nltk.download('wordnet')
```

Then;

```bash
./get_data.sh
```

This will create two directories; `dictionaries` and `wordnets`. Definition files that are used by the unsupervised methods are in `wordnets/ready`, they come in pairs, `a_to_b.def` and `b_to_a.def` for wordnet definitions in language `a` and `b`. The pairs are aligned linewise; definitions on the same line for either file belong to the same wordnet synset, in the respective language.

<details><summary>Language pairs and number of available aligned glosses</summary>
<p>

Source Language | Target Language | # of Pairs
--- | ---  | ---:
English | Bulgarian | 4959
English | Greek | 18136
English | Italian | 12688
English | Romanian | 58754
English | Slovenian | 3144
English | Albanian | 4681
Bulgarian | Greek | 2817
Bulgarian | Italian | 2115
Bulgarian | Romanian | 4701
Greek | Italian | 4801
Greek | Romanian | 2144
Greek | Albanian | 4681
Italian | Romanian | 10353
Romanian | Slovenian | 2085
Romanian | Albanian | 4646

</p>
</details>

## Acquiring The Embeddings

We use [VecMap](https://github.com/artetxem/vecmap) on [fastText](https://fasttext.cc/) embeddings. You can skip this step if you are providing your own polylingual embeddings.

Otherwise,

* initialize and update the VecMap submodule;

```bash
git submodule init && git submodule update
```

* make sure `./get_data` is already run and `dictionaries` directory is present.

* run;

```bash
./get_embeddings.sh
```

Bear in mind that this will require around 50 GB free space. The mapped embeddings are stored under `bilingual_embedings` using the same naming scheme that `.def` files use.

## Quick Demo

`demo.sh` is included, downloads data for 2 languages and runs WMD (Word Mover's Distance) and SNK (Sinkhorn Distance) methods in matching and retrieval paradigms.

```bash
./demo.sh
```

## Usage

### WMD.py - Word Mover's Distance and Sinkhorn Distance

Aligns definitions using WMD or SNK metrics and matching or retrieval paradigms.

```
usage: WMD.py [-h] [-b] [-n INSTANCES]
              source_lang target_lang source_vector target_vector source_defs
              target_defs {all,wmd,snk} {all,retrieval,matching}

align dictionaries using wmd and wasserstein distance

positional arguments:
  source_lang           source language short name
  target_lang           target language short name
  source_vector         path of the source vector
  target_vector         path of the target vector
  source_defs           path of the source definitions
  target_defs           path of the target definitions
  {all,wmd,snk}         which methods to run
  {all,retrieval,matching}
                        which paradigms to align with

optional arguments:
  -h, --help            show this help message and exit
  -b, --batch           running in batch (store results in csv) or running a
                        single instance (output the results)
  -n INSTANCES, --instances INSTANCES
                        number of instances in each language to retrieve
```

Example;

```bash
python WMD.py en bg bilingual_embeddings/en_to_bg.vec bilingual_embeddings/bg_to_en.vec wordnets/ready/en_to_bg.def wordnets/ready/bg_to_en.def all all
```

Will run on English and Bulgarian definitions, using WMD and SNK for matching and retrieval, for a total of 4 times.
