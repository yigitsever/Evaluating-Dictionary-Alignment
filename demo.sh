#!/bin/bash
#
# Copyright © 2019 Yiğit Sever <yigit.sever@tedu.edu.tr>
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
#

ROOT="$(pwd)"
SCRIPTS="${ROOT}/scripts"

WNET="${ROOT}/wordnets"
TAB_DIR="${WNET}/tab_files"
READY="${WNET}/ready"

DICT="${ROOT}/dictionaries"
TRAIN_DIR="${DICT}/train"
TEST_DIR="${DICT}/test"

EMBS="${ROOT}/embeddings"
MAP_TO="${ROOT}/bilingual_embeddings"

# create wordnets directory and download a single wordnet
mkdir -p "${WNET}"
wget -nc -q http://compling.hss.ntu.edu.sg/omw/wns/bul.zip -P "${WNET}"
unzip -o -q "${WNET}/bul.zip" -d "${WNET}"

# create tab directory and export a single .tab file
mkdir -p "${TAB_DIR}"
"${SCRIPTS}/tab_creator.pl" "${WNET}/bul/wn-data-bul.tab" "${TAB_DIR}"

# create ready directory and create two .def files
mkdir -p "${READY}"
python "${SCRIPTS}/prep_lookup.py" -s "en" -t "bg"
mv "${ROOT}"/*.def "${READY}"

# create dictionaries directory and download a single dictionary
mkdir -p "${DICT}"
wget -nc -q https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/dic/bg-en.dic.gz -P "${DICT}" # Bulgarian - English
gunzip -q "${DICT}/bg-en.dic.gz"

export LC_CTYPE=en_US.UTF-8
export LC_ALL=en_US.UTF-8

# create a train and a test seed lexicon
perl "${SCRIPTS}/train_dic_creator.pl" "en" "bg" "${DICT}"
mkdir -p "${TRAIN_DIR}"
mkdir -p "${TEST_DIR}"
mv "${DICT}"/*.train "${TRAIN_DIR}"
mv "${DICT}"/*.test "${TEST_DIR}"
rm -f "${DICT}"/*.dic

# download two monolingual embeddings
wget -nc -q https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.bg.300.vec.gz -P "${EMBS}" # Bulgarian
wget -nc -q https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip -P "${EMBS}" # English
gunzip "${EMBS}/cc.bg.300.vec.gz"
mv "${EMBS}/cc.bg.300.vec" "${EMBS}/bg.vec"
unzip -q "${EMBS}/crawl-300d-2M.vec.zip" -d "${EMBS}"
mv "${EMBS}/crawl-300d-2M.vec" "${EMBS}/en.vec"

# truncate two embeddings
for lang_code in bg en; do
    sed -i '1,500001!d' "${EMBS}/${lang_code}.vec" # one line on top for the <number of tokens> <dimensions>
    sed -i '1 s/^.*$/500000 300/' "${EMBS}/${lang_code}.vec"
done

# map two embeddings
python "${ROOT}/vecmap/map_embeddings.py" --supervised \
    "${TRAIN_DIC_DIR}/en_bg.train" \
    "${EMBS}/en.vec" \
    "${EMBS}/bg.vec" \
    "${MAP_TO}/en_to_bg.vec" \
    "${MAP_TO}/bg_to_en.vec" > /dev/null 2>&1

mkdir -p "${MAP_TO}" # create bilingual embeddings directory
source_lang="en"
target_lang="bg"

python "${ROOT}/vecmap/map_embeddings.py" --supervised \
    "${TRAIN_DIC_DIR}/${source_lang}_${target_lang}.train" \
    "${EMBS}/${source_lang}.vec" \
    "${EMBS}/${target_lang}.vec" \
    "${MAP_TO}/${source_lang}_to_${target_lang}.vec" \
    "${MAP_TO}/${target_lang}_to_${source_lang}.vec" > /dev/null 2>&1
