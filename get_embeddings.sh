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

set -o errexit -o pipefail -o noclobber -o nounset

ROOT="$(pwd)"
EMBS="${ROOT}/embeddings"
mkdir -p "${EMBS}"

echo "Downloading embeddings"

wget -nc -q https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.sq.300.vec.gz -P "${EMBS}" # Albanian
wget -nc -q https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.bg.300.vec.gz -P "${EMBS}" # Bulgarian
wget -nc -q https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip -P "${EMBS}" # English
wget -nc -q https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.el.300.vec.gz -P "${EMBS}" # Greek
wget -nc -q https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.it.300.vec.gz -P "${EMBS}" # Italian
wget -nc -q https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ro.300.vec.gz -P "${EMBS}" # Romanian
wget -nc -q https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.sl.300.vec.gz -P "${EMBS}" # Slovenian

echo "Extracting embeddings"

for lang_code in sq bg el it ro sl; do
        gunzip "${EMBS}/cc.${lang_code}.300.vec.gz"
        mv "${EMBS}/cc.${lang_code}.300.vec" "${EMBS}/${lang_code}.vec"
done

unzip -q "${EMBS}/crawl-300d-2M.vec.zip" -d "${EMBS}"
mv "${EMBS}/crawl-300d-2M.vec" "${EMBS}/en.vec"
rm -f "${EMBS}/crawl-300d-2M.vec.zip"

# truncate to top 500k tokens for efficiency
for lang_code in bg en el it ro sl sq; do
    sed -i '1,500001!d' "${EMBS}/${lang_code}.vec"
    sed -i '1 s/^.*$/500000 300/' "${EMBS}/${lang_code}.vec"
done

if [ ! "$(ls -A "${ROOT}/vecmap/")" ]; then
    echo "VecMap directory seems empty, did you run git submodule init && git submodule update?"; exit
fi

if [ ! -d "${ROOT}/dictionaries" ]; then
    echo "Dictionaries directory does not exist, did you run ./get_data.sh?"; exit
fi

if [ ! "$(ls -A "${ROOT}/dictionaries/")" ]; then
    echo "Dictionaries directory seems empty, did you run ./get_data.sh?"; exit
fi

TRAIN_DIC_DIR="${ROOT}/dictionaries/train"
MAP_TO="${ROOT}/bilingual_embeddings"

mkdir -p "${MAP_TO}"

for i in en,bg en,el en,it, en,ro, en,sl en,sq, bg,el bg,it bg,ro el,it el,ro el,sq it,ro ro,sl ro,sq; do
    IFS=',' read -r source_lang target_lang <<< "${i}"
    python "${ROOT}/vecmap/map_embeddings.py" --supervised \
        "${TRAIN_DIC_DIR}/${source_lang}_${target_lang}.train" \
        "${EMBS}/${source_lang}.vec" \
        "${EMBS}/${target_lang}.vec" \
        "${MAP_TO}/${source_lang}_to_${target_lang}.vec" \
        "${MAP_TO}/${target_lang}_to_${source_lang}.vec" > /dev/null
done
