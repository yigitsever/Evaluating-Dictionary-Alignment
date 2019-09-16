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

for LANG in sq bg el it ro sl; do
        gunzip -fc "${EMBS}/cc.${LANG}.300.vec.gz" > "${EMBS}/${LANG}.1M.vec"
        rm -f "${EMBS}/cc.${LANG}.300.vec.gz"
done
