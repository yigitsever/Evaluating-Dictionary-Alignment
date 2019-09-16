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
WNET="${ROOT}/data"
mkdir -p "${WNET}"

echo "Downloading wordnet data"

wget -nc -q http://compling.hss.ntu.edu.sg/omw/wns/als.zip -P "${WNET}"
wget -nc -q http://compling.hss.ntu.edu.sg/omw/wns/bul.zip -P "${WNET}"
wget -nc -q http://compling.hss.ntu.edu.sg/omw/wns/ell.zip -P "${WNET}"
wget -nc -q http://compling.hss.ntu.edu.sg/omw/wns/ita.zip -P "${WNET}"
wget -nc -q http://compling.hss.ntu.edu.sg/omw/wns/ron.zip -P "${WNET}"
wget -nc -q http://compling.hss.ntu.edu.sg/omw/wns/slv.zip -P "${WNET}"

echo "Unzipping wordnet data"

for LANG in als bul ell ita ron slv; do
        unzip -ofq "${WNET}/${LANG}" -d "${WNET}"
        rm -f "${WNET}/${LANG}.zip"
done

rm -rf "${WNET}/ita" # comes alongside iwn, not useful for us

echo "Downloading dictionaries"

DICT="${ROOT}/dictionaries"

wget -nc -q https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/dic/en-sq.dic.gz -P "${DICT}" # English - Albanian
wget -nc -q https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/dic/bg-en.dic.gz -P "${DICT}" # Bulgarian - English
wget -nc -q https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/dic/el-en.dic.gz -P "${DICT}" # Greek - English
wget -nc -q https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/dic/en-it.dic.gz -P "${DICT}" # English - Italian
wget -nc -q https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/dic/en-ro.dic.gz -P "${DICT}" # English - Romanian
wget -nc -q https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/dic/en-sl.dic.gz -P "${DICT}" # English - Slovenian
wget -nc -q https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/dic/bg-el.dic.gz -P "${DICT}" # Bulgarian - Greek
wget -nc -q https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/dic/bg-it.dic.gz -P "${DICT}" # Bulgarian - Italian
wget -nc -q https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/dic/bg-ro.dic.gz -P "${DICT}" # Bulgarian - Romanian
wget -nc -q https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/dic/el-it.dic.gz -P "${DICT}" # Greek - Italian
wget -nc -q https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/dic/el-ro.dic.gz -P "${DICT}" # Greek - Romanian
wget -nc -q https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/dic/el-sq.dic.gz -P "${DICT}" # Greek - Albanian
wget -nc -q https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/dic/it-ro.dic.gz -P "${DICT}" # Italian - Romanian
wget -nc -q https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/dic/ro-sl.dic.gz -P "${DICT}" # Romanian - Albanian
wget -nc -q https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/dic/ro-sq.dic.gz -P "${DICT}" # Romanian - Albanian
