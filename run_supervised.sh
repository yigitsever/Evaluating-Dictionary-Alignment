#!/bin/bash

set -o errexit -o pipefail -o noclobber -o nounset

ROOTDIR="$(pwd)"
ready_vectors_path="${ROOTDIR}/bilingual_embeddings"
tsv_path="${ROOTDIR}/wordnets/tsv_files"

for i in  en,bg en,el en,it, en,ro, en,sl en,sq, bg,el bg,it bg,ro el,it el,ro el,sq it,ro ro,sl ro,sq; do
    IFS=',' read -r source_lang target_lang <<< "${i}"
    source_vec="${ready_vectors_path}/${source_lang}_to_${target_lang}.vec"
    target_vec="${ready_vectors_path}/${target_lang}_to_${source_lang}.vec"
    data_file="${tsv_path}/${source_lang}_to_${target_lang}.tsv"
    python "${ROOTDIR}/learn_and_predict.py" -sl "${source_lang}" -tl "${target_lang}" -df "${data_file}" -es "${source_vec}" -et "${target_vec}" -b
    sleep 5
done
