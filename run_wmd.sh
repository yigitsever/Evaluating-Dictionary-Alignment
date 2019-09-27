#!/bin/bash

set -o errexit -o pipefail -o noclobber -o nounset

ROOTDIR="$(pwd)"
ready_vectors_path="${ROOTDIR}/bilingual_embeddings"
wordnets_path="${ROOTDIR}/wordnets/ready"

for i in  en,bg en,el en,it, en,ro, en,sl en,sq, bg,el bg,it bg,ro el,it el,ro el,sq it,ro ro,sl ro,sq; do
    IFS=',' read -r source_lang target_lang <<< "${i}"
    echo "WMD + SNK: ${source_lang} - ${target_lang}"
    source_vec="${ready_vectors_path}/${source_lang}_to_${target_lang}.vec"
    target_vec="${ready_vectors_path}/${target_lang}_to_${source_lang}.vec"
    source_def="${wordnets_path}/${source_lang}_to_${target_lang}.def"
    target_def="${wordnets_path}/${target_lang}_to_${source_lang}.def"
    python "${ROOTDIR}/WMD.py" "${source_lang}" "${target_lang}" "${source_vec}" "${target_vec}" "${source_def}" "${target_def}" all all -n 1000 -b
    sleep 5
done
