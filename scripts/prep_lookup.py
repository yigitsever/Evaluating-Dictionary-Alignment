import argparse
from pathlib import Path
import collections
import os

def en_and_other(other, dirname):
    from nltk.corpus import wordnet as wn
    other_file = os.path.join(dirname, other + "." + 'tab')
    lookup = collections.defaultdict(dict)

    with open(other_file, 'r') as f:
        for line in f:
            (pos, offset, rest) = line.split(' ', 2)
            offset = int(offset)
            # part of speech + offset is unique, so keys are combination of both
            en_def = wn.synset_from_pos_and_offset(pos, offset).definition()
            lookup[(pos, offset)]['en'] = en_def
            lookup[(pos,offset)][other] = rest.rstrip()
    return lookup

def both_lookup(source, target, dirname):
    from_file = os.path.join(dirname, source + "." + 'tab')
    to_file = os.path.join(dirname, target + "." + 'tab')
    lookup = collections.defaultdict(dict)

    for tab_file, lang_code in zip((from_file, to_file), (source, target)):
        with open(tab_file, 'r') as f:
            for line in f:
                (pos, offset, rest) = line.split(' ', 2)
                offset = int(offset)
                # part of speech + offset is unique, so keys are combination of both
                lookup[(pos,offset)][lang_code] = rest.rstrip()
    return lookup

def main(args):

    dirname = args.tab_directory
    source_lang = args.source_lang
    target_lang = args.target_lang

    if (source_lang == 'en'):
            lookup = en_and_other(target_lang, dirname)
    elif (target_lang == 'en'):
            lookup = en_and_other(source_lang, dirname)
    else:
            lookup = both_lookup(source_lang, target_lang, dirname)

    with open(f'{source_lang}_to_{target_lang}.def', 'w') as sf, open(f'{target_lang}_to_{source_lang}.def', 'w') as tf:
        for (pos, offset), overlap in lookup.items():
            if source_lang in overlap and target_lang in overlap:
                print(overlap[source_lang], file=sf)
                print(overlap[target_lang], file=tf)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a pair of .def files for 2 given languages')
    parser.add_argument('--tab_directory', help='directory of the .tab files', default='wordnets/tab_files')
    parser.add_argument('-s', '--source_lang', help='source language 2 letter code')
    parser.add_argument('-t', '--target_lang', help='target language 2 letter code')
    args = parser.parse_args()

    main(args)
