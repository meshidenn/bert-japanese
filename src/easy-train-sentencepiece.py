#!/usr/bin/env python3

import configparser
import glob
import os
import sentencepiece as sp

CURDIR = os.path.dirname(os.path.abspath(__file__))
CONFIGPATH = os.path.join(CURDIR, os.pardir, 'config.ini')
config = configparser.ConfigParser()
config.read(CONFIGPATH)

TEXTDIR = config['DATA']['TEXTDIR']
PREFIX = config['SENTENCEPIECE']['PREFIX']
VOCABSIZE = config['SENTENCEPIECE']['VOCABSIZE']
CTLSYMBOLS = config['SENTENCEPIECE']['CTLSYMBOLS']
SENTENCESIZE = config['SENTENCEPIECE']['SENTENCESIZE']


def _get_aasd_text_file(text_dir=TEXTDIR):
    file_list = glob.glob(f'{text_dir}/**/all.txt', recursive=True)
    print(file_list)
    # patent = os.path.join(f'{text_dir}', 'NTCIR')
    # file_list.extend(glob.glob(f'{patent}/*'))
    # print(file_list)
    wiki_subset = os.path.join(f'{text_dir}', 'wiki_subset.txt')
    if os.path.exists(wiki_subset):
        file_list.append(wiki_subset)

    print(file_list)
    files = ",".join(file_list)
    return files


def _get_general_text_file(text_dir=TEXTDIR):
    file_list = glob.glob(f'{text_dir}/wiki/**/all.txt', recursive=True)
    file_list.extend(glob.glob(f'{text_dir}/asahi/**/all.txt', recursive=True))
    file_list.extend(glob.glob(f'{text_dir}/aozora/**/all.txt', recursive=True))
    file_list.extend(glob.glob(f'{text_dir}/BCCWJ/*.txt'))
    files = ",".join(file_list)
    return files


def _get_long_open_text_file(text_dir=TEXTDIR):
    file_list = glob.glob(f'{text_dir}/wiki/**/all.txt', recursive=True)
    file_list.extend(glob.glob(f'{text_dir}/asahi/**/all.txt', recursive=True))
    file_list.extend(glob.glob(f'{text_dir}/aozora/**/all.txt', recursive=True))
    files = ",".join(file_list)
    return files


def _get_marine_text_file(text_dir=TEXTDIR):
    file_list = glob.glob(f'{text_dir}/wiki/**/all.txt', recursive=True)
    file_list.extend(glob.glob(f'{text_dir}/asahi/**/all.txt', recursive=True))
    file_list.extend(glob.glob(f'{text_dir}/customer/**/all.txt', recursive=True))
    files = ",".join(file_list)
    return files


def train(prefix=PREFIX, vocab_size=VOCABSIZE, ctl_symbols=CTLSYMBOLS, sentencesize=SENTENCESIZE):
    files = _get_long_open_text_file()
    command = f'--input={files} --model_prefix={prefix} --vocab_size={vocab_size} --control_symbols={ctl_symbols} --input_sentence_size={sentencesize}'
    sp.SentencePieceTrainer.Train(command)


def main():
    train()


if __name__ == "__main__":
    main()
