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


def _get_text_file(text_dir=TEXTDIR):
    file_list = glob.glob(f'{text_dir}/**/*.txt')
    files = ",".join(file_list)
    return files


def train(prefix=PREFIX, vocab_size=VOCABSIZE, ctl_symbols=CTLSYMBOLS, sentencesize=SENTENCESIZE):
    files = _get_text_file()
    command = f'--input={files} --model_prefix={prefix} --vocab_size={vocab_size} --control_symbols={ctl_symbols} --input_sentence_size={sentencesize}'
    sp.SentencePieceTrainer.Train(command)


def main():
    train()


if __name__ == "__main__":
    main()
