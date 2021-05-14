#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Arne Neumann

"""Wrapper script for splitting an input plaintext file into paragraphs, sentences and EDUs.

Can produce inline and standoff json output. The inline format is used
by the StageDP RST parser (Wang 2017).
"""


import argparse
import json
import logging
import pickle
import re
from pathlib import PosixPath
import sys

from intervaltree import IntervalTree
import spacy

import neuralseg
from neuralseg.rst_edu_reader import RSTData
from neuralseg.atten_seg import AttnSegModel

import pretty_errors

# This matches two or more empty lines.
# We enclose the regex in () "capturing brackets", so that re.split
# will return the matching as well as the non-matching substrings,
# cf. https://www.technomancy.org/python/strings-that-dont-match-regex/
NEWLINES_RE_CAPTURING = re.compile(r"(\n{2,})")

NEWLINES_ONLY_RE = re.compile(r"\n{2,}")

INSTALL_ROOT_PATH = PosixPath(neuralseg.__file__).parent.parent # root directory of the installed package

# These are the default arguments used by the original NeuralEDUSeg 
# implementation (cf. segment.py).
DEFAULT_ARGS = argparse.Namespace(
    batch_size=32, dropout_keep_prob=0.9, ema_decay=0.9999, gpu=None,
    hidden_size=200, learning_rate=0.001, max_grad_norm=5.0,
    optim='adam', segment=False, weight_decay=0.0001, window_size=5,
    model_dir=INSTALL_ROOT_PATH.joinpath('data/models'),
    output_file=sys.stdout,
    result_dir=INSTALL_ROOT_PATH.joinpath('data/results'), 
    word_vocab_path=INSTALL_ROOT_PATH.joinpath('data/vocab/word.vocab'))


def parse_args(namespace=DEFAULT_ARGS):
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser('split text into paragraphs, sentences and EDUs.')
    parser.add_argument('--debug', action='store_true')
    
    parser.add_argument(
        '--output_format', default='json',
        help="output format: 'json', 'tokenized' or 'inline' (default: inline)")
    
    parser.add_argument('input_file')
    parser.add_argument('output_file', nargs='?', default=sys.stdout)

    args = parser.parse_args(namespace=namespace)
    return args


def load_models(args):
    """Load models needed for EDU segmentation."""
    print("Loading NeuralEDUSeg models...")
    rst_data = RSTData()
    with open(args.word_vocab_path, 'rb') as fin:
        word_vocab = pickle.load(fin)
    rst_data.word_vocab = word_vocab

    model = AttnSegModel(args, word_vocab)
    model.restore('best', args.model_dir)
    if model.use_ema:
        model.sess.run(model.ema_backup_op)
        model.sess.run(model.ema_assign_op)

    spacy_nlp = spacy.blank("en")
    spacy_nlp.add_pipe(spacy_nlp.create_pipe("sentencizer"))
    print("Finished loading NeuralEDUSeg models.")
    return rst_data, model, spacy_nlp


def text2paragraphs(text, debug=False):
    """Split text into paragraphs.

    Paragraphs are the substrings created by splitting the input string on
    two or more consecutive newlines.
    """
    paragraphs = []
    raw_paragraphs = []
    offset = 0

    for i, match_str in enumerate(NEWLINES_RE_CAPTURING.split(text)):
        match_len = len(match_str)
        if not NEWLINES_ONLY_RE.match(match_str):  # if this matches the text (and not the paragraph breaks)
            raw_paragraphs.append(match_str)
            paragraph = {'start': offset, 'end': offset+match_len}
            if debug:
                paragraph['text'] = match_str
            paragraphs.append(paragraph)

        offset += match_len
    return paragraphs, raw_paragraphs


def text2sentences(spacy_nlp, json_paragraphs, raw_text, debug=False):
    raw_paragraphs = [raw_text[para['start']:para['end']] for para in json_paragraphs]

    sentences = []
    raw_sentences = []
    spacy_sentences = []

    for para_idx, para in enumerate(spacy_nlp.pipe(raw_paragraphs, batch_size=1000, n_threads=5)):
        para_start = json_paragraphs[para_idx]['start']
        para_end = json_paragraphs[para_idx]['end']

        for sent in para.sents:
            if not sent.string.strip():
                continue  # ignore superfluous empty lines

            sent_start = para_start + sent.start_char
            sent_end = para_start + sent.end_char
            sent_dict = {'start': sent_start, 'end': sent_end}
            if debug:
                sent_dict['text'] = sent.text
            sentences.append(sent_dict)
            raw_sentences.append(sent.text)

            fixed_sent = []
            for token in sent:
                absolute_offset = token.idx + para_start
                fixed_sent.append((token, absolute_offset))

            spacy_sentences.append(fixed_sent)
    return sentences, raw_sentences, spacy_sentences


def generate_samples(spacy_sentences):
    """
    Converts the preprocessed input (split into sentences using spacy)
    into the format needed by the actual EDU semgentation
    (AttnSegModel.segment).

    Parameters
    ----------
    spacy_sentences: List(List(Tuple(spacy.tokens.token.Token, int)))
        A list of sentences, where each sentence is represented by a
        list of (token, offset from the beginning of the input text) tuples.

    Returns
    -------
    samples: List(Dict)
        A list of sentences, where each sentence is represented by a
        dict with these keys:
        - words: List(str)
            A list of all tokens in the sentence.
        - spacy_tokens: List(spacy.tokens.token.Token)
            A list of all tokens in the sentence.
        - absolute_offsets: List(int)
            A list of character offsets (in relatition to the beginning of
            the input text), one for each token in the sentence.
        - edu_seg_indices: List
            An empty list (that will be filled with EDU break positions)
            later on.
    """
    samples = []
    for sent in spacy_sentences:
        words = []
        spacy_tokens = []
        absolute_offsets = []

        for token, absolute_offset in sent:
            words.append(token.text)
            spacy_tokens.append(token)
            absolute_offsets.append(absolute_offset)

        samples.append({'words': words,
                        'spacy_tokens': spacy_tokens,
                        'absolute_offsets': absolute_offsets,
                        'edu_seg_indices': []})
    return samples


def segment_batches(model, data_batches, input_text, debug=False):
    """Segments the preprocessed input text into EDUs.

    Parameters
    ----------
    model: neuralseg.atten_seg.AttnSegModel
        EDU segmentation model
    data_batches: generator(dict)
        Each dict contains these keys:
        - raw_data: List(dict)
            A list of sentences, where each sentence is represented by a
            dict with these keys:
            - words: List(str)
                A list of all tokens in the sentence.
            - spacy_tokens: List(spacy.tokens.token.Token)
                A list of all tokens in the sentence.
            - absolute_offsets: List(int)
                A list of character offsets (in relation to the beginning of
                the input text), one for each token in the sentence.
            - edu_seg_indices: List
                An empty list (that will be filled with EDU break positions)
                later on.

        - word_ids: List(List(int))
            A list of sentences, where each sentence is represented as a
            list of integers, i.e. a vector representing the tokens in the
            sentence. All vectors in the batch have the same length and
            are zero-padded at the end.
        - length: List(int)
            number of tokens per sentence
        - seg_labels: List(List(int))
            A list of sentences, where each sentence is represented as a
            list of integers. Initially, all values are 0.

    Returns
    -------
    edus: List(str)
        A list of EDUs, where each EDU is a plain-text string.
    spacy_edus: List(List(spacy.tokens.token.Token))
        A list of EDUs, where each EDU is a list of spacy tokens.
    """
    edus = []
    spacy_edus = []
    for batch in data_batches:
        batch_edu_breaks = model.segment(batch)

        for sent, pred_segs in zip(batch['raw_data'], batch_edu_breaks):
            edu_tokens = []
            spacy_edu_tokens = []

            for word_idx, word in enumerate(sent['words']):
                if word_idx in pred_segs:  # there's an EDU boundary before this token
                    edus.append(' '.join(edu_tokens))
                    spacy_edus.append(spacy_edu_tokens)

                    # reset EDU token lists
                    edu_tokens = []
                    spacy_edu_tokens = []

                edu_tokens.append(word)
                token_offset = sent['absolute_offsets'][word_idx]
                spacy_token = sent['spacy_tokens'][word_idx]
                spacy_edu_tokens.append((spacy_token, token_offset))

            if edu_tokens:
                edus.append(' '.join(edu_tokens))
                spacy_edus.append(spacy_edu_tokens)

    json_edus = []
    for i, spacy_edu in enumerate(spacy_edus):
        first_token, edu_start = spacy_edu[0]
        last_token, last_token_offset = spacy_edu[-1]
        edu_end = last_token_offset + len(last_token.string)
        raw_edu = input_text[edu_start:edu_end]

        json_edu = {'start': edu_start, 'end': edu_end}
        if debug:
            json_edu['raw_edu'] = raw_edu
            json_edu['tokenized_edu'] = edus[i]

        json_edus.append(json_edu)
    return edus, spacy_edus, json_edus


def segment_text(text, rst_data, model, spacy_nlp, output_format='json', debug=False):
    json_paragraphs, raw_paragraphs = text2paragraphs(text, debug=debug)
    json_sentences, raw_sentences, spacy_sentences = text2sentences(spacy_nlp, json_paragraphs, text, debug=debug)

    samples = generate_samples(spacy_sentences)
    rst_data.test_samples = samples

    data_batches = rst_data.gen_mini_batches(batch_size=32, test=True, shuffle=False)
    edus, spacy_edus, json_edus = segment_batches(model, data_batches, text, debug=debug)

    json_result = {
        'paragraphs': json_paragraphs,
        'sentences': json_sentences,
        'edus': json_edus,
    }

    nested_paras = []
    sent_ranges = IntervalTree.from_tuples([(s['start'], s['end']) for s in json_sentences])
    edu_ranges = IntervalTree.from_tuples([(e['start'], e['end']) for e in json_edus])
    for para in json_paragraphs:
        sents_in_para = sorted(sent_ranges.overlap(para['start'], para['end']))
        
        sents = []
        for sent_in_para in sents_in_para:
            edus_in_sent = sorted(edu_ranges.overlap(sent_in_para.begin, sent_in_para.end))
            sents.append((sent_in_para, edus_in_sent))
        nested_paras.append((para, sents))

    nested_str = ""
    for (para, sents) in nested_paras:
        nested_str += '<P>'
        for (sent_in_para, edus_in_sent) in sents:
            nested_str += '<S>'
            edus_str = "\n".join(text[interval.begin:interval.end] for interval in edus_in_sent)
            nested_str += edus_str
            nested_str += '\n'

    if output_format == 'json':
        return json.dumps(json_result)
    elif output_format == 'inline':
        return nested_str
    return '\n'.join(edus)


def main():
    args = parse_args()

    with open(args.input_file, 'r') as infile:
        input_text = infile.read()

    rst_data, model, spacy_nlp = load_models(args)

    result = segment_text(input_text, rst_data, model, spacy_nlp, output_format=args.output_format, debug=args.debug)

    if isinstance(args.output_file, str):  # write to file
        with open(args.output_file, 'w') as output_file:
            output_file.write(result)
    else:  # write to STDOUT
        args.output_file.write(result)
    

if __name__ == "__main__":
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    main()
