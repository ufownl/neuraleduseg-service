#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Arne Neumann <nlpbox.programming@arne.cl>

import argparse
import json
import logging
from pathlib import PosixPath, Path
import sys

import pytest
import requests

from neuralseg.splitter import DEFAULT_ARGS, load_models, segment_text


REPO_ROOT_PATH = PosixPath(__file__).parent.parent # root directory of the repo
REPO_PACKAGE_PATH = REPO_ROOT_PATH.joinpath('neuralseg')
FIXTURES_PATH = REPO_ROOT_PATH.joinpath('tests/fixtures')


@pytest.fixture(scope="session", autouse=True)
def parser_models():
    print("starting parser / loading models...")
    # silence warnings
    logging.getLogger("tensorflow").setLevel(logging.ERROR)

    # We need to mess with the PATH, since the code that unpickles
    # `word.vocab` expects to be able to `import vocab`
    # (instead of `import neuralseg.vocab`.
    sys.path.append(REPO_PACKAGE_PATH.as_posix())
    
    # provide the fixture values
    rst_data, model, spacy_nlp = load_models(DEFAULT_ARGS)
    return rst_data, model, spacy_nlp


def test_segmentation_short(parser_models):
    """NeuralEDUSeg produces the expected segmentation output."""
    rst_data, model, spacy_nlp = parser_models
    
    input_text = FIXTURES_PATH.joinpath('input_short.txt').read_text()

    expected_output_json = FIXTURES_PATH.joinpath('output_short.json').read_text()
    json_result = segment_text(input_text, rst_data, model, spacy_nlp, output_format='json')
    assert json_result == expected_output_json

    expected_output_json_debug = FIXTURES_PATH.joinpath('output_short.debug.json').read_text()
    json_result = segment_text(input_text, rst_data, model, spacy_nlp, output_format='json', debug=True)
    assert json_result == expected_output_json_debug

    expected_output_tokenized = FIXTURES_PATH.joinpath('output_short.tokenized').read_text()
    json_result = segment_text(input_text, rst_data, model, spacy_nlp, output_format='tokenized')
    assert json_result == expected_output_tokenized

    expected_output_inline = FIXTURES_PATH.joinpath('output_short.inline').read_text()
    json_result = segment_text(input_text, rst_data, model, spacy_nlp, output_format='inline')
    assert json_result == expected_output_inline


def test_segmentation_long(parser_models):
    """NeuralEDUSeg produces the expected segmentation output."""
    rst_data, model, spacy_nlp = parser_models
    
    input_text = FIXTURES_PATH.joinpath('input_long.txt').read_text()

    expected_output_json = FIXTURES_PATH.joinpath('output_long.json').read_text()
    json_result = segment_text(input_text, rst_data, model, spacy_nlp, output_format='json', debug=False)
    assert json_result == expected_output_json

    expected_output_json_debug = FIXTURES_PATH.joinpath('output_long.debug.json').read_text()
    json_result = segment_text(input_text, rst_data, model, spacy_nlp, output_format='json', debug=True)
    assert json_result == expected_output_json_debug

    expected_output_tokenized = FIXTURES_PATH.joinpath('output_long.tokenized').read_text()
    json_result = segment_text(input_text, rst_data, model, spacy_nlp, output_format='tokenized', debug=False)
    assert json_result == expected_output_tokenized

    expected_output_inline = FIXTURES_PATH.joinpath('output_long.inline').read_text()
    json_result = segment_text(input_text, rst_data, model, spacy_nlp, output_format='inline', debug=False)
    assert json_result == expected_output_inline
