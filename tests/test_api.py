#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Arne Neumann <nlpbox.programming@arne.cl>

import pexpect
import pytest
import requests
import sh

from test_cli import FIXTURES_PATH, REPO_PACKAGE_PATH


@pytest.fixture(scope="session", autouse=True)
def start_api():
    print("starting API...")
    # ~ import pudb; pudb.set_trace()

    api_path = REPO_PACKAGE_PATH.joinpath('splitter_api.py')                                                                                                                                                                                             
    child = pexpect.spawn(f'hug -f {api_path}')
    # provide the fixture value (we don't need it, but it marks the
    # point when the 'setup' part of this fixture ends).
    yield child.expect('(?i)Serving on :8000')
    print("stopping API...")
    child.close()


def test_api_short_json():
    input_text = FIXTURES_PATH.joinpath('input_short.txt').read_text()
    expected_output = FIXTURES_PATH.joinpath('output_short.json').read_text()

    # ~ import pudb; pudb.set_trace()

    res = requests.post(
        f'http://localhost:8000/parse?format=json',
        files={'input': input_text})
    assert expected_output == res.content.decode('utf-8')

def test_api_short_json_debug():
    input_text = FIXTURES_PATH.joinpath('input_short.txt').read_text()
    expected_output = FIXTURES_PATH.joinpath('output_short.debug.json').read_text()

    res = requests.post(
        f'http://localhost:8000/parse?format=json&debug=True',
        files={'input': input_text})
    assert expected_output == res.content.decode('utf-8')

def test_api_short_tokenized():
    input_text = FIXTURES_PATH.joinpath('input_short.txt').read_text()
    expected_output = FIXTURES_PATH.joinpath('output_short.tokenized').read_text()

    res = requests.post(
        f'http://localhost:8000/parse?format=tokenized',
        files={'input': input_text})
    assert expected_output == res.content.decode('utf-8')

def test_api_short_inline():
    input_text = FIXTURES_PATH.joinpath('input_short.txt').read_text()
    expected_output = FIXTURES_PATH.joinpath('output_short.inline').read_text()

    res = requests.post(
        f'http://localhost:8000/parse?format=inline',
        files={'input': input_text})
    assert expected_output == res.content.decode('utf-8')

    # check that 'inline' is also the default format
    res = requests.post(
        f'http://localhost:8000/parse',
        files={'input': input_text})
    assert expected_output == res.content.decode('utf-8')



def test_api_long_json():
    input_text = FIXTURES_PATH.joinpath('input_long.txt').read_text()
    expected_output = FIXTURES_PATH.joinpath('output_long.json').read_text()

    res = requests.post(
        f'http://localhost:8000/parse?format=json',
        files={'input': input_text})
    assert expected_output == res.content.decode('utf-8')

def test_api_long_json_debug():
    input_text = FIXTURES_PATH.joinpath('input_long.txt').read_text()
    expected_output = FIXTURES_PATH.joinpath('output_long.debug.json').read_text()

    res = requests.post(
        f'http://localhost:8000/parse?format=json&debug=True',
        files={'input': input_text})
    assert expected_output == res.content.decode('utf-8')

def test_api_long_tokenized():
    input_text = FIXTURES_PATH.joinpath('input_long.txt').read_text()
    expected_output = FIXTURES_PATH.joinpath('output_long.tokenized').read_text()

    res = requests.post(
        f'http://localhost:8000/parse?format=tokenized',
        files={'input': input_text})
    assert expected_output == res.content.decode('utf-8')

def test_api_long_inline():
    input_text = FIXTURES_PATH.joinpath('input_long.txt').read_text()
    expected_output = FIXTURES_PATH.joinpath('output_long.inline').read_text()

    res = requests.post(
        f'http://localhost:8000/parse?format=inline',
        files={'input': input_text})
    assert expected_output == res.content.decode('utf-8')

    # check that 'inline' is also the default format
    res = requests.post(
        f'http://localhost:8000/parse',
        files={'input': input_text})
    assert expected_output == res.content.decode('utf-8')

