import os
import logging
import sys

from tensorflow.python.framework.errors_impl import InvalidArgumentError

from falcon import HTTP_200, HTTP_422
import hug

from neuralseg.splitter import DEFAULT_ARGS, load_models, parse_args, segment_text


"""REST API for NeuralEDUSeg

Examples:

    hug -f splitter_api.py # run the server

    curl -X POST -F "input=@/tmp/input.txt" http://localhost:8000/parse # defaults to 'inline' output format
    curl -X POST -F "input=@/tmp/input.txt" http://localhost:8000/parse?format=inline

    curl -X POST -F "input=@/tmp/input.txt" http://localhost:8000/parse?format=tokenized
    curl -X POST -F "input=@/tmp/input.txt" http://localhost:8000/parse?format=json
    curl -X POST -F "input=@/tmp/input.txt" "http://localhost:8000/parse?format=json&debug=True"
"""


logging.getLogger("tensorflow").setLevel(logging.ERROR)

INPUT_FILEPATH = '/tmp/input.txt'
OUTPUT_FILEPATH = '/tmp/output.txt'

sys.argv = ['splitter_api.py', INPUT_FILEPATH, OUTPUT_FILEPATH]
ARGS = parse_args(namespace=DEFAULT_ARGS)
RST_DATA, MODEL, SPACY_NLP = load_models(ARGS)


@hug.response_middleware()
def process_data(request, response, resource):
    """This is a middleware function that gets called for every request a hug API processes.
    It will allow Javascript clients on other hosts / ports to access the API (CORS request).
    """
    response.set_header('Access-Control-Allow-Origin', '*')


@hug.post(
    '/parse', output=hug.output_format.file,
    examples=['format=inline', 'format=json', 'format=tokenized'])
def call_parser(body, format: hug.types.text = 'inline', debug: hug.types.boolean = False, response: hug.response.Response = response):
    if 'input' in body:
        input_file_content = body['input']
        input_text = input_file_content.decode('utf-8')

        # ~ from pudb.remote import set_trace; set_trace(term_size=(160, 40), host='0.0.0.0', port=6900)
        try:
            output_text = segment_text(input_text, RST_DATA, MODEL, SPACY_NLP, output_format=format, debug=debug)
        except InvalidArgumentError as err:
            response.status = HTTP_422  # Unprocessable Entity
            output_text = f"Input text too short for trained NeuralEDUSeg model. Error: {err}"

        with open(OUTPUT_FILEPATH, 'w') as output_file:
            output_file.write(output_text)

        return OUTPUT_FILEPATH

    else:
        return {'body': body}

@hug.get('/status')
def get_status():
    return HTTP_200
