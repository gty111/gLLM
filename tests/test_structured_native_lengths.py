"""Native XGrammar length bounds through the sampling integration."""
import json

import jsonschema
import pytest

from test_structured_string_lengths import accepts, backend, byte_compiler, title_schema


@pytest.mark.parametrize('length', [0, 1, 127, 128, 129, 319, 320, 321])
def test_recap_bounds(backend, length):
    schema = title_schema({'minLength': 1, 'maxLength': 320})
    assert accepts(backend, schema, json.dumps({'title': 'a' * length})) == (1 <= length <= 320)


@pytest.mark.parametrize('bounds', [
    {'minLength': 128, 'maxLength': 130},
    {'minLength': 129},
    {'minLength': 0, 'maxLength': 320},
    {'minLength': 0, 'maxLength': 1025},
])
def test_native_lower_and_upper_bounds(backend, bounds):
    schema = title_schema(bounds)
    for length in [0, 128, 129, 130, 131]:
        value = {'title': 'a' * length}
        assert accepts(backend, schema, json.dumps(value)) == jsonschema.Draft202012Validator(schema).is_valid(value)


def test_native_lengths_with_many_fields(backend):
    properties = {f'field_{i}': {'type': 'string', 'minLength': 1, 'maxLength': 320} for i in range(8)}
    schema = {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}
    assert accepts(backend, schema, json.dumps({name: 'x' for name in properties}))
    assert not accepts(backend, schema, json.dumps({name: '' for name in properties}))
