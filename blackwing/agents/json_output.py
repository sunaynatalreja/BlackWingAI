"""Conservative JSON syntax repair; never invent values or close truncated data."""
from __future__ import annotations

import json
import re


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON object key')
        result[key] = value
    return result


def parse_agent_json(raw: str):
    """Allow only missing separators, trailing commas and a whole JSON fence.

    Every value, key, citation and ordering remains exactly as supplied.
    Missing colons, unescaped/unfinished strings, unmatched containers,
    bare identifiers, and duplicate keys remain errors for the bounded retry.
    """
    try:
        return json.loads(raw, object_pairs_hook=_unique_pairs), []
    except json.JSONDecodeError as exc:
        failure = exc
    candidate = raw.strip()
    repairs = []
    fence = re.fullmatch(r'```(?:json)?\s*(.*?)\s*```', candidate, re.I | re.S)
    if fence:
        candidate = fence.group(1)
        repairs.append('removed_json_fence')
    if len(candidate) > 32_000:
        raise failure
    decoder = json.JSONDecoder()
    tokens = []
    position = 0
    try:
        while position < len(candidate):
            char = candidate[position]
            if char.isspace():
                position += 1
                continue
            if char in '{}[],:':
                tokens.append((char, char))
                position += 1
                continue
            if char != '"' and char not in '-0123456789tfn':
                raise ValueError('Not a JSON token')
            value, end = decoder.raw_decode(candidate, position)
            if isinstance(value, (dict, list)):
                raise ValueError('Unexpected container token')
            tokens.append(('string' if isinstance(value, str) else 'value', candidate[position:end]))
            position = end
        cursor = 0
        output = []

        def kind(offset=0):
            at = cursor + offset
            return tokens[at][0] if at < len(tokens) else None

        def take(expected=None):
            nonlocal cursor
            if cursor >= len(tokens) or (expected is not None and kind() != expected):
                raise ValueError('Missing JSON token')
            output.append(tokens[cursor][1])
            cursor += 1

        def change(name):
            if len(repairs) >= 4:
                raise ValueError('Too many syntax changes')
            repairs.append(name)

        def value(depth=0):
            nonlocal cursor
            if depth > 48:
                raise ValueError('Nested JSON is too deep')
            if kind() == '{':
                take('{')
                if kind() == '}':
                    take('}')
                    return
                while True:
                    take('string')
                    take(':')
                    value(depth + 1)
                    if kind() == '}':
                        take('}')
                        return
                    if kind() == ',':
                        if kind(1) == '}':
                            cursor += 1
                            change('removed_trailing_comma')
                            take('}')
                            return
                        else:
                            take(',')
                    elif kind() == 'string' and kind(1) == ':':
                        output.append(',')
                        change('inserted_missing_comma')
                    else:
                        raise ValueError('Ambiguous object syntax')
            elif kind() == '[':
                take('[')
                if kind() == ']':
                    take(']')
                    return
                while True:
                    value(depth + 1)
                    if kind() == ']':
                        take(']')
                        return
                    if kind() == ',':
                        if kind(1) == ']':
                            cursor += 1
                            change('removed_trailing_comma')
                            take(']')
                            return
                        else:
                            take(',')
                    elif kind() in {'string', 'value', '{', '['}:
                        output.append(',')
                        change('inserted_missing_comma')
                    else:
                        raise ValueError('Ambiguous array syntax')
            elif kind() in {'string', 'value'}:
                take()
            else:
                raise ValueError('Missing JSON value')

        value()
        if cursor != len(tokens) or not repairs:
            raise ValueError('Incomplete or ambiguous JSON')
        return json.loads(''.join(output), object_pairs_hook=_unique_pairs), repairs
    except (ValueError, IndexError, RecursionError):
        raise failure


def syntax_context(raw, position):
    """Show JSON punctuation around an error without revealing response text."""
    fragment = raw[max(0, position - 65):position + 65]
    masked = ''.join(char if char in '{}[],:" \t\r\n' else '?' for char in fragment)
    return re.sub(r'\?+', '<value>', masked)
