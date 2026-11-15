# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, host-managed episode memory; never runs code or accepts file paths.

Reuses saved observations and execution records. Each CodexPolicy owns
a separate instance for its episode.
"""
import base64
import json
from pathlib import Path

from ..io import InputError, write_json

MAX_NOTES_BYTES = 64 * 1024
MAX_RECALL_IMAGES = 12
MAX_MEMORY_CALLS = 16


def memory_tool_specs():
    def spec(name, description, properties, required=()):
        return dict(type='function', name=name, description=description,
            inputSchema=dict(type='object', properties=properties,
                             required=list(required), additionalProperties=False))
    limit = dict(limit=dict(type='integer', minimum=1, maximum=100))
    return [
        spec('memory_list_observations', 'List saved observations from this episode.', limit),
        spec('memory_get_observation', 'Recall a saved observation and selected images.',
             dict(observation_id=dict(type='integer', minimum=0),
                  image_indices=dict(type='array', items=dict(type='integer', minimum=0),
                                     maxItems=MAX_RECALL_IMAGES, uniqueItems=True)),
             ('observation_id',)),
        spec('memory_read_history', 'Read recent observation and host tool events.', limit),
        spec('memory_read_notes', 'Read persistent notes for this episode.', {}),
        spec('memory_write_notes', 'Replace persistent notes for this episode.',
             dict(text=dict(type='string', maxLength=MAX_NOTES_BYTES)), ('text',)),
    ]


MEMORY_TOOL_NAMES = frozenset(s['name'] for s in memory_tool_specs())


class EpisodeMemory:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir()
        self.notes = self.root/'NOTES.md'
        self.notes.write_text('# Agent episode notes\n')
        self.observations = []
        self.history = []

    def record_event(self, event):
        self.history.append(event)
        with (self.root/'history.jsonl').open('a') as stream:
            stream.write(json.dumps(event, ensure_ascii=False)+'\n')

    def record_observation(self, packet):
        # Only host-produced packets are registered. Agent arguments cannot
        # substitute paths, another episode, or a future observation.
        index = len(self.observations)
        write_json(self.root/f'observation_{index:06d}.json', packet)
        record = dict(observation_id=index, step_id=packet['step_id'],
                      image_count=len(packet['images']))
        self.observations.append(record)
        write_json(self.root/'observation_index.json', self.observations)
        self.record_event(dict(event='observation', **record))
        return index

    def call(self, name, arguments):
        specs = {s['name']: s['inputSchema'] for s in memory_tool_specs()}
        schema = specs.get(name)
        if (schema is None or not isinstance(arguments, dict)
                or set(arguments) - set(schema['properties'])
                or set(schema['required']) - set(arguments)):
            raise InputError('Invalid memory tool arguments')
        items = []
        if name in ('memory_list_observations', 'memory_read_history'):
            limit = arguments.get('limit', 20)
            if type(limit) is not int or not 1 <= limit <= 100:
                raise InputError('limit must be an integer in 1..100')
            result = ({'observations': self.observations[-limit:]}
                      if name == 'memory_list_observations' else
                      {'events': self.history[-limit:]})
        elif name == 'memory_get_observation':
            index = arguments['observation_id']
            if type(index) is not int or not 0 <= index < len(self.observations):
                raise InputError('observation_id does not exist in this episode')
            packet = json.loads((self.root/f'observation_{index:06d}.json').read_text())
            images = packet['images']
            indices = arguments.get('image_indices', list(range(len(images))))
            if (not isinstance(indices, list) or len(indices) > MAX_RECALL_IMAGES
                    or any(type(i) is not int or not 0 <= i < len(images) for i in indices)
                    or len(set(indices)) != len(indices)):
                raise InputError('Select at most 12 unique image indices')
            result = dict(observation_id=index, observation=packet)
            for i in indices:
                descriptor = images[i]
                data = base64.b64encode(Path(descriptor['path']).read_bytes()).decode('ascii')
                mime = descriptor.get('mime_type', 'image/png')
                items.append(dict(type='inputImage', imageUrl=f'data:{mime};base64,{data}'))
        elif name == 'memory_read_notes':
            result = self.notes.read_text()
        else:
            text = arguments['text']
            if not isinstance(text, str) or not text.strip() or len(text.encode('utf-8')) > MAX_NOTES_BYTES:
                raise InputError('Notes must be nonempty and at most 64 KiB UTF-8')
            self.notes.write_text(text.rstrip()+'\n')
            result = 'Episode notes saved.'
        self.record_event(dict(event='memory_call', tool=name, arguments=arguments))
        return [dict(type='inputText', text=result if isinstance(result, str) else
                     json.dumps(result, ensure_ascii=False)), *items]
