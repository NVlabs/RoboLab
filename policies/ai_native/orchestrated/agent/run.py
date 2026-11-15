# SPDX-FileCopyrightText: Copyright (c) 2026 Yu-Mool Shu and Lipxin Zheng
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
# Adapted from GPT-as-Policy (https://github.com/anonymous-report-421/GPT-as-Policy),
# hybrid_rollout/robolab/skill/run.py.

"""Add three rollout services to a full, persistent Codex policy agent."""
import base64
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from ..io import InputError, write_json
from ..settings import EFFORT, MODEL
from .episode_memory import MAX_MEMORY_CALLS, MEMORY_TOOL_NAMES, EpisodeMemory, memory_tool_specs
from .transport import StdioAppServer
from .workspace import prepare_workspace

NATIVE_WORK_ITEMS = frozenset(('commandExecution', 'fileChange', 'imageView',
    'webSearch', 'mcpToolCall', 'collabAgentToolCall', 'subAgentActivity',
    'sleep', 'imageGeneration'))


def inline_robot_context(skill_root):
    """Include robot and observation guidance in the persistent agent context."""
    return "\n\n".join((skill_root / "context" / name).read_text() for name in
                         ("teacher_context.md", "eef_control.md"))


def toml_value(value):
    """Codex -c values are TOML; JSON objects/quoted dotted keys are not equivalent."""
    if isinstance(value, dict):
        return '{' + ', '.join(json.dumps(k) + ' = ' + toml_value(v) for k, v in value.items()) + '}'
    return json.dumps(value)


def agent_config(audit, model=MODEL, effort=EFFORT):
    """Limit the session to the host-supplied robot and memory tools."""
    return dict(model=model, model_reasoning_effort=effort,
        sqlite_home=str(audit/'runtime_db'), log_dir=str(audit/'runtime_logs'),
        default_permissions=':read-only', **{
            'features.shell_tool': False, 'features.view_image': False,
            'features.js_repl': False, 'features.code_mode': False,
            'features.tool_search': False, 'features.tool_suggest': False,
            'features.skill_search': False, 'features.current_time_reminder': False,
            'features.memories': False, 'features.memory_tool': False,
            'features.multi_agent': False, 'features.apps': False,
            'features.plugins': False, 'features.goals': False,
            'web_search': 'disabled',
            'tools.experimental_request_user_input.enabled': False,
            'tools.update_plan.enabled': False,
        })


def content_items(packet, *, images):
    items = [dict(type='inputText', text=json.dumps(packet, separators=(',', ':')))]
    if images:
        for item in packet.get('images', []):
            data = base64.b64encode(Path(item['path']).read_bytes()).decode('ascii')
            mime = item.get('mime_type', 'image/png')
            items.append(dict(type='inputImage', imageUrl=f'data:{mime};base64,' + data))
    return items


class CodexPolicy:
    def __init__(self, workspace, codex, *, timeout=900, transport_factory=StdioAppServer,
                 skill_root, specs, tool_handlers, workspace_factory=prepare_workspace,
                 proposal_tool='policy_infer', provider='openai',
                 method='cosmos3_plus_gpt', model=MODEL, effort=EFFORT, bounded_memory=False):
        self.tool_handlers = tool_handlers
        self.proposal_tool = proposal_tool
        self.provider = provider
        self.model = model
        self.effort = effort
        self.method = method
        self.permissions = ':read-only'
        self.turn_count = 0
        self.memory_calls_this_decision = 0
        skill_root = Path(skill_root)
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=False)
        self.memory = EpisodeMemory(self.workspace/'episode_memory') if bounded_memory else None
        (self.workspace/'agent_events.jsonl').touch()
        self.agent_workspace = workspace_factory(self.workspace, skill_root)
        self.timeout = timeout
        skill = (skill_root/'SKILL.md').read_text()
        gate = (skill_root/'gate_prompt.md').read_text()
        capability = ('Only the supplied robot service and bounded memory tools are available. '
            if bounded_memory else 'Only the supplied robot service tools are available. ')
        capability += (
            'Use attached camera images and measured state directly. '
            'Keep same-episode history in this persistent conversation. '
            'Do not request shell, file, image-inspection, planning or delegation tools. '
            'Static robot context is included below; paths in service arguments '
            'are host-managed identifiers, not files you need to read.\n')
        skill = skill.replace('{{capability}}', capability)
        context = inline_robot_context(skill_root)
        self.prompt = skill + '\n\n# Unchanged baseline gate prompt\n\n' + gate + '\n\n' + context
        if self.memory is not None:
            self.prompt += ('\n\nUse memory_* tools to recall earlier observations, read execution history, '
                'and maintain episode notes. These host-managed tools never move the robot '
                'or execute code. At most 16 memory calls are allowed per robot decision.\n')
        (self.agent_workspace/'AGENTS.md').write_text(capability)
        self.prompt_sha256 = hashlib.sha256(self.prompt.encode()).hexdigest()
        (self.workspace/'SKILL.md').write_text(skill)
        (self.workspace/'gate_prompt.md').write_text(gate)
        (self.workspace/'PROMPT.md').write_text(self.prompt)
        specs = list(specs)
        if self.memory is not None:
            specs += memory_tool_specs()
        write_json(self.workspace/'tools.json', specs)
        version = subprocess.run([codex, '--version'], capture_output=True, text=True, check=True).stdout.strip()
        config = agent_config(self.workspace, self.model, self.effort)
        argv = [codex, 'app-server', '--stdio', '--strict-config']
        for key, value in config.items():
            argv += ['-c', key + '=' + toml_value(value)]
        write_json(self.workspace/'launch.json', dict(argv=argv, config=config))
        self.transport = transport_factory(argv, self.workspace)
        try:
            self.transport.request('initialize', dict(
                clientInfo=dict(name='robolab-hybrid-policy', version='1'),
                capabilities=dict(experimentalApi=True)), timeout)
            self.transport.notify('initialized', {})
            response = self.transport.request('thread/start', dict(
                cwd=str(self.agent_workspace), model=self.model, modelProvider=self.provider,
                config=dict(model_reasoning_effort=self.effort), developerInstructions=self.prompt,
                dynamicTools=specs, ephemeral=False,
                allowProviderModelFallback=False, approvalPolicy='never',
                approvalsReviewer='auto_review', permissions=self.permissions,
                runtimeWorkspaceRoots=[str(self.agent_workspace)]), timeout)
            if response.get('model') != self.model or response.get('reasoningEffort') != self.effort:
                raise RuntimeError(f'Codex model/effort differs from the required {self.model}/{self.effort}: '
                                   f'{response.get("model")}/{response.get("reasoningEffort")}')
            self.thread_id = response['thread']['id']
            write_json(self.workspace/'worker.json', dict(model=self.model, model_provider=self.provider,
                evaluation_method=self.method, reasoning_effort=self.effort,
                codex_version=version, thread_id=self.thread_id, pid=self.transport.process.pid,
                prompt_sha256=self.prompt_sha256, no_rollback=True, tools=[s['name'] for s in specs],
                agent_workspace=str(self.agent_workspace), native_tools_enabled=False,
                bounded_memory_enabled=self.memory is not None, permissions=self.permissions,
                persistent_episode_thread=True,
                action_execution='blocking_tool_reply',
                prompt_geometry=('gripper_camera' if self.method == 'cosmos3_plus_gpt' else None)))
        except BaseException:
            self.close()
            raise

    def _timing(self, event, **fields):
        record = dict(schema='robolab.astra_timing.v1', event=event,
            recorded_at=datetime.now(timezone.utc).isoformat(), **fields)
        with (self.workspace/'timing.jsonl').open('a') as stream:
            stream.write(json.dumps(record, ensure_ascii=False)+'\n')

    def _turn(self, text, *, images=()):
        self.turn_count += 1
        started = time.monotonic()
        result = self.transport.request('turn/start', dict(threadId=self.thread_id,
            model=self.model, effort=self.effort, approvalPolicy='never', approvalsReviewer='auto_review',
            permissions=self.permissions, cwd=str(self.agent_workspace),
            runtimeWorkspaceRoots=[str(self.agent_workspace)], input=[dict(type='text', text=text)] + [
                dict(type='localImage', path=item['path'], detail='high') for item in images]), self.timeout)
        self._timing('turn_started', turn_id=result['turn']['id'],
            request_seconds=time.monotonic()-started)
        return result['turn']['id']

    def run(self, rollout):
        destination = (
            "OpenAI Codex"
            if self.provider.lower() == "openai"
            else f"the configured {self.provider} model provider through Codex"
        )
        capabilities = ('Only the supplied robot service and bounded memory tools, when enabled, are available. '
            'Use attached images and measured state; retain history in the conversation. ')
        turn_id = self._turn('Act as the autonomous policy agent for this single simulation rollout. '
            + capabilities +
            'The user authorizes sending this episode\'s camera images, proprio, robot-only FK, '
            f'task text and same-episode history to {destination} for online decisions. '
            'First call: ' + json.dumps(rollout.next_call()))
        call_index = 0
        errors = 0
        continuations = 0
        proposal_ready_at = None
        deadline = time.monotonic() + self.timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Codex made no completed service or native tool call within the configured timeout')
            event = self.transport.next_message(remaining)
            method, params = event.get('method'), event.get('params', {})
            if method == 'item/tool/call':
                if params.get('threadId') != self.thread_id or params.get('turnId') != turn_id:
                    raise RuntimeError('Tool call belongs to another thread or turn')
                name, arguments = params['tool'], params['arguments']
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                received_at = time.monotonic()
                received_wall = datetime.now(timezone.utc).isoformat()
                print(json.dumps(dict(event='tool_start', tool=name, call=call_index,
                                      phase=rollout.phase, step_id=rollout.tick)), flush=True)
                audit = dict(call_id=params['callId'], tool=name, arguments=arguments)
                write_json(self.workspace/f'call_{call_index:04d}_request.json', audit)
                if self.memory is not None and name in MEMORY_TOOL_NAMES:
                    try:
                        self.memory_calls_this_decision += 1
                        if self.memory_calls_this_decision > MAX_MEMORY_CALLS:
                            raise InputError('Memory-tool call limit reached for this decision')
                        items = self.memory.call(name, arguments)
                        memory_result = dict(success=True, contentItems=items)
                    except InputError as error:
                        errors += 1
                        memory_result = dict(success=False, contentItems=[
                            dict(type='inputText', text=str(error))])
                    self.transport.reply(event['id'], memory_result)
                    write_json(self.workspace/f'call_{call_index:04d}_result.json', memory_result)
                    call_index += 1
                    # Memory lookup never invokes a rollout handler or changes
                    # proposal timing / simulator counters.
                    if memory_result["success"]:
                        deadline = time.monotonic() + self.timeout
                    elif errors >= 5:
                        raise RuntimeError(
                            "Five rejected tool calls; stopping without further execution")
                    continue
                handlers = self.tool_handlers(rollout)
                decision_index = len(getattr(rollout, 'history', []))
                handler_started = time.monotonic()
                try:
                    if name not in handlers or not isinstance(arguments, dict):
                        raise InputError('Unknown host service')
                    packet = handlers[name](**arguments)
                except InputError as error:
                    handler_seconds = time.monotonic()-handler_started
                    errors += 1
                    packet = dict(error=str(error), no_execution=True, next_call=rollout.next_call())
                    self._timing('service_tool_completed', tool=name, step_id=rollout.tick,
                        success=False, handler_seconds=handler_seconds,
                        error_type=type(error).__name__)
                    held = False
                    if (self.method == 'cosmos3_plus_gpt' and name == 'robolab_execute'
                            and rollout.phase == 'execute' and errors >= 3):
                        self._timing('astra_decision_rejected', decision=decision_index,
                            step_id=rollout.tick, error=str(error))
                        packet = rollout.hold_after_invalid_action(error=str(error), **arguments)
                        if self.memory is not None:
                            packet['observation_id'] = self.memory.record_observation(packet)
                            self.memory.record_event(dict(event='host_execution',
                                tool='invalid_action_hold', step_id=packet['step_id'], error=str(error)))
                        proposal_ready_at = None
                        self.memory_calls_this_decision = 0
                        errors = 0
                        held = True
                    self.transport.reply(event['id'], dict(success=False,
                        contentItems=content_items(packet, images=held)))
                    if errors >= 5:
                        raise RuntimeError('Five rejected tool calls; stopping without further execution') from error
                except BaseException as error:
                    self._timing('service_tool_completed', tool=name, step_id=rollout.tick,
                        success=False, handler_seconds=time.monotonic()-handler_started,
                        error_type=type(error).__name__)
                    raise
                else:
                    handler_seconds = time.monotonic()-handler_started
                    if name == 'robolab_execute' and proposal_ready_at is not None:
                        response = arguments.get('response', {})
                        self._timing('astra_decision_received', tool=name,
                            step_id=rollout.tick,
                            decision=response.get('decision', decision_index),
                            mode=response.get('mode'),
                            received_at=received_wall,
                            astra_decision_seconds=received_at-proposal_ready_at)
                        proposal_ready_at = None
                    self._timing('service_tool_completed', tool=name, step_id=rollout.tick,
                        success=True, handler_seconds=handler_seconds)
                    errors = 0
                    if self.memory is not None:
                        if name != self.proposal_tool:
                            packet['observation_id'] = self.memory.record_observation(packet)
                        self.memory.record_event(dict(event='host_execution', tool=name,
                            step_id=packet['step_id'], response=arguments.get('response')))
                    if name == 'robolab_execute':
                        self.memory_calls_this_decision = 0
                    self.transport.reply(event['id'], dict(success=True,
                        contentItems=content_items(packet, images=name != self.proposal_tool)))
                    if name == self.proposal_tool:
                        proposal_ready_at = time.monotonic()
                        self._timing('proposal_ready', tool=name, step_id=rollout.tick,
                            cosmos_inference_seconds=packet.get('inference_seconds'),
                            cosmos_server_timing=packet.get('server_timing'))
                write_json(self.workspace/f'call_{call_index:04d}_result.json', packet)
                call_index += 1
                print(json.dumps(dict(event='tool_done', tool=name, step_id=rollout.tick,
                    phase=rollout.phase, counters=rollout.counters)), flush=True)
                deadline = time.monotonic() + self.timeout
                if (self.method == 'cosmos3_plus_gpt' and rollout.phase != 'done'
                        and rollout.decision_budget_exhausted()):
                    rollout.drain_decision_budget()
                    return
            elif method in ('item/started', 'item/completed'):
                if params.get('threadId') != self.thread_id or params.get('turnId') != turn_id:
                    continue
                item = params.get('item', {})
                kind = item.get('type')
                if kind in NATIVE_WORK_ITEMS or (kind == 'agentMessage' and method == 'item/completed'):
                    # Public outputs only: never copy private reasoning items.
                    with (self.workspace/'agent_events.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(method=method, **params), ensure_ascii=False)+'\n')
                    event_record = dict(event='agent_activity', phase=method.split('/')[-1],
                        item_type=kind, item_id=item.get('id'), step_id=rollout.tick,
                        status=item.get('status'))
                    if kind == 'agentMessage':
                        event_record['text'] = item.get('text', '')
                    elif kind == 'commandExecution':
                        event_record['command'] = item.get('command')
                        event_record['exit_code'] = item.get('exitCode')
                    print(json.dumps(event_record, ensure_ascii=False), flush=True)
                    if kind in NATIVE_WORK_ITEMS:
                        raise RuntimeError(f'Codex attempted disabled native tool: {kind}')
            elif method == 'turn/completed' and params.get('threadId') == self.thread_id:
                turn = params.get('turn', {})
                if turn.get('id') != turn_id:
                    continue
                write_json(self.workspace/f'turn_{self.turn_count:06d}_completed.json', turn)
                self._timing('turn_completed', turn_id=turn_id,
                    status=turn.get('status'), usage=turn.get('usage'))
                if turn.get('status') != 'completed':
                    raise RuntimeError(f'Codex turn ended: {turn.get("status")}')
                if rollout.phase == 'done':
                    return
                if continuations >= 2:
                    raise RuntimeError('Codex ended repeatedly before completing the episode')
                # Context-only continuation, never a human action judgement.
                continuations += 1
                turn_id = self._turn('The same episode is unfinished. Continue the skill; next call: ' +
                                     json.dumps(rollout.next_call()))
                deadline = time.monotonic() + self.timeout
            elif method in ('error', 'turn/failed'):
                if method == 'error' and params.get('willRetry') is True:
                    # App-server owns model transport retry; this does not
                    # repeat inference, a tool handler, or physical execution.
                    print(json.dumps(dict(event='codex_transport_retry',
                        error=params.get('error'), step_id=rollout.tick)), flush=True)
                    continue
                raise RuntimeError(f'Codex error: {params}')
            elif 'id' in event and 'method' in event:
                raise RuntimeError(f'Unexpected Codex capability request: {method}')

    def close(self):
        self.transport.close()
