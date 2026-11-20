import json
from types import SimpleNamespace

import pytest

from integrations.hermes import delegation, runtime

MODEL = "synthetic-arena-model"


def test_profile_has_requested_delegation_limits(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime.os, 'environ', {})
    runtime.prepare_home(tmp_path, model=MODEL)
    config = (tmp_path / 'config.yaml').read_text()
    assert 'max_concurrent_children: 1000\n' in config
    assert 'oneshot_max_children: 1000\n' in config
    assert 'max_spawn_depth: 10\n' in config


def test_research_search_backend_receives_existing_proxy(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime.os, 'environ', {'HTTPS_PROXY': 'http://internet-proxy:3128'})
    runtime.prepare_home(tmp_path, model=MODEL)
    assert runtime.os.environ['DDGS_PROXY'] == 'http://internet-proxy:3128'


def test_solver_home_does_not_invent_internet_proxy(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime.os, 'environ', {})
    runtime.prepare_home(tmp_path, model=MODEL)
    assert 'DDGS_PROXY' not in runtime.os.environ


def test_fork_snapshot_keeps_completed_tools_and_excludes_pending_batch():
    messages = [
        {'role': 'user', 'content': 'research goal'},
        {'role': 'assistant', 'content': None, 'reasoning_content': 'reason',
         'tool_calls': [{'id': 'read'}]},
        {'role': 'tool', 'tool_call_id': 'read', 'content': 'evidence'},
        {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'spawn'}]},
    ]
    history = delegation.fork_history(messages)
    assert history == messages[:3]
    history[0]['content'] = 'changed by child'
    assert messages[0]['content'] == 'research goal'


def test_seeded_child_inherits_history_once():
    calls = []
    child = SimpleNamespace(run_conversation=lambda **kw: calls.append(kw))
    history = [{'role': 'user', 'content': 'prior'}]
    delegation.seed_child(child, history)
    child.run_conversation(user_message='implementation')
    child.run_conversation(user_message='followup')
    assert calls[0]['conversation_history'] == history
    assert 'conversation_history' not in calls[1]


def test_format_repair_uses_ready_result_without_research_or_tools():
    calls = []
    message = SimpleNamespace(content='{"result":"evidence"}', tool_calls=None)
    child = SimpleNamespace(
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kw: calls.append(kw) or SimpleNamespace(
                choices=[SimpleNamespace(message=message)],
                usage=SimpleNamespace(prompt_tokens=13, completion_tokens=7, total_tokens=20),
            )
        ))),
    )
    result = {'final_response': 'result: evidence', 'api_calls': 7, 'messages': []}
    schema = {'type': 'object', 'required': ['result']}

    def validate(value, contract):
        assert contract == schema
        try:
            parsed = json.loads(value)
        except ValueError:
            return False, ['not JSON']
        return 'result' in parsed, []

    valid, errors, corrections = delegation.repair_output(
        child, result, schema, validate, model=MODEL, parameters={
            "reasoning_effort": "low", "temperature": 0.25,
        },
    )
    assert (valid, errors, corrections) == (True, [], 1)
    assert result['final_response'] == message.content
    assert result['api_calls'] == 8
    assert child.session_prompt_tokens == 13
    assert child.session_completion_tokens == 7
    assert child.session_total_tokens == 20
    assert len(calls) == 1
    request = calls[0]
    assert request['model'] == MODEL
    assert request['reasoning_effort'] == 'low'
    assert request['temperature'] == 0.25
    assert request['tool_choice'] == 'none'
    assert 'tools' not in request
    prompt = json.loads(request['messages'][1]['content'])
    assert prompt == {'answer': 'result: evidence', 'schema': schema, 'errors': ['not JSON']}
    assert len(result['messages']) == 2


def test_valid_child_result_never_triggers_format_repair():
    child = SimpleNamespace()
    result = {'final_response': '{"result":1}'}
    assert delegation.repair_output(
        child, result, {}, lambda *_: (True, []), model=MODEL,
    ) == (True, [], 0)


def test_harness_diagnostic_does_not_edit_answer():
    events = []
    child = SimpleNamespace(
        _turn_failed_file_mutations={'file': 'failed'},
        _file_mutation_verifier_enabled=lambda: True,
        _file_mutations_still_failed=lambda value: value,
        _format_file_mutation_failure_footer=lambda value: 'File-mutation verifier: failure',
    )
    answer = '{"answer":"File-mutation verifier is quoted data"}  \n'
    assert delegation.mutation_diagnostic(child, answer, events) == answer
    assert events == [{'kind': 'file_mutation', 'message': 'File-mutation verifier: failure'}]


def test_empty_or_interrupted_child_is_not_reformatted():
    for result in [{'final_response': ''}, {'final_response': 'partial', 'interrupted': True}]:
        actual = delegation.repair_output(
            SimpleNamespace(), result, {}, lambda *_: (False, ['bad']), model=MODEL,
        )
        assert actual == (False, ['bad'], 0)


def test_invalid_fork_mode_rejected_before_building_child():
    calls = []
    parent = SimpleNamespace(_arena_fork_history=[{'role': 'user', 'content': 'goal'}])
    def build(*args, **kw):
        calls.append(args)
        return [], None
    result = delegation.build_children(build, [{'goal': 'new', 'context_mode': 'invalid'}],
                                       parent_agent=parent)
    assert result == ([], 'context_mode must be fresh or fork')
    assert calls == []


def test_missing_fork_history_is_explicit():
    with pytest.raises(ValueError, match='history'):
        delegation.fork_history(None)


def test_format_repair_rejects_profile_that_can_enable_tools():
    result = {'final_response': 'ready evidence'}
    with pytest.raises(ValueError, match='model parameters'):
        delegation.repair_output(
            SimpleNamespace(), result, {}, lambda *_: (False, ['bad']), model=MODEL,
            parameters={'tools': [{'type': 'function'}]},
        )
    assert result == {'final_response': 'ready evidence'}
