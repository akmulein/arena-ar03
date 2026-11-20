import copy
import importlib
import json
from functools import wraps
from typing import Any


def fork_history(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError('fork conversation history is unavailable')
    pending: set[str] = set()
    start = len(messages)
    for index, message in enumerate(messages):
        calls = message.get('tool_calls') or []
        if calls:
            if not pending:
                start = index
            pending.update(call['id'] for call in calls)
        if message.get('role') == 'tool':
            pending.discard(message.get('tool_call_id'))
    return copy.deepcopy(messages[:start] if pending else messages)


def seed_child(child: Any, history: list[dict[str, Any]]) -> None:
    original = child.run_conversation
    history = copy.deepcopy(history)

    @wraps(original)
    def run(*args, **kwargs):
        nonlocal history
        inherited, history = history, []
        if inherited and 'conversation_history' not in kwargs:
            kwargs['conversation_history'] = inherited
        return original(*args, **kwargs)

    child.run_conversation = run


def build_children(original: Any, task_list: list[dict[str, Any]], *args, **kwargs):
    if any(task.get('context_mode', 'fresh') not in ('fresh', 'fork') for task in task_list):
        return [], 'context_mode must be fresh or fork'
    history: list[dict[str, Any]] = []
    if any(task.get('context_mode') == 'fork' for task in task_list):
        try:
            history = fork_history(getattr(kwargs['parent_agent'], '_arena_fork_history', None))
        except ValueError as error:
            return [], str(error)
    children, error = original(task_list, *args, **kwargs)
    if error is None:
        for _, task, child in children:
            if task.get('context_mode') == 'fork':
                seed_child(child, history)
    return children, error


def repair_output(
    child: Any, result: dict[str, Any], schema: dict[str, Any], validate: Any, *,
    model: str, parameters: dict[str, Any] | None = None,
):
    text = result.get('final_response') or ''
    valid, errors = validate(text, schema)
    if valid or not text.strip() or result.get('interrupted'):
        return valid, errors, 0
    from integrations.hermes.runtime import model_parameters

    parameters = model_parameters(parameters)
    messages = [
        {'role': 'system', 'content': (
            'Reformat the completed research answer to the supplied JSON Schema. '
            'Preserve its conclusions and evidence. Do not investigate, call tools, '
            'invent missing facts or repeat the task. Return only the formatted answer.'
        )},
        {'role': 'user', 'content': json.dumps(
            {'answer': text, 'schema': schema, 'errors': errors}, ensure_ascii=False,
        )},
    ]
    result['api_calls'] = int(result.get('api_calls') or 0) + 1
    response = child.client.chat.completions.create(
        **((parameters or {}) | {
            'model': model, 'messages': messages, 'stream': False, 'tool_choice': 'none',
        }),
    )
    child.session_api_calls = getattr(child, 'session_api_calls', 0) + 1
    usage = getattr(response, 'usage', None)
    if usage is not None:
        for name in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            value = getattr(usage, name, None)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                field = 'session_' + name
                setattr(child, field, getattr(child, field, 0) + value)
    message = response.choices[0].message
    corrected = message.content
    if message.tool_calls or not isinstance(corrected, str) or not corrected.strip():
        return False, [*errors, 'format correction did not return text without tools'], 1
    result['final_response'] = corrected
    if isinstance(result.get('messages'), list):
        result['messages'].extend([
            messages[1], {'role': 'assistant', 'content': corrected},
        ])
    valid, errors = validate(corrected, schema)
    return valid, errors, 1


def mutation_diagnostic(agent: Any, answer: str, events: list[dict[str, str]]) -> str:
    failed = getattr(agent, '_turn_failed_file_mutations', None) or {}
    if failed and agent._file_mutation_verifier_enabled():
        failed = agent._file_mutations_still_failed(failed)
        message = agent._format_file_mutation_failure_footer(failed)
        if message:
            events.append({'kind': 'file_mutation', 'message': message})
    return answer


def install(
    agent_class: Any, events: list[dict[str, str]], *, model: str,
    parameters: dict[str, Any] | None = None,
) -> None:
    finalizer: Any = importlib.import_module("agent.turn_finalizer")
    delegate: Any = importlib.import_module("tools.delegate_tool")
    child_run: Any = importlib.import_module("tools.delegate_tool_child_run")
    validate_output = importlib.import_module("tools.delegation_output_schema").validate_output

    original_execute = agent_class._execute_tool_calls
    original_build = delegate._build_children
    original_description = delegate._build_top_level_description
    original_finalize = finalizer.finalize_turn

    @wraps(original_execute)
    def execute(self, assistant_message, messages, *args, **kwargs):
        self._arena_fork_history = fork_history(messages)
        try:
            return original_execute(self, assistant_message, messages, *args, **kwargs)
        finally:
            self._arena_fork_history = None

    @wraps(original_build)
    def build(task_list, *args, **kwargs):
        return build_children(original_build, task_list, *args, **kwargs)

    def validate(child, result, task_index, child_task_id, relay_child_text):
        schema = getattr(child, '_delegate_output_schema', None)
        if not isinstance(schema, dict):
            return child_run._SchemaOutcome(schema, None, [], 0)
        valid, errors, corrections = repair_output(
            child, result, schema, validate_output, model=model, parameters=parameters,
        )
        return child_run._SchemaOutcome(schema, valid, errors, corrections)

    def diagnostic(agent, final_response, logger):
        local_events: list[dict[str, str]] = []
        answer = mutation_diagnostic(agent, final_response, local_events)
        agent._arena_diagnostics = local_events
        events.extend(local_events)
        return answer

    @wraps(original_finalize)
    def finalize(agent, *args, **kwargs):
        agent._arena_diagnostics = []
        result = original_finalize(agent, *args, **kwargs)
        result['diagnostics'] = list(agent._arena_diagnostics)
        return result

    @wraps(original_description)
    def describe(*args, **kwargs):
        text = original_description(*args, **kwargs)
        text = text.replace(
            "Children know nothing of this conversation: pass everything needed via 'context'",
            "Fresh children have no conversation history: pass needed background via 'context'",
        )
        return text + (
            '\nChoose context_mode=fork for implementation or analysis inheriting your completed '
            'conversation; choose fresh for independent ideas with only explicit task context. '
            'Every child may delegate within the configured depth. '
            'Return concise findings and paths.'
        )

    task_schema = delegate.DELEGATE_TASK_SCHEMA['parameters']['properties']['tasks']
    properties = task_schema['items']['properties']
    properties['context_mode'] = {
        'type': 'string', 'enum': ['fresh', 'fork'], 'default': 'fresh',
        'description': 'Fresh uses explicit context; fork inherits completed parent history.',
    }
    properties['goal']['description'] = 'Specific task for this child, with its scope and result.'
    properties['context']['description'] = (
        'Background for fresh children, or focused instructions supplementing fork history.'
    )
    agent_class._execute_tool_calls = execute
    delegate._build_children = build
    delegate._build_top_level_description = describe
    delegate._validate_child_output_schema = validate
    child_run._validate_child_output_schema = validate
    finalizer._append_file_mutation_footer = diagnostic
    finalizer.finalize_turn = finalize
