"""
Generate conversation-format JSON training data from EntityInstance objects.

Provides loaders for NER (load_data_uniner_from_instances), point-marked NER
(load_point_data_uniner_from_instances), and all-spans NER
(load_data_uniner_from_instances_all_spans).
"""

import json
import copy
from collections import Counter
import random


def draw_with_probability(p):
    return random.choices([0, 1], weights=[1-p, p])[0]


def normalize_entity_types(instances, entity_type_map):
    seen_entity_types = set()
    if entity_type_map is None:
        return instances, []
    normalized_instances = []
    for inst in instances:
        if len(inst.gold_entities) == 0:
            normalized_instances.append(inst)
            continue
        if type(inst.gold_entities[0]) == type([]):
            normalized_gold_entity_sets = []
            for entity_set in inst.gold_entities:
                normalized_gold_entity_set = []
                for t, s in entity_set:
                    if t in entity_type_map:
                        normalized_gold_entity_set.append((entity_type_map[t], s))
                        seen_entity_types.add(entity_type_map[t])
                    else:
                        normalized_gold_entity_set.append((t, s))
                        seen_entity_types.add(t)
                normalized_gold_entity_sets.append(normalized_gold_entity_set)
            normalized_inst = copy.deepcopy(inst)
            normalized_inst.gold_entities = normalized_gold_entity_sets
        else:
            normalized_gold_entity_set = []
            for t, s in inst.gold_entities:
                if t in entity_type_map:
                    normalized_gold_entity_set.append((entity_type_map[t], s))
                    seen_entity_types.add(entity_type_map[t])
                else:
                    normalized_gold_entity_set.append((t, s))
            normalized_inst = copy.deepcopy(inst)
            try:
                normalized_inst.gold_entities = list(set(normalized_gold_entity_set))
            except Exception:
                normalized_inst.gold_entities = []
        normalized_instances.append(normalized_inst)
    return normalized_instances, seen_entity_types


def load_data_uniner_from_instances(instances, train_instances, json_to_write, entity_type_map=None,
                                    negative_sampling_multiplier=0.3, do_negative_sampling=True,
                                    uniform_negative_sampling=False):
    original_length = len(instances)
    instances, _ = normalize_entity_types(instances, entity_type_map)
    train_instances, _ = normalize_entity_types(train_instances, entity_type_map)
    assert len(instances) == original_length

    all_types = []
    all_types_counter = []
    for inst in train_instances:
        for t, s in inst.gold_entities:
            all_types_counter.append(t)
            if t not in all_types:
                all_types.append(t)
    all_types_proportions = {}
    num_spans = len(all_types_counter)
    all_types_counter = Counter(all_types_counter)
    for t in all_types:
        all_types_proportions[t] = all_types_counter[t] / num_spans

    conversation_set = []
    num_negative_samples = 0

    for s_idx, inst in enumerate(instances):
        messages = []
        messages.append({"from": "human", "value": "{}".format(inst.sentence)})
        messages.append({"from": "gpt", "value": "I've read this text."})

        for type_ in all_types:
            messages_ = copy.deepcopy(messages)
            if len(inst.gold_entities) == 0:
                spans_str = '[]'
                spans = []
            else:
                str_format = True
                if type(inst.gold_entities[0]) == type([]):
                    spans = []
                    for entity_set in inst.gold_entities:
                        set_type = entity_set[0][0]
                        if set_type == type_:
                            spans.append(list(set([s for t, s in entity_set])))
                    str_format = False
                else:
                    spans = [s for t, s in inst.gold_entities if t == type_]

                spans_ = []
                for span in spans:
                    if type(span) == str:
                        if span.endswith(']'):
                            span = span[:-1]
                        spans_.append(span)
                    else:
                        span_set = []
                        for s in span:
                            if s.endswith(']'):
                                s = s[:-1]
                            span_set.append(s)
                        spans_.append(span_set)

                if str_format:
                    spans_ = list(set(spans_))
                    spans_str = '[' + ','.join(['"{}"'.format(s) for s in spans_]) + ']'
                else:
                    spans_str = spans_

            skip_example = True
            if len(spans) == 0:
                if do_negative_sampling:
                    if draw_with_probability(all_types_proportions[type_] * negative_sampling_multiplier) == 1:
                        skip_example = False
                if skip_example and do_negative_sampling:
                    continue

            if len(spans) == 0:
                num_negative_samples += 1

            messages_.append({"from": "human", "value": "What describes {} in the text?".format(type_)})
            messages_.append({"from": "gpt", "value": spans_str})
            id_ = str(s_idx) + '_' + type_
            conversation_set.append({'id': id_, 'conversations': messages_})

    with open(json_to_write, "w") as jsonl_file:
        print('writing {} conversations to {}'.format(len(conversation_set), json_to_write))
        print('number of negative samples', num_negative_samples)
        json.dump(conversation_set, jsonl_file, indent=2)


def load_point_data_uniner_from_instances(instances, json_to_write, entity_type_map=None, openai_format=False):
    instances, all_types = normalize_entity_types(instances, entity_type_map)
    user_role = 'human' if not openai_format else 'user'
    assistant_role = 'gpt' if not openai_format else 'assistant'
    conversation_set = []

    for inst in instances:
        messages = []
        if len(inst.messages) == 0:
            continue
        if 'entity_type' in inst.messages and entity_type_map is not None:
            inst.messages['entity_type'] = entity_type_map.get(inst.messages['entity_type'], inst.messages['entity_type'])

        if openai_format:
            messages.append({"role": user_role, "content": "{}".format(inst.messages['text'])})
            messages.append({"role": assistant_role, "content": "I've read this text."})
            messages.append({"role": user_role, "content": inst.messages['system_message']})
            messages.append({"role": assistant_role, "content": inst.messages['gold_lozenged_span']})
        else:
            messages.append({"from": user_role, "value": "{}".format(inst.messages['text'])})
            messages.append({"from": assistant_role, "value": "I've read this text."})
            messages.append({"from": user_role, "value": inst.messages['system_message']})
            messages.append({"from": assistant_role, "value": inst.messages['gold_lozenged_span']})

        if inst.messages['entity_type'] is None:
            continue
        id_ = str(inst.article_id).replace('_', '') + '_' + inst.messages['entity_type']

        if not openai_format:
            conversation_set.append({'id': id_, 'conversations': messages})
        else:
            conversation_set.append({'messages': messages, 'id': id_})

    if not openai_format:
        unique_article_ids = set()
        for conv in conversation_set:
            unique_article_ids.add(conv['id'].split('_')[0])
        print(f'Unique article ids in the conversation set: {len(unique_article_ids)}')

    if openai_format:
        with open(json_to_write, "w") as jsonl_file:
            print('writing {} conversations to {}'.format(len(conversation_set), json_to_write))
            for conv in conversation_set:
                jsonl_file.write(json.dumps(conv) + '\n')
        return conversation_set
    else:
        if len(json_to_write) > 0:
            with open(json_to_write, "w") as jsonl_file:
                print('writing {} conversations to {}'.format(len(conversation_set), json_to_write))
                json.dump(conversation_set, jsonl_file, indent=4)
    return conversation_set


def load_data_uniner_from_instances_all_spans(instances, json_to_write, entity_type_map=None,
                                               generic_span_type='policy design attributes',
                                               alternate_spans=False, openai_format=False):
    """Prompt for all spans in the text with entity types in tuple format."""
    instances, _ = normalize_entity_types(instances, entity_type_map)
    user_role = 'human' if not openai_format else 'user'
    assistant_role = 'gpt' if not openai_format else 'assistant'
    entity_types_str = ""
    conversation_set = []

    for s_idx, inst in enumerate(instances):
        messages = []
        if openai_format:
            messages.append({"role": user_role, "content": "{}".format(inst.sentence)})
            messages.append({"role": assistant_role, "content": "I've read this text."})
        else:
            messages.append({"from": user_role, "value": "{}".format(inst.sentence)})
            messages.append({"from": assistant_role, "value": "I've read this text."})

        if alternate_spans:
            spans = [[s for t, s in entity_set] for entity_set in inst.gold_entities]
            types_ = [entity_set[0][0] for entity_set in inst.gold_entities]
        else:
            spans = [s for t, s in inst.gold_entities]
            types_ = [t for t, s in inst.gold_entities]

        try:
            if spans[-1][-1] == ']':
                spans = spans[:-1] + [spans[-1][:-1]]
        except Exception:
            pass

        assert len(spans) == len(types_)
        spans_str = '[' + ','.join(['("{}","{}") '.format(s, t) for s, t in zip(spans, types_)]) + ']'

        if openai_format:
            if len(entity_types_str) > 0:
                messages.append({"role": user_role, "content": f"""Your task is to extract all entities and identify their entity types. The output should be in a list of tuples of the following format:[("entity 1", "type of entity 1"), ...]. The candidate entity types are: {entity_types_str}."""})
            else:
                messages.append({"role": user_role, "content": """Your task is to extract all entities and identify their entity types. The output should be in a list of tuples of the following format:[("entity 1", "type of entity 1"), ...]."""})
            messages.append({"role": assistant_role, "content": spans_str})
        else:
            if len(entity_types_str) > 0:
                messages.append({"from": user_role, "value": f"""Your task is to extract all entities and identify their entity types. The output should be in a list of tuples of the following format:[("entity 1", "type of entity 1"), ...]. The candidate entity types are: {entity_types_str}."""})
            else:
                messages.append({"from": user_role, "value": """Your task is to extract all entities and identify their entity types. The output should be in a list of tuples of the following format:[("entity 1", "type of entity 1"), ...]."""})
            messages.append({"from": assistant_role, "value": spans_str})

        id_ = str(s_idx)
        if not openai_format:
            conversation_set.append({'id': id_, 'conversations': messages})
        else:
            conversation_set.append({'messages': messages, 'id': id_})

    if openai_format:
        with open(json_to_write, "w") as jsonl_file:
            print('writing {} conversations to {}'.format(len(conversation_set), json_to_write))
            for conv in conversation_set:
                jsonl_file.write(json.dumps(conv) + '\n')
        return conversation_set
    else:
        with open(json_to_write, "w") as jsonl_file:
            print('writing {} conversations to {}'.format(len(conversation_set), json_to_write))
            json.dump(conversation_set, jsonl_file, indent=4)
    return conversation_set
