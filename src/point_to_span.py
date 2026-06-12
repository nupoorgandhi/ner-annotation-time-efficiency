"""
Convert gold-span instances to point-marked (⧫) instances for P2S training.

Key functions: convert_instances_to_point, insert_lozenges,
point_to_span_estimation_instances_singlespan, convert_instances_to_span_typing.
"""

import random
import numpy as np
import copy


def convert_instances_to_point(instances, use_gold_spans_insertion=True, insertion_method='uniform'):
    point_instances = point_to_span_estimation_instances_singlespan(instances, insertion_method=insertion_method)
    return point_instances


def insert_lozenges(span_list, sentence, instance, insertion_method='uniform'):
    """Insert ⧫ markers at a sampled position within each span occurrence.

    Handles repeated spans independently and tracks offset from prior insertions.
    """
    if len(instance.gold_point_positions) > 0:
        lozenge_positions = instance.gold_point_positions
    else:
        lozenge_positions = []
        for span in span_list:
            start_idx = 0
            while True:
                start_idx = sentence.find(span, start_idx)
                if start_idx == -1:
                    break
                if insertion_method == 'uniform':
                    lozenge_position = start_idx + random.randint(0, len(span) - 1)
                elif insertion_method == 'gaussian':
                    mean = start_idx + len(span) // 2
                    std_dev = max(1, len(span) // 4)
                    lozenge_position = int(np.random.normal(mean, std_dev))
                    lozenge_position = max(0, min(lozenge_position, len(sentence) - 1))
                elif insertion_method == 'center':
                    lozenge_position = start_idx + len(span) // 2
                else:
                    raise ValueError(f"Unknown insertion method: {insertion_method}")
                lozenge_positions.append(lozenge_position)
                start_idx += len(span)
    lozenge_positions = sorted(lozenge_positions)
    for i, pos in enumerate(lozenge_positions):
        sentence = sentence[:pos + i] + '⧫' + sentence[pos + i:]
    return sentence


def point_to_span_estimation_instances_singlespan(instances, entity_type_map=None, insertion_method='uniform'):
    """Create one instance per unique (entity_type, span) pair with a ⧫ inserted into the span."""
    new_instances = []
    for instance in instances:
        unique_spans = list(set(instance.gold_entities))

        for entity_type, span_text in unique_spans:
            new_instance = copy.deepcopy(instance)
            new_instance.gold_entities = [(entity_type, span_text)]

            assert '⧫' not in new_instance.sentence, "Sentence already contains lozenge character."
            text = copy.deepcopy(new_instance.sentence)
            text = insert_lozenges([span_text], text, new_instance, insertion_method=insertion_method)

            display_entity_type = entity_type
            if entity_type_map is not None and entity_type in entity_type_map:
                display_entity_type = entity_type_map[entity_type]

            system_message = f"""
Extract the span marked by ⧫ character that describes {display_entity_type}.
"""
            new_instance.messages = {
                'system_message': system_message,
                'assistant_message': system_message + f'["{span_text}"]',
                'user_message': '',
                'retrieval_item': text,
                'entity_type': display_entity_type,
                'gold_lozenged_span': span_text,
                'text': text
            }
            new_instances.append(new_instance)

    return new_instances


def convert_instances_to_span_typing(instances, entity_type_map=None):
    """Create one instance per unique (entity_type, span) pair for span typing (given span, predict type)."""
    new_instances = []
    for instance in instances:
        unique_spans = list(set(instance.gold_entities))
        for entity_type, span_text in unique_spans:
            new_instance = copy.deepcopy(instance)
            new_instance.gold_entities = [(entity_type, span_text)]

            display_entity_type = entity_type
            if entity_type_map is not None and entity_type in entity_type_map:
                display_entity_type = entity_type_map[entity_type]

            new_instance.messages = {
                'system_message': f"What is the entity type of '{span_text}'?",
                'gold_lozenged_span': display_entity_type,
                'entity_type': display_entity_type,
                'text': instance.sentence,
                'user_message': '',
                'retrieval_item': instance.sentence,
            }
            new_instances.append(new_instance)
    return new_instances
