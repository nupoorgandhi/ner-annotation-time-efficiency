"""
Evaluate a trained NER or P2S model using vLLM inference.

Usage:
    python -m src.evaluate \
        --model_path <base_model> \
        --lora_path <lora_dir> \
        --test_data_path <test.json> \
        --result_filepath results.csv \
        [--from_p2s] \
        [--from_spantyping] \
        [--silver_data_path silver_in.json] \
        [--negative_data_path neg.json] \
        [--to_overwrite_data_path overwrite.json] \
        [--entity_type_map map.json] \
        [--tensor_parallel_size 1] \
        [--debug]
"""

import argparse
import ast
import csv
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

import src.data.evaluate_utils as evalfn
from src.data.instances import EntityInstance
from src.data.conversation import load_data_uniner_from_instances_all_spans

try:
    from rapidfuzz import fuzz
    USE_RAPIDFUZZ = True
except ImportError:
    print("Warning: rapidfuzz not installed. Install with: pip install rapidfuzz")
    USE_RAPIDFUZZ = False


# ---------------------------------------------------------------------------
# Message-format helpers (support both 'conversations' and 'messages' keys)
# ---------------------------------------------------------------------------

def first_msg_content(example):
    """Return the content of the first message in an example dict.

    Supports both ``conversations`` (list of dicts with 'value' key) and
    ``messages`` (list of dicts with 'content' key) formats.
    """
    if 'conversations' in example:
        return example['conversations'][0]['value']
    elif 'messages' in example:
        return example['messages'][0]['content']
    raise KeyError(f"Example has neither 'conversations' nor 'messages' key: {list(example.keys())}")


def last_msg_content(example):
    """Return the content of the last message in an example dict."""
    if 'conversations' in example:
        return example['conversations'][-1]['value']
    elif 'messages' in example:
        return example['messages'][-1]['content']
    raise KeyError(f"Example has neither 'conversations' nor 'messages' key: {list(example.keys())}")


def remove_last_message(example):
    """Return a shallow copy of *example* with the last message removed."""
    example = dict(example)
    if 'conversations' in example:
        example['conversations'] = example['conversations'][:-1]
    elif 'messages' in example:
        example['messages'] = example['messages'][:-1]
    return example


# ---------------------------------------------------------------------------
# NER output parsing
# ---------------------------------------------------------------------------

def parser(text):
    if type(text) != str:
        return text
    try:
        match = re.match(r'\[(.*?)\]', text)
        if match:
            text = match.group()
        else:
            text = '[]'
        formatted_items = json.loads(text)
        return formatted_items
    except Exception:
        return []


def postprocess_ner(res):
    """Try ast.literal_eval on a model output string; return list or []."""
    try:
        result = ast.literal_eval(res)
        if isinstance(result, list):
            return result
        return []
    except Exception:
        return parser(res)


# ---------------------------------------------------------------------------
# P2S postprocessing
# ---------------------------------------------------------------------------

def process_single_p2s_fast(args):
    """Optimised span-matching for a single example (no RapidFuzz)."""
    example, gold, output = args

    sentence = first_msg_content(example).replace('⧫', '').replace('\xa0', ' ')

    if isinstance(output, str):
        candidate_str = output.strip()
    else:
        candidate_str = str(output).strip()

    if not candidate_str:
        return ''

    # Fast path: exact match
    if candidate_str in sentence:
        return candidate_str

    # Fast path 2: case-insensitive match
    candidate_lower = candidate_str.lower()
    sentence_lower = sentence.lower()
    if candidate_lower in sentence_lower:
        idx = sentence_lower.find(candidate_lower)
        return sentence[idx:idx + len(candidate_str)]

    target_len = len(candidate_str)
    if target_len > len(sentence):
        return ''

    minL = max(1, target_len - 2)
    maxL = min(len(sentence), target_len + 2)

    def quick_distance(a, b):
        if len(a) != len(b):
            return abs(len(a) - len(b)) + sum(c1 != c2 for c1, c2 in zip(a, b))
        return sum(c1 != c2 for c1, c2 in zip(a, b))

    best = ''
    best_score = float('inf')
    target = candidate_str

    max_positions = min(500, len(sentence))
    step = max(1, len(sentence) // max_positions)

    for L in range(minL, maxL + 1):
        for i in range(0, len(sentence) - L + 1, step):
            substr = sentence[i:i + L]
            dist = quick_distance(target, substr)
            if dist < best_score:
                best_score = dist
                best = substr
                if best_score <= 2:
                    return best

    if best_score < len(target) * 0.3:
        return best
    return ''


def process_single_p2s_rapidfuzz(args):
    """Ultra-fast span-matching using RapidFuzz."""
    example, gold, output = args

    sentence = first_msg_content(example).replace('⧫', '').replace('\xa0', ' ')

    if isinstance(output, str):
        candidate_str = output.strip()
    else:
        candidate_str = str(output).strip()

    if not candidate_str:
        return ''

    if candidate_str in sentence:
        return candidate_str

    if not USE_RAPIDFUZZ:
        return candidate_str

    target = candidate_str
    target_len = len(target)

    minL = max(1, target_len - 2)
    maxL = min(len(sentence), target_len + 2)

    best_match = None
    best_score = 0

    max_samples = 500
    sentence_len = len(sentence)

    for L in range(minL, maxL + 1):
        num_positions = sentence_len - L + 1
        if num_positions <= 0:
            continue

        if num_positions > max_samples:
            step = num_positions // max_samples
            positions = range(0, num_positions, step)
        else:
            positions = range(num_positions)

        for i in positions:
            substr = sentence[i:i + L]
            score = fuzz.ratio(target, substr)

            if score > best_score:
                best_score = score
                best_match = substr

                if score > 90:
                    return best_match

    if best_match and best_score > 70:
        return best_match

    return ''


def postprocess_p2s(examples, golds, outputs, use_threading=True, n_threads=None):
    """
    Two-phase postprocessing: exact matching first, then fuzzy on remainders.
    Uses RapidFuzz when available, otherwise the fast pure-Python fallback.
    """
    print(f"Postprocessing {len(outputs)} examples...")
    sys.stdout.flush()

    # Phase 1: exact matching
    print("Phase 1: Exact matching...")
    processed_outputs = []
    fuzzy_indices = []
    fuzzy_args = []

    for idx, (example, gold, output) in enumerate(zip(examples, golds, outputs)):
        sentence = first_msg_content(example).replace('⧫', '').replace('\xa0', ' ')

        if isinstance(output, str):
            candidate_str = output.strip()
        else:
            candidate_str = str(output).strip()

        if '<|eot_id|>' in candidate_str:
            candidate_str = candidate_str.split('<|eot_id|>')[0].strip()

        if not candidate_str:
            processed_outputs.append('')
        elif candidate_str in sentence:
            processed_outputs.append(candidate_str)
        elif candidate_str.lower() in sentence.lower():
            idx_match = sentence.lower().find(candidate_str.lower())
            processed_outputs.append(sentence[idx_match:idx_match + len(candidate_str)])
        else:
            processed_outputs.append(None)
            fuzzy_indices.append(idx)
            fuzzy_args.append((example, gold, output))

    exact_matches = len(outputs) - len(fuzzy_args)
    print(f"Exact matches: {exact_matches}/{len(outputs)} ({exact_matches / len(outputs) * 100:.1f}%)")
    sys.stdout.flush()

    if not fuzzy_args:
        return processed_outputs

    # Phase 2: fuzzy matching on the remainder
    print(f"Phase 2: Fuzzy matching {len(fuzzy_args)} examples...")
    sys.stdout.flush()

    process_func = process_single_p2s_rapidfuzz if USE_RAPIDFUZZ else process_single_p2s_fast

    if use_threading and len(fuzzy_args) > 20:
        if n_threads is None:
            n_threads = min(os.cpu_count() or 1, 16)

        fuzzy_results = [None] * len(fuzzy_args)

        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            future_to_idx = {
                executor.submit(process_func, args): i
                for i, args in enumerate(fuzzy_args)
            }

            for future in tqdm(as_completed(future_to_idx),
                               total=len(fuzzy_args),
                               desc='Fuzzy matching'):
                i = future_to_idx[future]
                fuzzy_results[i] = future.result()

        for fuzzy_idx, result in zip(fuzzy_indices, fuzzy_results):
            processed_outputs[fuzzy_idx] = result
    else:
        for fuzzy_idx, args in tqdm(zip(fuzzy_indices, fuzzy_args),
                                    total=len(fuzzy_args),
                                    desc='Fuzzy matching'):
            processed_outputs[fuzzy_idx] = process_func(args)

    return processed_outputs


# ---------------------------------------------------------------------------
# Silver instance creation
# ---------------------------------------------------------------------------

def convert_to_silver_instances(examples, golds, preds, from_p2s=False):
    """Aggregate predictions at (article_id, sentence) level and return EntityInstances."""
    sentence_to_entities_map = defaultdict(list)
    silver_type_counts = Counter()

    for example, gold, pred in zip(examples, golds, preds):
        if not from_p2s:
            gold = parser(gold)
            pred = parser(pred)

        article_id = example['id'].split('_')[0]
        sentence = first_msg_content(example).replace('⧫', '').replace('\xa0', ' ')
        entity_type = '_'.join(example['id'].split('_')[1:])
        silver_type_counts[entity_type] += 1
        sentence_to_entities_map[(article_id, sentence)].extend(pred)

    instances = []
    for (article_id, sentence), entities in sentence_to_entities_map.items():
        instance = EntityInstance(sentence=sentence)
        instance.article_id = article_id
        instance.gold_entities = entities
        try:
            instance.gold_entities = [e for e in entities if len(e) == 2]
            instance.gold_entities = [(t, s) for s, t in instance.gold_entities]
        except Exception:
            print('issue', instance.gold_entities)
            continue
        instances.append(instance)

    final_silver_type_counts = Counter()
    for inst in instances:
        entity_types = set([e[0] for e in inst.gold_entities])
        for entity_type in entity_types:
            final_silver_type_counts[entity_type] += 1
    print('final silver type counts:', final_silver_type_counts)
    return instances


def convert_to_silver_instances_from_p2s(examples, golds, preds):
    """Create EntityInstances from P2S predictions."""
    sentence_to_gold_entities_map = defaultdict(list)
    pre_silver_type_counts = Counter()

    for example, _, pred in zip(examples, golds, preds):
        article_id = example['id'].split('_')[0]
        sentence = first_msg_content(example).replace('⧫', '').replace('\xa0', ' ')
        entity_type = '_'.join(example['id'].split('_')[1:])
        if type(pred) != str:
            print('type of pred is not string', pred)
        sentence_to_gold_entities_map[(article_id, sentence)].append((entity_type, str(pred)))
        pre_silver_type_counts[entity_type] += 1

    instances = []
    for (article_id, sentence), gold_entities in sentence_to_gold_entities_map.items():
        instance = EntityInstance(sentence=sentence)
        instance.article_id = article_id
        instance.gold_entities = gold_entities
        instances.append(instance)

    final_silver_type_counts = Counter()
    for inst in instances:
        entity_types = [e[0] for e in inst.gold_entities]
        for entity_type in entity_types:
            final_silver_type_counts[entity_type] += 1

    print('len instances', len(instances), 'len examples', len(examples))
    print('pre_silver', pre_silver_type_counts)
    return instances


# ---------------------------------------------------------------------------
# Instance / data utilities
# ---------------------------------------------------------------------------

def examples_to_instances(examples):
    """Convert a list of conversation-format example dicts to EntityInstance objects."""
    instances = []
    for example in examples:
        instance = EntityInstance(sentence=first_msg_content(example))
        instance.article_id = example['id'].split('_')[0]
        entity_type = '_'.join(example['id'].split('_')[1:])
        try:
            golds = ast.literal_eval(last_msg_content(example))
        except Exception:
            golds = []
        instance.gold_entities = [(span, entity_type) for entity_type, span in golds]
        instances.append(instance)
    return instances


def load_instances(json_data_path):
    """Load EntityInstances from a JSON file."""
    with open(json_data_path, 'r') as fh:
        examples = json.load(fh)
    return examples_to_instances(examples)


def load_negative_instances(data_path):
    """Load only examples whose gold response is an empty list."""
    with open(data_path, 'r') as fh:
        examples = json.load(fh)
    negative_examples = []
    for example in examples:
        gold_response = last_msg_content(example)
        try:
            gold_response = ast.literal_eval(gold_response)
        except Exception:
            continue
        if len(gold_response) == 0:
            negative_examples.append(example)
    return negative_examples


def overwrite_gold_entities(instances1, instances2):
    """For each instance in instances1, overwrite its gold_entities from instances2 if sentences match."""
    num_overwritten = 0
    for inst1 in instances1:
        for inst2 in instances2:
            if inst1.sentence.replace('\xa0', ' ').replace(' ', '') == inst2.sentence.replace('\xa0', ' ').replace(' ', ''):
                inst1.gold_entities = inst2.gold_entities
                num_overwritten += 1
                break
    print('[overwrite_gold_entities] number of overwritten', num_overwritten, 'out of', len(instances1))
    return instances1


# ---------------------------------------------------------------------------
# Checkpoint and tokenizer helpers
# ---------------------------------------------------------------------------

def get_highest_checkpoint(directory):
    """Return (checkpoint_path, base_directory) for the checkpoint inside *directory*."""
    print('directory currently is', directory)
    checkpoint_dirs = [d for d in os.listdir(directory) if d.startswith('checkpoint-')]
    print('checkpoint dirs found were', checkpoint_dirs)
    if len(checkpoint_dirs) == 0:
        print("No checkpoint directory found.")
        return '', directory
    return os.path.join(directory, checkpoint_dirs[0]), directory


def preprocess_instance(conversations, tokenizer):
    """Format a conversation using tokenizer.apply_chat_template (Llama-3 format)."""
    messages = [{'role': 'user' if i % 2 == 0 else 'assistant', 'content': turn['value']}
                for i, turn in enumerate(conversations)]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ---------------------------------------------------------------------------
# vLLM inference
# ---------------------------------------------------------------------------

def run_vllm_inference(examples, model_path, lora_path='', tensor_parallel_size=1, debug=False):
    """Run vLLM inference and return a list of output strings.

    vLLM is imported lazily so the rest of the module works without it.
    """
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        enable_lora=bool(lora_path),
        max_model_len=4500,
    )
    print('[run_vllm_inference] LLM initialised successfully!')
    sys.stdout.flush()

    if lora_path:
        checkpoint_folder, _ = get_highest_checkpoint(lora_path)
        lora_request = None if not checkpoint_folder else LoRARequest('lora_adapter', 1, checkpoint_folder)
    else:
        lora_request = None

    print('[run_vllm_inference] lora_request:', lora_request)

    if debug:
        examples = examples[:500]

    sampling_params = SamplingParams(temperature=0.0, max_tokens=512)

    # Build prompts: remove generation turn (last message already removed upstream)
    # Each example is expected to already have the last (assistant) message stripped.
    # We format using the tokenizer's chat template via a simple reconstruction.
    # vLLM accepts raw text prompts; we serialise each conversation as a plain string.
    # For Llama-3 instruct models the canonical prompt is built via apply_chat_template.
    # We do a best-effort conversion here without requiring a tokenizer object.

    def _build_prompt(example):
        """Convert conversation/messages list to a plain text prompt."""
        if 'conversations' in example:
            turns = example['conversations']
            roles = ['user', 'assistant']
            parts = []
            for i, turn in enumerate(turns):
                role = roles[i % 2]
                content = turn['value']
                if role == 'user':
                    parts.append(f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>")
                else:
                    parts.append(f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>")
            parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
            return ''.join(parts)
        elif 'messages' in example:
            turns = example['messages']
            parts = []
            for turn in turns:
                role = turn['role']
                content = turn['content']
                if role == 'user':
                    parts.append(f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>")
                else:
                    parts.append(f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>")
            parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
            return ''.join(parts)
        raise KeyError(f"Example has neither 'conversations' nor 'messages': {list(example.keys())}")

    prompts = [_build_prompt(ex) for ex in examples]

    generate_kwargs = dict(sampling_params=sampling_params)
    if lora_request is not None:
        generate_kwargs['lora_request'] = lora_request

    results = llm.generate(prompts, **generate_kwargs)
    outputs = [r.outputs[0].text for r in results]
    return outputs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser_arg = argparse.ArgumentParser(description="Evaluate NER or P2S model with vLLM.")
    parser_arg.add_argument('--model_path', type=str, required=True)
    parser_arg.add_argument('--lora_path', type=str, default='')
    parser_arg.add_argument('--test_data_path', type=str, default='')
    parser_arg.add_argument('--silver_data_path', type=str, default='')
    parser_arg.add_argument('--negative_data_path', type=str, default='')
    parser_arg.add_argument('--to_overwrite_data_path', type=str, default='')
    parser_arg.add_argument('--result_filepath', type=str, required=True)
    parser_arg.add_argument('--entity_type_map', type=str, default='',
                            help='Path to a JSON file mapping entity types.')
    parser_arg.add_argument('--from_p2s', action='store_true')
    parser_arg.add_argument('--from_spantyping', action='store_true')
    parser_arg.add_argument('--tensor_parallel_size', type=int, default=1)
    parser_arg.add_argument('--debug', action='store_true')
    args = parser_arg.parse_args()

    entity_type_map = None
    if args.entity_type_map:
        with open(args.entity_type_map, 'r') as fh:
            entity_type_map = json.load(fh)

    print('[eval] model path is', args.model_path)

    data_paths = {
        'test': args.test_data_path,
        'silver': args.silver_data_path,
    }

    for split, data_path in list(data_paths.items()):
        if not data_path:
            print(f"Skipping {split} split as data path is empty.")
            continue

        result_filepath_ = (args.result_filepath.replace('test.csv', 'silver.csv')
                            if split == 'silver' else args.result_filepath)

        # Load examples (support both .json and .jsonl)
        if data_path.endswith('.jsonl'):
            with open(data_path, 'r') as fh:
                examples = [json.loads(line) for line in fh if line.strip()]
        else:
            with open(data_path, 'r') as fh:
                examples = json.load(fh)

        print('example[0]:', examples[0])
        sys.stdout.flush()

        golds = [last_msg_content(ex) for ex in examples]
        inference_examples = [remove_last_message(ex) for ex in examples]

        outputs = run_vllm_inference(
            inference_examples,
            model_path=args.model_path,
            lora_path=args.lora_path,
            tensor_parallel_size=args.tensor_parallel_size,
            debug=args.debug,
        )

        # Quick sanity print
        for i in range(min(5, len(outputs))):
            print(f'outputs[{i}]:', outputs[i])
            print(f'golds[{i}]:', golds[i])
            print('--' * 20)
        sys.stdout.flush()

        percent_nonempty_outputs = sum(1 for o in outputs if len(o) > 2) / len(outputs)
        percent_nonempty_golds   = sum(1 for g in golds   if len(g) > 2) / len(golds)
        print(f"Percent non-empty outputs: {percent_nonempty_outputs:.2%}")
        print(f"Percent non-empty golds:   {percent_nonempty_golds:.2%}")
        sys.stdout.flush()

        # ------------------------------------------------------------------
        # Evaluation
        # ------------------------------------------------------------------
        if args.from_spantyping:
            type_tp = Counter()
            type_fp = Counter()
            type_fn = Counter()
            golds_str   = [str(g) for g in golds]
            outputs_str = [str(o).strip().split('\n')[0] for o in outputs]
            for g, o in zip(golds_str, outputs_str):
                g_key = g.strip()
                o_key = o.strip()
                if g_key.lower() == o_key.lower():
                    type_tp[g_key] += 1
                else:
                    type_fn[g_key] += 1
                    type_fp[o_key] += 1

            all_types = set(type_tp) | set(type_fn) | set(type_fp)
            per_type_f1 = {}
            for t in all_types:
                tp = type_tp[t]; fp = type_fp[t]; fn = type_fn[t]
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                per_type_f1[t] = {'precision': prec, 'recall': rec, 'f1': f1,
                                  'tp': tp, 'fp': fp, 'fn': fn}

            n_correct  = sum(type_tp.values())
            n_total    = len(golds_str)
            total_tp   = sum(type_tp.values())
            total_fp   = sum(type_fp.values())
            total_fn   = sum(type_fn.values())
            micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            micro_rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            micro_f1   = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0
            accuracy   = n_correct / n_total if n_total > 0 else 0.0
            macro_f1   = float(np.mean([v['f1'] for v in per_type_f1.values()])) if per_type_f1 else 0.0

            eval_result = {
                'accuracy':        accuracy,
                'micro_f1':        micro_f1,
                'micro_precision': micro_prec,
                'micro_recall':    micro_rec,
                'macro_f1':        macro_f1,
                'n_correct':       n_correct,
                'n_total':         n_total,
                'per_type_f1':     str(per_type_f1),
            }
            print(f'[span typing] accuracy={accuracy:.4f}  micro-F1={micro_f1:.4f}  macro-F1={macro_f1:.4f}  ({n_correct}/{n_total})')

        elif args.from_p2s:
            outputs = postprocess_p2s(examples, golds, outputs)
            print('outputs after postprocess:', outputs[:5])
            sys.stdout.flush()
            eval_result = evalfn.collect_results_strings(examples, golds, outputs)
            print('eval_result after postprocess:', eval_result)

        else:
            # Standard NER evaluation
            processed_outputs = [postprocess_ner(o) for o in outputs]
            eval_result = evalfn.collect_results(examples, golds, processed_outputs)

        sys.stdout.flush()

        # ------------------------------------------------------------------
        # Silver data generation (only for the 'silver' split)
        # ------------------------------------------------------------------
        if split == 'silver':
            if args.from_p2s and not args.from_spantyping:
                instances = convert_to_silver_instances_from_p2s(examples, golds, outputs)

                if args.negative_data_path:
                    negative_instances = load_negative_instances(args.negative_data_path)
                    instances += examples_to_instances(negative_instances)

                if args.to_overwrite_data_path:
                    to_overwrite_instances = load_instances(args.to_overwrite_data_path)
                    instances = overwrite_gold_entities(instances, to_overwrite_instances)

                random.shuffle(instances)
            else:
                instances = convert_to_silver_instances(
                    examples, golds, outputs,
                    from_p2s=args.from_p2s,
                )

            if 'train' in data_path:
                silver_out_path = os.path.join(
                    os.path.dirname(data_path),
                    os.path.basename(data_path).replace('train', 'silver'),
                )
            else:
                print('unsure what is happening')
                print('data_path:', data_path)
                sys.exit(0)

            assert silver_out_path != data_path, \
                "Silver data path should not be the same as the original data path"
            print('writing to silver data path:', silver_out_path)

            type_counts = Counter()
            for inst in instances:
                type_counts.update([e[0] for e in inst.gold_entities])
            print('pre-writing type counts', type_counts)

            written_examples = load_data_uniner_from_instances_all_spans(
                instances, silver_out_path,
                entity_type_map=None, alternate_spans=False,
            )

            final_type_counts = Counter()
            for ex in written_examples:
                try:
                    gold_entities = ast.literal_eval(last_msg_content(ex))
                except Exception:
                    continue
                final_type_counts.update([e[1] for e in gold_entities])
            print()
            print('post-writing type counts', final_type_counts)

        # ------------------------------------------------------------------
        # Write CSV results
        # ------------------------------------------------------------------
        with open(result_filepath_, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=eval_result.keys())
            writer.writeheader()
            writer.writerow(eval_result)
        print(f'[eval] results written to {result_filepath_}')


if __name__ == "__main__":
    main()
