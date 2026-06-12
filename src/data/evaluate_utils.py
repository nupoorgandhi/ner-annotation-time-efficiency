"""
NER evaluation metrics: micro/macro F1 for span detection and entity typing.

Entry point is collect_results(), which dispatches over span-level or type-level
scoring depending on the gold format.
"""

from difflib import SequenceMatcher
import re
import string
from collections import defaultdict
import sklearn.metrics as typing_metrics
import json
import copy
import ast
from tqdm import tqdm
import sys


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def parser(text):
    if type(text) == type([]):
        return text
    try:
        match = re.match(r'\[(.*?)\]', text)
        if match:
            text = match.group()
        else:
            text = '[]'
        items = json.loads(text)
        formatted_items = []
        for item in items:
            if isinstance(item, list) or isinstance(item, tuple):
                item = tuple([normalize_answer(element) for element in item])
            else:
                item = normalize_answer(item)
            if item not in formatted_items:
                formatted_items.append(item)
        return formatted_items
    except Exception:
        return []


def clf_parser(text):
    if type(text) == type([]):
        if len(text) == 0:
            return ''
        else:
            return normalize_answer(text[0])
    try:
        return normalize_answer(text)
    except:
        return ''


class NEREvaluator:

    def is_hard_match(self, t, gold_tuples):
        matched_gold_span = None
        for span in gold_tuples:
            if t == span:
                matched_gold_span = span
                break
        return matched_gold_span is not None, matched_gold_span

    def is_soft_match(self, t, gold_tuples):
        def overlap(x1, x2):
            return SequenceMatcher(None, str(x2), str(x1)).ratio() > .7

        for span in gold_tuples:
            if overlap(span, t):
                return True, span
        return False, None

    def soft_match_score(self, pred_tuples, gold_tuples):
        def overlap(x1, x2):
            matcher = SequenceMatcher(None, x2, x1)
            match = matcher.find_longest_match(0, len(x2), 0, len(x1))
            return match.size >= len(x2) / 2

        scores = []
        for t in pred_tuples:
            t_scores = []
            for span in gold_tuples:
                t_scores.append((span, t, overlap(span, t)))
            t_scores.sort(key=lambda x: x[2], reverse=True)
            if len(t_scores) > 0:
                scores.append(t_scores[0])
        return scores

    def is_hard_match_pool(self, t, gold_pool):
        """Check if predicted tuple matches any gold span in the pool."""
        matched_gold_span = None
        for span in gold_pool:
            if t == span:
                matched_gold_span = span
                break
        return matched_gold_span is not None, matched_gold_span

    def is_soft_match_pool(self, t, gold_pool):
        """Check if predicted tuple soft matches any gold span in the pool."""
        def overlap(x1, x2):
            return SequenceMatcher(None, str(x2), str(x1)).ratio() > .7

        for span in gold_pool:
            if overlap(span, t):
                return True, span
        return False, None

    def evaluate(self, preds: list, golds: list, soft_match: bool):
        n_correct, n_pos_gold, n_pos_pred = 0, 0, 0

        def make_hashable(x):
            if isinstance(x, dict):
                return tuple(sorted((k, make_hashable(v)) for k, v in x.items()))
            if isinstance(x, (list, tuple, set)):
                return tuple(make_hashable(i) for i in x)
            return x

        preds = [list({make_hashable(i) for i in p}) for p in preds]
        golds = [list({make_hashable(i) for i in g}) for g in golds]

        def to_tuple(x):
            if isinstance(x, list):
                return tuple(to_tuple(i) for i in x)
            return x

        for pred_tuples, gold_tuples in zip(preds, golds):
            if len(gold_tuples) != len(set(gold_tuples)):
                print('Gold tuples have duplicates, this is not expected')
            gold_tuples_ = copy.deepcopy(gold_tuples)
            for t in pred_tuples:
                n_pos_pred += 1
                if soft_match:
                    is_match, matched_gold_span = self.is_soft_match(t, gold_tuples_)
                else:
                    is_match, matched_gold_span = self.is_hard_match(t, gold_tuples_)
                if is_match:
                    n_correct += 1
                    gold_tuples_.remove(matched_gold_span)

            n_pos_gold += len(gold_tuples)

        prec = n_correct / (n_pos_pred + 1e-10)
        recall = n_correct / (n_pos_gold + 1e-10)
        f1 = 2 * prec * recall / (prec + recall + 1e-10)
        if soft_match:
            return {
                'soft-match-precision': prec,
                'soft-match-recall': recall,
                'soft-match-f1': f1,
            }
        else:
            return {
                'hard-match-precision': prec,
                'hard-match-recall': recall,
                'hard-match-f1': f1,
            }

    def evaluate_with_pool(self, preds: list, golds: list, soft_match: bool):
        """Evaluate with pool-based gold standards (each gold is a list of acceptable spans)."""
        n_correct, n_pos_gold, n_pos_pred = 0, 0, 0

        def make_hashable(x):
            if isinstance(x, dict):
                return tuple(sorted((k, make_hashable(v)) for k, v in x.items()))
            if isinstance(x, (list, tuple, set)):
                return tuple(make_hashable(i) for i in x)
            return x

        preds = [list({make_hashable(i) for i in p}) for p in preds]

        def to_tuple(x):
            if isinstance(x, list):
                return tuple(to_tuple(i) for i in x)
            return x

        golds_filtered_ = []
        for g in golds:
            g_filtered = []
            gold_span_set = set()
            for span_pool in g:
                if len(span_pool) == 0:
                    continue
                if span_pool[0] not in gold_span_set:
                    g_filtered.append(span_pool)
                    gold_span_set.add(span_pool[0])
            golds_filtered_.append(g_filtered)
        golds = golds_filtered_

        processed_golds = []
        for gold_pool in golds:
            flattened_pool = []
            for gold_list in gold_pool:
                flattened_pool.extend([to_tuple(g) for g in gold_list])
            processed_golds.append(list(set(flattened_pool)))

        for pred_tuples, gold_pool in zip(preds, processed_golds):
            matched_golds = set()

            for t in pred_tuples:
                if soft_match:
                    is_match, matched_gold_span = self.is_soft_match_pool(t, gold_pool)
                    if is_match and matched_gold_span not in matched_golds:
                        n_correct += 1
                        matched_golds.add(matched_gold_span)
                else:
                    is_match, matched_gold_span = self.is_hard_match_pool(t, gold_pool)
                    if is_match and matched_gold_span not in matched_golds:
                        n_correct += 1
                        matched_golds.add(matched_gold_span)

                n_pos_pred += 1

        n_pos_gold = sum([len(g) for g in golds])
        prec = n_correct / (n_pos_pred + 1e-10)
        recall = n_correct / (n_pos_gold + 1e-10)
        f1 = 2 * prec * recall / (prec + recall + 1e-10)
        if soft_match:
            return {
                'soft-match-precision': prec,
                'soft-match-recall': recall,
                'soft-match-f1': f1,
            }
        else:
            return {
                'hard-match-precision': prec,
                'hard-match-recall': recall,
                'hard-match-f1': f1,
            }


def typing_scores(examples, outputs, golds):
    outputs = [clf_parser(m) for m in outputs]
    golds = [clf_parser(m) for m in golds]

    eval_result = {}
    for ave in ['micro', 'macro']:
        eval_result['{}_typing_f1'.format(ave)] = typing_metrics.f1_score(golds, outputs,
                                labels=list(set(golds)),
                                average=ave)
        eval_result['{}_typing_p'.format(ave)] = typing_metrics.precision_score(golds, outputs,
                                labels=list(set(golds)),
                                average=ave)
        eval_result['{}_typing_r'.format(ave)] = typing_metrics.recall_score(golds, outputs,
                                labels=list(set(golds)),
                                average=ave)

    all_types = list(set(golds))
    for type_ in all_types:
        for ave in ['micro', 'macro']:
            outputs_ = [1 if o == type_ else 0 for o in outputs]
            golds_ = [1 if g == type_ else 0 for g in golds]
            eval_result['{}_typing_f1_{}'.format(ave, type_)] = typing_metrics.f1_score(golds_, outputs_,
                                    labels=list(set(golds_)),
                                    average=ave)
            eval_result['{}_typing_p_{}'.format(ave, type_)] = typing_metrics.precision_score(golds_, outputs_,
                                    labels=list(set(golds_)),
                                    average=ave)
            eval_result['{}_typing_r_{}'.format(ave, type_)] = typing_metrics.recall_score(golds_, outputs_,
                                    labels=list(set(golds_)),
                                    average=ave)
            outputs_ = [1 if o != type_ else 0 for o in outputs]
            golds_ = [1 if g != type_ else 0 for g in golds]
            eval_result['{}_typing_f1_not{}'.format(ave, type_)] = typing_metrics.f1_score(golds_, outputs_,
                                    labels=list(set(golds_)),
                                    average=ave)
            eval_result['{}_typing_p_not{}'.format(ave, type_)] = typing_metrics.precision_score(golds_, outputs_,
                                    labels=list(set(golds_)),
                                    average=ave)
            eval_result['{}_typing_r_not{}'.format(ave, type_)] = typing_metrics.recall_score(golds_, outputs_,
                                    labels=list(set(golds_)),
                                    average=ave)

    xcl_to_indices = defaultdict(list)
    for idx, example in enumerate(examples):
        components = example['id'].split('_')
        if len(components) < 3:
            continue
        else:
            xcl = components[2]
            xcl_to_indices[xcl].append(idx)

    for xcl_, indices in xcl_to_indices.items():
        golds_ = [golds[i] for i in indices]
        outputs_ = [outputs[i] for i in indices]
        for ave in ['micro', 'macro']:
            eval_result['{}_{}_typing_f1'.format(xcl_, ave)] = typing_metrics.f1_score(golds_, outputs_,
                                    labels=list(set(golds_)),
                                    average=ave)
            eval_result['{}_{}_typing_p'.format(xcl_, ave)] = typing_metrics.precision_score(golds_, outputs_,
                                    labels=list(set(golds_)),
                                    average=ave)
            eval_result['{}_{}_typing_r'.format(xcl_, ave)] = typing_metrics.recall_score(golds_, outputs_,
                                    labels=list(set(golds_)),
                                    average=ave)
        golds_ = [g for ix, g in enumerate(golds) if ix not in indices]
        outputs_ = [o for ix, o in enumerate(outputs) if ix not in indices]
        for ave in ['micro', 'macro']:
            eval_result['not{}_{}_typing_f1'.format(xcl_, ave)] = typing_metrics.f1_score(golds_, outputs_,
                                    labels=list(set(golds_)),
                                    average=ave)
            eval_result['not{}_{}_typing_p'.format(xcl_, ave)] = typing_metrics.precision_score(golds_, outputs_,
                                    labels=list(set(golds_)),
                                    average=ave)
            eval_result['not{}_{}_typing_r'.format(xcl_, ave)] = typing_metrics.recall_score(golds_, outputs_,
                                    labels=list(set(golds_)),
                                    average=ave)

    return eval_result


def flatten(lst):
    return [item for sublist in lst for item in sublist]


def micro_ner_scores(outputs, golds, soft_match, prefix=''):
    outputs = [parser(m) for m in outputs]
    golds = [parser(m) for m in golds]

    eval_result_ = NEREvaluator().evaluate(outputs, golds, soft_match)
    eval_result = {}
    for k in eval_result_.keys():
        eval_result[prefix+'{}_{}'.format('micro-ner', k)] = eval_result_[k]
    return eval_result


def micro_md_scores(examples, outputs, golds, soft_match, prefix=''):
    sidx_to_indices = defaultdict(list)
    for idx, example in enumerate(examples):
        sidx = example['id'].split('_')[0]
        sidx_to_indices[sidx].append(idx)
    golds_sidx_level = []
    outputs_sidx_level = []
    for sidx, indices in sidx_to_indices.items():
        golds_ = flatten([parser(golds[i]) for i in indices])
        outputs_ = flatten([parser(outputs[i]) for i in indices])
        golds_sidx_level.append(golds_)
        outputs_sidx_level.append(outputs_)

    eval_result_ = NEREvaluator().evaluate(outputs_sidx_level, golds_sidx_level, soft_match)
    eval_result = {}
    for k in eval_result_.keys():
        eval_result[prefix+'{}_{}'.format('micro-md', k)] = eval_result_[k]
    return eval_result


def micro_ner_scores_pool(outputs, golds, soft_match):
    """Micro NER scores with pool-based gold standards."""
    outputs = [parser(m) for m in outputs]
    eval_result_ = NEREvaluator().evaluate_with_pool(outputs, golds, soft_match)
    eval_result = {}
    for k in eval_result_.keys():
        eval_result['{}_{}'.format('micro-ner-pool', k)] = eval_result_[k]
    return eval_result


def micro_md_scores_pool(examples, outputs, golds, soft_match):
    """Micro MD scores with pool-based gold standards."""
    sidx_to_indices = defaultdict(list)
    for idx, example in enumerate(examples):
        sidx = example['id'].split('_')[0]
        sidx_to_indices[sidx].append(idx)

    golds_sidx_level = []
    outputs_sidx_level = []

    for sidx, indices in sidx_to_indices.items():
        merged_pool = []
        for i in indices:
            gold_pool = golds[i]
            for gold_list in gold_pool:
                merged_pool.append(gold_list)
            golds_sidx_level.append(merged_pool)

            outputs_ = flatten([parser(outputs[i]) for i in indices])
            outputs_sidx_level.append(outputs_)

    eval_result_ = NEREvaluator().evaluate_with_pool(outputs_sidx_level, golds_sidx_level, soft_match)
    eval_result = {}
    for k in eval_result_.keys():
        eval_result['{}_{}'.format('micro-md-pool', k)] = eval_result_[k]
    return eval_result


def macro_ner_scores(examples, outputs, golds, soft_match):
    type_to_indices = defaultdict(list)
    for idx, example in enumerate(examples):
        try:
            ent_type = example['id'].split('_')[1]
        except:
            ent_type = "NA"
        type_to_indices[ent_type].append(idx)
    metrics = []
    eval_result = {}
    for type_, indices in tqdm(type_to_indices.items(), 'Computing macro NER scores'):
        golds_ = [parser(golds[i]) for i in indices]
        outputs_ = [parser(outputs[i]) for i in indices]
        eval_result_ = NEREvaluator().evaluate(outputs_, golds_, soft_match=soft_match)
        metrics = eval_result_.keys()
        for k in eval_result_.keys():
            eval_result['{}_{}_{}'.format('macro-ner', type_, k)] = eval_result_[k]

    for k in metrics:
        macro_score = 0.0
        for type_, _ in type_to_indices.items():
            macro_score += eval_result['{}_{}_{}'.format('macro-ner', type_, k)]
        macro_score /= len(type_to_indices)
        eval_result['macro-ner_{}'.format(k)] = macro_score

    return eval_result


def macro_md_scores(examples, outputs, golds, soft_match):
    sidx_to_indices = defaultdict(list)
    for idx, example in enumerate(examples):
        sidx = example['id'].split('_')[0]
        sidx_to_indices[sidx].append(idx)

    type_to_indices = defaultdict(list)
    for idx, example in enumerate(examples):
        try:
            ent_type = example['id'].split('_')[1]
        except:
            ent_type = "NA"
        type_to_indices[ent_type].append(idx)

    metrics = []
    eval_result = {}

    for type_, type_indices in type_to_indices.items():
        golds_ = []
        outputs_ = []
        for sidx, sidx_indices in sidx_to_indices.items():
            indices = list(set(type_indices) & set(sidx_indices))
            golds_.append([parser(golds[i]) for i in indices])
            outputs_.append([parser(outputs[i]) for i in indices])

        eval_result_ = NEREvaluator().evaluate(outputs_, golds_, soft_match=soft_match)
        metrics = eval_result_.keys()
        for k in eval_result_.keys():
            eval_result['{}_{}_{}'.format('macro-md', type_, k)] = eval_result_[k]

        golds_ = []
        outputs_ = []
        for sidx, sidx_indices in sidx_to_indices.items():
            indices = list(set(sidx_indices) - set(type_indices))
            golds_.append([parser(golds[i]) for i in indices])
            outputs_.append([parser(outputs[i]) for i in indices])
        eval_result_ = NEREvaluator().evaluate(outputs_, golds_, soft_match=soft_match)
        metrics = eval_result_.keys()
        for k in eval_result_.keys():
            eval_result['{}_not{}_{}'.format('macro-md', type_, k)] = eval_result_[k]

    for k in metrics:
        macro_score = 0.0
        for type_, _ in type_to_indices.items():
            macro_score += eval_result['{}_{}_{}'.format('macro-md', type_, k)]
        macro_score /= len(type_to_indices)
        eval_result['macro-md_{}'.format(k)] = macro_score

    return eval_result


def is_list_of_lists(golds):
    for g in golds:
        if len(g) > 0 and type(g[0]) == list:
            return True
    return False


def is_list_of_stringlists(golds):
    for g in golds:
        if type(g) != list:
            continue
        for g_ in g:
            if g_.startswith('[') and g_.endswith(']'):
                return True
    return False


def transform_to_type_level(gold_entity_type_span_tuples, output_entity_type_span_tuples, examples):
    gold_type_set = set()
    for gold in gold_entity_type_span_tuples:
        for t in gold:
            entity_type = t[1]
            if isinstance(entity_type, list):
                entity_type = tuple(entity_type)
            gold_type_set.add(entity_type)

    num_ood_output_types = 0
    for output_ in output_entity_type_span_tuples:
        if type(output_) == str:
            output_ = []
        if type(output_) != list:
            output_ = []
        for t in output_:
            try:
                if len(t) < 2:
                    continue
            except:
                continue
            t = list(t)
            entity_type = t[1]
            if isinstance(entity_type, list):
                entity_type = tuple(entity_type)
            if entity_type not in gold_type_set:
                num_ood_output_types += 1

    print('number of ood output types', num_ood_output_types)

    type_level_golds = []
    type_level_outputs = []
    type_level_examples = []
    for example, golds, outputs in zip(examples, gold_entity_type_span_tuples, output_entity_type_span_tuples):
        outputs = list(outputs)
        for t in gold_type_set:
            golds_ = [g[0] for g in golds if g[1] == t]
            outputs = [list(o) for o in outputs if o is not None and o is not ... and isinstance(o, (list, tuple, str))]
            outputs_ = [o[0] for o in outputs if o is not None and len(o) == 2 and o[1] == t]
            if len(golds_) > 0 or len(outputs_) > 0:
                type_level_golds.append(golds_)
                type_level_outputs.append(outputs_)
                example_ = copy.deepcopy(example)
                example_['id'] = '{}_{}'.format(example['id'], t)
                type_level_examples.append(example_)
    return type_level_golds, type_level_outputs, type_level_examples


def is_list_tuples(golds):
    for g in golds:
        if len(g) > 0 and type(g[0]) == tuple:
            return True
    return False


def preprocess_golds(golds):
    golds_ = []
    for g in golds:
        if isinstance(g, str) and g.startswith('[') and g.endswith(']'):
            try:
                golds_.append(ast.literal_eval(g))
            except:
                golds_.append([])
        else:
            golds_.append(g)
    return golds_


def collect_results_strings(examples, golds, outputs, typed_eval=False):
    """Aggregate string-level P2S predictions by article id and compute metrics."""
    golds_aggregated = defaultdict(list)
    outputs_aggregated = defaultdict(list)
    for example, gold, output in zip(examples, golds, outputs):
        golds_aggregated[example['id'].split('_')[0]].append(gold)
        outputs_aggregated[example['id'].split('_')[0]].append(output)

    golds_doc_level = []
    outputs_doc_level = []
    article_ids = list(golds_aggregated.keys())
    for article_id in article_ids:
        golds_doc_level.append(golds_aggregated[article_id])
        outputs_doc_level.append(outputs_aggregated[article_id])

    all_results = []
    for soft_match in [True, False]:
        print(f"Evaluating with soft_match={soft_match} at doc level")
        sys.stdout.flush()
        all_results.append(micro_ner_scores(outputs_doc_level,
                                           golds_doc_level,
                                           soft_match,
                                           prefix='doclevel-'))
        if typed_eval:
            print(f"Evaluating with soft_match={soft_match} at macro level")
            sys.stdout.flush()
            all_results.append(macro_ner_scores(examples, outputs, golds, soft_match))

    eval_results = {}
    for r in all_results:
        eval_results.update(r)
    for k, v in eval_results.items():
        print(f"{k}: {v:.4f}")
    return eval_results


def collect_results(examples, golds, outputs, clf=False, typed_eval=False, debug=False):
    print('[initial] example output', outputs[0], type(outputs[0]))

    golds = preprocess_golds(golds)
    compute_pooling_metrics = False
    if is_list_of_lists(golds):
        golds_baseline = []
        for gold in golds:
            golds_baseline.append([g[0] for g in gold])
        compute_pooling_metrics = True
    elif is_list_tuples(golds):
        print('Transforming to type level evaluation since golds are list of tuples')
        golds_baseline, outputs, examples = transform_to_type_level(golds, outputs, examples)
        if is_list_of_stringlists(golds_baseline):
            golds_baseline_baseline = []
            golds_pooled = []
            for g in golds_baseline:
                entity_sets = preprocess_golds(g)
                golds_baseline_baseline.append([set_[0] for set_ in entity_sets if len(set_) > 0])
                golds_pooled.append(entity_sets)
            golds = golds_pooled
            golds_baseline = golds_baseline_baseline
            compute_pooling_metrics = True
    else:
        golds_baseline = golds

    all_results = []
    
    for soft_match in [True, False]:
        if not clf:
            all_results.append(micro_ner_scores(outputs, golds_baseline, soft_match))
            all_results.append(micro_md_scores(examples, outputs, golds_baseline, soft_match))
            if typed_eval:
                all_results.append(macro_ner_scores(examples, outputs, golds_baseline, soft_match))
        else:
            all_results.append(typing_scores(examples, outputs, golds_baseline))

    if compute_pooling_metrics:
        for soft_match in [True, False]:
            all_results.append(micro_ner_scores_pool(outputs, golds, soft_match))
            all_results.append(micro_md_scores_pool(examples, outputs, golds, soft_match))

    eval_results = {}
    for r in all_results:
        eval_results.update(r)
    for k, v in eval_results.items():
        print(f"{k}: {v:.4f}")
    return eval_results
