"""
Load NER datasets (CRAFT, GENIA, FewNERD, POLIANNA) into EntityInstance lists.
"""

import pandas as pd
import os
import re
import xml.etree.ElementTree as ET
import nltk
from datasets import load_dataset
from src.data.instances import EntityInstance, get_few_nerd_data

nltk.download('punkt')


def split_text_into_sentences(text):
    sentences = nltk.sent_tokenize(text)
    indices = [(text.index(sentence), text.index(sentence) + len(sentence)) for sentence in sentences]
    return sentences, indices


def parse_curation(curation_str, keep_indices=False):
    """Parse a POLIANNA curation string into (span_type, span_text) tuples."""
    if isinstance(curation_str, list):
        entities = []
        for sp in curation_str:
            span_type = f"{sp.layer}_{sp.feature}_{sp.tag}"
            span_type = span_type.replace('_2', '')
            span_text = sp.text.strip()
            start = sp.start
            stop = sp.stop
            if not keep_indices:
                entities.append((span_type, span_text))
            else:
                entities.append((span_type, span_text, start, stop))
        return entities

    if not isinstance(curation_str, str):
        return []

    pattern = r'layer:(\w+)\s+feature:(\w+)\s+tag:(\w+)\s+start:(\d+)\s+stop:(\d+)\s+text:([^,]+)'
    matches = re.findall(pattern, curation_str)

    entities = []
    for layer, feature, tag, start, stop, span_text in matches:
        span_type = f"{layer}_{feature}_{tag}"
        span_type = span_type.replace('_2', '')
        if not keep_indices:
            entities.append((span_type, span_text.strip()))
        else:
            start = int(start)
            stop = int(stop)
            entities.append((span_type, span_text.strip(), start, stop))
    return entities


def clean_instances(instances, include_alternate_spans=False):
    """Normalize non-breaking spaces in span texts and sentences."""
    for inst in instances:
        if not include_alternate_spans:
            inst.gold_entities = [(span_type, span_text.replace(' ', '')) for span_type, span_text in inst.gold_entities]
        else:
            new_gold_entities = []
            for g in inst.gold_entities:
                new_gold_entities.append([(span_type, span_text.replace(' ', '')) for span_type, span_text in g])
            inst.gold_entities = new_gold_entities
        inst.sentence = inst.sentence.replace(' ', '')
    return instances


def extract_entities_from_bio(tokens, ner_tags, label_list):
    """Extract (entity_type, span_text) tuples from BIO-tagged tokens."""
    entities = []
    current_entity = []
    current_type = None

    for token, tag_idx in zip(tokens, ner_tags):
        tag = label_list[tag_idx]

        if tag.startswith('B-'):
            if current_entity:
                entities.append((current_type, ' '.join(current_entity)))
            current_entity = [token]
            current_type = tag[2:]
        elif tag.startswith('I-'):
            entity_type = tag[2:]
            if current_type == entity_type:
                current_entity.append(token)
            else:
                if current_entity:
                    entities.append((current_type, ' '.join(current_entity)))
                current_entity = [token]
                current_type = entity_type
        else:
            if current_entity:
                entities.append((current_type, ' '.join(current_entity)))
                current_entity = []
                current_type = None

    if current_entity:
        entities.append((current_type, ' '.join(current_entity)))

    return entities


def load_craft(data_dir, split_sentences=True, schema='CHEBI', chunk_size=-1, filter_empty=False):
    articles_plaintext_dir = os.path.join(data_dir, 'articles', 'txt')

    def get_article_text(article_id):
        with open(os.path.join(articles_plaintext_dir, f"{article_id}.txt"), 'r') as f:
            text = f.read()
        return text

    def parse_craft_annotations(xml_str):
        """Parse CRAFT XML annotations into (entity_type, span_text, start, end) tuples."""
        try:
            root = ET.fromstring(xml_str)
        except ET.ParseError as e:
            print(f"Error parsing XML: {e}")
            return []

        mention_to_class = {}
        for class_mention in root.findall('classMention'):
            mention_id = class_mention.get('id')
            mention_class = class_mention.find('mentionClass')
            if mention_class is not None:
                class_id = mention_class.get('id')
                class_name = mention_class.text
                mention_to_class[mention_id] = {
                    'class_id': class_id,
                    'class_name': class_name
                }

        entities = []
        for annotation in root.findall('annotation'):
            mention_elem = annotation.find('mention')
            span_elem = annotation.find('span')
            spanned_text_elem = annotation.find('spannedText')

            if mention_elem is not None and span_elem is not None and spanned_text_elem is not None:
                mention_id = mention_elem.get('id')
                start = int(span_elem.get('start'))
                end = int(span_elem.get('end'))
                span_text = spanned_text_elem.text.strip()

                entity_type = 'UNKNOWN'
                if mention_id in mention_to_class:
                    entity_type = mention_to_class[mention_id]['class_name']

                entities.append((entity_type, span_text, start, end))
        return entities

    concept_annotation_dir = os.path.join(data_dir, 'concept-annotation', schema, schema, 'knowtator')
    entity_instances = []

    for fname in os.listdir(concept_annotation_dir):
        if fname.endswith('.knowtator.xml'):
            article_id = fname.split('.')[0]
            article_text = get_article_text(article_id)

            annotation_file = os.path.join(concept_annotation_dir, fname)
            with open(annotation_file, 'r', encoding='utf-8') as f:
                xml_content = f.read()

            gold_entities_with_indices = parse_craft_annotations(xml_content)

            if split_sentences:
                prechunked_entity_instances = []
                sentences, sentence_indices = split_text_into_sentences(article_text)
                for sentence, bounds in zip(sentences, sentence_indices):
                    new_instance = EntityInstance(sentence)
                    new_instance.article_id = article_id
                    new_instance.sentence_bounds = bounds
                    new_instance.gold_entities = [
                        (span_type, span_text) for span_type, span_text, start, stop in gold_entities_with_indices
                        if span_text in sentence and start >= bounds[0] and stop <= bounds[1]
                    ]
                    prechunked_entity_instances.append(new_instance)
                if chunk_size > 0:
                    for i in range(0, len(prechunked_entity_instances), chunk_size):
                        sentences = '\n'.join([inst.sentence for inst in prechunked_entity_instances[i:i+chunk_size]])
                        new_instance = EntityInstance(sentences)
                        new_instance.article_id = article_id
                        new_instance.gold_entities = []
                        for inst in prechunked_entity_instances[i:i+chunk_size]:
                            new_instance.gold_entities.extend(inst.gold_entities)
                        entity_instances.append(new_instance)
                else:
                    entity_instances.extend(prechunked_entity_instances)
            else:
                instance = EntityInstance(article_text)
                instance.article_id = article_id
                instance.gold_entities = [(span_type, span_text) for span_type, span_text, _, _ in gold_entities_with_indices]
                entity_instances.append(instance)

    if filter_empty:
        entity_instances = [inst for inst in entity_instances if len(inst.gold_entities) > 0]
    return entity_instances


def load_genia():
    """Load GENIA NER data from HuggingFace (all splits combined)."""
    tag_list = ['O', 'B-DNA', 'I-DNA', 'B-RNA', 'I-RNA', 'B-cell line',
                'I-cell line', 'B-cell type', 'I-cell type', 'B-protein', 'I-protein']

    entity_instances = []
    for split in ['train', 'validation', 'test']:
        dataset = load_dataset("chufangao/GENIA-NER", split=split)
        for example in dataset:
            tokens = example['tokens']
            sentence = ' '.join(tokens)
            new_instance = EntityInstance(sentence)
            new_instance.article_id = f"genia_{split}_{example['id']}"
            entities = extract_entities_from_bio(tokens, example['ner_tags'], tag_list)
            new_instance.gold_entities = entities
            entity_instances.append(new_instance)

    return entity_instances




def load_polianna(filepath, split_sentences=False, entity_type_map=None, include_alternate_spans=False):
    """Load POLIANNA from a pickle file."""
    df = pd.read_pickle(filepath)
    print(f"Loaded {len(df)} rows from {filepath}")
    print(f"Average document length: {df['Text'].apply(lambda x: len(x.split())).mean()} words")

    entity_instances = []
    for article_id, row in df.iterrows():
        instance = EntityInstance(row['Text'])
        instance.gold_entities = parse_curation(row['Curation'], keep_indices=True)
        if include_alternate_spans:
            for annotator in ['A', 'C', 'F', 'B', 'E', 'G', 'D']:
                if annotator in row:
                    if row[annotator] != None:
                        instance.alternate_entities.extend(parse_curation(row[annotator], keep_indices=True))

        instance.article_id = article_id
        entity_instances.append(instance)

    unique_article_ids = len(set([instance.article_id for instance in entity_instances]))
    print(f"Number of unique article ids: {unique_article_ids}")

    entity_instances = [instance for instance in entity_instances if len(instance.gold_entities) > 0]
    print(f"Number of instances with at least one gold entity: {len(entity_instances)}")

    entity_instances_split = []
    if split_sentences:
        for instance in entity_instances:
            sentences, sentence_indices = split_text_into_sentences(instance.sentence)
            for sentence, bounds in zip(sentences, sentence_indices):
                assert sentence in df.loc[instance.article_id]['Text'], f"Sentence '{sentence}' not found in article text for article id {instance.article_id}"
                new_instance = EntityInstance(sentence)
                new_instance.article_id = instance.article_id
                new_instance.sentence_bounds = bounds

                if include_alternate_spans:
                    gold_curation_entities = [
                        (span_type, span_text, start, stop) for span_type, span_text, start, stop in instance.gold_entities
                        if span_text in sentence and start >= bounds[0] and stop <= bounds[1]
                    ]
                    alternate_include_gold_entities = []
                    for span_type, span_text, start, stop in gold_curation_entities:
                        alternate_entities = [
                            (alt_span_type, alt_span_text, alt_start, alt_stop) for alt_span_type, alt_span_text, alt_start, alt_stop in instance.alternate_entities
                            if alt_span_text in sentence and alt_start >= bounds[0] and alt_stop <= bounds[1]
                        ]
                        alternate_entities = [
                            (alt_span_type, alt_span_text) for alt_span_type, alt_span_text, alt_start, alt_stop in alternate_entities
                            if max(start, alt_start) < min(stop, alt_stop)
                        ]
                        gold_entities = [(span_type, span_text)] + alternate_entities
                        if entity_type_map is not None:
                            gold_entities = [
                                (entity_type_map.get(span_type, span_type), span_text) for span_type, span_text in gold_entities
                            ]
                        alternate_include_gold_entities.append(gold_entities)
                    new_instance.gold_entities = alternate_include_gold_entities

                else:
                    new_instance.gold_entities = [
                        (span_type, span_text) for span_type, span_text, start, stop in instance.gold_entities
                        if span_text in sentence and start >= bounds[0] and stop <= bounds[1]
                    ]
                    if entity_type_map is not None:
                        new_instance.gold_entities = [
                            (entity_type_map.get(span_type, span_type), span_text)
                            for span_type, span_text in new_instance.gold_entities
                        ]
                entity_instances_split.append(new_instance)
        entity_instances_split = clean_instances(entity_instances_split, include_alternate_spans=include_alternate_spans)
        return entity_instances_split

    entity_instances = clean_instances(entity_instances, include_alternate_spans=include_alternate_spans)
    return entity_instances
