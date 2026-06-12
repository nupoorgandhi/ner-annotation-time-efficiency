"""
EntityInstance data class and FewNERD dataset loader.
"""

from datasets import load_dataset


class EntityInstance:
    """Maintains information for entity typing/NER task."""

    def __init__(self, sentence):
        self.raw_responses = []
        if sentence is not None:
            if type(sentence) == str:
                self.sentence = sentence
        self.pred_entities = []
        self.gold_entities = []
        self.alternate_entities = []
        self.gold_span_bounds = []
        self.gold_point_positions = []
        self.article_id = None
        self.annotator = None
        self.xcl = None
        self.entity_type = None
        self.span = None
        self.span_xcls = None
        self.clf = False
        self.messages = {}
        self.sentence_bounds = None


def get_few_nerd_data(subset_size=5000, split='train', config='supervised'):
    dataset = load_dataset("DFKI-SLT/few-nerd", config, split=split)
    dataset = dataset.shuffle(seed=42).select(range(subset_size))
    instances = []
    coarse_tags = ['0', 'art', 'building', 'event', 'location', 'organization', 'other', 'person', 'product']

    for i, entry in enumerate(dataset):
        if i >= subset_size:
            break
        tokens = entry['tokens']
        ner_tags = entry['ner_tags']
        sentence_str = " ".join(tokens)

        spans = []
        current_entity = None
        current_span_start = None

        for j, tag in enumerate(ner_tags):
            if tag != 0:
                coarse_tag = coarse_tags[tag]
                if current_entity is None:
                    current_span_start = j
                    current_entity = coarse_tag
                elif current_entity != coarse_tag:
                    spans.append((current_entity, ' '.join(tokens[current_span_start:j])))
                    current_span_start = j
                    current_entity = coarse_tag
            else:
                if current_entity is not None:
                    spans.append((current_entity, ' '.join(tokens[current_span_start:j])))
                    current_entity = None
                    current_span_start = None

        if current_entity is not None:
            spans.append((current_entity, ' '.join(tokens[current_span_start:])))

        inst = EntityInstance(None)
        inst.sentence = sentence_str
        inst.gold_entities = spans
        instances.append(inst)

    return instances
