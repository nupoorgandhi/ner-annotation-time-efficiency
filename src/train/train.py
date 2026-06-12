
# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""
Supervised fine-tuning script for causal LMs.

Adapted from tatsu-lab/stanford_alpaca and lm-sys/FastChat. Supports full
fine-tuning with ChatML and Llama-2 masking styles.
"""

from dataclasses import dataclass, field
import json
import math
import pathlib
from typing import Dict, Optional, Sequence
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother
import os
from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to allow for custom models defined on the Hub in their own modeling files"
        },
    )
    padding_side: str = field(
        default="right", metadata={"help": "The padding side in tokenizer"}
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    secondary_data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")

    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


local_rank = None


def rank0_print(*args):
    if local_rank is None or local_rank == 0:
        print(*args)


def trainer_save_model_safe(trainer: transformers.Trainer):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        trainer.model, StateDictType.FULL_STATE_DICT, save_policy
    ):
        trainer.save_model()

def find_subsequence(sequence: list, subsequence: list) -> list:
    """Find all starting positions of subsequence in sequence."""
    positions = []
    subseq_len = len(subsequence)

    for i in range(len(sequence) - subseq_len + 1):
        if sequence[i:i + subseq_len] == subsequence:
            positions.append(i)

    return positions

def mask_falcon_style(conversation, input_ids, target, tokenizer, conv_idx):
    """
    Mask targets for Falcon style conversations.
    Only unmasks the second assistant response.
    """
    target[:] = IGNORE_TOKEN_ID
    conv_tokens = tokenizer(conversation, add_special_tokens=False).input_ids
    assistant_role = "Assistant:"
    assistant_tokens = tokenizer(assistant_role, add_special_tokens=False).input_ids
    user_role = "User:"
    user_tokens = tokenizer(user_role, add_special_tokens=False).input_ids
    # Find all Assistant: positions
    assistant_indices = []
    start = 0
    while start < len(conv_tokens):
        try:
            idx = conv_tokens.index(assistant_tokens[0], start)
        except ValueError:
            break
        if conv_tokens[idx:idx+len(assistant_tokens)] == assistant_tokens:
            assistant_indices.append(idx)
            start = idx + len(assistant_tokens)
        else:
            start = idx + 1
    # Only unmask the second assistant message (index 1)
    if len(assistant_indices) > 1:
        unmask_start = assistant_indices[1] + len(assistant_tokens)
        # Find next User: after this Assistant:
        try:
            unmask_end = conv_tokens.index(user_tokens[0], unmask_start)
        except ValueError:
            unmask_end = len(conv_tokens)
        target[unmask_start:unmask_end] = input_ids[unmask_start:unmask_end]

def mask_llama2_style(input_ids, target, tokenizer, conv_idx):
    """
    Mask targets for Mistral/LLAMA2 style conversations.
    Format: [INST] user_message [/INST] assistant_message </s>
    """
    inst_close_with_space = tokenizer(" [/INST]", add_special_tokens=False).input_ids
    inst_close_no_space = tokenizer("[/INST]", add_special_tokens=False).input_ids

    eos_token_id = tokenizer.eos_token_id

    target[:] = IGNORE_TOKEN_ID

    input_list = input_ids.tolist()

    inst_positions_with_space = find_subsequence(input_list, inst_close_with_space)
    inst_positions_no_space = find_subsequence(input_list, inst_close_no_space)

    if len(inst_positions_with_space) > 0:
        inst_positions = inst_positions_with_space
        inst_close = inst_close_with_space
    elif len(inst_positions_no_space) > 0:
        inst_positions = inst_positions_no_space
        inst_close = inst_close_no_space
    else:
        inst_positions = []

    for turn_idx, inst_pos in enumerate(inst_positions):
        start_pos = inst_pos + len(inst_close)

        end_pos = len(input_list)
        for j in range(start_pos, len(input_list)):
            if input_list[j] == eos_token_id:
                end_pos = j + 1
                break

        # Only unmask the second assistant response (turn_idx == 1)
        if turn_idx == 1 and start_pos < len(target) and end_pos > start_pos:
            end_pos = min(end_pos, len(target))
            target[start_pos:end_pos] = input_ids[start_pos:end_pos]


def mask_chatml_style(input_ids, target, tokenizer, conv_idx):
    """
    Mask targets for Qwen/CHATML style conversations.
    Format: <|im_start|>user\nmessage<|im_end|>\n<|im_start|>assistant\nmessage<|im_end|>
    Only unmasks the second assistant response (similar to Mistral setup).
    """
    assistant_start = tokenizer("<|im_start|>assistant\n", add_special_tokens=False).input_ids
    im_end = tokenizer("<|im_end|>", add_special_tokens=False).input_ids

    target[:] = IGNORE_TOKEN_ID

    input_list = input_ids.tolist()

    assistant_positions = find_subsequence(input_list, assistant_start)

    for turn_idx, assist_pos in enumerate(assistant_positions):
        start_pos = assist_pos + len(assistant_start)

        end_pos = len(input_list)
        for j in range(start_pos, len(input_list)):
            if input_list[j:j+len(im_end)] == im_end:
                end_pos = j + len(im_end)
                break

        # Only unmask the second assistant response (turn_idx == 1)
        if turn_idx == 1 and start_pos < len(target) and end_pos > start_pos:
            end_pos = min(end_pos, len(target))
            target[start_pos:end_pos] = input_ids[start_pos:end_pos]


def mask_llama3_style(conversation, input_ids, target, tokenizer, conv):
    """
    Mask targets for Llama-3 style conversations.
    Uses separator-based splitting.
    Only unmasks the second assistant response (similar to Mistral and Qwen setup).
    """
    sep = conv.sep + conv.roles[1]
    turns = conversation.split(conv.sep2)

    target[:] = IGNORE_TOKEN_ID

    cur_len = 1
    assistant_turn_count = 0

    for i, turn in enumerate(turns):
        if turn == "":
            break

        turn_len = len(tokenizer(turn).input_ids)
        parts = turn.split(sep)

        if len(parts) != 2:
            break

        parts[0] += sep
        instruction_len = len(tokenizer(parts[0]).input_ids) - 2

        response_start = cur_len + instruction_len
        response_end = cur_len + turn_len

        # Only unmask the second assistant response (assistant_turn_count == 1)
        if assistant_turn_count == 1:
            target[response_start:response_end] = input_ids[response_start:response_end]

        assistant_turn_count += 1
        cur_len += turn_len


def setup_conversation_template(model_name, tokenizer):
    """Set up the appropriate conversation template based on model name."""
    if model_name and any(variant in model_name.lower() for variant in ["mistral", "mixtral"]):
        conv = get_conversation_template("mistral")

        if tokenizer.eos_token_id is not None:
            if conv.stop_token_ids is None:
                conv.stop_token_ids = [tokenizer.eos_token_id]
            elif tokenizer.eos_token_id not in conv.stop_token_ids:
                conv.stop_token_ids.append(tokenizer.eos_token_id)

        eos_str_tokens = tokenizer("</s>", add_special_tokens=False).input_ids
        for eos_str_token in eos_str_tokens:
            if eos_str_token not in (conv.stop_token_ids or []):
                if conv.stop_token_ids is None:
                    conv.stop_token_ids = [eos_str_token]
                else:
                    conv.stop_token_ids.append(eos_str_token)

        rank0_print(f"Using Mistral template for: {model_name}")
        rank0_print(f"Stop tokens: {conv.stop_token_ids}")

    elif model_name and 'falcon' in model_name.lower():
        conv = get_conversation_template("falcon")
        tokenizer.pad_token = tokenizer.eos_token

        rank0_print(f"Using Falcon template for: {model_name}")

    elif model_name and "qwen" in model_name.lower():
        conv = get_conversation_template("qwen-7b-chat")

        if tokenizer.eos_token_id is not None:
            if conv.stop_token_ids is None:
                conv.stop_token_ids = [tokenizer.eos_token_id]
            elif tokenizer.eos_token_id not in conv.stop_token_ids:
                conv.stop_token_ids.append(tokenizer.eos_token_id)

        im_end_tokens = tokenizer("<|im_end|>", add_special_tokens=False).input_ids
        for im_end_token in im_end_tokens:
            if im_end_token not in (conv.stop_token_ids or []):
                if conv.stop_token_ids is None:
                    conv.stop_token_ids = [im_end_token]
                else:
                    conv.stop_token_ids.append(im_end_token)

        rank0_print(f"Using Qwen template for: {model_name}")
        rank0_print(f"Stop tokens: {conv.stop_token_ids}")

    else:
        conv = get_conversation_template("llama-3")
        rank0_print(f"Using Llama-3 template for: {model_name}")

    return conv


def build_conversations(sources, conv):
    """Build conversation strings from sources using the conversation template."""
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    conversations = []

    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"Role mismatch at conversation {i}"
            conv.append_message(role, sentence["value"])

        conversations.append(conv.get_prompt())

    return conversations


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    model_name: str = None,
) -> Dict:
    """
    Preprocess conversations for training.
    Handles Mistral, Qwen, and Llama-3 style conversations.
    """
    conv = setup_conversation_template(model_name, tokenizer)

    conversations = build_conversations(sources, conv)

    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids

    targets = input_ids.clone()

    rank0_print(f'Conversation template: {conv.name}')
    rank0_print(f'Sep style: {conv.sep_style}')
    rank0_print(f'Separators: "{conv.sep}", "{conv.sep2}"')
    rank0_print(f'Roles: {conv.roles}')

    num_exceeding_max = 0
    for conv_idx, (conversation, target) in enumerate(zip(conversations, targets)):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        if conv.name == "falcon":
            mask_falcon_style(conversation, input_ids[conv_idx], target, tokenizer, conv_idx)
        elif conv.sep_style == SeparatorStyle.LLAMA2:
            mask_llama2_style(input_ids[conv_idx], target, tokenizer, conv_idx)
        elif conv.sep_style == SeparatorStyle.CHATML:
            mask_chatml_style(input_ids[conv_idx], target, tokenizer, conv_idx)
        else:
            mask_llama3_style(conversation, input_ids[conv_idx], target, tokenizer, conv)
        if total_len >= tokenizer.model_max_length:
            num_exceeding_max += 1
    print(f"WARNING: {num_exceeding_max}/{len(sources)} conversations exceed max length")
    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, model_name: str = None):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer, model_name)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]
        self.num_examples = len(sources)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
            num_examples=self.num_examples
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, model_name: str = None):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.model_name = model_name

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer, self.model_name)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args, model_name: str = None
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))

    if data_args.secondary_data_path is not None:
        secondary_train_json = json.load(open(data_args.secondary_data_path, "r"))
        train_json.extend(secondary_train_json)
        rank0_print(f"Combined primary and secondary data. Total training examples: {len(train_json)}")

    train_dataset = dataset_cls(train_json, tokenizer=tokenizer, model_name=model_name)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer, model_name=model_name)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side=model_args.padding_side,
        use_fast=False,
        trust_remote_code=model_args.trust_remote_code,
    )

    # Set up padding token appropriately for different models
    if tokenizer.pad_token is None:
        if any(mistral_variant in model_args.model_name_or_path.lower() for mistral_variant in ["mistral", "mixtral"]):
            tokenizer.pad_token = tokenizer.eos_token
            rank0_print(f"Set pad_token to eos_token for Mistral model: {tokenizer.pad_token}")
        elif "qwen" in model_args.model_name_or_path.lower():
            tokenizer.pad_token = tokenizer.eos_token
            rank0_print(f"Set pad_token to eos_token for Qwen model: {tokenizer.pad_token}")
        else:
            tokenizer.pad_token = tokenizer.unk_token
            rank0_print(f"Set pad_token to unk_token for non-Mistral/non-Qwen model: {tokenizer.pad_token}")
    elif tokenizer.pad_token != tokenizer.unk_token and not any(variant in model_args.model_name_or_path.lower() for variant in ["mistral", "mixtral", "qwen"]):
        tokenizer.pad_token = tokenizer.unk_token
        rank0_print(f"Updated pad_token to unk_token: {tokenizer.pad_token}")

    # Load data
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, model_name=model_args.model_name_or_path)

    # Start trainer
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save model
    model.config.use_cache = True
    trainer.save_state()
    if trainer.is_deepspeed_enabled:
        trainer.save_model()
    else:
        trainer_save_model_safe(trainer)


if __name__ == "__main__":
    train()
