# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import itertools
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

from modules import (WEIGHTS_NAME, BertConfig,
                     BertForQuestionAnswering, BertTokenizer,
                     XLMConfig, XLMForQuestionAnswering,
                     XLMTokenizer, XLNetConfig,
                     XLNetForQuestionAnswering,
                     XLNetTokenizer,
                     DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer)
from transformers import AdamW, WarmupLinearSchedule

from utils.utils_squad import (read_squad_examples, convert_examples_to_features,
                               RawResult, write_predictions,
                               RawResultExtended, write_predictions_extended)

# The follwing import is the official SQuAD evaluation script (2.0).
# You can remove it from the dependencies if you are using this script outside of the library
# We've added it here for automated tests (see examples/test_examples.py file)
from utils.utils_squad_evaluate import EVAL_OPTS, main as evaluate_on_squad

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) \
                  for conf in (BertConfig, XLNetConfig, XLMConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForQuestionAnswering, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForQuestionAnswering, XLMTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer)
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def create_filter_conditions(args, model):
    unfreeze_layer_idxs = [str(len(model.bert.encoder.layer) - 1 - i) for i in range(args.unfreeze_top_k_bert_layer)]
    unfreeze_layernorm_idxs = [str(len(model.bert.encoder.layer) - 1 - i) for i in range(args.unfreeze_top_k_layernorm)]
    fns = [lambda x: x.startswith("qa_outputs")]
    if args.elmo_style:
        fns.append(lambda x: x.startswith("elmo"))
    if args.apply_first_adapter_in_layer or args.apply_second_adapter_in_layer or args.apply_adapter_between_layer:
        fns.append(lambda x: "adapter" in x)
    fns.append(lambda x: any(idx in x for idx in unfreeze_layer_idxs))
    fns.append(lambda x: any(idx in x and "LayerNorm" in x for idx in unfreeze_layernorm_idxs))
    return lambda x: any(fn(x) for fn in fns)


def train(args, train_dataset, model, tokenizer):
    summary_dir_name = construct_folder_name(args)

    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(f"{args.logging_dir}/{summary_dir_name}")

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    # add sampling
    # bp()
    if args.num_sample > -1:
        random_indices = np.arange(len(train_dataset))
        np.random.shuffle(random_indices)
        random_indices = random_indices[:args.num_sample]
        train_dataset = torch.utils.data.TensorDataset(*train_dataset[random_indices])

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    condition_fn = create_filter_conditions(args, model)
    optimizer_grouped_parameters = [{'params': [], 'weight_decay': args.weight_decay},
                                    {'params': [], 'weight_decay': 0.0}]
    optimizer_parameters_name = []
    for n, p in model.named_parameters():
        # print(n)
        if condition_fn(n):
            p.requires_grad = True
            optimizer_parameters_name.append(n)
            if any(nd in n for nd in no_decay):
                optimizer_grouped_parameters[1]['params'].append(p)
            else:
                optimizer_grouped_parameters[0]['params'].append(p)
        else:
            p.requires_grad = False
    print(f"Parameters to be optimized: {optimizer_parameters_name}")

    # bp()
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0

    # only fine-tune the last layer
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        # bp()
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'start_positions': batch[3],
                      'end_positions': batch[4]}
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = None if args.model_type == 'xlm' else batch[2]
            if args.model_type in ['xlnet', 'xlm']:
                inputs.update({'cls_index': batch[5],
                               'p_mask': batch[6]})
            # bp()
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            # results = evaluate(args, model, tokenizer)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoi1nt
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model,
                                                            'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_results = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1]
                      }
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = None if args.model_type == 'xlm' else batch[2]  # XLM don't use segment_ids
            example_indices = batch[3]
            if args.model_type in ['xlnet', 'xlm']:
                inputs.update({'cls_index': batch[4],
                               'p_mask': batch[5]})
            outputs = model(**inputs)

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            if args.model_type in ['xlnet', 'xlm']:
                # XLNet uses a more complex post-processing procedure
                result = RawResultExtended(unique_id=unique_id,
                                           start_top_log_probs=to_list(outputs[0][i]),
                                           start_top_index=to_list(outputs[1][i]),
                                           end_top_log_probs=to_list(outputs[2][i]),
                                           end_top_index=to_list(outputs[3][i]),
                                           cls_logits=to_list(outputs[4][i]))
            else:
                result = RawResult(unique_id=unique_id,
                                   start_logits=to_list(outputs[0][i]),
                                   end_logits=to_list(outputs[1][i]))
            all_results.append(result)

    # Compute predictions
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))
    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

    if args.model_type in ['xlnet', 'xlm']:
        # XLNet uses a more complex post-processing procedure
        write_predictions_extended(examples, features, all_results, args.n_best_size,
                                   args.max_answer_length, output_prediction_file,
                                   output_nbest_file, output_null_log_odds_file, args.predict_file,
                                   model.config.start_n_top, model.config.end_n_top,
                                   args.version_2_with_negative, tokenizer, args.verbose_logging)
    else:
        write_predictions(examples, features, all_results, args.n_best_size,
                          args.max_answer_length, args.do_lower_case, output_prediction_file,
                          output_nbest_file, output_null_log_odds_file, args.verbose_logging,
                          args.version_2_with_negative, args.null_score_diff_threshold)

    # Evaluate with the official SQuAD script
    evaluate_options = EVAL_OPTS(data_file=args.predict_file,
                                 pred_file=output_prediction_file,
                                 na_prob_file=output_null_log_odds_file)
    results = evaluate_on_squad(evaluate_options)
    return results


def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    input_file = args.predict_file if evaluate else args.train_file
    # bp()
    cached_features_file = os.path.join(os.path.dirname(input_file), 'cached_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache and not output_examples:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", input_file)
        examples = read_squad_examples(input_file=input_file,
                                       is_training=not evaluate,
                                       version_2_with_negative=args.version_2_with_negative)
        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                max_seq_length=args.max_seq_length,
                                                doc_stride=args.doc_stride,
                                                max_query_length=args.max_query_length,
                                                is_training=not evaluate)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
    # sys.exit(0)
    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
    if evaluate:
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_example_index, all_cls_index, all_p_mask)
    else:
        all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_start_positions, all_end_positions,
                                all_cls_index, all_p_mask)

    if output_examples:
        return dataset, examples, features
    return dataset


def parse_range(string: str):
    """Return a tuple that represent the range, inclusive on both sides"""
    string = string.strip()
    assert '-' in string
    assert '[' == string[0]
    assert ']' == string[-1]
    pair = string.split("-")
    assert len(pair) == 2
    lower = int(pair[0][1:])
    upper = int(pair[1][:-1])
    return list(range(lower, upper + 1))


def main(args):
    if args.num_sample == 0:
        raise ValueError("Either -1 to indicate not using sampling, "
                         "or > 0 to indicate the legitimate number of samples")
    if args.adapter_activation not in [0, 1]:
        raise ValueError("Only 0/1 are legit")
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            f"cuda:{args.gpu_id}" if torch.cuda.is_available() and not args.no_cuda else "cpu")  # TODO: delete :0
        args.n_gpu = 1  # TODO:  torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_proBertForQuestionAnsweringcess_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    # bp()

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    # customized config
    # elmo_style argument
    config.output_hidden_states = args.elmo_style  # TODO: experiment weighting different layers(ELMo style)
    config.elmo_style = args.elmo_style  # TODO: experiment weighting different layers(ELMo style)
    # adapter argument
    config.adapter_range = parse_range(args.adapter_range)
    config.adapter_activation = args.adapter_activation
    config.apply_first_adapter_in_layer = args.apply_first_adapter_in_layer  # apply first adapter in layer
    config.apply_second_adapter_in_layer = args.apply_second_adapter_in_layer  # apply first adapter in layer
    config.apply_adapter_between_layer = args.apply_adapter_between_layer  # apply first adapter in layer
    config.bottleneck_size = args.bottleneck_size  # set bottleneck size
    config.init_scale = args.init_scale  # set bottleneck size
    assert args.init_scale >= 0
    config.init_scale = args.init_scale  # standard deviation for normal
    # bidaf argument
    config.top_layer = args.top_layer
    config.dropout = args.dropout
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case)
    config.sep = tokenizer._convert_token_to_id('[SEP]')
    # bp()
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, 'einsum')
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    print(args)

    # return
    # Training
    # bp()
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Save the trained model and the tokenizer
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)

            # Evaluate
            result = evaluate(args, model, tokenizer, prefix=global_step)

            result = dict((k + ('_{}'.format(global_step) if global_step else ''), v) for k, v in result.items())
            results.update(result)

    logger.info("Results: {}".format(results))

    return results


def construct_folder_name(args):
    model_dir_name = f"lr{args.learning_rate}.unfreeze_top_{args.unfreeze_top_k_bert_layer}_bert_layer." \
                     f"epoch{args.num_train_epochs}.bs{args.per_gpu_train_batch_size * args.gradient_accumulation_steps}"
    if args.apply_first_adapter_in_layer:
        model_dir_name += f".adapter{args.bottleneck_size}.FirstAdaInLayer"
    if args.apply_second_adapter_in_layer:
        model_dir_name += f".adapter{args.bottleneck_size}.SecondAdaInLayer"
    if args.apply_adapter_between_layer:
        model_dir_name += f".adapter{args.bottleneck_size}.AdaBetweenLayer"
    if args.adapter_activation == 0:
        model_dir_name += f".adapterNoActivation"
    model_dir_name += f".adapterRange{args.adapter_range}"
    if args.num_sample > -1:
        model_dir_name += f".sample{args.num_sample}"
    if args.check:
        model_dir_name += ".check"
    return model_dir_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_file", default=None, type=str, required=True,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--predict_file", default=None, type=str, required=True,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
    parser.add_argument("--run_mode", default=None, type=str, required=True,
                        help="single_run | adapter_random_search | ...")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--logging_dir", default=None, type=str, required=True,
                        help="Logging dir")
    parser.add_argument("--check", action="store_true",
                        help="flag to indicate if current execution is for checking")
    parser.add_argument("--gpu_id", default=None, type=int, required=True,
                        help="The index of gpu to use")

    ## Other parameters
    # elmo_style + freezing/unfreezing
    # parser.add_argument("--freeze_pretrained", action='store_true',
    #                     help="Store the pretrained BERT weights and only train the last layer")
    parser.add_argument("--elmo_style", action='store_true',
                        help="Have our output to be weighted in a ELMo-like fashion")  # TODO: This is not working
    parser.add_argument("--unfreeze_top_k_bert_layer", default=0, type=int,
                        help="unfreeze top k transformer layers")
    parser.add_argument("--unfreeze_top_k_layernorm", default=0, type=int,
                        help="unfreeze top k transformer layers")
    # adapter parameter
    parser.add_argument("--adapter_range", default="[0-11]", type=str,
                        help="In put has to be '[x-y]' format, 0 <= x <= y <= 11")
    parser.add_argument("--apply_first_adapter_in_layer", action='store_true',
                        help="choose to whether apply adapter in each layer")
    parser.add_argument("--apply_second_adapter_in_layer", action='store_true',
                        help="choose to whether apply adapter in each layer")
    parser.add_argument("--apply_adapter_between_layer", action='store_true',
                        help="choose to whether apply adapter between each layer")
    parser.add_argument("--bottleneck_size", default=64, type=int,
                        help="the size of bottleneck design")
    parser.add_argument("--init_scale", default=1e-3, type=float,
                        help="the std of the truncated normal distribution used for initialization")
    parser.add_argument("--adapter_activation", default=1, type=int,
                        help="0/1 flag to decide if apply non-linearity function on bottleneck")
    # bidaf parameter
    parser.add_argument("--top_layer", default="linear", type=str,
                        help="The type of top layer")
    parser.add_argument('--dropout', default=0.2, type=float)

    # other
    parser.add_argument("--num_sample", default=-1, type=int,
                        help="Number of samples(w/o replacement) to draw, presumably less than "
                             "the entire dataset")

    # official
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--version_2_with_negative', action='store_true',
                        help='If true, the SQuAD examples contain some that do not have an answer.')
    parser.add_argument('--null_score_diff_threshold', type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")

    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the "
                             "nbest_predictions.json output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name"
                             " ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

    out_dir = args.output_dir

    if args.run_mode == "single_run":
        model_dir_name = construct_folder_name(args)
        args.output_dir = out_dir + f"/{model_dir_name}"
        print(f"lr: {args.learning_rate} \t num_train_epochs: {args.num_train_epochs}")
        print(args.output_dir)
        main(args)
    elif args.run_mode == "adapter_random_search":
        lrs = [3e-5, 1e-4, 3e-4, 1e-3]
        epochs = [3, 10, 20]
        num_tune = 5

        all_hypers = list(itertools.product(lrs, epochs))
        print(f"All Hypers: {all_hypers}")
        hyper_idxs = set()
        while len(hyper_idxs) < num_tune:
            hyper_idxs.add(np.random.randint(0, len(all_hypers)))
        print(f"Hyper_idxs: {hyper_idxs}")
        for i, idx in enumerate(hyper_idxs):
            print(f"The {i}th set of hyperparameters")

            lr, epoch = all_hypers[idx]
            args.learning_rate = lr
            args.num_train_epochs = epoch
            model_dir_name = construct_folder_name(args)
            args.output_dir = out_dir + f"/{model_dir_name}"
            print(f"lr: {args.learning_rate} \t num_train_epochs: {args.num_train_epochs}")
            summary_name = f"lr{args.learning_rate}.unfreeze_top_{args.unfreeze_top_k_bert_layer}_bert_layer." \
                           f"epoch{args.num_train_epochs}.bs{args.per_gpu_train_batch_size * args.gradient_accumulation_steps}."
            if args.apply_adapter:
                summary_name += f"adapter{args.bottleneck_size}."
            if args.adapter_activation == 0:
                model_dir_name += f".adapterNoActivation"
            if args.num_sample > -1:
                model_dir_name += f".sample{args.num_sample}"
            print(f"{args.logging_dir}/{summary_name}")
            print(args.output_dir)
            main(args)
