from collections import defaultdict
import time
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from math import ceil
import torch
from torch import nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
import numpy as np
import os
import shutil
import sys
from tqdm import tqdm
from src.model import LOTClassModel
from config.configs_interface import Configs
from src.data_utils.utils import load_stop_words
import warnings
import jieba, re
import itertools
from src.logers import LOGS

warnings.filterwarnings("ignore")


class WoBertTokenizer(BertTokenizer):
    def _tokenize(self, text):
        cut_words = jieba.lcut(text)
        return_words = []
        for w in cut_words:
            if w in self.vocab:
                # will not [UNK]
                return_words.append(w)
            else:
                # will be [UNK]
                w = list(w)
                return_words.extend(w)

        return return_words

    def tokenize(self, text, **kwargs):
        all_special_tokens = self.all_special_tokens

        def lowercase_text(t):
            # convert non-special tokens to lowercase
            escaped_special_toks = [re.escape(s_tok) for s_tok in all_special_tokens]
            pattern = r"(" + r"|".join(escaped_special_toks) + r")|" + r"(.+?)"
            return re.sub(pattern, lambda m: m.groups()[0] or m.groups()[1].lower(), t)

        if self.init_kwargs.get("do_lower_case", False):
            text = lowercase_text(text)

        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                sub_text = sub_text.rstrip()
                if i == 0 and not sub_text:
                    result += [tok]
                elif i == len(split_text) - 1:
                    if sub_text:
                        result += [sub_text]
                    else:
                        pass
                else:
                    if sub_text:
                        result += [sub_text]
                    result += [tok]
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []
            if not tok_list:
                return self._tokenize(text)

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.unique_added_tokens_encoder:
                        tokenized_text += split_on_token(tok, sub_text)
                    else:
                        tokenized_text += [sub_text]
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        self._tokenize(token) if token not in self.unique_added_tokens_encoder else [token]
                        for token in tokenized_text
                    )
                )
            )

        added_tokens = self.unique_added_tokens_encoder
        tokenized_text = split_on_tokens(added_tokens, text)
        return tokenized_text


class LOTClassTrainer(object):

    def __init__(self, args: Configs):
        self.args = args
        self.max_len = args.train_args.MAX_LEN
        self.dataset_dir = args.data.DATASET
        self.dist_port = args.train_args.dist_port
        self.num_cpus = min(4, cpu_count() - 4) if cpu_count() > 1 else 1
        self.world_size = args.train_args.GPUS
        self.train_batch_size = args.train_args.TRAIN_BATCH
        self.eval_batch_size = args.train_args.EVAL_BATCH
        self.accum_steps = args.train_args.ACCUM_STEP
        eff_batch_size = self.train_batch_size * self.world_size * self.accum_steps
        assert abs(
            eff_batch_size - 256) < 10, f"Make sure the effective training batch size is around 256, current: {eff_batch_size}"
        LOGS.log.debug(f"Effective training batch size: {eff_batch_size}")
        self.pretrained_lm = args.train_args.pretrained_weights_path
        jieba.load_userdict(os.path.join(self.pretrained_lm, 'vocab.txt'))
        self.tokenizer = WoBertTokenizer.from_pretrained(self.pretrained_lm, do_lower_case=True)
        '''
        测试词汇级分词是否成功：
        text = '家里的大蒜居然是很好的护发素'
        cc = self.tokenizer.tokenize(text=text)
        cc1 = self.tokenizer.encode(text=text)
        cc2 = self.tokenizer.decode(token_ids=cc1)
        '''
        self.vocab = self.tokenizer.get_vocab()
        self.vocab_size = len(self.vocab)
        self.mask_id = self.vocab[self.tokenizer.mask_token]
        self.inv_vocab = {k: v for v, k in self.vocab.items()}
        self.read_label_names(args.data.DATASET, args.data.LABEL_NAME_FILE)
        self.num_class = len(self.label_name_dict)
        self.model = LOTClassModel.from_pretrained(self.pretrained_lm,
                                                   output_attentions=False,
                                                   output_hidden_states=False,
                                                   num_labels=self.num_class)
        self.read_data(args.data.DATASET, args.data.TRAIN_CORPUS, args.data.TEST_CORPUS, args.data.TRAIN_LABEL,
                       args.data.TEST_LABEL)
        self.with_test_label = True if args.data.TEST_LABEL is not None else False
        self.temp_dir = f'tmp_{self.dist_port}'
        self.mcp_loss = nn.CrossEntropyLoss()
        self.st_loss = nn.KLDivLoss(reduction='batchmean')
        self.update_interval = args.train_args.update_interval
        self.early_stop = args.train_args.early_stop

    # set up distributed training
    def set_up_dist(self, rank):
        dist.init_process_group(
            backend='nccl',
            init_method=f'tcp://localhost:{self.dist_port}',
            world_size=self.world_size,
            rank=rank
        )
        # create local model
        model = self.model.to(rank)
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        return model

    # get document truncation statistics with the defined max length
    def corpus_trunc_stats(self, docs):
        doc_len = []
        for doc in docs:
            input_ids = self.tokenizer.encode(doc, add_special_tokens=True)
            doc_len.append(len(input_ids))
        LOGS.log.debug(
            f"Document max length: {np.max(doc_len)}, avg length: {np.mean(doc_len)}, std length: {np.std(doc_len)}")
        trunc_frac = np.sum(np.array(doc_len) > self.max_len) / len(doc_len)
        LOGS.log.debug(f"Truncated fraction of all documents: {trunc_frac}")

    # convert a list of strings to token ids
    def encode(self, docs):
        encoded_dict = self.tokenizer.batch_encode_plus(docs, add_special_tokens=True,
                                                        max_length=self.max_len,
                                                        padding='max_length',
                                                        pad_to_max_length=True,
                                                        return_attention_mask=True,
                                                        truncation=True,
                                                        return_tensors='pt')
        input_ids = encoded_dict['input_ids']
        attention_masks = encoded_dict['attention_mask']
        return input_ids, attention_masks

    # convert list of token ids to list of strings
    def decode(self, ids):
        strings = self.tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return strings

    # convert dataset into tensors
    def create_dataset(self, dataset_dir, text_file, label_file, loader_name, find_label_name=False,
                       label_name_loader_name=None):
        loader_file = os.path.join(dataset_dir, loader_name)
        if os.path.exists(loader_file):
            LOGS.log.debug(f"Loading encoded texts from {loader_file}")
            data = torch.load(loader_file)
        else:
            LOGS.log.debug(f"Reading texts from {os.path.join(dataset_dir, text_file)}")
            corpus = open(os.path.join(dataset_dir, text_file), encoding="utf-8")
            docs = [doc.strip() for doc in corpus.readlines()]
            LOGS.log.debug(f"Converting texts into tensors.")
            chunk_size = ceil(len(docs) / self.num_cpus)
            chunks = [docs[x:x + chunk_size] for x in range(0, len(docs), chunk_size)]
            results = Parallel(n_jobs=self.num_cpus)(delayed(self.encode)(docs=chunk) for chunk in chunks)
            input_ids = torch.cat([result[0] for result in results])
            attention_masks = torch.cat([result[1] for result in results])
            LOGS.log.debug(f"Saving encoded texts into {loader_file}")
            if label_file is not None:
                LOGS.log.debug(f"Reading labels from {os.path.join(dataset_dir, label_file)}")
                truth = open(os.path.join(dataset_dir, label_file))
                labels = [int(label.strip()) for label in truth.readlines()]
                labels = torch.tensor(labels)
                data = {"input_ids": input_ids, "attention_masks": attention_masks, "labels": labels}
            else:
                data = {"input_ids": input_ids, "attention_masks": attention_masks}
            torch.save(data, loader_file)
        if find_label_name:
            loader_file = os.path.join(dataset_dir, label_name_loader_name)
            if os.path.exists(loader_file):
                LOGS.log.debug(f"Loading texts with label names from {loader_file}")
                label_name_data = torch.load(loader_file)
            else:
                LOGS.log.debug(f"Reading texts from {os.path.join(dataset_dir, text_file)}")
                corpus = open(os.path.join(dataset_dir, text_file), encoding="utf-8")
                docs = [doc.strip() for doc in corpus.readlines()]
                LOGS.log.debug("Locating label names in the corpus.")
                chunk_size = ceil(len(docs) / self.num_cpus)
                chunks = [docs[x:x + chunk_size] for x in range(0, len(docs), chunk_size)]
                results = Parallel(n_jobs=self.num_cpus)(
                    delayed(self.label_name_occurrence)(docs=chunk) for chunk in chunks)
                input_ids_with_label_name = torch.cat([result[0] for result in results])
                attention_masks_with_label_name = torch.cat([result[1] for result in results])
                label_name_idx = torch.cat([result[2] for result in results])
                assert len(input_ids_with_label_name) > 0, "No label names appear in corpus!"
                label_name_data = {"input_ids": input_ids_with_label_name,
                                   "attention_masks": attention_masks_with_label_name, "labels": label_name_idx}
                loader_file = os.path.join(dataset_dir, label_name_loader_name)
                LOGS.log.debug(f"Saving texts with label names into {loader_file}")
                torch.save(label_name_data, loader_file)
            return data, label_name_data
        else:
            return data

    # find label name indices and replace out-of-vocab label names with [MASK]
    def label_name_in_doc(self, doc):
        doc = self.tokenizer.tokenize(doc)
        label_idx = -1 * torch.ones(self.max_len, dtype=torch.long)
        new_doc = []
        wordpcs = []
        idx = 1  # index starts at 1 due to [CLS] token
        for i, wordpc in enumerate(doc):
            wordpcs.append(wordpc[2:] if wordpc.startswith("##") else wordpc)
            if idx >= self.max_len - 1:  # last index will be [SEP] token
                break
            if i == len(doc) - 1 or not doc[i + 1].startswith("##"):
                word = ''.join(wordpcs)
                if word in self.label2class:
                    label_idx[idx] = self.label2class[word]
                    # replace label names that are not in tokenizer's vocabulary with the [MASK] token
                    if word not in self.vocab:
                        wordpcs = [self.tokenizer.mask_token]
                new_word = ''.join(wordpcs)
                if new_word != self.tokenizer.unk_token:
                    idx += len(wordpcs)
                    new_doc.append(new_word)
                wordpcs = []
        if (label_idx >= 0).any():
            return ' '.join(new_doc), label_idx
        else:
            return None

    # find label name occurrences in the corpus
    def label_name_occurrence(self, docs):
        text_with_label = []
        label_name_idx = []
        for doc in docs:
            result = self.label_name_in_doc(doc)
            if result is not None:
                text_with_label.append(result[0])
                label_name_idx.append(result[1].unsqueeze(0))
        if len(text_with_label) > 0:
            encoded_dict = self.tokenizer.batch_encode_plus(text_with_label,
                                                            add_special_tokens=True,
                                                            max_length=self.max_len,
                                                            pad_to_max_length=True,
                                                            padding='max_length',
                                                            return_attention_mask=True,
                                                            truncation=True,
                                                            return_tensors='pt')
            input_ids_with_label_name = encoded_dict['input_ids']
            attention_masks_with_label_name = encoded_dict['attention_mask']
            label_name_idx = torch.cat(label_name_idx, dim=0)
        else:
            input_ids_with_label_name = torch.ones(0, self.max_len, dtype=torch.long)
            attention_masks_with_label_name = torch.ones(0, self.max_len, dtype=torch.long)
            label_name_idx = torch.ones(0, self.max_len, dtype=torch.long)
        return input_ids_with_label_name, attention_masks_with_label_name, label_name_idx

    # read text corpus and labels from files
    def read_data(self, dataset_dir, train_file, test_file, train_label_file, test_label_file):
        self.train_data, self.label_name_data = self.create_dataset(dataset_dir, train_file, train_label_file,
                                                                    "train.pt",
                                                                    find_label_name=True,
                                                                    label_name_loader_name="label_name_data.pt")
        if test_file is not None:
            self.test_data = self.create_dataset(dataset_dir, test_file, test_label_file, "test.pt")

    # read label names from file
    def read_label_names(self, dataset_dir, label_name_file):
        label_name_file = open(os.path.join(dataset_dir, label_name_file))
        label_names = label_name_file.readlines()
        self.label_name_dict = {i: [word.lower().strip() for word in category_words.strip().split()] for
                                i, category_words in
                                enumerate(label_names)}
        LOGS.log.debug(f"Label names used for each class are: {self.label_name_dict}")
        self.label2class = {}
        self.all_label_name_ids = [self.mask_id]
        self.all_label_names = [self.tokenizer.mask_token]
        for class_idx in self.label_name_dict:
            for word in self.label_name_dict[class_idx]:
                # assert word not in self.label2class, f"\"{word}\" used as the label name by multiple classes!"
                self.label2class[word] = class_idx
                if word in self.vocab:
                    self.all_label_name_ids.append(self.vocab[word])
                    self.all_label_names.append(word)

    # create dataset loader
    def make_dataloader(self, rank, data_dict, batch_size):
        if "labels" in data_dict:
            dataset = TensorDataset(data_dict["input_ids"], data_dict["attention_masks"], data_dict["labels"])
        else:
            dataset = TensorDataset(data_dict["input_ids"], data_dict["attention_masks"])
        sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=rank)
        dataset_loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, shuffle=False)
        return dataset_loader

    # filter out stop words and words in multiple categories
    def filter_keywords(self, category_vocab_size=256):
        all_words = defaultdict(list)
        sorted_dicts = {}
        for i, cat_dict in self.category_words_freq.items():
            sorted_dict = {k: v for k, v in
                           sorted(cat_dict.items(), key=lambda item: item[1], reverse=True)[:category_vocab_size]}
            sorted_dicts[i] = sorted_dict
            for word_id in sorted_dict:
                all_words[word_id].append(i)
        repeat_words = []
        for word_id in all_words:
            if len(all_words[word_id]) > 1:
                repeat_words.append(word_id)
        self.category_vocab = {}
        for i, sorted_dict in sorted_dicts.items():
            self.category_vocab[i] = np.array(list(sorted_dict.keys()))
        stopwords_vocab = load_stop_words(os.path.join(self.args.data.DATASET, self.args.data.stop_words))
        for i, word_list in self.category_vocab.items():
            delete_idx = []
            for j, word_id in enumerate(word_list):
                word = self.inv_vocab[word_id]
                if word in self.label_name_dict[i]:
                    continue
                if not word.isalpha() or len(word) == 1 or word in stopwords_vocab or word_id in repeat_words:
                    delete_idx.append(j)
            self.category_vocab[i] = np.delete(self.category_vocab[i], delete_idx)

    # construct category vocabulary (distributed function)
    def category_vocabulary_dist(self, rank, top_pred_num=50, loader_name="category_vocab.pt"):
        model = self.set_up_dist(rank)
        model.eval()
        label_name_dataset_loader = self.make_dataloader(rank, self.label_name_data, self.eval_batch_size)
        category_words_freq = {i: defaultdict(float) for i in range(self.num_class)}
        wrap_label_name_dataset_loader = tqdm(label_name_dataset_loader) if rank == 0 else label_name_dataset_loader
        try:
            for batch in wrap_label_name_dataset_loader:
                with torch.no_grad():
                    input_ids = batch[0].to(rank)
                    input_mask = batch[1].to(rank)
                    label_pos = batch[2].to(rank)
                    match_idx = label_pos >= 0
                    predictions = model(input_ids,
                                        pred_mode="mlm",
                                        token_type_ids=None,
                                        attention_mask=input_mask)
                    _, sorted_res = torch.topk(predictions[match_idx], top_pred_num, dim=-1)
                    label_idx = label_pos[match_idx]
                    for i, word_list in enumerate(sorted_res):
                        for j, word_id in enumerate(word_list):
                            category_words_freq[label_idx[i].item()][word_id.item()] += 1
            save_file = os.path.join(self.temp_dir, f"{rank}_" + loader_name)
            torch.save(category_words_freq, save_file)
        except RuntimeError as err:
            self.cuda_mem_error(err, "eval", rank)

    # construct category vocabulary
    def category_vocabulary(self, top_pred_num=50, category_vocab_size=100, loader_name="category_vocab.pt"):
        loader_file = os.path.join(self.dataset_dir, loader_name)
        if os.path.exists(loader_file):
            LOGS.log.debug(f"Loading category vocabulary from {loader_file}")
            if loader_name[-3:] == '.pt':
                self.category_vocab = torch.load(loader_file)
            else:
                self.category_vocab = {}
                with open(loader_file, mode='r', encoding='utf-8') as wf:
                    for i, line in enumerate(wf.readlines()):
                        words = line.strip().split(' ')
                        token_words = [self.vocab[w] for w in words if w in self.vocab]
                        self.category_vocab[i] = np.array(token_words)
        else:
            LOGS.log.debug("Contructing category vocabulary.")
            if not os.path.exists(self.temp_dir):
                os.makedirs(self.temp_dir)
            mp.spawn(self.category_vocabulary_dist, nprocs=self.world_size, args=(top_pred_num, loader_name))
            gather_res = []
            for f in os.listdir(self.temp_dir):
                if f[-3:] == '.pt':
                    gather_res.append(torch.load(os.path.join(self.temp_dir, f)))
            assert len(gather_res) == self.world_size, "Number of saved files not equal to number of processes!"
            self.category_words_freq = {i: defaultdict(float) for i in range(self.num_class)}
            for i in range(self.num_class):
                for category_words_freq in gather_res:
                    for word_id, freq in category_words_freq[i].items():
                        self.category_words_freq[i][word_id] += freq
            self.filter_keywords(category_vocab_size)
            torch.save(self.category_vocab, loader_file)
            with open(loader_file.replace('.pt', '.txt'), mode='w', encoding='utf-8') as wf:
                for i, wk in self.category_vocab.items():
                    wk = wk.tolist()
                    wk = [str(self.inv_vocab[w]) for w in wk]
                    wl = ' '.join(wk)
                    wf.write(wl + '\n')

            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        for i, category_vocab in self.category_vocab.items():
            LOGS.log.debug(f"Class {self.label_name_dict[i]} category vocabulary: {[self.inv_vocab[w] for w in category_vocab]}\n")

    # prepare self supervision for masked category prediction (distributed function)
    def prepare_mcp_dist(self, rank, top_pred_num=50, match_threshold=20, loader_name="mcp_train.pt"):
        model = self.set_up_dist(rank)
        model.eval()
        train_dataset_loader = self.make_dataloader(rank, self.train_data, self.eval_batch_size)
        all_input_ids = []
        all_mask_label = []
        all_input_mask = []
        category_doc_num = defaultdict(int)
        wrap_train_dataset_loader = tqdm(train_dataset_loader) if rank == 0 else train_dataset_loader
        try:
            for batch in wrap_train_dataset_loader:
                with torch.no_grad():
                    input_ids = batch[0].to(rank)
                    input_mask = batch[1].to(rank)
                    predictions = model(input_ids,
                                        pred_mode="mlm",
                                        token_type_ids=None,
                                        attention_mask=input_mask)
                    _, sorted_res = torch.topk(predictions, top_pred_num, dim=-1)
                    for i, category_vocab in self.category_vocab.items():
                        match_idx = torch.zeros_like(sorted_res).bool()
                        for word_id in category_vocab:
                            match_idx = (sorted_res == word_id) | match_idx
                        match_count = torch.sum(match_idx.int(), dim=-1)
                        valid_idx = (match_count > match_threshold) & (input_mask > 0)
                        valid_doc = torch.sum(valid_idx, dim=-1) > 0
                        if valid_doc.any():
                            mask_label = -1 * torch.ones_like(input_ids)
                            mask_label[valid_idx] = i
                            all_input_ids.append(input_ids[valid_doc].cpu())
                            all_mask_label.append(mask_label[valid_doc].cpu())
                            all_input_mask.append(input_mask[valid_doc].cpu())
                            category_doc_num[i] += valid_doc.int().sum().item()
            all_input_ids = torch.cat(all_input_ids, dim=0)
            all_mask_label = torch.cat(all_mask_label, dim=0)
            all_input_mask = torch.cat(all_input_mask, dim=0)
            save_dict = {
                "all_input_ids": all_input_ids,
                "all_mask_label": all_mask_label,
                "all_input_mask": all_input_mask,
                "category_doc_num": category_doc_num,
            }
            if len(all_input_ids) == 0:
                raise ValueError('len(all_input_ids) == 0')
            save_file = os.path.join(self.temp_dir, f"{rank}_" + loader_name)
            torch.save(save_dict, save_file)
        except RuntimeError as err:
            self.cuda_mem_error(err, "eval", rank)

    # prepare self supervision for masked category prediction
    def prepare_mcp(self, top_pred_num=50, match_threshold=20, loader_name="mcp_train.pt"):
        loader_file = os.path.join(self.dataset_dir, loader_name)
        if os.path.exists(loader_file):
            LOGS.log.debug(f"Loading masked category prediction data from {loader_file}")
            self.mcp_data = torch.load(loader_file)
        else:
            loader_file = os.path.join(self.dataset_dir, loader_name)
            LOGS.log.debug("Preparing self supervision for masked category prediction.")
            if not os.path.exists(self.temp_dir):
                os.makedirs(self.temp_dir)
            mp.spawn(self.prepare_mcp_dist, nprocs=self.world_size, args=(top_pred_num, match_threshold, loader_name))
            gather_res = []
            for f in os.listdir(self.temp_dir):
                if f[-3:] == '.pt':
                    gather_res.append(torch.load(os.path.join(self.temp_dir, f)))
            assert len(gather_res) == self.world_size, "Number of saved files not equal to number of processes!"
            all_input_ids = torch.cat([res["all_input_ids"] for res in gather_res], dim=0)
            all_mask_label = torch.cat([res["all_mask_label"] for res in gather_res], dim=0)
            all_input_mask = torch.cat([res["all_input_mask"] for res in gather_res], dim=0)
            category_doc_num = {i: 0 for i in range(self.num_class)}
            for i in category_doc_num:
                for res in gather_res:
                    if i in res["category_doc_num"]:
                        category_doc_num[i] += res["category_doc_num"][i]
            LOGS.log.debug(
                f"Number of documents with category indicative terms found for each category is: {category_doc_num}")
            self.mcp_data = {"input_ids": all_input_ids, "attention_masks": all_input_mask, "labels": all_mask_label}
            torch.save(self.mcp_data, loader_file)
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
            for i in category_doc_num:
                assert category_doc_num[
                           i] > 10, f"Too few ({category_doc_num[i]}) documents with category indicative terms found for category {i}; " \
                                    "try to add more unlabeled documents to the training corpus (recommend) or reduce `--match_threshold` (not recommend)"
        LOGS.log.debug(f"There are totally {len(self.mcp_data['input_ids'])} documents with category indicative terms.")

    # masked category prediction (distributed function)
    def mcp_dist(self, rank, epochs=5, loader_name="mcp_model.pt"):
        model = self.set_up_dist(rank)
        mcp_dataset_loader = self.make_dataloader(rank, self.mcp_data, self.train_batch_size)
        total_steps = len(mcp_dataset_loader) * epochs / self.accum_steps
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * total_steps,
                                                    num_training_steps=total_steps)
        try:
            for i in range(epochs):
                model.train()
                total_train_loss = 0
                if rank == 0:
                    LOGS.log.debug(f"Epoch {i + 1}:")
                wrap_mcp_dataset_loader = tqdm(mcp_dataset_loader) if rank == 0 else mcp_dataset_loader
                model.zero_grad()
                for j, batch in enumerate(wrap_mcp_dataset_loader):
                    input_ids = batch[0].to(rank)
                    input_mask = batch[1].to(rank)
                    labels = batch[2].to(rank)
                    mask_pos = labels >= 0
                    labels = labels[mask_pos]
                    # mask out category indicative words
                    input_ids[mask_pos] = self.mask_id
                    logits = model(input_ids,
                                   pred_mode="classification",
                                   token_type_ids=None,
                                   attention_mask=input_mask)
                    logits = logits[mask_pos]
                    loss = self.mcp_loss(logits.view(-1, self.num_class), labels.view(-1)) / self.accum_steps
                    total_train_loss += loss.item()
                    loss.backward()
                    if (j + 1) % self.accum_steps == 0:
                        # Clip the norm of the gradients to 1.0.
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                        model.zero_grad()
                avg_train_loss = torch.tensor([total_train_loss / len(mcp_dataset_loader) * self.accum_steps]).to(rank)
                gather_list = [torch.ones_like(avg_train_loss) for _ in range(self.world_size)]
                dist.all_gather(gather_list, avg_train_loss)
                avg_train_loss = torch.tensor(gather_list)
                if rank == 0:
                    LOGS.log.debug(f"Average training loss: {avg_train_loss.mean().item()}")
            if rank == 0:
                loader_file = os.path.join(self.dataset_dir, loader_name)
                torch.save(model.module.state_dict(), loader_file)
        except RuntimeError as err:
            self.cuda_mem_error(err, "train", rank)

    # masked category prediction
    def mcp(self, top_pred_num=50, match_threshold=20, epochs=5, loader_name="mcp_model.pt"):
        loader_file = os.path.join(self.dataset_dir, loader_name)
        if os.path.exists(loader_file):
            LOGS.log.debug(f"\nLoading model trained via masked category prediction from {loader_file}")
        else:
            self.prepare_mcp(top_pred_num, match_threshold)
            LOGS.log.debug(f"\nTraining model via masked category prediction.")
            mp.spawn(self.mcp_dist, nprocs=self.world_size, args=(epochs, loader_name))
        self.model.load_state_dict(torch.load(loader_file))

    # prepare self training data and target distribution
    def prepare_self_train_data(self, rank, model, idx):
        target_num = min(self.world_size * self.train_batch_size * self.update_interval * self.accum_steps,
                         len(self.train_data["input_ids"]))
        if idx + target_num >= len(self.train_data["input_ids"]):
            select_idx = torch.cat((torch.arange(idx, len(self.train_data["input_ids"])),
                                    torch.arange(idx + target_num - len(self.train_data["input_ids"]))))
        else:
            select_idx = torch.arange(idx, idx + target_num)
        assert len(select_idx) == target_num
        idx = (idx + len(select_idx)) % len(self.train_data["input_ids"])
        select_dataset = {"input_ids": self.train_data["input_ids"][select_idx],
                          "attention_masks": self.train_data["attention_masks"][select_idx]}
        dataset_loader = self.make_dataloader(rank, select_dataset, self.eval_batch_size)
        input_ids, input_mask, preds = self.inference(model, dataset_loader, rank, return_type="data")
        gather_input_ids = [torch.ones_like(input_ids) for _ in range(self.world_size)]
        gather_input_mask = [torch.ones_like(input_mask) for _ in range(self.world_size)]
        gather_preds = [torch.ones_like(preds) for _ in range(self.world_size)]
        dist.all_gather(gather_input_ids, input_ids)
        dist.all_gather(gather_input_mask, input_mask)
        dist.all_gather(gather_preds, preds)
        input_ids = torch.cat(gather_input_ids, dim=0).cpu()
        input_mask = torch.cat(gather_input_mask, dim=0).cpu()
        all_preds = torch.cat(gather_preds, dim=0).cpu()
        weight = all_preds ** 2 / torch.sum(all_preds, dim=0)
        target_dist = (weight.t() / torch.sum(weight, dim=1)).t()
        all_target_pred = target_dist.argmax(dim=-1)
        agree = (all_preds.argmax(dim=-1) == all_target_pred).int().sum().item() / len(all_target_pred)
        self_train_dict = {"input_ids": input_ids, "attention_masks": input_mask, "labels": target_dist}
        return self_train_dict, idx, agree

    # train a model on batches of data with target labels
    def self_train_batches(self, rank, model, self_train_loader, optimizer, scheduler, test_dataset_loader):
        model.train()
        total_train_loss = 0
        wrap_train_dataset_loader = tqdm(self_train_loader) if rank == 0 else self_train_loader
        model.zero_grad()
        try:
            for j, batch in enumerate(wrap_train_dataset_loader):
                input_ids = batch[0].to(rank)
                input_mask = batch[1].to(rank)
                target_dist = batch[2].to(rank)
                logits = model(input_ids,
                               pred_mode="classification",
                               token_type_ids=None,
                               attention_mask=input_mask)
                logits = logits[:, 0, :]
                preds = nn.LogSoftmax(dim=-1)(logits)
                loss = self.st_loss(preds.view(-1, self.num_class),
                                    target_dist.view(-1, self.num_class)) / self.accum_steps
                total_train_loss += loss.item()
                loss.backward()
                if (j + 1) % self.accum_steps == 0:
                    # Clip the norm of the gradients to 1.0.
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
            if self.with_test_label:
                acc = self.inference(model, test_dataset_loader, rank, return_type="acc")
                gather_acc = [torch.ones_like(acc) for _ in range(self.world_size)]
                dist.all_gather(gather_acc, acc)
                acc = torch.tensor(gather_acc).mean().item()
            avg_train_loss = torch.tensor([total_train_loss / len(wrap_train_dataset_loader) * self.accum_steps]).to(
                rank)
            gather_list = [torch.ones_like(avg_train_loss) for _ in range(self.world_size)]
            dist.all_gather(gather_list, avg_train_loss)
            avg_train_loss = torch.tensor(gather_list)
            if rank == 0:
                LOGS.log.debug(f"lr: {optimizer.param_groups[0]['lr']:.4g}")
                LOGS.log.debug(f"Average training loss: {avg_train_loss.mean().item()}")
                if self.with_test_label:
                    LOGS.log.debug(f"Test acc: {acc}")
        except RuntimeError as err:
            self.cuda_mem_error(err, "train", rank)

    # self training (distributed function)
    def self_train_dist(self, rank, epochs, loader_name="final_model.pt"):
        model = self.set_up_dist(rank)
        test_dataset_loader = self.make_dataloader(rank, self.test_data,
                                                   self.eval_batch_size) if self.with_test_label else None
        total_steps = int(
            len(self.train_data["input_ids"]) * epochs / (self.world_size * self.train_batch_size * self.accum_steps))
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-6, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * total_steps,
                                                    num_training_steps=total_steps)
        idx = 0
        if self.early_stop:
            agree_count = 0
        for i in range(int(total_steps / self.update_interval)):
            self_train_dict, idx, agree = self.prepare_self_train_data(rank, model, idx)
            # early stop if current prediction agrees with target distribution for 3 consecutive updates
            if self.early_stop:
                if 1 - agree < 1e-3:
                    agree_count += 1
                else:
                    agree_count = 0
                if agree_count >= 3:
                    break
            self_train_dataset_loader = self.make_dataloader(rank, self_train_dict, self.train_batch_size)
            self.self_train_batches(rank, model, self_train_dataset_loader, optimizer, scheduler, test_dataset_loader)
        if rank == 0:
            loader_file = os.path.join(self.dataset_dir, loader_name)
            LOGS.log.debug(f"Saving final model to {loader_file}")
            torch.save(model.module.state_dict(), loader_file)

    # self training
    def self_train(self, epochs, loader_name="final_model.pt"):
        loader_file = os.path.join(self.dataset_dir, loader_name)
        if os.path.exists(loader_file):
            LOGS.log.debug(f"\nFinal model {loader_file} found, skip self-training")
        else:
            rand_idx = torch.randperm(len(self.train_data["input_ids"]))
            self.train_data = {"input_ids": self.train_data["input_ids"][rand_idx],
                               "attention_masks": self.train_data["attention_masks"][rand_idx]}
            LOGS.log.debug(f"\nStart self-training.")
            mp.spawn(self.self_train_dist, nprocs=self.world_size, args=(epochs, loader_name))

    # use a model to do inference on a dataloader
    def inference(self, model, dataset_loader, rank, return_type):
        if return_type == "data":
            all_input_ids = []
            all_input_mask = []
            all_preds = []
        elif return_type == "acc":
            pred_labels = []
            truth_labels = []
        elif return_type == "pred":
            pred_labels = []
        model.eval()
        try:
            for batch in dataset_loader:
                with torch.no_grad():
                    input_ids = batch[0].to(rank)
                    input_mask = batch[1].to(rank)
                    logits = model(input_ids,
                                   pred_mode="classification",
                                   token_type_ids=None,
                                   attention_mask=input_mask)
                    logits = logits[:, 0, :]
                    if return_type == "data":
                        all_input_ids.append(input_ids)
                        all_input_mask.append(input_mask)
                        all_preds.append(nn.Softmax(dim=-1)(logits))
                    elif return_type == "acc":
                        labels = batch[2]
                        pred_labels.append(torch.argmax(logits, dim=-1).cpu())
                        truth_labels.append(labels)
                    elif return_type == "pred":
                        pred_labels.append(torch.argmax(logits, dim=-1).cpu())
            if return_type == "data":
                all_input_ids = torch.cat(all_input_ids, dim=0)
                all_input_mask = torch.cat(all_input_mask, dim=0)
                all_preds = torch.cat(all_preds, dim=0)
                return all_input_ids, all_input_mask, all_preds
            elif return_type == "acc":
                pred_labels = torch.cat(pred_labels, dim=0)
                truth_labels = torch.cat(truth_labels, dim=0)
                samples = len(truth_labels)
                acc = (pred_labels == truth_labels).float().sum() / samples
                return acc.to(rank)
            elif return_type == "pred":
                pred_labels = torch.cat(pred_labels, dim=0)
                return pred_labels
        except RuntimeError as err:
            self.cuda_mem_error(err, "eval", rank)

    # use trained model to make predictions on the test set
    def write_results(self, loader_name="final_model.pt", out_file="out.txt"):
        loader_file = os.path.join(self.dataset_dir, loader_name)
        assert os.path.exists(loader_file)
        LOGS.log.debug(f"\nLoading final model from {loader_file}")
        self.model.load_state_dict(torch.load(loader_file))
        self.model.to(0)
        test_set = TensorDataset(self.test_data["input_ids"], self.test_data["attention_masks"])
        test_dataset_loader = DataLoader(test_set, sampler=SequentialSampler(test_set), batch_size=self.eval_batch_size)
        pred_labels = self.inference(self.model, test_dataset_loader, 0, return_type="pred")
        out_file = os.path.join(self.dataset_dir, out_file)
        LOGS.log.debug(f"Writing prediction results to {out_file}")
        f_out = open(out_file, 'w')
        for label in pred_labels:
            f_out.write(str(label.item()) + '\n')

    # print error message based on CUDA memory error
    def cuda_mem_error(self, err, mode, rank):
        if rank == 0:
            LOGS.log.debug(err)
            if "CUDA out of memory" in str(err):
                if mode == "eval":
                    LOGS.log.debug(
                        f"Your GPUs can't hold the current batch size for evaluation, try to reduce `--eval_batch_size`, current: {self.eval_batch_size}")
                else:
                    LOGS.log.debug(
                        f"Your GPUs can't hold the current batch size for training, try to reduce `--train_batch_size`, current: {self.train_batch_size}")
        sys.exit(1)
