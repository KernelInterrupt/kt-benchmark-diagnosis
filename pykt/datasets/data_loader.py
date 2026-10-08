#!/usr/bin/env python
# coding=utf-8

import os, sys
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from .cache_utils import (
    atomic_pickle_dump,
    cache_lock,
    try_gpu_qtest_prebuild,
    try_gpu_sequence_prebuild,
    _force_gpu_prebuild,
    _inmemory_gpu_prebuild,
    _effective_prebuild_device,
)
from .batch_fetch_utils import batch_select, expand_empty_batch, shifted_masked_batch

if torch.cuda.is_available():
    from torch.cuda import FloatTensor, LongTensor
else:
    from torch import FloatTensor, LongTensor


CTW_FLOAT_SEQ_COLS = {
    "ctw_pseqs",
    "ctw_logitseqs",
    "ctw_depthseqs",
    "ctw_totalseqs",
    "ctw_posseqs",
    "ctw_negseqs",
}

SCALAR_LONG_COLS = {
    "uid",
}

class KTDataset(Dataset):
    """Dataset for KT
        can use to init dataset for: (for models except dkt_forget)
            train data, valid data
            common test data(concept level evaluation), real educational scenario test data(question level evaluation).
    Args:
        file_path (str): train_valid/test file path
        input_type (list[str]): the input type of the dataset, values are in ["questions", "concepts"]
        folds (set(int)): the folds used to generate dataset, -1 for test data
        qtest (bool, optional): is question evaluation or not. Defaults to False.
    """
    def __init__(self, file_path, input_type, folds, qtest=False):
        super(KTDataset, self).__init__()
        sequence_path = file_path
        self.input_type = input_type
        self.qtest = qtest
        folds = sorted(list(folds))
        folds_str = "_" + "_".join([str(_) for _ in folds])
        if self.qtest:
            processed_data = file_path + folds_str + "_qtest.pkl"
        else:
            processed_data = file_path + folds_str + ".pkl"

        headers = set(pd.read_csv(sequence_path, nrows=0).columns)
        needs_uid = "uid" in headers

        force_prebuild = _force_gpu_prebuild()
        built_inmem = self._try_inmemory_cudf_build(sequence_path, folds)
        if not built_inmem:
            if self.qtest:
                if force_prebuild or not os.path.exists(processed_data):
                    try_gpu_qtest_prebuild(
                        csv_path=sequence_path,
                        folds=folds,
                        input_type=self.input_type,
                        processed_data=processed_data,
                        variant="base",
                    )
            else:
                if force_prebuild or not os.path.exists(processed_data):
                    try_gpu_sequence_prebuild(
                        csv_path=sequence_path,
                        folds=folds,
                        input_type=self.input_type,
                        processed_data=processed_data,
                    )

            with cache_lock(processed_data):
                if not os.path.exists(processed_data):
                    print(f"Start preprocessing {file_path} fold: {folds_str}...")
                    if self.qtest:
                        self.dori, self.dqtest = self.__load_data__(sequence_path, folds)
                        atomic_pickle_dump([self.dori, self.dqtest], processed_data)
                    else:
                        self.dori = self.__load_data__(sequence_path, folds)
                        atomic_pickle_dump(self.dori, processed_data)
                else:
                    print(f"Read data from processed file: {processed_data}")
                    if self.qtest:
                        self.dori, self.dqtest = pd.read_pickle(processed_data)
                    else:
                        self.dori = pd.read_pickle(processed_data)
                        for key in self.dori:
                            self.dori[key] = self.dori[key]#[:100]
                    if needs_uid and "uid" not in self.dori:
                        print(f"Reprocessing {file_path} because cached file is missing uid.")
                        if self.qtest:
                            self.dori, self.dqtest = self.__load_data__(sequence_path, folds)
                            atomic_pickle_dump([self.dori, self.dqtest], processed_data)
                        else:
                            self.dori = self.__load_data__(sequence_path, folds)
                            atomic_pickle_dump(self.dori, processed_data)
        else:
            print(f"Built in-memory cuDF cache for: {file_path}, qtest={self.qtest}")
        print(f"file path: {file_path}, qlen: {len(self.dori['qseqs'])}, clen: {len(self.dori['cseqs'])}, rlen: {len(self.dori['rseqs'])}")
        self._scalar_keys = [key for key in self.dori if key in SCALAR_LONG_COLS]
        self._seq_keys = [key for key in self.dori if key not in SCALAR_LONG_COLS and key not in {"masks", "smasks"}]
        self._empty_seq_keys = [key for key in self._seq_keys if len(self.dori[key]) == 0]
        self._nonempty_seq_keys = [key for key in self._seq_keys if len(self.dori[key]) != 0]

    def _try_inmemory_cudf_build(self, sequence_path, folds):
        if not _inmemory_gpu_prebuild():
            return False
        try:
            from .cudf_inmem_build import build_qtest_cache_inmem, build_sequence_cache_inmem
        except Exception as exc:
            print(f"In-memory cuDF builder unavailable for {sequence_path}: {exc}")
            return False

        try:
            device = int(_effective_prebuild_device())
        except ValueError:
            device = 0

        try:
            if self.qtest:
                self.dori, self.dqtest, _ = build_qtest_cache_inmem(sequence_path, self.input_type, folds, device=device)
            else:
                self.dori, _ = build_sequence_cache_inmem(sequence_path, self.input_type, folds, device=device)
            return True
        except Exception as exc:
            print(f"In-memory cuDF build failed for {sequence_path}: {exc}")
            return False

    def __len__(self):
        """return the dataset length
        Returns:
            int: the length of the dataset
        """
        return len(self.dori["rseqs"])

    def __getitem__(self, index):
        """
        Args:
            index (int): the index of the data want to get
        Returns:
            (tuple): tuple containing:
            
            - **q_seqs (torch.tensor)**: question id sequence of the 0~seqlen-2 interactions
            - **c_seqs (torch.tensor)**: knowledge concept id sequence of the 0~seqlen-2 interactions
            - **r_seqs (torch.tensor)**: response id sequence of the 0~seqlen-2 interactions
            - **qshft_seqs (torch.tensor)**: question id sequence of the 1~seqlen-1 interactions
            - **cshft_seqs (torch.tensor)**: knowledge concept id sequence of the 1~seqlen-1 interactions
            - **rshft_seqs (torch.tensor)**: response id sequence of the 1~seqlen-1 interactions
            - **mask_seqs (torch.tensor)**: masked value sequence, shape is seqlen-1
            - **select_masks (torch.tensor)**: is select to calculate the performance or not, 0 is not selected, 1 is selected, only available for 1~seqlen-1, shape is seqlen-1
            - **dcur (dict)**: used only self.qtest is True, for question level evaluation
        """
        dcur = dict()
        mseqs = self.dori["masks"][index]
        for key in self._scalar_keys:
            dcur[key] = self.dori[key][index]
        for key in self._empty_seq_keys:
            dcur[key] = self.dori[key]
            dcur["shft_"+key] = self.dori[key]
        for key in self._nonempty_seq_keys:
            seqs = self.dori[key][index][:-1] * mseqs
            shft_seqs = self.dori[key][index][1:] * mseqs
            dcur[key] = seqs
            dcur["shft_"+key] = shft_seqs
        dcur["masks"] = mseqs
        dcur["smasks"] = self.dori["smasks"][index]
        # print("tseqs", dcur["tseqs"])
        if not self.qtest:
            return dcur
        else:
            dqtest = dict()
            for key in self.dqtest:
                dqtest[key] = self.dqtest[key][index]
            return dcur, dqtest

    def fetch_batch(self, batch_indices):
        dcur = dict()
        mseqs = batch_select(self.dori["masks"], batch_indices)
        for key in self._scalar_keys:
            dcur[key] = batch_select(self.dori[key], batch_indices)
        for key in self._empty_seq_keys:
            empty = expand_empty_batch(self.dori[key], mseqs.shape[0])
            dcur[key] = empty
            dcur["shft_" + key] = empty
        for key in self._nonempty_seq_keys:
            batch_value = batch_select(self.dori[key], batch_indices)
            seqs, shft_seqs = shifted_masked_batch(batch_value, mseqs, apply_mask=True)
            dcur[key] = seqs
            dcur["shft_" + key] = shft_seqs
        dcur["masks"] = mseqs
        dcur["smasks"] = batch_select(self.dori["smasks"], batch_indices)
        if not self.qtest:
            return dcur
        dqtest = {key: batch_select(self.dqtest[key], batch_indices) for key in self.dqtest}
        return dcur, dqtest

    def __load_data__(self, sequence_path, folds, pad_val=-1):
        """
        Args:
            sequence_path (str): file path of the sequences
            folds (list[int]): 
            pad_val (int, optional): pad value. Defaults to -1.
        Returns: 
            (tuple): tuple containing
            - **q_seqs (torch.tensor)**: question id sequence of the 0~seqlen-1 interactions
            - **c_seqs (torch.tensor)**: knowledge concept id sequence of the 0~seqlen-1 interactions
            - **r_seqs (torch.tensor)**: response id sequence of the 0~seqlen-1 interactions
            - **mask_seqs (torch.tensor)**: masked value sequence, shape is seqlen-1
            - **select_masks (torch.tensor)**: is select to calculate the performance or not, 0 is not selected, 1 is selected, only available for 1~seqlen-1, shape is seqlen-1
            - **dqtest (dict)**: not null only self.qtest is True, for question level evaluation
        """
        dori = {"qseqs": [], "cseqs": [], "rseqs": [], "tseqs": [], "utseqs": [], "smasks": []}

        # seq_qids, seq_cids, seq_rights, seq_mask = [], [], [], []
        df = pd.read_csv(sequence_path)#[0:1000]
        df = df[df["fold"].isin(folds)]
        optional_seq_cols = [col for col in df.columns if col in CTW_FLOAT_SEQ_COLS]
        for col in optional_seq_cols:
            dori[col] = []
        interaction_num = 0
        # seq_qidxs, seq_rests = [], []
        dqtest = {"qidxs": [], "rests":[], "orirow":[]}
        for i, row in df.iterrows():
            #use kc_id or question_id as input
            if "concepts" in self.input_type:
                dori["cseqs"].append([int(_) for _ in row["concepts"].split(",")])
            # For question-level evaluation on concept-only datasets (e.g. pseudo question=concept),
            # still load the serialized questions column if present in the qtest file.
            if "questions" in self.input_type or (self.qtest and "questions" in row.index):
                dori["qseqs"].append([int(_) for _ in row["questions"].split(",")])
            if "timestamps" in row:
                dori["tseqs"].append([int(_) for _ in row["timestamps"].split(",")])
            if "usetimes" in row:
                dori["utseqs"].append([int(_) for _ in row["usetimes"].split(",")])
                
            dori["rseqs"].append([int(_) for _ in row["responses"].split(",")])
            dori["smasks"].append([int(_) for _ in row["selectmasks"].split(",")])
            if "uid" in row:
                dori.setdefault("uid", []).append(int(row["uid"]))
            for col in optional_seq_cols:
                dori[col].append([float(_) for _ in row[col].split(",")])

            interaction_num += dori["smasks"][-1].count(1)

            if self.qtest:
                dqtest["qidxs"].append([int(_) for _ in row["qidxs"].split(",")])
                dqtest["rests"].append([int(_) for _ in row["rest"].split(",")])
                dqtest["orirow"].append([int(_) for _ in row["orirow"].split(",")])
        for key in dori:
            if key in SCALAR_LONG_COLS:
                dori[key] = LongTensor(dori[key])
                continue
            if key in ["rseqs"] or key in CTW_FLOAT_SEQ_COLS:
                dori[key] = FloatTensor(dori[key])
            else:#in ["smasks", "tseqs"]:
                dori[key] = LongTensor(dori[key])

        mask_seqs = (dori["cseqs"][:,:-1] != pad_val) * (dori["cseqs"][:,1:] != pad_val)
        dori["masks"] = mask_seqs

        dori["smasks"] = (dori["smasks"][:, 1:] != pad_val)
        print(f"interaction_num: {interaction_num}")
        # print("load data tseqs: ", dori["tseqs"])

        if self.qtest:
            for key in dqtest:
                dqtest[key] = LongTensor(dqtest[key])[:, 1:]
            
            return dori, dqtest
        return dori
