#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 03/05/2018 2:56 PM
import logging
import numpy as np
import os
import random
import tensorflow as tf
from neuralseg.config import parse_args
from neuralseg.api import prepare, train, evaluate, segment


if __name__ == '__main__':
    args = parse_args()

    logger = logging.getLogger("SegEDU")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        ' %(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    if args.log_path:
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info('Running with args : {}'.format(args))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    if args.prepare:
        prepare(args)
    if args.train:
        train(args)
    if args.evaluate:
        evaluate(args)
    if args.segment:
        segment(args)
