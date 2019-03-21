#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import shutil

from config import FLAGS


def delete_file(filename):
    """
    Delete file or dir
    :param filename:
    :return:
    """
    if os.path.isfile(filename):
        os.remove(filename)
    elif os.path.isdir(filename):
        shutil.rmtree(filename)


def clean_path(path):
    """
    Delete all sub files under path
    :param path:
    :return:
    """
    filenames = os.listdir(path)
    for filename in filenames:
        if filename != '.gitignore':
            delete_file(os.path.join(path, filename))


def main():
    # clean generated files
    delete_file(os.path.join('../', FLAGS.raw_data_path, 'format_data.txt'))
    delete_file(os.path.join('../', FLAGS.raw_data_path, 'split_data.txt'))

    paths = [FLAGS.datasets_path, FLAGS.vocab_path, FLAGS.tfrecords_path, FLAGS.checkpoint_path,
             FLAGS.tensorboard_path, FLAGS.freeze_graph_path, FLAGS.saved_model_path]
    for path in paths:
        clean_path(os.path.join('../', path))


if __name__ == '__main__':
    main()
