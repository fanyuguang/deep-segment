#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
from hdfs3 import HDFileSystem


class HdfsUtils(object):

    def __init__(self, host, port, user):
        self.hdfs = HDFileSystem(host=host, port=port, user=user)


    def hdfs_download(self, hdfs_path, local_path):
        '''
        Download file or dir from hdfs
        :param hdfs_path:
        :param local_path:
        :return:
        '''
        hdfs_path = os.path.normpath(hdfs_path)
        local_path = os.path.normpath(local_path)
        local_parent_path = os.path.dirname(local_path)
        if not self.hdfs.exists(hdfs_path):
            raise Exception('hdfs file not exists: ' + hdfs_path)
        if not os.path.exists(local_parent_path):
            raise Exception('local parent folder not exists: ' + local_parent_path)
        if os.path.exists(local_path):
            raise Exception('local file exists: ' + local_path)

        if self.hdfs.isfile(hdfs_path):
            print('is file')
            self.hdfs.get(hdfs_path, local_path)
        elif self.hdfs.isdir(hdfs_path):
            print('is dir')
            os.mkdir(local_path)
            for (root, dirnames, filenames) in self.hdfs.walk(hdfs_path):
                relative_path = os.path.relpath(root, hdfs_path)
                for dirname in dirnames:
                    current_local_dir_path = os.path.join(local_path, relative_path, dirname)
                    os.makedirs(current_local_dir_path)
                for filename in filenames:
                    current_hdfs_file_path = os.path.join(root, filename)
                    current_local_file_path = os.path.join(local_path, relative_path, filename)
                    self.hdfs.get(current_hdfs_file_path, current_local_file_path)
        else:
            raise Exception('parameters invalid')
        print('Done.')


    def hdfs_upload(self, local_path, hdfs_path):
        '''
        Upload file or dir to hdfs
        :param local_path:
        :param hdfs_path:
        :return:
        '''
        local_path = os.path.normpath(local_path)
        hdfs_path = os.path.normpath(hdfs_path)
        hdfs_parent_path = os.path.dirname(hdfs_path)
        if not os.path.exists(local_path):
            raise Exception('local file not exists: ' + local_path)
        if not self.hdfs.exists(hdfs_parent_path):
            raise Exception('hdfs parent folder not exists: ' + hdfs_parent_path)
        if self.hdfs.exists(hdfs_path):
            raise Exception('hdfs file exists: ' + hdfs_path)

        if os.path.isfile(local_path):
            print('is file')
            self.hdfs.put(local_path, hdfs_path)
        elif os.path.isdir(local_path):
            print('is dir')
            self.hdfs.mkdir(hdfs_path)
            for (root, dirnames, filenames) in os.walk(local_path):
                relative_path = os.path.relpath(root, local_path)
                for dirname in dirnames:
                    current_hdfs_dir_path = os.path.join(hdfs_path, relative_path, dirname)
                    self.hdfs.mkdir(current_hdfs_dir_path)
                for filename in filenames:
                    current_local_file_path = os.path.join(root, filename)
                    current_hdfs_file_path = os.path.join(hdfs_path, relative_path, filename)
                    self.hdfs.put(current_local_file_path, current_hdfs_file_path)
        else:
            raise Exception('parameters invalid')
        print('Done.')


    def hdfs_delete(self, hdfs_path):
        '''
        Delete file or dir at hdfs
        :param hdfs_path:
        :param local_path:
        :return:
        '''
        hdfs_path = os.path.normpath(hdfs_path)
        if self.hdfs.exists(hdfs_path):
            self.hdfs.rm(hdfs_path)
        print('Done.')


    def hdfs_mv(self, source_hdfs_path, target_hdfs_path):
        self.hdfs(source_hdfs_path, target_hdfs_path)