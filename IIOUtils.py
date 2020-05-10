import os
import csv
import json

def get_data_filenames(data_dir):
    return [file for file in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, file))]
    
def get_data_filepaths(data_dir):
    '''
    get all file path from given folder
    not support recursive
    if data_dir is a file path, return this path
    '''
    if os.path.isdir(data_dir):
        return [os.path.join(data_dir, file) for file in get_data_filenames(data_dir)]
    else:
        return [data_dir]

def get_data_row(data_dir):
    for input_filepath in get_data_filepaths(data_dir):
        for line in get_data_row_from_file(input_filepath):
            yield line

def get_data_row_from_file(input_filepath):
    with open(input_filepath, 'rt', encoding='utf-8') as f:
        line = f.readline()
        while line:
            yield line
            line = f.readline()

def write_json_lines(file_path, data):
    write_lines(file_path, [json.dumps(item) for item in data])

def write_lines(file_path, data):
    '''
    data: string list
    file_path: string
    '''
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(line + '\n')