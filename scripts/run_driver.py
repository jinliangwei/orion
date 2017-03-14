#!/usr/bin/env python3

import os
from os.path import dirname
from os.path import join
import time
import sys

import argparse
import configparser
from pprint import pprint
import subprocess

def parse_command_line():
    parser = argparse.ArgumentParser(description="Lunching application")
    parser.add_argument('-c', '--config_file', action='store')
    args = parser.parse_args()

    return args

def parse_config_file (config_file_path, config):
    fconfig = configparser.ConfigParser()
    fconfig.read(config_file_path)
    for key, value in fconfig.items():
        if key not in config.keys():
            continue
        for ckey, cvalue in value.items():
                config[key][ckey] = cvalue
    print (config)
    return config

def get_default_config():
    config = { }
    config['master'] = {
        'ip' : "127.0.0.1",
        'port' : "10000",
        'comm_buff_capacity' : "1024"
    }
    config['log'] = {
        'log_dir' : "",
        'logtostderr' : "true",
        'minloglevel' : "INFO",
        'v' : "0",
        'stderrthreshold' : "ERROR",
        'alsologtostderr' : "false",
        'logbuflevel' : '-1'
    }
    return config
def get_env_str(pargs):
    env_vars = {
        'GLOG_logtostderr': pargs['log']['logtostderr'],
        'GLOG_v': pargs['log']['v'],
        'GLOG_minloglevel': pargs['log']['minloglevel'],
        'GLOG_stderrthreshold': pargs['log']['stderrthreshold'],
        'GLOG_log_dir' : pargs['log']['log_dir'],
        'GLOG_alsologtostderr': pargs['log']['alsologtostderr'],
        'GLOG_logbuflevel': pargs['log']['logbuflevel']
    }

    return "".join([" %s=%s" % (k, v) for (k, v) in env_vars.items()])

def get_arg_strs(args, pargs):
    args = {
        'master_ip' : pargs['master']['ip'],
        'master_port' : pargs['master']['port'],
        'comm_buff_capacity' : pargs['master']['comm_buff_capacity']
    }
    arg_str = "".join([" --%s=%s" % (k, v) for (k, v) in args.items()])
    return arg_str

if __name__ == "__main__":
    args = parse_command_line()
    pargs = get_default_config()
    if args.config_file is not None:
        pargs = parse_config_file(args.config_file, pargs)
    env_vars_str = get_env_str(pargs)
    arg_str = get_arg_strs(args, pargs)

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
    path_driver = project_dir + "/bin/bosen/driver"

    cmd_driver = env_vars_str + " " + path_driver + " " + arg_str
    driver_proc = subprocess.Popen(cmd_driver, shell=True)
    print(cmd_driver)
    driver_proc.wait()
