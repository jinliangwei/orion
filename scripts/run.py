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
    parser.add_argument('bin', help='the absolute path to the bin directory '
                        'where the executables are located')
    parser.add_argument('-a', '--app_name', action='store')
    parser.add_argument('-c', '--config_file', action='store')
    parser.add_argument('-m', '--machine_file', action='store')
    parser.add_argument('--deploy_mode', action='store', default='local',
                        help='local or cluster')
    parser.add_argument('-p', '--profile', action='store')
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
    config['driver'] = {
        'ip' : "127.0.0.1",
        'port' : "10000",
        'comm_buff_capacity' : "1024"
    }
    config['worker'] = {
        'port' : "11000",
        'num_executors_per_worker' : "1"
    }
    config['log'] = {
        'log_dir' : "",
        'logtostderr' : "true",
        'minloglevel' : "INFO",
        'v' : "0",
        'stderrthreshold' : "ERROR",
        'alsologtostderr' : "false"
    }
    config['strace'] = {
        'driver_output' : "/tmp/driver.strace",
        'worker_output' : "/tmp/worker.strace",
        'summary' : "false",
        'trace_set' : ""
    }
    config['valgrind'] = {
        'leak-check' : 'yes',
        'track-origins' : 'yes',
        'callgrind' : 'false'
    }
    config['app'] = { }
    return config

def get_env_str(pargs):
    env_vars = {
        'GLOG_logtostderr': pargs['log']['logtostderr'],
        'GLOG_v': pargs['log']['v'],
        'GLOG_minloglevel': pargs['log']['minloglevel'],
        'GLOG_stderrthreshold': pargs['log']['stderrthreshold'],
        'GLOG_log_dir' : pargs['log']['log_dir'],
        'GLOG_alsologtostderr': pargs['log']['alsologtostderr']
    }

    return "".join([" %s=%s" % (k, v) for (k, v) in env_vars.items()])

def get_arg_strs(args, pargs):
    hosts = []
    num_executors_total = pargs['worker']['num_executors_per_worker']
    if args.deploy_mode == "cluster":
        with open(args.machine_file, 'r') as fobj:
            for line in fobj:
                hosts.append(line)
        num_executors_total = int(pargs['worker']['num_executors_per_worker']) * len(hosts)

    driver_args = {
        'driver_ip' : pargs['driver']['ip'],
        'driver_port' : pargs['driver']['port'],
        'driver_num_executors' : num_executors_total,
        'comm_buff_capacity' : pargs['driver']['comm_buff_capacity']
    }

    worker_args = {
        'worker_driver_ip' : pargs['driver']['ip'],
        'worker_driver_port' : pargs['driver']['port'],
        'comm_buff_capacity' : pargs['driver']['comm_buff_capacity'],
        'worker_num_executors_per_worker' : pargs['worker']['num_executors_per_worker'],
        'worker_port' : pargs['worker']['port']
    }

    driver_arg_str = "".join([" --%s=%s" % (k, v) for (k, v) in driver_args.items()])
    worker_arg_str = "".join([" --%s=%s" % (k, v) for (k, v) in worker_args.items()])
    return driver_arg_str, worker_arg_str, hosts

def get_strace_arg_str(strace_config):
    if strace_config['summary'] == 'true':
        driver_str = "-c"
        worker_str = "-c"
    else:
        driver_str = "-tt -T -o " + strace_config['driver_output']
        worker_str = "-tt -T -o " + strace_config['worker_output']

    if not strace_config['trace_set'] == "":
        driver_str += " -e trace=" + strace_config['trace_set']
        worker_str += " -e trace=" + strace_config['trace_set']

    return driver_str, worker_str

def get_valgrind_arg_str(valgrind_config):
    if valgrind_config['callgrind'] == "true":
        arg_str = "--tool=callgrind"
        return arg_str
    arg_str = "".join([" --%s=%s" % (k, v) for (k, v) in valgrind_config.items() \
                       if not k == "callgrind"])
    return arg_str

if __name__ == "__main__":
    args = parse_command_line()
    pargs = get_default_config()
    if args.config_file is not None:
        pargs = parse_config_file(args.config_file, pargs)

    env_vars_str = get_env_str(pargs)
    driver_arg_str, worker_arg_str, hosts = get_arg_strs(args, pargs)

    path_driver = args.bin + "/driver"
    path_worker = args.bin + "/worker"

    if args.profile is None:
        cmd_driver = env_vars_str + " " + path_driver + " " + driver_arg_str
        cmd_worker = env_vars_str + " " + path_worker + " " + worker_arg_str
    elif args.profile == "strace":
        worker_str, driver_str = get_strace_arg_str(pargs['strace'])
        cmd_driver = env_vars_str + " strace " + driver_str + " " + path_driver \
          + " " + driver_arg_str
        cmd_worker = env_vars_str + " strace " + worker_str + " " + path_worker \
          + " " + worker_arg_str
    elif args.profile == "valgrind":
        valgrind_arg_str = get_valgrind_arg_str(pargs['valgrind'])
        cmd_driver = env_vars_str + " valgrind " + valgrind_arg_str + " " + path_driver \
          + " " + driver_arg_str
        cmd_worker = env_vars_str + " valgrind " + valgrind_arg_str + " " + path_worker \
          + " " + worker_arg_str
    else:
        print ("unsupported profile option %s" % args.profile)
        sys.exit(1)

    if args.deploy_mode == "local":
        cmd_worker += " --worker_ip=127.0.0.1"
    else:
        print ("Warning: profiling in cluster mode might not work")

    print(cmd_driver)
    print(cmd_worker)

    driver_proc = subprocess.Popen(cmd_driver, shell=True, stdout=subprocess.PIPE)
    while True:
        line = driver_proc.stdout.readline()
        if line.decode("utf-8").strip() == "Driver is ready!":
            print ("Driver is ready; starting workers now!")
            break

    if args.deploy_mode == 'local':
        subprocess.Popen(cmd_worker, shell=True)
    else:
        worker_id = 0
        for host in hosts:
            print("starting worker ", worker_id)
            ssh_cmd_worker = cmd_worker + " --worker_ip=" + host.strip() \
              + " --worker_id=" + str(worker_id)
            print(ssh_cmd_worker)
            subprocess.Popen(["ssh", "-oStrictHostKeyChecking=no",
                             "-oUserKnownHostsFile=/dev/null",
                             "-oLogLevel=quiet",
                             "%s" % host,
                              ssh_cmd_worker], shell=False)
            worker_id += 1
