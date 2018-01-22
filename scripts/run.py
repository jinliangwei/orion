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
    #print (config)
    if not config['worker']['orion_home']:
        sys.exit("orion home is not set, abort!")
    return config

def get_default_config():
    config = { }
    config['master'] = {
        'ip' : "127.0.0.1",
        'port' : "10000",
        'comm_buff_capacity' : "1024"
    }
    config['worker'] = {
        'port' : "11000",
        'num_executors_per_worker' : "1",
        'num_servers_per_worker' : "1",
        'executor_thread_pool_size' : "4",
        'julia_bin' : "/home/ubuntu/julia-0.5.1/usr/bin",
        'partition_size_mb' : "1",
        'orion_home' : None
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
    config['strace'] = {
        'master_output' : "/tmp/master.strace",
        'worker_output' : "/tmp/worker.strace",
        'summary' : "false",
        'trace_set' : ""
    }
    config['valgrind'] = {
        'leak-check' : 'yes',
        'track-origins' : 'yes',
        'callgrind' : 'false'
    }
    config['hdfs'] = {
        'name_node' : "hdfs://localhost:9000",
        'hadoop_classpath_file' : None
    }
    config['googleprof'] = {
        'master_output_dir' : '/tmp/master.prof',
        'worker_output_dir' : '/tmp/worker.prof'
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
        'GLOG_logbuflevel': pargs['log']['logbuflevel'],
        'JULIA_HOME' : pargs['worker']['julia_bin']
    }

    if pargs['hdfs']['hadoop_classpath_file'] is not None:
        with open(pargs['hdfs']['hadoop_classpath_file'], 'r') as fobj:
            env_vars['CLASSPATH'] = fobj.read().strip()

    return "".join([" %s=%s" % (k, v) for (k, v) in env_vars.items()])

def get_arg_strs(args, pargs):
    hosts = []
    num_executors_total = pargs['worker']['num_executors_per_worker']
    num_servers_total = pargs['worker']['num_servers_per_worker']
    if args.deploy_mode == "cluster":
        with open(args.machine_file, 'r') as fobj:
            for line in fobj:
                hosts.append(line)
        num_executors_total = int(pargs['worker']['num_executors_per_worker']) * len(hosts)
        num_servers_total = int(pargs['worker']['num_servers_per_worker']) * len(hosts)

    master_args = {
        'master_ip' : pargs['master']['ip'],
        'master_port' : pargs['master']['port'],
        'num_executors' : num_executors_total,
        'num_servers' : num_servers_total,
        'comm_buff_capacity' : pargs['master']['comm_buff_capacity'],
        'orion_home' : pargs['worker']['orion_home']
    }

    worker_args = {
        'master_ip' : pargs['master']['ip'],
        'master_port' : pargs['master']['port'],
        'comm_buff_capacity' : pargs['master']['comm_buff_capacity'],
        'num_executors_per_worker' : pargs['worker']['num_executors_per_worker'],
        'num_servers_per_worker' : pargs['worker']['num_servers_per_worker'],
        'num_executors' : num_executors_total,
        'num_servers' : num_servers_total,
        'worker_port' : pargs['worker']['port'],
        'executor_thread_pool_size' : pargs['worker']['executor_thread_pool_size'],
        'partition_size_mb' : pargs['worker']['partition_size_mb'],
        'orion_home' : pargs['worker']['orion_home'],
        'hdfs_name_node' : pargs['hdfs']['name_node']
    }

    master_arg_str = "".join([" --%s=%s" % (k, v) for (k, v) in master_args.items()])
    worker_arg_str = "".join([" --%s=%s" % (k, v) for (k, v) in worker_args.items()])
    return master_arg_str, worker_arg_str, hosts

def get_strace_arg_str(strace_config):
    if strace_config['summary'] == 'true':
        master_str = "-c"
        worker_str = "-c"
    else:
        master_str = "-tt -T -o " + strace_config['master_output']
        worker_str = "-tt -T -o " + strace_config['worker_output']

    if not strace_config['trace_set'] == "":
        master_str += " -e trace=" + strace_config['trace_set']
        worker_str += " -e trace=" + strace_config['trace_set']

    return master_str, worker_str

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
    master_arg_str, worker_arg_str, hosts = get_arg_strs(args, pargs)

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))

    path_master = project_dir + "/bin/bosen/master"
    path_worker = project_dir + "/bin/bosen/worker"

    if args.profile is None:
        cmd_master = env_vars_str + " " + path_master + " " + master_arg_str
        cmd_worker = env_vars_str + " " + path_worker + " " + worker_arg_str
    elif args.profile == "strace":
        worker_str, master_str = get_strace_arg_str(pargs['strace'])
        cmd_master = env_vars_str + " strace " + master_str + " " + path_master \
          + " " + master_arg_str
        cmd_worker = env_vars_str + " strace " + worker_str + " " + path_worker \
          + " " + worker_arg_str
    elif args.profile == "valgrind":
        valgrind_arg_str = get_valgrind_arg_str(pargs['valgrind'])
        cmd_master = env_vars_str + " valgrind " + valgrind_arg_str + " " + path_master \
          + " " + master_arg_str
        cmd_worker = env_vars_str + " valgrind " + valgrind_arg_str + " " + path_worker \
          + " " + worker_arg_str
    elif args.profile == "googleprof":
        cmd_master = env_vars_str + " LD_PRELOAD=" + pargs['googleprof']['profiler_lib'] \
                     + " CPUPROFILE=" + pargs['googleprof']['master_output_dir'] \
                     + " " + path_master + " " + master_arg_str
        cmd_worker = env_vars_str + " LD_PRELOAD=" + pargs['googleprof']['profiler_lib'] \
                     + " CPUPROFILE=" + pargs['googleprof']['worker_output_dir'] \
                     + " " + path_worker + " " + worker_arg_str
    else:
        print ("unsupported profile option %s" % args.profile)
        sys.exit(1)

    if args.deploy_mode == "local":
        cmd_worker += " --worker_ip=127.0.0.1"
    else:
        print ("Warning: profiling in cluster mode might not work")

    print(cmd_master)
    print(cmd_worker)

    master_proc = subprocess.Popen(cmd_master, stdout=subprocess.PIPE, shell=True)
#    time.sleep(5)
    while True:
        line = master_proc.stdout.readline()
        if line.decode("utf-8").strip() == "Master is ready to receive connection from executors!":
            print ("Master is ready; starting workers now!")
            break

    num_executors_per_worker = int(pargs['worker']['num_executors_per_worker'])
    num_servers_per_worker = int(pargs['worker']['num_servers_per_worker'])
    if args.deploy_mode == 'local':
        for i in range(0, num_executors_per_worker + num_servers_per_worker):
            is_server = "false"
            if i >= num_executors_per_worker:
                is_server = "true"
            curr_cmd_worker = cmd_worker + " --local_executor_index=" + str(i) \
                               + " --is_server=" + is_server
            subprocess.Popen(curr_cmd_worker, shell=True)
    else:
        worker_id = 0
        for host in hosts:
            for i in range(0, num_executors_per_worker + num_servers_per_worker):
                print("starting %d-th executor on worker %d" % (i, worker_id))
                is_server = "false"
                if i >= num_executors_per_worker:
                    is_server = "true"
                ssh_cmd_worker = "cd " + project_dir + "; " + cmd_worker \
                                 + " --worker_ip=" + host.strip() \
                                + " --worker_id=" + str(worker_id) \
                                + " --local_executor_index=" + str(i) \
                                + " --is_server=" + is_server
                print(ssh_cmd_worker)
                worker_proc = subprocess.Popen(["ssh", "-oStrictHostKeyChecking=no",
                                                "-oUserKnownHostsFile=/dev/null",
                                                "-oLogLevel=QUIET",
                                                "%s" % host,
                                                ssh_cmd_worker],
                                               shell=False)
                if (i + 1) % 8 == 0:
                    time.sleep(1)
            worker_id += 1
    while True:
        line = master_proc.stdout.readline().decode("utf-8").strip()
        if line == "Your Orion cluster is ready!":
            print ("Your Orion cluster is ready!")
            line = master_proc.stdout.readline().decode("utf-8").strip()
            print (line)
            break
    master_proc.wait()
