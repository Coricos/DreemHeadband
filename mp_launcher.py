# DINDIN Meryll
# June 04, 2018
# Dreem Headband Sleep Phases Classification Challenge

import argparse

from package.imports import *

# Defines the multi-threaded auto-encoders training
# storage pinpoints to the database repository
# channels refers to which channel to train on
def ae_launcher(test_size=0.0, storage='./dataset', channels=['eeg_1', 'eeg_2', 'eeg_3', 'eeg_4', 'po_r', 'po_ir']):

    thread_list = channels

    while len(thread_list) != 0:

        # Defines the gpu on which to launch the process
        while True:

            gpu = GPUtil.getAvailable(order='first')

            if len(gpu) != 0:
                gpu = gpu[0]
                break

            time.sleep(10)

        # Launch the thread through multi-threading
        cnl = thread_list.pop(0)
        arg = '{} {} {} {} ATE'.format(test_size, storage, cnl, gpu)
        cmd = 'python3 -W ignore ae_builder.py {}'.format(arg)
        cmd = ['nohup'] + shlex.split(cmd)
        out = open('./models/ATE_{}.out'.format(cnl), 'w')
        log = open('./models/ATE_logger.out', 'a')
        subprocess.Popen(cmd, stdout=out, stderr=log)
        print('# Launched:', arg)
        time.sleep(30)

# Defines the multi-threaded conv1D channels training
# storage pinpoints to the database repository
# channels refers to which channel to train on
def c1_launcher(test_size=0.0, storage='./dataset', channels=['eeg_1', 'eeg_2', 'eeg_3', 'eeg_4', 'po_r', 'po_ir']):

    thread_list = channels

    while len(thread_list) != 0:

        # Defines the gpu on which to launch the process
        while True:

            gpu = GPUtil.getAvailable(order='first')

            if len(gpu) != 0:
                gpu = gpu[0]
                break

            time.sleep(10)

        # Launch the thread through multi-threading
        cnl = thread_list.pop(0)
        arg = '{} {} {} {} CV1'.format(test_size, storage, cnl, gpu)
        cmd = 'python3 -W ignore ae_builder.py {}'.format(arg)
        cmd = ['nohup'] + shlex.split(cmd)
        out = open('./models/CV1_{}.out'.format(cnl), 'w')
        log = open('./models/CV1_logger.out', 'a')
        subprocess.Popen(cmd, stdout=out, stderr=log)
        print('# Launched:', arg)
        time.sleep(30)

# Main algorithm

if __name__ == '__main__':

    # Initialize the arguments
    prs = argparse.ArgumentParser()
    # Mandatory arguments
    prs.add_argument('-s', '--storage', help='Refers to the databases repository', type=str, default='./dataset')
    prs.add_argument('-t', '--test_size', help='Test_size for training', type=float, default=0.0)
    prs.add_argument('-o', '--objectif', help='Objectif', type=str, default='ATE')
    # Parse the arguments
    prs = prs.parse_args()

    # Launch the model
    if prs.objectif == 'ATE': 
        ae_launcher(test_size=prs.test_size, storage=prs.storage, channels=['eeg_1', 'eeg_2', 'eeg_3', 'eeg_4', 'po_r', 'po_ir'])
    if prs.objectif == 'CV1':
        c1_launcher(test_size=prs.test_size, storage=prs.storage, channels=['eeg_1', 'eeg_2', 'eeg_3', 'eeg_4', 'po_r', 'po_ir'])