# DINDIN Meryll
# June 04, 2018
# Dreem Headband Sleep Phases Classification Challenge

import argparse

# Defines the multi-threaded auto-encoders training
# storage pinpoints to the database repository
# channels refers to which channel to train on
def ae_launcher(storage='./dataset', channels=['eeg_1', 'eeg_2', 'eeg_3', 'eeg_4', 'po_r', 'po_ir']):

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
        arg = '{} {} {}'.format(storage, cnl, gpu)
        cmd = 'python3 -W ignore ae_builder.py {}'.format(arg)
        cmd = ['nohup'] + shlex.split(cmd)
        out = open('./models/AE_{}.out'.format(cnl), 'w')
        log = open('./models/AE_logger.out', 'a')
        subprocess.Popen(cmd, stdout=out, stderr=log)
        print('# Launched:', arg)
        time.sleep(20)

# Main algorithm

if __name__ == '__main__':

    # Initialize the arguments
    prs = argparse.ArgumentParser()
    # Mandatory arguments
    prs.add_argument('-s', '--storage', help='Refers to the databases repository', type=str, default=None)
    # Parse the arguments
    prs = prs.parse_args()

    # Launch the model
    ae_launcher(storage=prs.storage, channels=['eeg_1', 'eeg_2', 'eeg_3', 'eeg_4', 'po_r', 'po_ir'])