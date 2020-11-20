#!/usr/bin/env python3
from logging.handlers import RotatingFileHandler
import logging
import subprocess
import select
import os, sys
import re
import argparse
import time
import csv

TIMEOUT = 180
NB_LAUNCH = 20
NB_THREADS = list(range(1,17))
CMD = "./CodeSequentiel ../img/%s %d" # param1 is image name and param2 is num_threads

def kill_project(logger):
    cmd = "ps -a -o pid,cmd | grep -P './CodeSequentiel ../PVM/img/[A-z]+.pgm [0-9]+' | awk '{print  $1}' | xargs -I{} kill -9 {}"
    logger.info(cmd)
    logger.info(subprocess.call(cmd, shell=True))

def call(popenargs, logger, is_stats=False, timeout=TIMEOUT, stdout_log_level=logging.DEBUG, stderr_log_level=logging.INFO, **kwargs):
    """
    Variant of subprocess.call that accepts a logger instead of stdout/stderr,
    and logs stdout messages via logger.debug and stderr messages via logger.error.

    author = bgreenlee
    src = https://gist.github.com/bgreenlee/1402841

    re-edit by Romain Chantrel
    """
    child = subprocess.Popen(popenargs, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)

    log_level = {child.stdout: stdout_log_level, child.stderr: stderr_log_level}

    def check_io():
        ready_to_read = select.select([child.stdout, child.stderr], [], [], 1000)[0]
        for io in ready_to_read:
            line = io.readline()
            if line[:-1] != "" and line[:-1] != b'':
                logger.log(log_level[io], line[:-1].decode())

    if not is_stats:
        # keep checking stdout/stderr until the child exits
        while child.poll() is None:
            check_io()
        check_io()  # check again to catch anything after the process exits

        return child.wait()
    else:
        try:
            return child.communicate()
        except Exception as e:
            logger.error(str(e))
            kill_project(logger)
            return child.communicate()

def create_logger():
    formatter = logging.Formatter('%(asctime)s :: %(name)s :: %(levelname)s :: %(message)s')
    logger = logging.getLogger()
    logger.setLevel(0)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Create logger file
    file_handler = RotatingFileHandler('stats/log/'+time.strftime("%Y-%m-%d")+'.log', backupCount=100)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


class Images(object):
    imgs = ["image1.pgm", "MontagneFoncee.pgm", "stavrovouni.pgm"]

    @staticmethod
    def create_stats(hostname):
        logger = create_logger()

        filename = "%s.csv" % hostname
        stats = open("stats/"+filename, "w+")
        writer = csv.writer(stats)

        writer.writerow(["num_threads","image","duration"])

        j=0
        for img in Images.imgs:
            for num_threads in NB_THREADS:
                for i in range(NB_LAUNCH):
                    j += 1

                    command = CMD % (img, num_threads)
                    logger.info("%d/%d\t"%(j, NB_LAUNCH*len(NB_THREADS)*len(Images.imgs))+command)
                    outs, errs = call(command, logger=logger, is_stats=True, shell=True)

                    # write results of this test
                    res = iter((outs + errs).decode().split("\n"))

                    duration = 0

                    # parse output
                    while res.__length_hint__() > 0:
                        c_line = next(res)

                        dur = re.match(r"Duration ([0-9]+)", c_line)
                        if dur is not None:
                            duration = dur.group(1)

                    writer.writerow([num_threads, img, duration])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("hostname", help="Host name", type=str)
    args = parser.parse_args()
    Images.create_stats(args.hostname)
