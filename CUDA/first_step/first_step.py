#!/usr/bin/env python3
from logging.handlers import RotatingFileHandler
import logging
import subprocess
import select
import os, sys
import re
import time
import csv

TIMEOUT = 180
CMD_CPU = "./VecteurCPU %d" # %s vector size
CMD_GPU = "./VecteurGPU %d" # %s vector size

def kill_project(logger):
    cmd = "ps -a -o pid,cmd | grep -P './Vecteur(C|G)PU [0-9]+' | awk '{print  $1}' | xargs -I{} kill -9 {}"
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
    file_handler = RotatingFileHandler('log/'+time.strftime("%Y-%m-%d")+'.log', backupCount=100)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

class Vectors(object):
    vectors = list(1000*10**x for x in range(6))

    @staticmethod
    def create_stats():
        logger = create_logger()

        # "cpu_or_gpu","vector_size","calcul_time","total_time"
        line = '%s,%d,%d'

        filename = "first_steps.csv"
        stats = open("stats/"+filename, "w+")
        writer = csv.writer(stats)

        writer.writerow(["cpu_or_gpu","vector_size","calcul_time","total_time"])

        Vectors.create_stats_cpu(logger, writer)
        Vectors.create_stats_gpu(logger, writer)

    @staticmethod
    def create_stats_cpu(logger, writer):
        for size in Vectors.vectors:
            command = CMD_CPU % size
            logger.info("%s\t"%command)
            outs, errs = call(command, logger=logger, is_stats=True, shell=True)

            # write results of this test
            res = iter((outs + errs).decode().split("\n"))

            duration = -1

            # parse output
            while res.__length_hint__() > 0:
                c_line = next(res)

                dur = re.match(r"Vecteur [0-9]+ => Temps calcul CPU ([0-9]+)", c_line)
                if dur is not None:
                    duration = dur.group(1)

            writer.writerow(["cpu", size, duration, ""])

    @staticmethod
    def create_stats_gpu(logger, writer):
        for size in Vectors.vectors:
            command = CMD_GPU % size
            logger.info("%s\t"%command)
            outs, errs = call(command, logger=logger, is_stats=True, shell=True)

            # write results of this test
            res = iter((outs + errs).decode().split("\n"))

            duration_gpu = -1
            duration_cpu = -1

            # parse output
            while res.__length_hint__() > 0:
                c_line = next(res)

                dur = re.match(r"Vecteur [0-9]+ => Temps calcul GPU ([0-9]+)", c_line)
                if dur is not None:
                    duration_gpu = dur.group(1)

                dur_c = re.match(r"Vecteur [0-9]+ => Temps total  CPU ([0-9]+)", c_line)
                if dur_c is not None:
                    duration_cpu = dur_c.group(1)

            writer.writerow(["gpu", size, duration_gpu, duration_cpu])

if __name__ == "__main__":
    Vectors.create_stats()
