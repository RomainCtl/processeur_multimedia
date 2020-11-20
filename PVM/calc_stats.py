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
CMD = "./CodeSequentiel ../img/%s" # %s is image name

def kill_project(logger):
	cmd = "ps -a -o pid,cmd | grep -P './CodeSequentiel img/[A-z]+.pgm' | awk '{print  $1}' | xargs -I{} kill -9 {}"
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
	def create_stats(nb_nodes):
		logger = create_logger()

		# "number_nodes","image","time","node_1_lines_handled","node_1_duration",...
		line = '%d,%s,%s' + (",%s,%s,%s,%s,%s"*nb_nodes)

		filename = "2_%d_nodes.csv"%nb_nodes
		stats = open("stats/"+filename, "w+")
		writer = csv.writer(stats)

		writer.writerow(("number_nodes,image,time(ms)"+ ''.join(",node_%d_lines_handled,node_%d_duration(ms),node_%d_name,node_%d_arch,node_%d_speed"%(i,i,i,i,i) for i in range(nb_nodes))).split(","))

		for img in Images.imgs:
			for i in range(NB_LAUNCH):
				command = CMD % img
				logger.info("%d/%d\t"%(i, NB_LAUNCH*len(Images.imgs))+command)
				outs, errs = call(command, logger=logger, is_stats=True, shell=True)

				# write results of this test
				res = iter((outs + errs).decode().split("\n"))

				nodes_lines_handled = []
				nodes_info = []
				duration = 0

				# parse output
				while res.__length_hint__() > 0:
					c_line = next(res)

					tasks = re.match(r"La tache [0-9]+ a traiter ([0-9]+) lignes en ([0-9]+)ms", c_line)
					if tasks is not None:
						nodes_lines_handled.append((tasks.group(1), tasks.group(2)))

					nodes = re.match(r"\tNoeud [0-9]+ :", c_line)
					if nodes is not None:
						next(res) # tid
						hi_name = next(res).split("= ")[1].strip() # name (ex: abe-a-43)
						hi_arch = next(res).split("= ")[1].strip() # arch (ex: LINUX64)
						hi_speed = next(res).split("= ")[1].strip() # speed (ex: 1000)

						nodes_info.append( (hi_name, hi_arch, hi_speed) )

					dur = re.match(r"Duration: ([0-9]+)ms", c_line)
					if dur is not None:
						duration = dur.group(1)

				_nodes_data = list(map(lambda x: x[0] + x[1], zip(nodes_lines_handled, nodes_info)))
				nodes_data = list(_nodes_data[0])
				for n in range(len(_nodes_data)-1):
					nodes_data.extend(list(_nodes_data[n]))

				lines = [nb_nodes, img, duration]
				lines.extend(nodes_data)

				writer.writerow((line % tuple(lines)).split(","))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("nodes", help="number of PVM nodes", type=int)
	args = parser.parse_args()
	Images.create_stats(args.nodes)

