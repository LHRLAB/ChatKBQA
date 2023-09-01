#!/usr/bin/env python3

# This script provides a convenient wrapper for the Virtuoso SPARQL server.
# Adapted from Sempre (https://github.com/percyliang/sempre)

import os
import sys
import subprocess
import argparse

virtuosoPath = "../virtuoso-opensource"
if not os.path.exists(virtuosoPath):
  print(f"{virtuosoPath} does not exist")
  sys.exit(1)

# Virtuoso has two services: the server (isql) and SPARQL endpoint
def isqlPort(port): return 10000 + port
def httpPort(port): return port

def run(command):
  print(f"RUNNING: {command}")
  res = subprocess.run(command, shell=True, stdout=subprocess.PIPE)
  return res.stdout

def start(dbPath, port):

  if not os.path.exists(dbPath):
    os.mkdir(dbPath)

  # Recommended: 70% of RAM, each buffer is 8K
  # Use a fraction of the free RAM. The result may vary across runs.
  # memFree = parseInt(`cat /proc/meminfo | grep MemFree | awk '{print $2}'`) # KB
  # Use a fraction of the total RAM. The result is the same across runs.
  memFree = int(run("cat /proc/meminfo | grep MemTotal | awk '{print $2}'")) # KB
  numberOfBuffers = memFree * 0.15 / 8
  maxDirtyBuffers = numberOfBuffers / 2
  print(f"{memFree} KB free, using {numberOfBuffers} buffers, {maxDirtyBuffers} dirty buffers")

  # Configuration options:
  #   http://docs.openlinksw.com/virtuoso/dbadm.html
  #   http://virtuoso.openlinksw.com/dataspace/doc/dav/wiki/Main/VirtConfigScale
  config = (
    f"[Database]\n"
    f"DatabaseFile = {dbPath}/virtuoso.db\n"
    f"ErrorLogFile = {dbPath}/virtuoso.log\n"
    f"LockFile = {dbPath}/virtuoso.lck\n"
    f"TransactionFile = {dbPath}/virtuoso.trx\n"
    f"xa_persistent_file = {dbPath}/virtuoso.pxa\n"
    f"ErrorLogLevel = 7\n"
    f"FileExtend = 200\n"
    f"MaxCheckpointRemap = 2000\n"
    f"Striping = 0\n"
    f"TempStorage = TempDatabase\n"
    f"\n"
    f"[TempDatabase]\n"
    f"DatabaseFile = {dbPath}/virtuoso-temp.db\n"
    f"TransactionFile = {dbPath}/virtuoso-temp.trx\n"
    f"MaxCheckpointRemap = 2000\n"
    f"Striping = 0\n"
    f"\n"
    f"[Parameters]\n"
    f"ServerPort = {isqlPort(port)}\n"
    f"LiteMode = 0\n"
    f"DisableUnixSocket = 1\n"
    f"DisableTcpSocket = 0\n"
    f"ServerThreads = 100 ; increased from 20\n"
    f"CheckpointInterval = 60\n"
    f"O_DIRECT = 1 ; increased from 0\n"
    f"CaseMode = 2\n"
    f"MaxStaticCursorRows = 100000\n"
    f"CheckpointAuditTrail = 0\n"
    f"AllowOSCalls = 0\n"
    f"SchedulerInterval = 10\n"
    f"DirsAllowed = .\n"
    f"ThreadCleanupInterval = 0\n"
    f"ThreadThreshold = 10\n"
    f"ResourcesCleanupInterval = 0\n"
    f"FreeTextBatchSize = 100000\n"
#    f"SingleCPU = 0\n"
    f"PrefixResultNames = 0\n"
    f"RdfFreeTextRulesSize = 100\n"
    f"IndexTreeMaps = 256\n"
    f"MaxMemPoolSize = 200000000\n"
    f"PrefixResultNames = 0\n"
    f"MacSpotlight = 0\n"
    f"IndexTreeMaps = 64\n"
    f"NumberOfBuffers = {numberOfBuffers}\n"
    f"MaxDirtyBuffers = {maxDirtyBuffers}\n"
    f"\n"
    f"[SPARQL]\n"
    f"ResultSetMaxRows = 50000\n"
    f"MaxQueryCostEstimationTime = 600 ; in seconds (increased)\n"
    f"MaxQueryExecutionTime = 180; in seconds (increased)\n"
    f"\n"
    f"[HTTPServer]\n"
    f"ServerPort = {httpPort(port)}\n"
    f"Charset = UTF-8\n"
    f"ServerThreads = 15 ; increased from unknown\n"
  ) 

  configPath = f"{dbPath}/virtuoso.ini"
  print(config)
  print()
  print(configPath)
  print(f"==== Starting Virtuoso server for {dbPath} on port {port}...")
  with open(configPath, 'w') as f:
      f.write(config)
  run(f"{virtuosoPath}/bin/virtuoso-t +configfile {configPath} +wait")

def stop(port):
  run(f"echo 'shutdown;' | {virtuosoPath}/bin/isql localhost:{isqlPort(port)}")

def status(port):
  run(f"echo 'status();' | {virtuosoPath}/bin/isql localhost:{isqlPort(port)}")

############################################################
# Main

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="manage Virtuoso services")
  parser.add_argument("action", type=str, help="start or stop")
  parser.add_argument("port", type=int, help="port for the SPARQL HTTP endpoint")
  parser.add_argument("-d", "--db-path", type=str, help="path to the db directory")

  args = parser.parse_args()
  if args.action == "start":
    if not args.db_path:
      print("please specify path to the db directory with -d")
      sys.exit()
      
    if not os.path.isdir(args.db_path):
      print("the path specified does not exist")
      sys.exit()

    start(args.db_path, args.port)
  elif args.action == "stop":
    stop(args.port)
  else:
    print(f"invalid action: ${args.action}")
    sys.exit()
