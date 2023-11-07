#! /bin/sh


# author: liuwenshun
echo top_submit

time=`date '+%Y-%m-%d-%H:%M:%S'`
name=train_by_single
logpath=logs/${name}-${time}.log
exefile=train_by_single.py

echo log path: $logpath
echo submit time: $time | tee -a $logpath
echo submit name: $name | tee -a $logpath
echo execute file: $exefile | tee -a $logpath


nohup python -u $exefile >> $logpath &

echo PID: $! | tee -a $logpath