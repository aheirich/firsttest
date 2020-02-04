#!/bin/bash -l
 
 
#BSUB -W 1:00                                                   # Job wall clock limit hh:mm
  
#BSUB -n 3                                                       # request 3 slots/cpu cores for this job
#BSUB -q slacgpu                                                 # queue
 
#BSUB -gpu "num=1:gmodel=TeslaV100_SXM2_32GB:mode=exclusive_process:j_exclusive=no:mps=no"  # request 1 gpu per -n (slot), ie 3 gpus total with 3 cpu cores

# enable site modules
export MODULEPATH=/usr/share/Modules/modulefiles:/etc/modulefiles:/opt/modulefiles:/afs/slac/package/singularity/modulefiles
 
module load cuda/10.0

singularity pull docker:aheirich/firsttest-nvidia:latest
singularity images
singularity run -u $(id -u):$(id -g) docker:aheirich/firsttest-nvidia
