alias drun='sudo docker run -it --rm --network=host --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $HOME/dockerx:/dockerx -v /data:/data'
# drun -w /dockerx/bert rocm/tensorflow:rocm3.5-tf1.15-dev
drun -w /dockerx/bert rocm/tensorflow:rocm3.5-tf2.1-dev