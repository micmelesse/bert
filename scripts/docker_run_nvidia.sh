alias nv_drun='sudo docker run -it --network=host --runtime=nvidia --ipc=host -v $HOME/dockerx:/dockerx'
nv_drun -w /dockerx/bert nvcr.io/nvidia/tensorflow:20.03-tf2-py3