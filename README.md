# CorrProxies

### Declaration
This repo is for paper: _Optimizing Machine Learning Inference Queries with Correlative Proxy Models_.

## Setup ENV

### Quick Start
1. We provide a fully ready Docker Image  ready to use out-of-box.
2. Optionally, you can also follow the steps to build your own testing environment.

### The Provided Docker Environment

#### Steps to run the Docker Environment
- Get the docker image from this [link](https://texera.ics.uci.edu/docker/corrproxies-image.tar).
- Load the docker image.
  `docker load -i corrproxies-image.tar`
- Run the docker image in a container.
  `docker run --name=CorrProxies -i -t -d corrproxies-image`
    - it will return you the docker container ID, for example `d979af9a17f23345cb2894b22dc8527680acdfd7a7e1aaed6a7a28ea134e66e6`.
- Use CLI to control the container with the specific ID generated.
  `docker exec -it d979af9a17f23345cb2894b22dc8527680acdfd7a7e1aaed6a7a28ea134e66e6 /bin/zsh`

#### ENV Spec
- Operating System: `ubuntu@16.04`
- Python ENV: 
  - `Python@3.6.6` with `Anaconda@4.5.11` distribution.
  - dependencies:
    - `numpy@1.19.5`
    - `tensorflow@1.14.0`
    - `torch@1.8.1`
    - see `requirements.txt` for more dependencies.
- Java ENV: `openjdk@1.8`

#### File structure:
  - The home directory for `CorrProxies` locates at `/home/CorrProxies`.
  - The Python executable locates at `/home/anaconda3/envs/condaenv/bin/python3`.
  - The models locate at `/home/CorrProxies/model`.
  - The datasets locate at `/home/CorrProxies/data`.
  - The starting scripts locate at `/home/CorrProxies/scripts`.

### Build Your Own Environment
This instruction is based on a clean distribution of `Ubuntu@16.04`
1. Install pre-requisites.

   `apt-get update && apt-get install -y build-essential`
2. Install `Anaconda`.

    - `wget https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh && bash Anaconda3-5.3.1-Linux-x86_64.sh -b -p <AnacondaHome>`
    - `export PATH="<AnacondaHome>/bin/:$PATH"`
3. Install `Python@3.6.6` with `Anaconda3`.

   `conda create -n condaenv python=3.6.6`
4. Activate the newly installed Python ENV.

   `conda activate condaenv`
5. Install dependencies with pip.
   
   `pip3 install -r requirements.txt`
6. Install `Java (openjdk-8)` (for `standford-nlp` usage).
   
    `apt-get install -y openjdk-8-jdk`

## Queries & Datasets
- We use `Twitter` text dataset, `COCO` image dataset and `UCF101` video dataset as our benchmark datasets.
Please see this page for [examples](https://github.com/ZhihuiYangCS/CorrProxies/wiki/Queries-and-Datasets) of detailed Queries and Datasets examples we use in our experiments.

- After you setup the environment, either manually or using the docker image provided by us, the next step is to download the datasets.
  - To get the `COCO` dataset: 
  `cd /home/CorrProxies/data/image/coco && ./get_coco_dataset.sh`
  - To get the `UCF101` dataset:
  `cd /home/CorrProxies/data/video/ucf101 && wget -c https://www.crcv.ucf.edu/data/UCF101/UCF101.rar && unrar x UCF101.rar`.

## Execution
### Please pull the latest code before executing the code. Command `cd /home/CorrProxies && git pull`
### Run Operators Individually
To run and see each operator we used in our experiment, simply execute `python3 <operator_name.py>`. For example: `python3 operators/ml_operators/image_video_operators/video_activity_recognition.py`.


### Run Experiments
We use `scripts/run.sh` to start experiments. The script will take in command line arguments.
- Text(Twitter)
  - Since we do not provide text dataset, we will skip the experiment.
- Image(COCO)
  
    Example: `./scripts/run.sh -w 2 -t 1 -i '1' -a 0.9 -s 3 -o 2 -e 1`
- Video(UCF101)

    Example: `./scripts/run.sh -w 2 -t 2 -i '1' -a 0.9 -s 3 -o 2 -e 1`

- arguments detail.
  - w     int: experiment type in `[1, 2, 3, 4]` referring to `/home/CorrProxies/ml_workflow/exps/WorkflowExp*.py`;
  - t     int: query type in `[0, 1, 2]`. Int `0, 1, 2` means queries on the `Twitter, COCO, and UCF101` datasets, respectively;
  - i     int: query index in `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`;
  - a     float: query accuracy;
  - s     int: scheme in `[0, 1, 2, 3, 4, 5, 6]`. Int `0, 1, 2, 3, 4, 5, 6` means `'ORIG', 'NS', 'PP', 'CORE', 'COREa', 'COREh' and 'REORDER'` schemes, respectively;
  - o     int: number of threads used in optimization phase;
  - e     int: number of threads used in execution phase after generating an optimized plan.
