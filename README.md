# Sorting via Pushing
This repository is an extension of the repository rp2024 from gergely-soti --> forked.

## Task

The workspace contains multiple objects of two different colors and areas corresponding to those colors. The task is for the robot to push all objects to their corresponding colored areas. The objects can be of different shapes, and the areas should be simple squares with colored borders able to fit all required objects. The objects and the areas should be placed randomly.

<figure>
<img src="project/block_push.gif" alt="exampleReplaceWithOurVideo" width="300"/>
<figcaption>https://diffusion-policy.cs.columbia.edu/</figcaption>
</figure>

view full Task under ``project\rp_Project_Assignment.pdf``

## Repository Structure

```rp2024
├── .gitignore
├── Dockerfile
├── README.md
├── build_image.sh
├── run_container.sh
├── assets
│   ├── meshes
│   │   ├── ...
│   ├── objects
│   │   ├── object.urdf
│   │   ├── cubes
│   │   │   ├── cube_green.urdf
│   │   │   └── cube_red.urdf
│   │   ├── goals
│   │   │   ├── goal_green.urdf
│   │   │   └── goal_red.urdf
│   │   └── signs
│   │       ├── plus_green.urdf
│   │       └── plus_red.urdf
│   ├── urdf
│   │   ├── robot.urdf
│   │   └── robot_without_gripper.urdf
│   └── util
│       ├── ...
├── src
│   ├── config.yaml
│   ├── changeUrdfRectangle.py
│   ├── transform.py
│   ├── plotScores.py
│   ├── pushToSort.py
│   ├── sortingLearn.py
│   ├── handleEnvironment.py
│   ├── sortingViaPushingEnv.py
│   ├── checkEnv.py
│   ├── __init__.py
│   ├── bullet_env
│   │   ├── bullet_robot.py
│   │   ├── ...
│   ├── __pycache__
│   │   ├── ...
│   ├── lib
│   │   ├── ...
├── project
│   ├── block_push.gif
│   └── rp_Project_Assignment.pdf
└── data
    ├── models
        ├── ...
    ├── logs
        ├── ...
    ├── train
        ├── ...
    └── scores
        ├── ...
```

## Setup environment
In `build_image.sh` choose between
* with graphics-card: `RENDER=nvidia`
* without graphics-card: `RENDER=base`

In `run_container.sh` choose between
* with graphics-card: add line: `--gpus all \`
* without graphics-card: remove line: `--gpus all \`

## Run project
This repo is dockerized, to run/use it in the docker container follow the steps bellow.

First run in terminal:
* `./build_image.sh`
* `./run_container.sh`

Config:

* `config.yaml`: set parameters for training, environment, ... here

Executable skripts:

* `python changeUrdfRectangle.py`: edit urdf files (goal size)
* `python sortingLearn.py`: start training
* `python checkEnv.py`: test the env in 3 different modes
* `python pushToSort.py`: use trained model
* `python plotScores.py`: show result

Non-Executable Skripts:

* `python sortingViaPushingEnv.py`: contains functions to create the environment
* `python handleEnvironment.py`: contains functions to handle the environment
* `python transform.py`: contains class Affine (for dealing with affine transformations)

## View training
* train the model
* in new terminal from docker path: `python3 -m tensorboard.main --logdir=data/logs`
* open tensorboard localhost via the created link