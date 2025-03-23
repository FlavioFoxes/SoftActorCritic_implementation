# Soft Actor-Critic implementation

This repo contains a Soft Actor-Critic implementation compatible with gymnasium environments. Furthermore, it contains a gymnasium environment linked with SPQR Team framework for the creation of KeyMotionFrame kicks.

## Overview
`main_sb.py` contains a call to SAC implementation by **stable-baselines3** applied on Pendulum-v1 environment.

`main_SAC.py` explores my custom implementation of SAC algorithm on Pendulum-v1 environment.

`main_RoboCup.py` calls my custom implementation of SAC algorithm on custom gym compatible RoboCup environment (`RoboCup_env.py`)

The folder `trained_models` contains some trained models for different environments (according to their names).

The folder `logs` contains tensorboard files which show the logs of the trainings.

The folder `src` contains all the implementation of my custom SAC algorithm.

## Results
**NOTE**: WORK IN PROGRESS
