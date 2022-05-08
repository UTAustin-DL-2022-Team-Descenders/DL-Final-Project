# Team Descenders - Group 1

## Team Members
- Jose Rojas
- Justin Gutter
- Gabriel Hart Rockwell

## How the Agent Works

The agent extracts different features from the current and past states of the 
environment and feeds each set of features to the respective actor 
(`SteeringActor|steer_net.pt`, `DriftActor|drift_net.pt`, and 
`SpeedActor|speed_net.pt`), who then calculate the particular actions given 
the features. The Planner (`planner_net.pt`) then calculates how much influence 
each sub-actor should have on the action given the states available. These 
actions are then fed to the Fine Tuned Planner (`ft_palnner_net.pt`) which 
filters the actions to better act in the environment.

## Description of Files

### Reinforce / Policy Gradient Agent
- `__init__.py` - initialization script for the `state_agent`
- `action_nets.py`- contains the various neural networks used by the `actors`
- `actors.py` - contains the low level actors: `SteeringActor`, `DriftActor`, `SpeedActor`
- `agents.py` - includes the agents, which are wrapers of the actors to allow them to act
in the environment.
- `core_utils.py` - a subset of `utils_agent` the `state_agent` uses for the grader
- `features.py` - contains the features that are extracted from the state of the 
environment and used in the action calculations
- `planners.py` - includes the Planner and Fine Tuned Planner classes, which instruct 
the agent on what action to take in which scenario.
- `reinforce.py` - REINFORCE implemntation by Jose, not used to train the model
- `remote.py` - copy of the tournament `remote` used in development to train the `state_agent`
- `rewards.py` - rewards for the policy gradient algorithm
- `runner.py` - copy of the tournament `runner` used in development to train the `state_agent`
- `train_arena.ipynb` - main tool used to train the actors using policy gradient
- `train_policy_gradient.py` - contains the main policy gradient algorithm used
to train the `state_agent`
- `train_reinforce.py` - REINFORCE implemntation by Justin, not used to train the model
- `train_runner.py` - A wrapper around runner and train for executing repetitve 
reinforement learning data collection and training. Not used extensively to 
in training the `state_agent`.
- `utils.py` - copy of the tournament `utils` used in development to train the `state_agent`
- `utils_agent.py` - all custom utilities used in development to train the `state_agent`

### Imitation Agent
All code below was used for trying to imitate agents in the environment. The
R&D proved unfruitful and the final `state_agent` did not use imitation learning
to train the agent.

- `train_imitation.py` - implentation of imitation algorithm by Justin
- `train_imitation_action_network.py` - neural nets used for the imitation agent
- `train_imitation_categorize_movements.py` - script to isolate segments of 
particular movements: moving backwards, moving to the puck, scoring a goal, etc.
- `train_imitation_extract_features_for_categories.py` - functions used to isolate
movements from a entire match. Used by `train_imitation_categorize_movements`.
- `train_imitation_improved.py` - improvement of `train_imitation` by Hart including bug fixes and batch gradient descent.
- `train_imitation_utils.py` - utilities used by `train_imitation_improved`

### JIT Scripts
Files listed below are the JIT Scripts for each actor that work in unison to act 
in the environment.

- `drift_net.pt`
- `ft_planner_net.pt`
- `planner_net.pt`
- `speed_net.pt`
- `steer_net.pt`