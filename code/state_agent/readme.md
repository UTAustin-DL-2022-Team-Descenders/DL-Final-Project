# Team Descenders - Group 1

## Team Members
- Jose Rojas
- Justin Gutter
- Gabriel Hart Rockwell

## How the Agent Works

Our team's agent is broken down into a heirarchy of individual subnetworks that
were trained independently for each particular task. The subnetworks were trained
exclusively on each task using reinforment learning via policy gradients.

At the highest structure, there are the agents that make up the team. Each agent is
broken into 'Actors' and at the bottom tiers are the 'Action Networks' which are PyTorch
models. The agent executes each Actor, which is responsible for extracting the
features for its respecitive Action network. The Networks/Actors communicate with each
other via a global feature object (`SoccerFeatures`). This decouples the networks and
allows for reconfiguration for easier training and debugging. The outputs of the networks
at the bottom of the heirarchy of Actors produce the final actions to manipulate the
cart without any controller logic or post-processing.

The agent extracts different features from the current and past states of the 
environment and feeds each set of features to the respective actor 
(`SteeringActor|steer_net.pt`, `DriftActor|drift_net.pt`, and 
`SpeedActor|speed_net.pt`), who then calculate the particular actions given 
the features. The Planner (`planner_net.pt`) then identifies the best policy
action to take given a small set of pre-trained scenarios (e.g., head to the puck,
move the puck to the goal, back up if stuck against a wall). The actions are
then fed to the Fine Tuned Planner (`ft_palnner_net.pt`) which filters the
actions to better act in the environment.

The Action Networks that generate the policy actions are all 2 layer linear
networks, seperated by a Tanh activation across the hidden network and have either raw outputs,
Tanh or sigmoid activations depending on what is best suited for the particular actuatiton
that is needed to execute the policy actions. For example, Tanh activations are used for
the steering action network to produce outputs from -1 to 1.

## Description of Files

### Reinforce / Policy Gradient Agent
- `__init__.py` - initialization script for the `state_agent`
- `action_nets.py`- contains the various neural networks used by the `actors`
- `actors.py` - contains the low level actors: `SteeringActor`, `DriftActor`, `SpeedActor`
which actuate the cart by taking policy decision from the planner actors.
- `agents.py` - includes the agents, which are wrapers of the actors to allow them to act
in the environment.
- `core_utils.py` - a subset of `utils_agent` the `state_agent` uses for the grader
- `features.py` - contains the features that are extracted from the state of the 
environment and used in the action calculations
- `planners.py` - includes the Planner and Fine Tuned Planner classes, which determine
the current state of the world and decide on what high level policy action is best. The planners
output speed and steering offsets for the Drift, Steering and Speed Actors to process for final output.
- `reinforce.py` - REINFORCE implemntation by Jose, not used to train the model
- `remote.py` - copy of the tournament `remote` used in development to train the `state_agent`
- `rewards.py` - rewards for the policy gradient algorithm
- `runner.py` - copy of the tournament `runner` used in development to train the `state_agent`
- `train_arena.ipynb` - main tool used to train the actors using policy gradients
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