# Temporal-Reasoning-Based Learning from Demonstration (TR-LfD)
Authors: Estuardo Carpio, Madison Clark-Turner 

This repository contains a temporal-reasoning-based architecture for the learning of high-level human interactions from demonstrations (LfD).

The framework has been evaluated with an Applied Behavioral Analysis (ABA) styled social greeting behavioral intervention (BI). The system collects demonstrations of the BI using a tele-operated robot and then extracts relevant spatial features of the interaction using two Convolutional Neural Networks (CNN). Meanwhile, the temporal structure of the interaction is learned by an Interval Temporal Bayesian Network (ITBN) based temporal reasoning model (TRM). Once the interaction has been learned the the CNN and TRM can used to deliver it autonomously using a NAO humanoid robot. 

Usage
=============

Usage of the system occurs in three steps:
1. Collection of training data using a tele-operated robot as it delivers the desired BI.
2. Learning of the spatial features of the BI using the CNN perception models.
3. Learning of the temporal structure of the BI using the ITBN-based TRM model. 
4. Execution of the learned BI using an autonomous system.

The implementation is designed for use with a social greeting BI. Which proceeds in the following manner:
1. The therapist/robot delivers a *Discriminative Stimuli* (The robot says "hello" and waves)
2. The participant provides a *response* that is either compliant (responding to the robot) or non-compliant (refusing to acknowledge the robot's command)
3. The robot reacts to the participants response:
   - Compliant: the robot delivers a *reward* congratulating the participant on following the intervention. The BI then continues to step 4
   - Non-compliant: the robot delivers a *prompt* instructing the participant to respond in a compliant manner (saying "<Participant>, say hello" and waving). The BI then returns to step 2 or if a prompt had already failed to elicit a compliant response then the BI proceeds tho step 4.
4. The robot ends the BI by saying "Good Bye"
 
Data Collection
--------------------

Data collection is performed using a tele-operated NAO humanoid robot. Demonstrations are first recorded as rosbags and then later converted into TFRecords to train the CNNs. The TRM model is trained from input data containing the name and start and end times of the events in the BI.

Operating the robot can be performed using the Wizard of Oz interface provided [here](https://github.com/AssistiveRoboticsUNH/deep_reinforcement_abstract_lfd/tree/master/interface). The interface can be opened using the following commands in separate terminals

```
roslaunch nao_bringup nao_full_py.launch
ros itbn_lfd itbn_lfd.launch
```

![Robot Interface](misc/interface.png)
  
The following buttons perform the following operations:

Action Functions (Blue)
- **Command:** delivers the *Discriminative Stimuli* (SD)
- **Prompt:** executes the *Prompt* (PMT) action
- **Reward:** executes the *Reward* (REW) action
- **Abort:** executes the *End Session* (END) action

Recording Functions (Green)
- **Start Record:** starts recording observations of the BI
- **Stop Record:** stops recording observations of the BI and outputs the generated rosbag to the "~/bag" directory

Stance Functions (Magenta)
- **Stand:** places the robot in a standing stance
- **Rest:** places the robot in a resting/crouching stance

Utility Functions (Yellow)
- **Angle Head:** angles the robot's head down so that the camera is focused on the participant
- **Toggle Life:** disables autonomous life

Autonomous Functions (Black)
- **Run:** has the robot deliver the learned BI autonomously.

When delivering the SD and PMT the robot will greet the participant by the name listed in the textbox. A live feed of what the robot observes is displayed in the interface and a clock displaying the current time (minutes and seconds) is provided for operations that require timing on the part of the system operator. 

To record training examples begin by selecting 'Start Record', then perform the desired function (e.g. the SD action followed by an observation period of several seconds, the REW action, and then the END action). Once the entire interaction has been observed select the 'Stop Record' button to generate a rosbag file (extension .bag) in the "~/bag" directory described in the installation instructions.

Once a collection of rosbag demonstrations have been recorded the files can be converted to TFRecords using

```
# file located in the /src/itbn_classifier/tools/ directory
python generate_tfrecord_from_rosbag.py
```

Training the TRM
--------------------
The ITBN-based TRM uses input files in which the name, start, and end times of each event in the intervention are listed. The ITBN model was implemented using a modified version of the pgmpy library that is included in this repository.

A TRM can be trained by creating a python script with the following lines:

```
# instantiate a new ITBN model
model = ITBN()

# learn nodes from data. Data is a pandas dataframe object. 
model.add_nodes_from(data)

# learn temporal relations from data
model.learn_temporal_relationships(data)

# learn model structure from data
hc = HillClimbSearchITBN(data, scoring_method=BicScore(data))
model = hc.estimate(start=model)

# learn model parameters
model.fit(data)

# learn observation nodes
model.add_edges_from(<set of observation node and model node tuples>)
model.add_cpds(<set of cpds for the observation nodes>)

# (optional) outputs resulting network to a png file
model.draw_to_file(<output directory>, include_obs=True)

# (optional) outputs resulting network to a nx file
nx.write_gpickle(model, <output directory>)
``` 

Training the CNNs
--------------------

The CNN models must be trained separately, this can be achieved by running the following command

```
# file is contained in the src/itbn_classifier/<CNN model> directory
python <model name>_trainer.py
```

The trainer will begin optimizing the network and will output a partially trained model every 5,000 iterations. The finalized network values will be output in the directory suffixed by "_final". 

Once the trained network can be evaluated by executing

```
# file is contained in the src/itbn_classifier/<CNN model> directory
python <model name>_validator.py
```

Execution of Autonomous System
--------------------


The automated BI can be executed by opening the WoZ interface and the action selector. The the autonomous BI is delivered by clicking the *Start* button or pressing the left bumper on the robot.

```
# file is contained in the src/ directory
python itbn_action_selector.py

roslaunch nao_bringup nao_full_py.launch
roslaunch itbn_lfd itbn_lfd.launch
```

Dependencies
=============
The following libraries are used by this application:
- [Tensorflow](https://www.tensorflow.org/) - Deep network library
- [NumPy](http://www.numpy.org/)
- [OpenCV2](https://opencv.org/) - Open Source image processing library
- [Librosa](https://librosa.github.io/librosa/index.html) - Music and audio analysis library
- [ROS Indigo](http://wiki.ros.org/) - Robot Operating System
- [ROS naoqi](https://github.com/ros-naoqi/nao_robot.git) - ROS interface for NAO
- [Qt](https://www.qt.io/) - GUI

Acknowledgements
=============

We borrowed code from several sources for this project:

- [Spectral Subtraction](https://github.com/tracek/Ornithokrites.git): Used during the pre-processing of the audio feed. 
- [pgmpy](https://github.com/pgmpy/pgmpy): Probabilistic graphical model library that was used to implement the ITBN model.