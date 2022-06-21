---
layout: post
title: Training a real robot to push objects (1/3)
date: 2022-06-20 15:09:00
description: In this series I document how I taught our UR3e robot to push objects to a target position. Part one focuses on creating the learning environment for in-vivo learning.
tags: robotics deepRL UR3e
categories: [robot learning]
comments: true
hide: true

img: assets/img/blog/ur-state-pusher/robot-primitive.gif
---

Deep Reinforcement Learning has made tremendous progress over the past years. Algorithmic advances have resulted in more stable learning,  less fragility to hyperparameter settings and some progress in data-efficiency. Together with the increase in computing power, this has enabled roboticist to tackle more challenging problems.  On the other hand, the field of Robot Learning is by far not as mature as say Computer Vision with Deep Learning, where relatively little knowledge is required to apply state-of-the-art techniques to a custom  dataset of interest. Using RL to train a real robot on a manipulation task remains a challenging endeavor.

In this tutorial I will describe how to set up and train a real robot (UR3e) to perform the simple task of **pushing an object to a target position**. You can see the physical setup below.

{% include figure.html path="assets/img/blog/ur-state-pusher/setup.jpg" class="img-fluid mx-auto d-block rounded z-depth-1" %}


The main goal of this post was to document this process for future me. This implies that the post might be a little rough around the edges. Nontheless, I hope this will provide some insights along the way that can help others as well.  Experienced roboticists (that work on robot learning) will not learn anything new here. However, **the nuances of real-world robot learning  are not always easily available to newcomers in the field as they are scattered across many papers and/or blogposts or just not written down at all**.   The tutorial assumes basic knowledge of RL. 

Training a robot to perform a task (with RL) has two main components. First you need to define the task at hand: What are the actions the robot can take?  What is the input to the control policy (i.e. the observations)? How do we tell the robot it is doing well? In a second stage, you need to select a learning  algorithm, find good hyperparameters and train the robot on the task.  
  
In the remainder of this blog post I will discuss the first part. A second blog post will focus on the learning part. In order to reduce the complexity further, I have made some design choices to simplify the problem further. **These choices make the learning environment less "generic" but are necessary to limit the number of interactions during training**. In the third part, I will discuss a few options to move back towards more generic formulations for robot learning environments and for dealing with the increased number of required samples that comes with undoing these simplifications.
   
## Defining the Task  
In this part we will dive into defining and implementing the task -pushing a disc to a target position- that we want the robot to solve. I will briefly introduce the mathematical formalism that is used in RL for this task specification, which will bring us to the design decisions we have to make, and I will introduce the commonly used Gym interface.
After that I will list some issues specific to real-world learning that will influence the design choices.  Finally we will create the learning environment.

### MDP Formalism  
in RL, tasks are defined as as [Markov  Decision Process (MDP)](https://en.wikipedia.org/wiki/Markov_decision_process).
An MDP is often defined as a 5-tupe

 $$\langle S,A,R,P,\rho_0 \rangle.$$

$$S$$ is the state space, $$A(s)$$ the (state-conditioned) action space, $$R(s,a,s')$$ is the reward fucntion, $$P(s'| s,a)$$ is the transition function and $$s_0 \sim \rho_0$$ is the initial state distribution. So to specify the task we want to robot to learn we need to define 
- a state space,
-  action space,
-  reward function,
- and initial distribution.

 The transition function is taken care of by the laws of physics (and a couple layers of abstraction). 

### Gym Interface
The dominant way of implementing the resulting MDP and providing the learning algorithm with an interface to the environment is through a [Gym interface](https://www.gymlibrary.ml/). 

Interacting with a Gym environment follows the iconic agent-environment interaction figure by Sutton & Barto and goes as follows:

 ```python
env = YourEnvironment()
observation = env.reset()
for _ in range(1000):
   action = policy(observation)  # User-defined policy function
   observation, reward, done, info = env.step(action)
   if done:
      observation, info = env.reset(return_info=True)
env.close()
```

so mapping to the previous section about task definition:
- define observation space -> implement function to obtain `observation`from the environment, hence implicitly defining the observation space.
- define action space -> implement function that makes the robot execute the action if it is part of $A(s)$ , hence implicitly defining the action space.
- define reward function -> implement function to calculate reward based
- define initial state distribution-> implement a reset function that brings the environment to a state according to the initial state distribution

One might argue that the control stack of the robot is part of the transition function, I've decided to list it as part of the action space implementation but I'm not so sure this is "correct", feel free to let me know if you feel that this should be part of the transition function.

### Difficulties with real-world robot learning  
There are some aspects that make RL for real-world robot learning trickier than say learning to play a video game.  These issues are well-known and a.o. discussed in an excellent paper called [How to Train Your Robot with  DeepRL - Lessons Learned](https://arxiv.org/pdf/2102.02915.pdf) .
  
A first issue is **how to obtain state information**.   In games or robotics simulators, we have access to the entire state of the environment but this is no longer true in a real-world setting. In general there are two options to tackle this:
1. Instrumentation of the environment.  
2. Providing the agent with sensor data instead of the full state. As the agent now does no longer have access to the exact state, we call this setting a Partial-Observable MDP (POMDP).  RGB(D) cameras are by far the most used sensing modality for robots due to their high information density and low cost.
  
Related to this is **how to compute the rewards for the agent**. The reward function $$r(s,a,s')$$ depends on the environment state, which is no longer available.  Again one could instrument the environment or try to come up with heuristics such as the difference in sensor space between the goal and current state. Another approach is learning a reward function  based on some sensor inputs that are available. The reward can be learned together with the policy (Inverse RL) or upfront from examples of the desired manipulation behavior.  
  
A third issue is related to autonomous operation: we want to minimize the number of human inventions during training.  An assumption of the RL framework is that we can **reset the environment** at the end of an *episode*. Although this is indeed easily done in games of simulators, it is not the case in the real-world where it requires instrumenting the environment or scripting reset policies for the robot to execute. Furthermore, some manipulation tasks are irreversible, think about preparing food.
  
Finally, in real-world settings **safety** is also an important factor. We do not want the robot to harm humans or to destroy itself or its environment. This is especially relevant in the beginning of training where the behavior will typically be random. 
  
### Designing the environment  
Now that we have summed up the decisions we need to make and  discussed the difficulties of real-world robot learning, we can create a gym environment. 
#### Observations 
To get the observations, we either need to instrument the environment to obtain the relevant state or use sensor readings as observation. I chose to go for the first option as it drastically reduces the input dimensionality of the RL agent.
The relevant state of the environment consists of 
-  the pose of the object,
-  the goal pose
- and the state of the robot. 

Typically we  want to express all poses relative to the robot's base frame or EEF frame. Here I chose to use the base frame, as this best suits the action space I had in mind (this process of creating the environment is by no means linear, even though it is layed out linearly in this post..)

To communicate with the UR3e robot, I use the [ur-rtde](https://sdurobotics.gitlab.io/ur_rtde/index.html) library, which is a lightweight wrapper around the RTDE interface of the robot. We can easily get the joint states as well as the End-Effector (EEF) pose.

The relevant pose of the object reduces to a 2D position as the object that I used is a disk, which is symmetric around the z-axis.
To obtain the position I first used a fiducial marker (aruco/charuco/..) to determine the pose of the camera w.r.t. that marker. I then measured the translation from that marker (aligning the marker manually to the robot frame) to the robot base on the table by moving the robot to the marker and querying the EEF pose.
Combining these two transformations gives us the pose of the camera w.r.t. the robot.

*This process of getting the camera-robot transform is called hand-eye calibration. There exist tools to solve this such as [Moveit_calibration](https://github.com/ros-planning/moveit_calibration) (not yet ported to ROS2) or [Opencv](). These methods would probably result in better performance as you do not need to eyeball the rotational alignment of the marker with the robot frame, although this is easy on our robot setup due to its layout.*

Now we can determine the pose of an object on the table by determining the position of the object on the image plane of the camera and then projecting the resulting image ray onto the table (or more precisely on the plane at the height of the object, parallel to the table). To accomplish this I used a [Zed2i](https://www.stereolabs.com/zed-2i/) camera and its python API as well as Opencv. The code is available in a [standalone repo](https://github.com/tlpss/camera-toolkit). Below you can find a sketch of the different transforms and steps to acquire them.
{% include figure.html path="assets/img/blog/ur-state-pusher/state-estimation.png" class="mx-auto d-block rounded z-depth-1" %}
I've used a 5x5 aruco marker of 5cm for the object and a 6x6 Aruco marker of 10cm for the camera calibration (bigger is better).  Make sure to use the highest resolution (2K in case of the Zed camera) to limit the reprojection errors. The resulting estimation of the positions were mostly within 1cm of the object's true position, which seems accurate enough for this task, although I'm rather sure that you could increase this accuracy by using a (large) charuco board for determining the camera extrinsics or by using a proper *hand-eye calibration* tool. One obvious limitation of this method is that it requires the marker to be visible from the camera at all times, however you can easily see how the robot could occlude the marker whilst interacting with the object. This is something that we will come back to during the action space design.


*Note that the Zed camera is actually a depth camera, so instead of manually reprojecting to a plane parallel to the table we could sample the depth image to get a 3D location. This proved however to be less accurate as we know the exact height in this case. Yet another option would be to use stereoview and find the intersection between the rays, but again if you know the exact height of the object there is no need to do this.*

The target position is typically known to the agent. Furthermore, it was fixed during this experiment to reduce complexity. Nonetheless, it is still included in the observation as my intuition was that this would make the task easier to learn for the network since it would not have to internalize the goal position.

As we will soon discuss, I chose my actions in such a way that the robot does not need to know its own position, so the observation space actually reduces to $$[\hat{x}_{object},\hat{y}_{object},x_{goal},y_{goal}]$$. 

This is implemented in the following snippet (with an additional try-catch on the aruco marker detection for robustness):
```python
def _get_observation(self) -> List:  
    """  
	gym observation 
	"""  
	object_position = self._get_object_position()  
  
    obs = object_position.tolist()  
    obs.extend(self.goal_position)  
    return obs
   
def _get_object_position(self):  
    """  
    Get object pose by reading in image frame, detecting keypoint and reprojecting it on the Z = object_height plane.
    """  
    img = self.camera.get_mono_rgb_image()  
    img = self.camera.image_shape_torch_to_opencv(img)  
  
    for i in range(3):  
        image_coords = get_aruco_marker_coords(img, cv2.aruco.DICT_5X5_250)  
        if image_coords is None:  
            if i == 2:  
                raise ValueError("Could not detect the aruco marker after 3 attempts. Is the marker visible?")  
            logger.error("could not detect object aruco marker")  
            time.sleep(1.0)  
        else:  
            break  
	aruco_frame_coords = reproject_to_ground_plane(  
        image_coords, self.cam_matrix, self.aruco_in_camera_transform, height=URPushState.object_height  
    )  
  
    robot_frame_coords = aruco_frame_coords + URPushState.robot_to_aruco_translation  
  
    if not self.position_is_in_object_space(robot_frame_coords[0][:2]):  # pushed object outside of goal_space  
	    raise ValueError(  
            "Object is outside of workspace.. this should not happen and is probably caused"  
			" by inaccurate object position which makes the primitive motion behave unexpected."  )  
  
    return robot_frame_coords[0][:2]
```

#### Episode Termination
Next to the observation and reward, the `step` function also returns whether the episode has terminated or not. This allows the learning algorithm to reset the environment as well as to do some bookkeeping for calculating the expected returns or storing the experiences in a replay buffer.
As we consider episodic RL, the episode terminates either if the goal is reached or if the maximum amount of steps has been reached:
```python
@staticmethod  
def _is_episode_finished(n_steps_in_episode, distance_to_target: float):  
    done = distance_to_target < PushStateConfig.goal_l2_margin  # goal reached  
	done = done or n_steps_in_episode >= PushStateConfig.max_episode_steps  
    return done
```
#### Actions  
Designing the action space was the most tricky part of this project. There are a number of aspects to keep in mind here:
- For the state estimation, the object should not be occluded after a step has been taken (or we should work around occlusions using multiple aruco markers on different sides etc..).
- The result of executing an action should be such that the robot can still reach the goal state from that state (Or the environment can at least be reset to an arbitrary initial state) to enable autonomous data collection.
- Although one of the defining aspects of RL is its ability to deal with long-horizon tasks with delayed feedback (sparse rewards), in practice it tends to work a lot better with short-horizon tasks and immediate feedback (dense rewards). This translates into a trend towards formulating robotic (manipulation) problems as short-horizon tasks by using pre-defined motion primitives. An example is grasping objects, where one could have the policy move the EEF (long-horizon) or have it determine a grasp location (short-horizion). 

Based on these observations, I chose to use a "push motion primitive": The robot will always push the object, but the agent has to decide how far and in what direction: $$(\theta,l)$$. In between actions, the robot can lift its EEF, hence avoiding occlusions of the aruco marker. 
{% include figure.html path="assets/img/blog/ur-state-pusher/motion-primitive.png" class="img-fluid.center mx-auto d-block rounded z-depth-1 zoomable=true " %}


The robot's workspace is defined by its mechanical design as well as by the control stack that is used for the robot: as I am not using a motion planner, the robot is simply moving linearly in cartesian space. However, this can easily result in collisions with the robot base or other links for poses that are close to the robot base. So I had to manually constrain the workspace of the robot to avoid self-collisions. After some experimenting (and collisions), I arrived at the following "safe" workspace for the robot:
{% include figure.html path="assets/img/blog/ur-state-pusher/robot-workspace.png" class="img-fluid  mx-auto d-block rounded z-depth-1 " %}
It is important to note that it is not always possible to design such "collision-safe action spaces". In environments with lots of objects, this would be very hard (and constantly changing). In such scenarios you should consider using an explicit motion planner ([Moveit2](https://github.com/ros-planning/moveit2)) if you know the locations of the obstacles. If you do not know the locations, your robot will inevitably make undesired contacts with the environment and you should consider using an (indirect) force controller at the low-level to avoid damage. (Such active-compliant controllers are often even better-suited for contact-rich tasks (e.g. insertion tasks) anyway. See [this paper](https://arxiv.org/pdf/1906.08880.pdf) for example).


To make sure the robot would be able to push the object in at least one direction during the next action, the object's end position should be not closer than the radius of the object + the radius of the EEF to the border of the robot's workspace (defined w.r.t. the toolpoint center or the center of the EEF). Actions that do not meet this constraint are considered invalid and will not be executed by the robot, although the episode will not finish. 
{% include figure.html path="assets/img/blog/ur-state-pusher/object-space.png" class="img-fluid mx-auto d-block rounded z-depth-1 " %}

Note that we can predict the outcome of the primitive for this simple task (we know the dynamics model), although this usually is not the case. We could then stop the action when the robot reaches the edge of the its workspace, which we can always detect at runtime if we have access to the proprioceptive state.

```python
def _execute_primitive_motion(self, angle: float, length: float) -> bool:  
    """  
 Do the "motion primitive": a push along the desired angle and over the specified distance  
 To avoid collisions this is executed as: - move to pre-start pose - move to start pose - push - move to post-push pose - move to "out-of-sight" pose (home)  
  
 angle: radians in [0,2Pi] lenght: value in [0, max_pushing_distance]  
 Returns True after executing if the motion was allowed (start robot position is in the robot workspace, end object position is in the block workspace) and False otherwise. """  
 
 # get current position of the object
	current_object_position = self._get_object_position()  
  
    # determine primitive motion start and endpoint  
    push_direction = np.array([np.cos(angle), np.sin(angle)])  
  
    block_start_point = current_object_position  
    robot_start_point = block_start_point - push_direction * (  
        URPushState.object_radius + URPushState.robot_flange_radius + URPushState.robot_motion_margin  
    )  
    block_end_point = block_start_point + length * push_direction  
    robot_end_point = block_end_point - push_direction * (  
        URPushState.object_radius + URPushState.robot_flange_radius  
    )  
  
    logger.debug(f"motion primitive: (angle:{angle},len:{length} ) - {block_start_point} -> {block_end_point}")  
    # calculate if the proposed primitive does not violate the robot's workspace  
  
  if not self._position_is_in_workspace(robot_start_point):  
        logger.debug(f"invalid robot startpoint for primitive {block_start_point} -> {block_end_point}")  
        return False  
 if not self.position_is_in_object_space(block_end_point, margin=0.01):  
        logger.debug(f"invalid  block endpoint for primitive {block_start_point} -> {block_end_point}")  
        return False  
  
  # move to start pose  
  self._move_robot(robot_start_point[0], robot_start_point[1], URPushState.robot_eef_z + 0.05)  
    # execute  
  self._move_robot(robot_start_point[0], robot_start_point[1], URPushState.robot_eef_z)  
    self._move_robot(robot_end_point[0], robot_end_point[1], URPushState.robot_eef_z)  
  
    # move back to home pose  
  self._move_robot(robot_end_point[0], robot_end_point[1], URPushState.robot_eef_z + 0.05)  
    self._move_robot(URPushState.home_pose[0], URPushState.home_pose[1], URPushState.home_pose[2])  
    return True
 ```
#### Resets  
The reset of our environment is fairly simple: we should move the object to a random position within the object space and we're done. This initial state of our MDP should be uniformly distributed over the object space (with a little extra margin to accomodate for state estimation errors), so I simply used rejection sampling to obtain them:
```python
@staticmethod  
def get_random_object_position(goal_position: np.ndarray) -> np.ndarray:  
    """  
	Brute-force sample positions until one is in the allowed object workspace """  
	while True:  
		 x = np.random.random() - 0.5  
		 y = np.random.random() * 0.25 - 0.45  
		 position = np.array([x, y])  
	     logger.debug(f"proposed object reset {position}")  
	     if (  
		      URPushState._position_is_in_workspace(  
		          position,  
	               margin=1.1  
					  * (URPushState.object_radius + URPushState.robot_flange_radius + URPushState.robot_motion_margin),  
			  )  
	      ):  
	            return position
```

#### Rewards  
We already know the relevant state of the environment (the pose of the object and the goal position) and hence we can simply formulate the reward as $r(s,a,s')$ instead of having to define it in terms of the available sensor readings. 

How to come up with an appropriate reward function? This is more of an art than a scientific process, but there are some guidelines to follow. The simplest reward is 1 if the goal is reached and 0 elsewhere. However, these sparse rewards create very hard exploration problems, as the agent needs to solve the complete task by accident (it is taking random actions at that time) before it ever receives any feedback... 
There exist techniques such as [Hindsight Experience Replay]() or [Curriculum learning methods](https://libstore.ugent.be/fulltxt/RUG01/003/014/816/RUG01-003014816_2021_0001_AC.pdf) that try to overcome this, but in general it is more pragmatical to add more information to the reward signal.  
An obvious candidate is often to include heuristics for the distance towards the goal. In order to make sure the reward engineering does not influence the optimal solution of the MDP, you typically want the reward components to be [potential-based functions](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf) . Even then, combining different components can be tricky as you need to carefully balance them. More about combining losses can be found in [this excellent blog](https://www.engraved.blog/why-machine-learning-algorithms-are-hard-to-tune/) post by Jonas Degraeve.

I settled for the following reward function:

 $$ - \frac{current\_distance}{max\_distance}.$$

Note that this is not a potential-based function, but the agent is discouraged from taking any unnecessary action as the reward is negative at each timestep. Also note that I do not additionally punish the robot for taking invalid actions, although I am not sure if this would be beneficial or not. If you do, feel free to let me know! You can see how this reward function provides feedback at every single step, making it a lot easier for the agent to learn.

We are now done with the environment setup. The implementation of the `step` function now looks as follows:
```python
def step(self, action: np.ndarray) -> Tuple[Any, float, bool, dict]:  
    """  
 performs action, returns observation, reward, is episode finished?, info dict (empty) """  
 
    self.n_steps_in_episode += 1  
  
    normalized_angle, normalized_length = action  
    angle = normalized_angle * 2 * np.pi  
    length = normalized_length * PushStateConfig.max_pushing_distance  
    logger.debug(f"taking action ({angle},{length})")  
    valid_action = self._execute_primitive_motion(angle, length)  
  
    new_observation = self._get_observation()  
    new_object_position = self._get_object_position()  
  
    distance_to_target = np.linalg.norm(new_object_position - self.goal_position)  
    done = self._is_episode_finished(self.n_steps_in_episode, distance_to_target)  
  
    # determine reward  
    reward = self.calculate_reward(valid_action, distance_to_target)  
  
    return new_observation, reward, done, {}
   ```
The next two sections briefly discuss logging and testing, two important yet often overlooked topics..

### Logging 
Logging is important for every robot system. Either to figure out what went wrong or to simply follow along with what the robot is doing.  In this environment I used the built-in [python logging](https://realpython.com/python-logging/) framework. I logged resets, actions taken, etc.. During testing these were printed to the main console for convenience but during training we can write them to a file to avoid cluttering the CLI. The following snippet shows how to customize what happens to the logged information: 

```python 
if __name__ == "__main__":
	# configure logging
	# https://realpython.com/python-logging/#basic-configurations  
	logging.basicConfig(  
	    filename="learn.log",  
	    level=logging.INFO,  
	    filemode="a",  
	    format="%(asctime)s - %s(name) - %(levelname)s - %(message)s",  
	    datefmt="%d-%b-%y %H:%M:%S",  
	)
	
	# learning stuff
	# ...
```
The logs proved to be invaluable both during testing and training, as it was not always easy to see when the robot was doing a "stupid action" or was simply resetting the environment.
I also found it convenient to display the logging information on a screen next to the robot. For a more involved project I would take some time to create a dedicated information window next to the robot, containing information such as: the episode step, the real-world training time, the action taken etc.

### Testing  
At this point we have an environment that we can interact with. Before starting to learn there a a few sanity checks we can (and should) perform.
{% include figure.html path="assets/img/blog/ur-state-pusher/robot-primitive.gif" class="mx-auto d-block rounded z-depth-1 " %}

- Although I took care of designing the action space so that the object's new position was still reachable by the robot, it is a good idea to test what happens when performing random actions: 

```python

for i in range(n_episodes):  
    obs = env.reset()  
    done = False  
    while not done:  
            angle = np.random.random()  
            length = np.random.random()  
            action = [angle, length]  
            obs, reward, done, info = env.step(action)  
            print(obs, reward, done)
```
The robot should be able to continue interacting with the environment. However, after a few steps an error was thrown as the object got out of the object space. After some messing around I found out this was due to imperfect state estimations, resulting in drift during the execution of the motion primitive. Luckily this seemed to be a rare event.

- The second check involves scripting a policy and seeing how this one behaves. Usually this won't be possible of course, but if you can it is very insightful and a valuable baseline to compare the performance of the learned policy later on.  For the task we are considering, and given the state information it is easy to script a policy (which we already did for resetting the object):

```python
def _calculate_optimal_primitive(self, position) -> Tuple[float, float]:  
    """  
	Calculate the optimal angle and length assuming perfect 	observation. """ 
	current_position = self._get_object_position()  
    vector = position - current_position  
    angle = np.arctan2(vector[1], vector[0])  
    length = np.linalg.norm(vector)  
  
    # from [-pi,pi ] to [0,2pi] for normalization  
	if angle < 0:  
        angle += 2 * np.pi  
    return angle, length
```

This policy performed rather well, although it was already notable that state estimation errors resulted in suboptimal behavior.


So now that we have defined the task and tested the implementation, we are ready to start learning! This will be discussed in part 2! The complete code can be found [here](https://github.com/tlpss/ur-gym/blob/main/ur_gym/pusher/state_pusher.py) for reference. 
