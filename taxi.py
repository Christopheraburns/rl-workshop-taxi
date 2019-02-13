import gym
import numpy as np
import random
import argparse
import sys
import time
import datetime

env = None                      # Our Gym Environment (default to Taxi)
r = None                   # The qtable that stores the exploration choices
qtableready = False             # Flag to determine if the qtable has been updated

# Hyper Params
total_train_episodes = 20000    # How many times do we attempt to train on the Taxi environment (known as episode)
total_test_episodes = 10        # Number of episodes to test our qtable
max_steps = 100                 # How many steps do we limit each episode to
current_steps = 0               # Holder for the current step of the current training episode
learning_rate = 0.7             # Learning rate for Bellman equation
gamma = 0.618                   # Gamma rate for the Bellman equation
total_reward = 0                # Holder for the Cumulative reward of the current episode
state = None                    # Current state - OR current 'position' of the Taxi in the environment
epsilontrackers = []            # List to hold each epsilon value change for future viewing
action_size = 0                 # Holder for the number of possible choices at each state
                                # Taxi-v2 has SIX possible discrete, deterministic actions:
                                # 0 - move south
                                # 1 - move north
                                # 2 - move east
                                # 3 - move west
                                # 4 - pickup passenger
                                # 5 - dropoff passenger

state_size = 0                  # Holder for the observable space size (Taxi-v2 has 500 observable states
                                # +---------+       The Taxi environment is a 5x5 grid (125 spaces).
                                # |R: | : :G|       There are 4 "locations" on the grid,
                                # | : : : : |       denoted with letters R, G, B, Y to randomize pickup and Drop-off
                                # | : : : : |       locations. The pickup spot can be anyone of the 4 letters so there
                                # | | : | : |       is in essence 4 versions of the 5x5 grid -- which gives us 500
                                # |Y| : |B: |       (5x5) x 4 spaces.
                                # +---------+

# Exploration Params
epsilon = 1                     # Greedy Epsilon
max_epsilon = 1                 # Maximum Epsilon value
min_epsilon = 0.1               # Minimum epsilon value
decay_rate = .01                # Amount to reduce epsilon value after each episode.


# Args
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--train", required=False, type=bool, default=False,
                help="Shall we train this RL Agent? True for yes, defaults to False")
ap.add_argument("-v", "--verbose", required=False, type=bool, default=False,
                help="Shall we show all the details, or just show the result? True for yes, defaults to False")
ap.add_argument("-q", "--qtable", required=False, type=bool, default=True,
                help="Shall we show the qTable? True for yes, defaults to False")
ap.add_argument("-a", "--actions", required=False, type=bool, default=False,
                help="Shall we show the available actions?  True for yes, defaults to False")
ap.add_argument("-e", "--environment", required=False, type=str, default="Taxi-v2",
                help="Override the default environment Taxi-v2?  \n Options: CartPole-v1 \n Acrobot-v1 /"
                     "\n MountainCar-v0 \n")
ap.add_argument("-r", "--render", required=False, type=bool, default=True,
                help="(Render) Show the steps as the agent trains")
ap.add_argument("-s", "--save", required=False, type=bool, default=True,
                help="(Save) Save the qtable as a numpy .npy file. In format (mmddyyyhhmmss.npy. Default is True")
args = vars(ap.parse_args())


def sendagent():
    global qtable
    global env
    env.reset()
    rewards = []
    steps = []
    try:
        for episode in range(total_test_episodes):
            state = env.reset()
            step = 0
            done = False
            total_rewards = 0

            for step in range(max_steps):
                #only show the first episode
                if episode is 0:
                    env.render()

                action = np.argmax(qtable[state, :])
       # take ^ action with ^ max expected future ^ reward

                new_state, reward, done, info = env.step(action)

                total_rewards += reward

                if done:
                    print("Episode {} complete!".format(episode))
                    print("Completed in {} steps with a reward of {}".format(step, total_rewards))
                    rewards.append(total_rewards)
                    steps.append(step)
                    break
                state = new_state
                if episode is 0:
                    time.sleep(.5)
        env.close()
        print("\n All test episodes complete:")
        print("Average number of steps per episode: {} average reward [score]: {}".
              format(sum(steps)/total_test_episodes, sum(rewards)/total_test_episodes))
    except Exception as err:
        print("Error in sendagent() function: {}".format(err))


def step(howmany):
    global env
    global current_steps
    global state
    global total_reward
    global current_steps
    global qtable
    global qtableready
    global epsilontrackers
    global epsilon

    if howmany is "single":
        action = env.action_space.sample()
        new_state, reward, done, info = env.step(action)
        total_reward = total_reward + reward
        actions = ["move south", "move north", "move east", "move west", "pick up", "drop off"]
        print("Taking action: {}".format(actions[action]))
        print("Action reward: {} cumulative reward is now: {}".format(reward, total_reward))
        print("Action info: {}".format(info))
        print("[Observation] State: {} of {}".format(new_state, state_size))

    elif howmany is "all":
        try:
            for episode in range(total_train_episodes):
                # Reset the environment
                state = env.reset()
                step = 0
                done = False

                for step in range(max_steps):
                    print(f"\r{step} of {max_steps} steps in episode {episode} of {total_train_episodes} total episodes"
                          , end="")
                    sys.stdout.flush()

                    # 3. Choose an action a in the current world state (s)
                    # First we randomize a number
                    exp_exp_tradeoff = random.uniform(0, 1)

                    # If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
                    if exp_exp_tradeoff > epsilon:
                        action = np.argmax(qtable[state, :])
                    # Else doing a random choice --> exploration
                    else:
                        action = env.action_space.sample()

                    # Take the action (a) and observe the outcome state(s') and reward (r)
                    new_state, reward, done, info = env.step(action)

                    # Probably the most important line of this sourcecode
                    # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
                    qtable[state, action] = \
                        qtable[state, action] + learning_rate * (reward + gamma *np.max(qtable[new_state, :]) - qtable[
                                                                                         state, action])
                    # Our new state is state
                    state = new_state

                    # If done (either 100 steps or passenger drop off) : finish episode
                    if done:
                        break

                # capture the current epsilon
                epsilontrackers.append(epsilon)

                # Reduce epsilon (because we need less and less exploration)
                epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        except Exception as err:
            print("Error in step function: {}".format(err))

    qtableready = True
    # Shall we save the qtable for later use?
    if args['save'] is True:
        filename = str(datetime.datetime.now())
        np.save(filename, qtable)
    print("\n****************  Populated QTable   **************")
    print(qtable)
    print("\n")
    prompt()


def prompt():
    global qtableready

    cmdfound = False
    command = input("Press 's' to observe one random step.\n"
                    "Press 't' to run through training.\n"
                    "Press 'a' to send the trained agent to complete the task (must complete  training first).\n")
    if command == 's':
        cmdfound = True
        # Take one step
        step('single')
    if command == 't':
        cmdfound = True
        # Run through the episode
        step('all')
    if command == 'a':
        cmdfound = True
        # Use our existing qtable to solve the episode
        if qtableready:
            sendagent()
        else:
            print("qtable is not ready. You must first populate the qtable by choosing option 't' for training")
    if cmdfound is False:
        print("{} is not an 'S' or an 'R'".format(command))
        prompt()


def start():
    global env
    global qtable
    global state_size
    global action_size
    global state

    try:
        # Prepare the environment
        env = None
        env = gym.make("Taxi-v2")
        action_size = env.action_space.n
        state_size = env.observation_space.n
        state = env.reset()
        print("***** An example environment*****")
        env.render()

        print("Blue = passenger")
        print("Magenta = destination")
        print("Yellow = empty taxi")
        print("Green = full taxi")
        print("Letters = Locations")
        print("\n\n")
        # qTable can be represented by a numpy matrix - with all available states of the observable space
        # as 'rows' and the number of actions per state as 'columns'
        # don't get these reversed, or bad things will happen.
        # set up the qtable

        qtable = np.zeros((state_size, action_size))
        if args["qtable"] == True:
            print("Q-table initialized with {} observable states and {} actions per state".format(
                state_size, action_size))
            print(qtable)
            print("\n")

        prompt()

    except Exception as err:
        print("Error! {}".format(err))


start()


