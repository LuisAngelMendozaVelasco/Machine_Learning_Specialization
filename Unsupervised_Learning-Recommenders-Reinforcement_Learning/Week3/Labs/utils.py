import base64
import random
from itertools import zip_longest

import imageio
import IPython
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import tensorflow as tf
from statsmodels.iolib.table import SimpleTable


SEED = 0              # seed for pseudo-random number generator
MINIBATCH_SIZE = 64   # mini-batch size
TAU = 1e-3            # soft update parameter
E_DECAY = 0.995       # ε decay rate for ε-greedy policy
E_MIN = 0.01          # minimum ε value for ε-greedy policy


random.seed(SEED)


def get_experiences(memory_buffer):
    experiences = random.sample(memory_buffer, k=MINIBATCH_SIZE)
    states = tf.convert_to_tensor(np.array([e.state for e in experiences if e is not None]),dtype=tf.float32)
    actions = tf.convert_to_tensor(np.array([e.action for e in experiences if e is not None]), dtype=tf.float32)
    rewards = tf.convert_to_tensor(np.array([e.reward for e in experiences if e is not None]), dtype=tf.float32)
    next_states = tf.convert_to_tensor(np.array([e.next_state for e in experiences if e is not None]),dtype=tf.float32)
    done_vals = tf.convert_to_tensor(np.array([e.done for e in experiences if e is not None]).astype(np.uint8),
                                     dtype=tf.float32)
    return (states, actions, rewards, next_states, done_vals)


def check_update_conditions(t, num_steps_upd, memory_buffer):
    if (t + 1) % num_steps_upd == 0 and len(memory_buffer) > MINIBATCH_SIZE:
        return True
    else:
        return False
    
    
def get_new_eps(epsilon):
    return max(E_MIN, E_DECAY*epsilon)


def get_action(q_values, epsilon=0):
    if random.random() > epsilon:
        return np.argmax(q_values.numpy()[0])
    else:
        return random.choice(np.arange(4))
    
    
def update_target_network(q_network, target_q_network):
    for target_weights, q_net_weights in zip(target_q_network.weights, q_network.weights):
        target_weights.assign(TAU * q_net_weights + (1.0 - TAU) * target_weights)
    

def plot_history(reward_history, rolling_window=20, lower_limit=None,
                 upper_limit=None, plot_rw=True, plot_rm=True):
    
    if lower_limit is None or upper_limit is None:
        rh = reward_history
        xs = [x for x in range(len(reward_history))]
    else:
        rh = reward_history[lower_limit:upper_limit]
        xs = [x for x in range(lower_limit,upper_limit)]
    
    df = pd.DataFrame(rh)
    rollingMean = df.rolling(rolling_window).mean()

    plt.figure(figsize=(10,7), facecolor='white')
    
    if plot_rw:
        plt.plot(xs, rh, linewidth=1, color='cyan')
    if plot_rm:
        plt.plot(xs, rollingMean, linewidth=2, color='magenta')

    text_color = 'black'
        
    ax = plt.gca()
    ax.set_facecolor('black')
    plt.grid()
#     plt.title("Total Point History", color=text_color, fontsize=40)
    plt.xlabel('Episode', color=text_color, fontsize=30)
    plt.ylabel('Total Points', color=text_color, fontsize=30)
    yNumFmt = mticker.StrMethodFormatter('{x:,}')
    ax.yaxis.set_major_formatter(yNumFmt)
    ax.tick_params(axis='x', colors=text_color)
    ax.tick_params(axis='y', colors=text_color)
    plt.show()
    
    
def display_table(initial_state, action, next_state, reward, done):

    action_labels = ["Do nothing", "Fire right engine", "Fire main engine", "Fire left engine"]
    
    # Do not use column headers
    column_headers = None

    with np.printoptions(formatter={'float': '{:.3f}'.format}):
        table_info = [("Initial State:", [f"{initial_state}"]),
                      ("Action:", [f"{action_labels[action]}"]),
                      ("Next State:", [f"{next_state}"]),
                      ("Reward Received:", [f"{reward:.3f}"]),
                      ("Episode Terminated:", [f"{done}"])]

    # Generate table  
    row_labels, data = zip_longest(*table_info)
    table = SimpleTable(data, column_headers, row_labels)

    return table


def embed_mp4(filename):
    """Embeds an mp4 file in the notebook."""
    video = open(filename,'rb').read()
    b64 = base64.b64encode(video)
    tag = '''
    <video width="840" height="480" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4">
    Your browser does not support the video tag.
    </video>'''.format(b64.decode())
    return IPython.display.HTML(tag)
        
        
def create_video(filename, env, q_network, fps=30):
    with imageio.get_writer(filename, fps=fps) as video:
        done = False
        state = env.reset()
        frame = env.render(mode="rgb_array")
        video.append_data(frame)
        while not done:    
            state = np.expand_dims(state, axis=0)
            q_values = q_network(state)
            action = np.argmax(q_values.numpy()[0])
            state, _, done, _ = env.step(action)
            frame = env.render(mode="rgb_array")
            video.append_data(frame)

def generate_rewards(num_states, each_step_reward, terminal_left_reward, terminal_right_reward):

    rewards = [each_step_reward] * num_states
    rewards[0] = terminal_left_reward
    rewards[-1] = terminal_right_reward
    
    return rewards 

def generate_transition_prob(num_states, num_actions, misstep_prob = 0):
    # 0 is left, 1 is right 
    
    p = np.zeros((num_states, num_actions, num_states))
    
    for i in range(num_states):        
        if i != 0:
            p[i, 0, i-1] = 1 - misstep_prob
            p[i, 1, i-1] = misstep_prob
            
        if i != num_states - 1:
            p[i, 1, i+1] = 1  - misstep_prob
            p[i, 0, i+1] = misstep_prob
        
    # Terminal States    
    p[0] = np.zeros((num_actions, num_states))
    p[-1] = np.zeros((num_actions, num_states))
    
    return p

def calculate_Q_value(num_states, rewards, transition_prob, gamma, V_states, state, action):
    q_sa = rewards[state] + gamma * sum([transition_prob[state, action, sp] * V_states[sp] for sp in range(num_states)])
    return q_sa

def evaluate_policy(num_states, rewards, transition_prob, gamma, policy):
    max_policy_eval = 10000 
    threshold = 1e-10
    
    V = np.zeros(num_states)
    
    for i in range(max_policy_eval):
        delta = 0
        for s in range(num_states):
            v = V[s]
            V[s] = calculate_Q_value(num_states, rewards, transition_prob, gamma, V, s, policy[s])
            delta = max(delta, abs(v - V[s]))
                       
        if delta < threshold:
            break
            
    return V

def improve_policy(num_states, num_actions, rewards, transition_prob, gamma, V, policy):
    policy_stable = True
    
    for s in range(num_states):
        q_best = V[s]
        for a in range(num_actions):
            q_sa = calculate_Q_value(num_states, rewards, transition_prob, gamma, V, s, a)
            if q_sa > q_best and policy[s] != a:
                policy[s] = a
                q_best = q_sa
                policy_stable = False
    
    return policy, policy_stable


def get_optimal_policy(num_states, num_actions, rewards, transition_prob, gamma):
    optimal_policy = np.zeros(num_states, dtype=int)
    max_policy_iter = 10000 

    for i in range(max_policy_iter):
        policy_stable = True

        V = evaluate_policy(num_states, rewards, transition_prob, gamma, optimal_policy)
        optimal_policy, policy_stable = improve_policy(num_states, num_actions, rewards, transition_prob, gamma, V, optimal_policy)

        if policy_stable:
            break
            
    return optimal_policy, V

def calculate_Q_values(num_states, rewards, transition_prob, gamma, optimal_policy):
    # Left and then optimal policy
    q_left_star = np.zeros(num_states)

    # Right and optimal policy
    q_right_star = np.zeros(num_states)
    
    V_star =  evaluate_policy(num_states, rewards, transition_prob, gamma, optimal_policy)

    for s in range(num_states):
        q_left_star[s] = calculate_Q_value(num_states, rewards, transition_prob, gamma, V_star, s, 0)
        q_right_star[s] = calculate_Q_value(num_states, rewards, transition_prob, gamma, V_star, s, 1)
        
    return q_left_star, q_right_star


def plot_optimal_policy_return(num_states, optimal_policy, rewards, V):
    actions = [r"$\leftarrow$" if a == 0 else r"$\rightarrow$" for a in optimal_policy]
    actions[0] = ""
    actions[-1] = ""
    
    fig, ax = plt.subplots(figsize=(2*num_states,2))

    for i in range(num_states):
        ax.text(i+0.5, 0.5, actions[i], fontsize=32, ha="center", va="center", color="orange")
        ax.text(i+0.5, 0.25, rewards[i], fontsize=16, ha="center", va="center", color="black")
        ax.text(i+0.5, 0.75, round(V[i],2), fontsize=16, ha="center", va="center", color="firebrick")
        ax.axvline(i, color="black")
    ax.set_xlim([0, num_states])
    ax.set_ylim([0, 1])

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_title("Optimal policy",fontsize = 16)

def plot_q_values(num_states, q_left_star, q_right_star, rewards):
    fig, ax = plt.subplots(figsize=(3*num_states,2))

    for i in range(num_states):
        ax.text(i+0.2, 0.6, round(q_left_star[i],2), fontsize=16, ha="center", va="center", color="firebrick")
        ax.text(i+0.8, 0.6, round(q_right_star[i],2), fontsize=16, ha="center", va="center", color="firebrick")

        ax.text(i+0.5, 0.25, rewards[i], fontsize=20, ha="center", va="center", color="black")
        ax.axvline(i, color="black")
    ax.set_xlim([0, num_states])
    ax.set_ylim([0, 1])

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_title("Q(s,a)",fontsize = 16)

def generate_visualization(terminal_left_reward, terminal_right_reward, each_step_reward, gamma, misstep_prob):
    num_states = 6
    num_actions = 2
    
    rewards = generate_rewards(num_states, each_step_reward, terminal_left_reward, terminal_right_reward)
    transition_prob = generate_transition_prob(num_states, num_actions, misstep_prob)
    
    optimal_policy, V = get_optimal_policy(num_states, num_actions, rewards, transition_prob, gamma)
    q_left_star, q_right_star = calculate_Q_values(num_states, rewards, transition_prob, gamma, optimal_policy)
    
    plot_optimal_policy_return(num_states, optimal_policy, rewards, V)
    plot_q_values(num_states, q_left_star, q_right_star, rewards)