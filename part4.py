import numpy as np

def prepare_data(file_path):
    """Prepare the raw data for training

    Args:
        file_path (string): input file path for preparation

    Returns:
        dictionary: state_to_idx - a mapping of each state to the index
        dictionary: observation_to_idx - a mapping of each observation to the index
        list: states - a list of unique states
        list: observations - a list of unique observations
        list: data - list of tuples (observation, state)

    """
    with open(file_path, 'r', encoding="utf-8") as file:
        lines = file.readlines()

    states = set()
    observations = set()
    data = []

    for line in lines:
        line = line.strip() # Remove trailing newline characters
        if line: # Non-empty line
            space_idx = line.rfind(" ")
            observation = line[:space_idx]
            state = line[space_idx + 1:]
            states.add(state)
            observations.add(observation)
            data.append((observation, state))

    state_to_idx = {state: idx for idx, state in enumerate(states)}
    observation_to_idx = {obs: idx for idx, obs in enumerate(observations)}

    return state_to_idx, observation_to_idx, list(states), list(observations), data

def initialize_parameters(states, observations):
    """Initialize the transition and emission probabilities

    Args:
        states (list): List of unique states
        observations (list): List of unique observations

    Returns:
        numpy array: transition_prob - transition probabilities between states
        numpy array: emission_prob - emission probabilities of observations given states

    """
    num_states = len(states)
    num_observations = len(observations)

    # Initialize transition and emission probabilities uniformly
    transition_prob = np.ones((num_states, num_states)) / num_states
    emission_prob = np.ones((num_states, num_observations)) / num_observations

    return transition_prob, emission_prob


def train_hmm(data, state_to_idx, observation_to_idx, transition_prob, emission_prob, iterations=10):
    """Train the Hidden Markov Model using the given data

    Args:
        data (list): List of tuples (observation, state)
        state_to_idx (dict): Mapping of states to indices
        observation_to_idx (dict): Mapping of observations to indices
        transition_prob (numpy array): Initial transition probabilities
        emission_prob (numpy array): Initial emission probabilities
        iterations (int): Number of training iterations

    Returns:
        numpy array: Updated transition probabilities
        numpy array: Updated emission probabilities

    """
    num_states = len(state_to_idx)
    num_observations = len(observation_to_idx)

    for iteration in range(iterations):
        # Counters for transition and emission probabilities
        transition_count = np.zeros((num_states, num_states))
        emission_count = np.zeros((num_states, num_observations))

        # Iterate through the data to update counts
        for i in range(len(data) - 1):
            obs_idx_curr = observation_to_idx[data[i][0]]
            obs_idx_next = observation_to_idx[data[i + 1][0]]
            state_idx_curr = state_to_idx[data[i][1]]
            state_idx_next = state_to_idx[data[i + 1][1]]

            transition_count[state_idx_curr, state_idx_next] += 1
            emission_count[state_idx_curr, obs_idx_curr] += 1

        # Update transition and emission probabilities
        transition_prob = transition_count / transition_count.sum(axis=1, keepdims=True)
        emission_prob = (emission_count + 1) / (emission_count.sum(axis=1, keepdims=True) + num_observations)

    return transition_prob, emission_prob

def viterbi(observation_sequence, state_to_idx, observation_to_idx, transition_prob, emission_prob):
    """Viterbi algorithm to predict the most likely sequence of states

    Args:
        observation_sequence (list): List of observations (words)
        state_to_idx (dict): Mapping of states to indices
        observation_to_idx (dict): Mapping of observations to indices
        transition_prob (numpy array): Transition probabilities between states
        emission_prob (numpy array): Emission probabilities of observations given states

    Returns:
        list: Predicted sequence of states (tags)

    """
    num_states = len(state_to_idx)
    num_observations = len(observation_sequence)
    dp = np.zeros((num_states, num_observations))
    backpointer = np.zeros((num_states, num_observations), dtype=int)

    # Initialization step
    obs_idx = observation_to_idx[observation_sequence[0]]
    dp[:, 0] = emission_prob[:, obs_idx]

    # Recursion step
    for t in range(1, num_observations):
        obs_idx = observation_to_idx[observation_sequence[t]]
        for s in range(num_states):
            trans_prob = transition_prob[:, s] * dp[:, t - 1]
            max_trans_prob = np.max(trans_prob) # Max transition probability - highest likelihood from previous state to current state
            backpointer[s, t] = np.argmax(trans_prob) # Backpointer - index of previous state with highest likelihood
            dp[s, t] = max_trans_prob * emission_prob[s, obs_idx] # Update likelihood of current state

    # Termination step
    best_path = np.zeros(num_observations, dtype=int)
    best_path[-1] = np.argmax(dp[:, -1])

    # Backtrack to find the best path
    for t in range(num_observations - 2, -1, -1):
        best_path[t] = backpointer[best_path[t + 1], t + 1]

    # Convert indices to state names
    idx_to_state = {idx: state for state, idx in state_to_idx.items()}
    predicted_states = [idx_to_state[idx] for idx in best_path]

    return predicted_states

def write_predictions_to_file(file_path, observation_sequence, predicted_states):
    """Write the observations and predicted states to a file

    Args:
        file_path (string): Output file path
        observation_sequence (list): List of observations (words)
        predicted_states (list): List of predicted states (tags)

    """
    with open(file_path, 'w') as file:
        for observation, state in zip(observation_sequence, predicted_states):
            file.write(f"{observation} {state}\n")


def main():
    file_path = 'Data\\ES\\train'
    state_to_idx, observation_to_idx, states, observations, data = prepare_data(file_path)
    transition_prob, emission_prob = initialize_parameters(states, observations)
    transition_prob, emission_prob = train_hmm(data, state_to_idx, observation_to_idx, transition_prob, emission_prob)

    # print('Transition probabilities:')
    # print(transition_prob)
    
    # print('Emission probabilities:')
    # print(emission_prob)
    
    # Test the model on the test set
    file_path = "Data\\ES\\dev.in"
    observation_sequence = []
    with open(file_path, 'r', encoding="utf-8") as file:
        initial_observation_sequence = file.read().splitlines() # splitlines() removes trailing newline characters
        
        # formatting observation sequence
        for observation in initial_observation_sequence:
            if observation: # Non-empty line
                observation = observation.strip() # remove trailing newline characters
                observation_sequence.append(observation)

    # print(observation_sequence)
    predicted_states = viterbi(observation_sequence, state_to_idx, observation_to_idx, transition_prob, emission_prob)
    
    print(predicted_states)
    
    # file_path = "Data\\ES\\dev.p4.out"
    # write_predictions_to_file(file_path, observation_sequence, predicted_states)
    
if __name__ == "__main__":
    main()
