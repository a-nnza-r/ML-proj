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
        string: train_data - training data as a whole string

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

    # add START and STOP states
    states.add("START")
    states.add("STOP")
    
    # add #UNK# observation
    observations.add("#UNK#")
    
    state_to_idx = {state: idx for idx, state in enumerate(sorted(states))}
    observation_to_idx = {obs: idx for idx, obs in enumerate(sorted(observations))}
    
    # read train_data
    with open(file_path, 'r', encoding="utf-8") as file:
        train_data = file.read()

    return state_to_idx, observation_to_idx, sorted(states), sorted(observations), train_data


def estimate_emission_parameters(train_data, states, observations, state_to_idx, observation_to_idx, k=1):
    """Estimate the emission probabilities

    Args:
        train_data (string): Training data as a whole string
        states (list): List of unique states
        observations (list): List of unique observations
        state_to_idx (dict): Mapping of each state to the index
        observation_to_idx (dict): Mapping of each observation to the index
        k (int): Smoothing parameter

    Returns:
        numpy array: Emission probabilities of shape (num_states, num_observations)

    """
    num_states = len(states)
    num_observations = len(observations)
    
    # Initialize counts
    emission_counts = np.zeros((num_states, num_observations))
    state_counts = np.zeros(num_states)

    # Split the training data into sentences
    sentences = train_data.strip().split('\n\n')

    for sentence in sentences:
        lines = sentence.strip().split('\n')
        for line in lines:
            observation, state = line.rsplit(' ', 1)
            if observation not in observation_to_idx:
                observation = "#UNK#"
            state_idx = state_to_idx[state]
            observation_idx = observation_to_idx[observation]
            emission_counts[state_idx, observation_idx] += 1
            state_counts[state_idx] += 1

    # Apply smoothing and calculate probabilities
    emission_probabilities = (emission_counts + k) / (state_counts[:, None] + k * num_observations)

    # Set emission probabilities for START and STOP states to 0
    emission_probabilities[state_to_idx["START"], :] = 0
    emission_probabilities[state_to_idx["STOP"], :] = 0

    return emission_probabilities

def estimate_transmission_parameters(train_data, states, state_to_idx):
    """Estimate the transmission probabilities

    Args:
        train_data (string): Training data as a whole string
        states (list): List of unique states
        state_to_idx (dict): Mapping of each state to the index

    Returns:
        numpy array: Transmission probabilities of shape (num_states, num_states)

    """
    num_states = len(states)
    
    # Initialize counts
    transition_counts = np.zeros((num_states, num_states))
    state_counts = np.zeros(num_states)

    # Split the training data into sentences
    sentences = train_data.strip().split('\n\n')

    for sentence in sentences:
        # print(sentence)
        lines = sentence.strip().split('\n')
        prev_state = "START"
        for line in lines:
            _, current_state = line.rsplit(' ', 1)
            transition_counts[state_to_idx[prev_state], state_to_idx[current_state]] += 1
            state_counts[state_to_idx[prev_state]] += 1
            prev_state = current_state
        # Transition to STOP state
        transition_counts[state_to_idx[prev_state], state_to_idx["STOP"]] += 1
        state_counts[state_to_idx[prev_state]] += 1

    # Calculate probabilities
    transmission_probabilities = transition_counts / state_counts[:, None]

    nan_mask = np.isnan(transmission_probabilities)
    transmission_probabilities[nan_mask] = 0 # set all transition probabilities from STOP to 0
    return transmission_probabilities

def viterbi(test_data, states, state_to_idx, observation_to_idx, emission_probabilities, transmission_probabilities):
    # Split the test data into sentences
    sentences = test_data.strip().split('\n\n')
    predicted_tags = []
    
    for sentence in sentences:
        words = sentence.strip().split('\n')
        num_words = len(words)
        num_states = len(states)

        # Initialize Viterbi matrix
        viterbi_matrix = np.zeros((num_states, num_words+2)) # the rows of the viterbi matrix correspond to the states while the columns correspond to the words

        # Initialization 
        start_idx = state_to_idx["START"]
        stop_idx = state_to_idx["STOP"]
        viterbi_matrix[start_idx, 0] = 1 # base case

        # Recursion step
        for j in range(1, num_words+1):
            word = words[j-1]
            if word not in observation_to_idx:
                word = "#UNK#"
            observation_idx = observation_to_idx[word]
            for s in range(num_states): # s is the current state
                
                if s == start_idx or s == stop_idx: # skip START and STOP states
                    continue
                
                trans_prob = transmission_probabilities[:, s] * viterbi_matrix[:, j-1] # find transition probabilities from all states to current state
                max_trans_prob = np.max(trans_prob) # most likely transition probability
                viterbi_matrix[s, j] = max_trans_prob * emission_probabilities[s, observation_idx]
                
        # Termination step
        # calculate transition probability from all states to STOP state
        stop_trans_prob = transmission_probabilities[:, stop_idx] * viterbi_matrix[:, num_words] # 0th index is START state
        max_stop_trans_prob = np.max(stop_trans_prob)
        viterbi_matrix[stop_idx, num_words+1] = max_stop_trans_prob # calculate probability from last word to STOP state
        
        best_last_tag = np.argmax(viterbi_matrix[:, -2]) # find the most likely last tag of the last word
        best_path = [best_last_tag]

        # Backtrack to find the best path
        for i in range(num_words, 1, -1):
            best_tag = np.argmax(viterbi_matrix[:, i-1])
            best_path.insert(0, best_tag)
        
        # Convert state indices to state names
        predicted_tags.append([states[s] for s in best_path])
    #print(predicted_tags)
    return predicted_tags

def write_predictions_to_file(file_path, predicted_sequences, test_data):
    with open(file_path, 'w', encoding="utf-8") as file:
        sentences = test_data.strip().split('\n\n')
        for i, sentence in enumerate(sentences):
            words = sentence.strip().split('\n')
            for j, word in enumerate(words):
                file.write(word + ' ' + predicted_sequences[i][j] + '\n')
            file.write('\n')

def main():
    file_path = "Data\\ES\\train"
    state_to_idx, observation_to_idx, states, observations, train_data = prepare_data(file_path)

    emission_prob = estimate_emission_parameters(train_data, states, observations, state_to_idx, observation_to_idx, k=1)
    
    transition_prob = estimate_transmission_parameters(train_data, states, state_to_idx)
    
    test_file_path = "Data\\ES\\dev.in"
    with open(test_file_path, 'r', encoding="utf-8") as file:
        test_data = file.read()
    
    # print(state_to_idx)
    # print(transition_prob)
    
    predicted_tags = viterbi(test_data, states, state_to_idx, observation_to_idx, emission_prob, transition_prob)
    # print(predicted_tags)
    
    write_predictions_to_file("Testing\\dev.test", predicted_tags, test_data)
    
if __name__ == "__main__":
    main()