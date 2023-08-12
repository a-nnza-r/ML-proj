import numpy as np
import evalResult

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
    emission_counts = np.zeros((num_states, num_observations)) # rows are states, columns are observations
    state_counts = np.zeros(num_states)

    # Split the training data into sentences
    sentences = train_data.strip().split('\n\n')

    for sentence in sentences:
        lines = sentence.strip().split('\n')
        for line in lines:
            observation, state = line.rsplit(' ', 1)
            state_idx = state_to_idx[state]
            observation_idx = observation_to_idx[observation]
            emission_counts[state_idx, observation_idx] += 1
            state_counts[state_idx] += 1

    # Calculate probabilities for known words
    emission_probabilities = (emission_counts + k) / (state_counts[:, None] + k * num_observations) # LAPLACE SMOOTHING
    
    # Calculate probabilities for unknown words
    unk_idx = observation_to_idx["#UNK#"]
    for i in range(num_states):
        emission_probabilities[i, unk_idx] = k / (state_counts[i] + k)

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
        lines = sentence.strip().split('\n') # separate each line as an element without the \n
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
    sentences = test_data.strip().split('\n\n')
    predicted_tags = []
    
    for sentence in sentences: # assume each chunk separated by a newline in the test data is a sentence
        words = sentence.strip().split('\n')
        num_words = len(words)
        num_states = len(states)
        
        # initialize viterbi matrix (tabulation table) and backpointers matrix (for convenience to backtrack later)
        viterbi_matrix = np.zeros((num_states, num_words+2))
        backpointers = np.zeros((num_states, num_words+2), dtype=int) # Backpointers matrix for recording the index of best previous state

        # step 1 - initialization
        start_idx = state_to_idx["START"]
        stop_idx = state_to_idx["STOP"]
        viterbi_matrix[start_idx, 0] = 1

        # step 2 - recurrence
        for j in range(1, num_words+1):
            word = words[j-1]
            if word not in observation_to_idx:
                word = "#UNK#"
            observation_idx = observation_to_idx[word]
            for s in range(num_states):
                if s == start_idx or s == stop_idx:
                    continue
                max_trans_prob = 0
                # finding the max transition probability from all previous states to current state
                for prev_s in range(num_states): 
                    trans_prob = transmission_probabilities[prev_s, s] * viterbi_matrix[prev_s, j-1]
                    if trans_prob > max_trans_prob:
                        max_trans_prob = trans_prob
                        
                # update viterbi matrix and backpointers matrix
                viterbi_matrix[s, j] = max_trans_prob * emission_probabilities[s, observation_idx]
                backpointers[s, j] = np.argmax(transmission_probabilities[:, s] * viterbi_matrix[:, j-1]) # Added backpointer

        # step 3 - termination
        stop_trans_prob = transmission_probabilities[:, stop_idx] * viterbi_matrix[:, num_words]
        max_stop_trans_prob = np.max(stop_trans_prob)
        viterbi_matrix[stop_idx, num_words+1] = max_stop_trans_prob
        
        best_last_tag = np.argmax(viterbi_matrix[:, -2])
        best_path = [best_last_tag]

        # Backtrack using backpointers
        for i in range(num_words, 1, -1): 
            best_tag = backpointers[best_last_tag, i]
            best_path.insert(0, best_tag)
            best_last_tag = best_tag
        
        predicted_tags.append([states[s] for s in best_path])

    return predicted_tags

def write_predictions_to_file(file_path, predicted_sequences, test_data):
    """write to file the predicted sequences

    Args:
        file_path (string): the file path and name to write to
        predicted_sequences (a list of sentences): each sentence is a list of predicted tags
        test_data (string): a string of the test data
    """
    with open(file_path, 'w', encoding="utf-8") as file:
        sentences = test_data.strip().split('\n\n')
        for i, sentence in enumerate(sentences):
            words = sentence.strip().split('\n')
            for j, word in enumerate(words):
                file.write(word + ' ' + predicted_sequences[i][j] + '\n')
            file.write('\n')

def compute_scores(gold_file_path, predicted_file_path):
    ## code adapted from evalResult ##
    gold = open(gold_file_path, "r", encoding="utf-8")
    predicted = open(predicted_file_path, "r", encoding="utf-8")

    #column separator
    separator = ' '

    #the column index for tags
    outputColumnIndex = 1
    
    #Read Gold data
    observed = evalResult.get_observed(gold, separator, outputColumnIndex)

    #Read Predction data
    predicted = evalResult.get_predicted(predicted, separator, outputColumnIndex)

    correct_entity, correct_sentiment, entity_prec, entity_rec, entity_f, sentiment_prec, sentiment_rec, sentiment_f = evalResult.compare_observed_to_predicted(observed, predicted)
    
    return correct_entity, correct_sentiment, entity_prec, entity_rec, entity_f, sentiment_prec, sentiment_rec, sentiment_f


def predict_es():
    file_path = "Data\\ES\\train"
    state_to_idx, observation_to_idx, states, observations, train_data = prepare_data(file_path)

    validation_file_path = "Data\\ES\\dev.in"
    with open(validation_file_path, 'r', encoding="utf-8") as file:
        validation_data = file.read()
        
    gold_file_path = "Data\\ES\\dev.out"
        
    # Define a range of k values to try
    k_values = [0.1 * i for i in range(1, 11)] # k = 0.1, 0.2, ..., 1.0

    transition_prob = estimate_transmission_parameters(train_data, states, state_to_idx)

    # Iterate through each k value
    for k in k_values:
        # Estimate emission probabilities with the current k
        emission_prob = estimate_emission_parameters(train_data, states, observations, state_to_idx, observation_to_idx, k=k)
        
        # Run Viterbi on the validation data
        predicted_tags = viterbi(validation_data, states, state_to_idx, observation_to_idx, emission_prob, transition_prob)
        output_file_path = "Testing\\es.dev.k" + str(k)
        write_predictions_to_file(output_file_path, predicted_tags, validation_data)
        
        correct_entity, correct_sentiment, entity_prec, entity_rec, entity_f, sentiment_prec, sentiment_rec, sentiment_f = compute_scores(gold_file_path, output_file_path)
        
        
        print("k = " + str(k))
        print("correct_entity = " + str(correct_entity))
        print("correct_sentiment = " + str(correct_sentiment))
        print("entity_prec = " + str(entity_prec))
        print("entity_rec = " + str(entity_rec))
        print("entity_f = " + str(entity_f))
        print("sentiment_prec = " + str(sentiment_prec))
        print("sentiment_rec = " + str(sentiment_rec))
        print("sentiment_f = " + str(sentiment_f))
        print()
        

    
def predict_ru():
    file_path = "Data\\RU\\train"
    state_to_idx, observation_to_idx, states, observations, train_data = prepare_data(file_path)

    validation_file_path = "Data\\RU\\dev.in"
    with open(validation_file_path, 'r', encoding="utf-8") as file:
        validation_data = file.read()
        
    gold_file_path = "Data\\RU\\dev.out"
        
    # Define a range of k values to try
    k_values = [0.1 * i for i in range(1, 11)] # k = 0.1, 0.2, ..., 1.0

    transition_prob = estimate_transmission_parameters(train_data, states, state_to_idx)

    # Iterate through each k value
    for k in k_values:
        # Estimate emission probabilities with the current k
        emission_prob = estimate_emission_parameters(train_data, states, observations, state_to_idx, observation_to_idx, k=k)
        
        # Run Viterbi on the validation data
        predicted_tags = viterbi(validation_data, states, state_to_idx, observation_to_idx, emission_prob, transition_prob)
        output_file_path = "Testing\\ru.dev.k" + str(k)
        write_predictions_to_file(output_file_path, predicted_tags, validation_data)
        
        correct_entity, correct_sentiment, entity_prec, entity_rec, entity_f, sentiment_prec, sentiment_rec, sentiment_f = compute_scores(gold_file_path, output_file_path)
        
        
        print("k = " + str(k))
        print("correct_entity = " + str(correct_entity))
        print("correct_sentiment = " + str(correct_sentiment))
        print("entity_prec = " + str(entity_prec))
        print("entity_rec = " + str(entity_rec))
        print("entity_f = " + str(entity_f))
        print("sentiment_prec = " + str(sentiment_prec))
        print("sentiment_rec = " + str(sentiment_rec))
        print("sentiment_f = " + str(sentiment_f))
        print()
        
    
if __name__ == "__main__":
    predict_es()
    predict_ru()