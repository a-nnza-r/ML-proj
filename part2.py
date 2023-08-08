import math

def readFile(filepath):
    y_count = {}
    emission_count = {}
    transition_counts = {}
    training_observations_x = set()


    with open(filepath, 'r') as file:
        y_i = "START"
        y_count["START"] = 1
        for line in file:
            line = line.strip()
            if not line:
                transition_counts[(y_i, "STOP")] = transition_counts.get((y_i, "STOP"), 0) + 1
                y_count["STOP"] = y_count.get("STOP", 0) + 1
                y_i = "START"
                y_count["START"] = y_count.get("START", 0) + 1
                continue

            last_space_idx = line.rfind(" ")
            x, y_j = line[:last_space_idx], line[last_space_idx + 1:]
            transition_counts[(y_i, y_j)] = transition_counts.get((y_i, y_j), 0) + 1
            emission_count[(y_j, x)] = emission_count.get((y_j, x), 0) + 1
            y_count[y_j] = y_count.get(y_j, 0) + 1
            training_observations_x.add(x)
            y_i = y_j
        
        if y_i != "START":
            transition_counts[(y_i, "STOP")] = transition_counts.get((y_i, "STOP"), 0) + 1
            y_count["STOP"] = y_count.get("STOP", 0) + 1
        else: 
            y_count["START"] -=1
    return y_count, emission_count, transition_counts, training_observations_x

def readDevIn(filePath):
    x_sequences = []
    current_sequence = []
    with open(filePath, 'r') as file:
        for line in file:
            x = line.strip()
            if not x:
                if current_sequence: 
                    x_sequences.append(current_sequence)
                    current_sequence = [] 
            else:
                current_sequence.append(x)
        if current_sequence:
            x_sequences.append(current_sequence)
    return x_sequences

def write_seq_pairs_to_file(file_path, list_of_sequences):
    with open(file_path, 'w') as file:
        for seq_pairs in list_of_sequences:
            for x, y in seq_pairs:
                 file.write(f"{x} {y}\n")
            file.write("\n")

def emision_parameters_updated(x_t, y_t, y_count, emission_count, k=1):
    if (y_t, x_t) in emission_count.keys():
        return math.log(emission_count[(y_t, x_t)] / (y_count[y_t] + k))
    else:
        return math.log(k / (y_count[y_t] + k))

def transition_parameters(y_i, y_j, transition_count, y_count):
    numerator = transition_count.get((y_i, y_j), 0)
    denominator = y_count[y_i]
    if numerator == 0:
        return float('-inf')
    return math.log(numerator / denominator)


def viterbi(y_count, emission_count, transition_counts, x_input_seq):
    scores = {}
    parent_pointer = {}
    n = len(x_input_seq)
    labels = ['O', 'B-positive', 'B-negative', 'B-neutral', 'I-neutral', 'I-positive', 'I-negative']
    states = ["START",'O', 'B-positive', 'B-negative', 'B-neutral', 'I-neutral', 'I-positive', 'I-negative']

    for k in range(0, n):
        for key in states:
            scores[(k, key)] = float("-inf")
    scores[(0, "START")] = 0.0

    # BOTTOM UP score build
    for k in range(0, n):
        for v in labels:
            max_u_score = float("-inf")
            parent = None
            for u in states:
                if((k == 0 and u == "START") or u in labels): 
                    emission_prob = emision_parameters_updated(x_input_seq[k], v, y_count, emission_count, 1)  # x sequence is indexed from 0 soo k -> k-1
                    transition_prob = transition_parameters(u, v, transition_counts, y_count)
                    possible_u_score = scores[(k, u)] + emission_prob + transition_prob
                    #print("emmsion prob: {0:.10f} transition prob {1:.10f} u_score {1:.10f}".format(emission_prob,transition_prob,possible_u_score))
                    if possible_u_score > max_u_score:
                        max_u_score = possible_u_score
                        parent = u
                    #print("k:{0} {1:>10}->{2:<10} P:{3} ms: {4:.10f} cs: {5:.10f}".format(k,u,v,parent, max_u_score, possible_u_score) )
            #print("current node: {0} parent: {1:} Max score: {2:.10f}".format(v,parent, max_u_score) )
            scores[(k+1, v)] = max_u_score
            parent_pointer[(k+1,v)] = parent
        print(parent_pointer)
    # Final step
    max_final_transition_score = float("-inf")
    stop_parent = None
    for v in labels:
        score = scores[(n, v)] + transition_parameters(v, "STOP", transition_counts, y_count)
        if score > max_final_transition_score:
            max_final_transition_score = score
            stop_parent = v
        #print("{0:>10}->{1:<10} P:{2} ms:{3:.10f} cs:{4:.10f}".format(v,"STOP",stop_parent, max_final_transition_score, score) )
        

    scores[(n+1, "STOP")] = max_final_transition_score
    parent_pointer[(n+1,"STOP")] = stop_parent
    predicted_labels = ["STOP"]
    current_label = "STOP"
    #print(scores)
    #print(parent_pointer)

    for i in range(n+1,0,-1):
        parent = parent_pointer.get((i,current_label))
        predicted_labels.insert(0,parent)
        current_label = parent
    #print(predicted_labels[1:-1])
    return predicted_labels[1:-1] , scores , parent_pointer

def buildModelNwrite(readDevInPath, y_count, emission_count, transition_count, writeFilePath):
    x_sequences = readDevIn(readDevInPath)
    total_sequences = len(x_sequences)
    list_of_sequences = []
    for x_input_seq in x_sequences: 
        predicted_labels , _ , _ = viterbi(y_count, emission_count, transition_count , x_input_seq)
        list_of_sequences.append(list(zip(x_input_seq,predicted_labels)))
    write_seq_pairs_to_file(writeFilePath, list_of_sequences)



# RU
y_count_RU, emission_count_RU, transition_count_RU,training_observations_x_RU = readFile("./Data/RU/train")
buildModelNwrite("./Data/RU/dev.in", y_count_RU, emission_count_RU, transition_count_RU, "./Data/RU/dev.p2.out")

#Entity in gold data: 389
#Entity in prediction: 426

#Correct Entity : 1
#Entity  precision: 0.0023
#Entity  recall: 0.0026
#Entity  F: 0.0025

#Correct Sentiment : 1
#Sentiment  precision: 0.0023
#Sentiment  recall: 0.0026
#Sentiment  F: 0.0025


# ES 
y_count_ES, emission_count_ES, transition_count_ES,training_observations_x_ES = readFile("./Data/ES/train")
buildModelNwrite("./Data/ES/dev.in", y_count_ES, emission_count_ES, transition_count_ES, "./Data/ES/dev.p2.out")

#Entity in gold data: 229
#Entity in prediction: 255

#Correct Entity : 9
#Entity  precision: 0.0353
#Entity  recall: 0.0393
#Entity  F: 0.0372

#Correct Sentiment : 7
#Sentiment  precision: 0.0275
#Sentiment  recall: 0.0306
#Sentiment  F: 0.0289


