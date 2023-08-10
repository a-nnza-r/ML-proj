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

def emission_parameters_updated(x_t,y_t,y_count,emission_count,training_observations_x,k=1):
    if(x_t in training_observations_x):
        if(emission_count.get((y_t,x_t),0) == 0 ):
            return float("-inf")
        return math.log(emission_count.get((y_t,x_t),0)/(y_count[y_t]+k))
    else:
        return math.log(k/(y_count[y_t]+k))

def transition_parameters(y_i, y_j, transition_count, y_count):
    numerator = transition_count.get((y_i, y_j), 0)
    denominator = y_count[y_i]
    if numerator == 0:
        return float('-inf')
    return math.log(numerator / denominator)

def k_best_viterbi(y_count, emission_count, transition_counts, training_observations_x, x_input_seq, k=1):
    n = len(x_input_seq)
    states = list(y_count.keys())

    scores = {}
    for i in range(n+1):
        for state in states: 
            score_list = []
            for j in range(k):
                score_list.append((float('-inf'), None , None))
            scores[(i,state)] = score_list

    scores[(n+1,"STOP")] = None 
    
    # for i in range(k):
    #     scores[(0, "START")][i] = [(0.0, None, None)]
    scores[(0, "START")] = [(0.0, None, None)]

    for t in range(1, n+1):
        for v in states:
            if (v == "START") or (v == "STOP"): continue 
            all_scores = []
            for u in states:
                if (u == "STOP") or (u == "START" and t != 1): continue   
                #for idxInParentScoreList in range(k):
                for idx , score_tup in enumerate(scores[(t-1, u)]): 
                    emission_prob = emission_parameters_updated(x_input_seq[t-1], v, y_count, emission_count, training_observations_x, 1)
                    transition_prob = transition_parameters(u, v, transition_counts, y_count)
                    current_v_score = score_tup[0] + emission_prob + transition_prob
                    #if(current_v_score > float('-inf')):
                    all_scores.append((current_v_score, u , idx)) #idxInParentScoreList))
                        #print("layer {0} ({1},{2})->{3} cs:{4}".format(t,u,idx,v,current_v_score))
            #extract the k best scores and update the memo 
            scores[(t, v)] = sorted(all_scores, reverse=True)[:k]
            #print("score list at ({0:>},{1:<}):{2}".format(t,v,scores[(t, v)]))
        #print("\n")
    
    # Final step to STOP
    all_scores = []
    for u in states:
        if (u == "START") or (u == "STOP"): continue 
        #for idxInParentScoreList in range(k):
        for idx , score_tup in enumerate(scores[(n, u)]): 
            #k_score_at_STOP, _ , _ = scores[(n, v)][idxInParentScoreList]
            transition_prob = transition_parameters(u, "STOP", transition_counts, y_count)
            current_v_score = score_tup[0] + transition_prob
            #if(current_v_score > float('-inf')):
            all_scores.append((current_v_score, u , idx)) #idxInParentScoreList))
                #print("layer {0} ({1},{2})->{3} cs:{4}".format(n+1,u,idx,"STOP",current_v_score))
            #all_scores.append((current_v_score, v, idx))
    scores[(n+1, "STOP")] = sorted(all_scores, reverse=True)[:k]
    #print("score list at ({0:>},{1:<}):{2}".format(n+1,"STOP",scores[(n+1, "STOP")]))

    # Reconstruct k-best paths
    k_best_paths = []
    score_list = [(n + 1, "STOP")]
    for idx_in_STOP_list in range(k):
        path = []
        #print("printing final score list:", scores[(n+1,"STOP")], idx_in_STOP_list)
        score , parent , idx_in_parent = scores[(n+1,"STOP")][idx_in_STOP_list]
        for i in range(n,0,-1):
            #print(score,parent,idx_in_parent)
            path.insert(0,parent)
            score , parent , idx_in_parent = scores[(i,parent)][idx_in_parent]
        #print(path)
        k_best_paths.append(path)
    #print(k_best_paths)
    return k_best_paths , scores

def buildModelNwrite(readDevInPath, y_count, emission_count, transition_count, training_observations_x,writeFilePathList,k_paths_to_extract):
    x_sequences = readDevIn(readDevInPath)
    k = max(k_paths_to_extract)
    list_of_sequences = {}
    for paths in k_paths_to_extract:
        list_of_sequences[paths] = []

    for x_input_seq in x_sequences: 
        k_best_paths , scores = k_best_viterbi(y_count, emission_count, transition_count ,training_observations_x ,x_input_seq,k)
        for paths in k_paths_to_extract:
            list_of_sequences[paths].append(list(zip(x_input_seq,k_best_paths[paths-1])))
    
    for write_loc , path_number in list(zip(writeFilePathList,k_paths_to_extract)):
        write_seq_pairs_to_file(write_loc,list_of_sequences[path_number])



# RU
y_count_RU, emission_count_RU, transition_count_RU,training_observations_x_RU = readFile("./Data/RU/train")
buildModelNwrite("./Data/RU/dev.in", y_count_RU, emission_count_RU, transition_count_RU, training_observations_x_RU,["./Data/RU/dev.p3.1st.out","./Data/RU/dev.p3.2nd.out","./Data/RU/dev.p3.8th.out"],[1,2,8])

#RU 1 st output 
#Entity in gold data: 389
#Entity in prediction: 490

#Correct Entity : 189
#Entity  precision: 0.3857
#Entity  recall: 0.4859
#Entity  F: 0.4300

#Correct Sentiment : 129
#Sentiment  precision: 0.2633
#Sentiment  recall: 0.3316
#Sentiment  F: 0.2935


#RU 2nd output 
#Entity in gold data: 389
#Entity in prediction: 696

#Correct Entity : 202
#Entity  precision: 0.2902
#Entity  recall: 0.5193
#Entity  F: 0.3724

#Correct Sentiment : 124
#Sentiment  precision: 0.1782
#Sentiment  recall: 0.3188
#Sentiment  F: 0.2286

#RU 8th output 
#Entity in gold data: 389
#Entity in prediction: 703

#Correct Entity : 172
#Entity  precision: 0.2447
#Entity  recall: 0.4422
#Entity  F: 0.3150

#Correct Sentiment : 94
#Sentiment  precision: 0.1337
#Sentiment  recall: 0.2416
#Sentiment  F: 0.1722




# ES 
y_count_ES, emission_count_ES, transition_count_ES,training_observations_x_ES = readFile("./Data/ES/train")
buildModelNwrite("./Data/ES/dev.in", y_count_ES, emission_count_ES, transition_count_ES,training_observations_x_ES ,["./Data/ES/dev.p3.1st.out","./Data/ES/dev.p3.2nd.out","./Data/ES/dev.p3.8th.out"],[1,2,8])

# ES 1 st 
#Entity in gold data: 229
#Entity in prediction: 542

#Correct Entity : 134
#Entity  precision: 0.2472
#Entity  recall: 0.5852
#Entity  F: 0.3476

#Correct Sentiment : 97
#Sentiment  precision: 0.1790
#Sentiment  recall: 0.4236
#Sentiment  F: 0.2516

# ES 2nd output
#Entity in gold data: 229
#Entity in prediction: 436

#Correct Entity : 117
#Entity  precision: 0.2683
#Entity  recall: 0.5109
#Entity  F: 0.3519

#Correct Sentiment : 65
#Sentiment  precision: 0.1491
#Sentiment  recall: 0.2838
#Sentiment  F: 0.1955

# ES 8th output
#Entity in gold data: 229
#Entity in prediction: 438

#Correct Entity : 102
#Entity  precision: 0.2329
#Entity  recall: 0.4454
#Entity  F: 0.3058

#Correct Sentiment : 56
#Sentiment  precision: 0.1279
#Sentiment  recall: 0.2445
#Sentiment  F: 0.1679