def readFile(filepath):
    y_count = {}
    emision_count = {}
    training_observations_x = []
    with open(filepath, 'r') as file:
        for line in file:
            if not line.strip():
                continue
            line = line.strip()
            last_space_idx = line.rfind(" ")
            x, y = line[:last_space_idx], line[last_space_idx + 1:]
            key = (y,x)
            if x not in training_observations_x: 
                training_observations_x.append(x)
            if y in y_count:
                y_count[y] += 1
            else:
                y_count[y] = 1
            if key in emision_count:
                emision_count[key] += 1
            else:
                emision_count[key] = 1
    return y_count, emision_count


def emision_parameters(x_t, y_t, y_count, emission_count):
    return emission_count.get((y_t, x_t), 0) / y_count.get(y_t, 1)

def emision_parameters_updated(x_t,y_t,y_count,emmision_count,k=1):
    if(emmision_count.get((y_t,x_t),-1) != -1):
        return emmision_count[(y_t,x_t)]/(y_count[y_t]+k)
    else:
        return k/(y_count[y_t]+k)

def simple_sentiment_analysis(x_seq, y_count, emission_count):
    Y = list(y_count.keys())
    seq_pair = []
    for x_s in x_seq:
        max_idx = 0
        max_emission_prob = 0
        for i in range(len(Y)):
            emission_prob = emision_parameters_updated(x_s, Y[i], y_count, emission_count)
            if emission_prob > max_emission_prob:
                max_idx = i
                max_emission_prob = emission_prob
        seq_pair.append((x_s, Y[max_idx]))
    return seq_pair


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

def predictNwrite(readDevInPath,y_count,transition_count,writeFilePath):
    x_sequences = readDevIn(readDevInPath)
    list_of_sequences = [] 
    for x_seq in x_sequences:
        seq_pairs = simple_sentiment_analysis(x_seq ,y_count , transition_count )
        list_of_sequences.append(seq_pairs)
    write_seq_pairs_to_file(writeFilePath,list_of_sequences)
    
    
# RU training 
y_count_RU, emision_count_RU = readFile("./Data/RU/train")
predictNwrite("./Data/RU/dev.in" ,y_count_RU,emision_count_RU , "./Data/RU/dev.p1.out" )

# ES training 
y_count_ES, transition_count_ES = readFile("./Data/ES/train")
predictNwrite("./Data/ES/dev.in",y_count_ES,transition_count_ES, "./Data/ES/dev.p1.out")


