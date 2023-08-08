def readFileEmissionParameters(filepath):
    y_count = {}
    emission_count = {}
    training_observations_x = []
    with open(filepath, 'r',encoding='utf-8') as file:
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
            if key in emission_count:
                emission_count[key] += 1
            else:
                emission_count[key] = 1
    return y_count, emission_count, training_observations_x


def emision_parameters(x_t, y_t, y_count, emission_count):
    return emission_count.get((y_t, x_t), 0) / y_count.get(y_t, 1)

def emision_parameters_updated(x_t,y_t,y_count,emmision_count,training_observations_x,k=1):
    if(x_t in training_observations_x):
        return emmision_count.get((y_t,x_t),0)/(y_count[y_t]+k)
    else:
        return k/(y_count[y_t]+k)

def simple_sentiment_analysis(x_seq, y_count, emission_count,training_observations_x):
    Y = list(y_count.keys())
    seq_pair = []
    for x_s in x_seq:
        max_y = None
        max_emission_prob = 0
        for y in Y:
            emission_prob = emision_parameters_updated(x_s, y, y_count, emission_count,training_observations_x)
            if emission_prob > max_emission_prob:
                max_y = y
                max_emission_prob = emission_prob
        seq_pair.append((x_s, max_y))
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

def predictNwrite(readDevInPath, y_count, emission_count, training_observations_x, writeFilePath):
    x_sequences = readDevIn(readDevInPath)
    list_of_sequences = [] 
    for x_seq in x_sequences:
        seq_pairs = simple_sentiment_analysis(x_seq, y_count, emission_count,training_observations_x)
        list_of_sequences.append(seq_pairs)
    write_seq_pairs_to_file(writeFilePath, list_of_sequences)
    
# RU training 
y_count_RU, emission_count_RU, training_observations_x_RU = readFileEmissionParameters("./Data/RU/train")
predictNwrite("./Data/RU/dev.in", y_count_RU, emission_count_RU, training_observations_x_RU, "./Data/RU/dev.p1.out")

#Entity in gold data: 389
#Entity in prediction: 1816

#Correct Entity : 266
#Entity  precision: 0.1465
#Entity  recall: 0.6838
#Entity  F: 0.2413

#Correct Sentiment : 129
#Sentiment  precision: 0.0710
#Sentiment  recall: 0.3316
#Sentiment  F: 0.1170

# ES training 
y_count_ES, emission_count_ES, training_observations_x_ES = readFileEmissionParameters("./Data/ES/train")
predictNwrite("./Data/ES/dev.in", y_count_ES, emission_count_ES, training_observations_x_ES, "./Data/ES/dev.p1.out")

#Entity in gold data: 229
#Entity in prediction: 1466

#Correct Entity : 178
#Entity  precision: 0.1214
#Entity  recall: 0.7773
#Entity  F: 0.2100

#Correct Sentiment : 97
#Sentiment  precision: 0.0662
#Sentiment  recall: 0.4236
#Sentiment  F: 0.1145

