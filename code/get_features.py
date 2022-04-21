import argparse
from nltk.stem import PorterStemmer

def getSetOfWords(filepath):
    '''
    Gets a set of transparent nouns  or support verbs from the filepath
    '''
    text_file = open(filepath, "r")
    lines = text_file.read().splitlines()
    items= set(lines)
    return items

def distanceFromArgument(argument_number, arg_token, token, update_flag, idx, start_sentence_idx, features, feature_lines):
    '''
    Calculate the forward and backward distance of a word from a given argument
    '''
    back_dist = argument_number + "_backward_dist"
    forward_dist = argument_number + "_forward_dist"
    if arg_token:
            #Update the distance from argument for words before it
            if not update_flag:
                for i in range(idx-1, start_sentence_idx-1, -1):
                    prev_features = feature_lines[i][1]
                    feature_lines[i][1][back_dist] = arg_token - int(prev_features["position"])
                #We only update the previous features when the current word is the predicate
                features[back_dist] = 0
                update_flag = True
            else:
                features[back_dist] = None
            
            #Set distance from predicate to words after it
            features[forward_dist] = int(token) - arg_token
    else:
        features[back_dist] = None
        features[forward_dist] = None
    
    return features, feature_lines, update_flag

def getFeatures(filename, args):
    '''
    Reads the input to form the features
    '''
    training_corpus = open(filename,'r')
    lines  = training_corpus.readlines()
    training_corpus.close()

    feature_lines = []                      #Each entry will have [word, features, label] for training file, and [word, features] for testing
    sent_beginning = True                   #Checks if the next word is the beginning of a sentence
    next_sent_beginning = True              #To treat the second word after the beginning of the sentence
    pred_token = None                       #If the predicate for the sentence has been found
    start_sentence_idx = 0                  #The index of the start of each sentence
    update_prev_distance = False            #Whether the distance from the predicate to tokens before it has been updated or not
    arg0_token = None                       #If arg0 has been found
    arg1_token = None                       #If arg1 has been found
    arg2_token = None                       #If arg2 has been found
    arg3_token = None                       #If arg3 has been found
    arg4_token = None                       #If arg4 has been found
    update_arg0_distance = False
    update_arg1_distance = False
    update_arg2_distance = False
    update_arg3_distance = False
    update_arg4_distance = False
    transparent_noun_token = None
    to_update_transparent_dist = False         
    

    if args["transparent_noun"]:
        transparent_nouns = getSetOfWords(args["transparent_noun_path"])
    
    if args["support_verb"]:
        support_verbs = getSetOfWords(args["support_verb_path"])
            

    for (idx,line) in enumerate(lines):
        
        #Handle end of line case
        if line=="\n":
            feature_lines.append([line])
            sent_beginning = True           #Next word is the beginning of the sentence
            next_sent_beginning = True      #To treat the second word after the beginning of the sentence
            pred_token = None               #Will re-initialize the predicate token for that sentence
            start_sentence_idx = idx+1      #The new sentence starts from the next word
            update_prev_distance = False    #For the new sentence
            update_arg0_distance = False
            update_arg1_distance = False
            update_arg2_distance = False
            update_arg3_distance = False
            update_arg4_distance = False
            to_update_transparent_dist = False
            arg0_token = None                   
            arg1_token = None                       
            arg2_token = None                     
            arg3_token = None                       
            arg4_token = None 
            transparent_noun_token = None                     
            continue

        components = line.split('\t')
        role = None
        if len(components)==5:
            #The line contains neither the argument nor the predicate nor the support (len ==5)
            word, word_POS, word_BIO, token, sent = components
        elif len(components)==6:
            #Line contains the argument/predicate/support
            #No roles seen for test features
            if args["test_features"]:
                 word, word_POS, word_BIO, token, _, role = components
            else:
                word, word_POS, word_BIO, token, sent, role = components
                # For the training, we need a label which is either "SUPPORT" or None 
        elif len(components) == 7:
            #Line contains predicate, and type of predicate
            if args["test_features"]:
                 word, word_POS, word_BIO, token, _, role, _ = components
            else:
                word, word_POS, word_BIO, token, sent, role, _ = components
        else:
            print("1, Error in line format on line {}".format(idx+1))
            exit(10)

        if role:
            role = role.strip()     #Remove training '\n' from role
        features = dict()

        #Features of the word
        features["word"] = word
        features["POS"] = word_POS
        features["BIO"] = word_BIO
        features["position"] = token

        ps = PorterStemmer()
        features["stem"] = ps.stem(word)

        features["is_noun"]= False   
        #Here we asssume that the predicate is always known by the system       
        features["is_pred"] = False
        if args["arguments_known"]:
            features["is_arg0"] = False
            features["is_arg1"] = False
            features["is_arg2"] = False
            features["is_arg3"] = False
            features["is_arg4"] = False
        
        label = None

        if word_POS in ["NN","NNS","NNP"]:
            #word is a Noun type
            features["is_noun"] = True
        
        #check if the word is a transparent noun
        if args["transparent_noun"]:
            if word.lower() in transparent_nouns or ps.stem(word) in transparent_nouns:
                features["is_transparent_noun"] = True
                transparent_noun_token = int(token)
                to_update_transparent_dist = True
            else:
                features["is_transparent_noun"] = False

            features["1_before_transparent"] = False
            features["2_before_transparent"] = False
            features["3_before_transparent"] = False
            features["1_after_transparent"] = False
            features["2_after_transparent"] = False
            features["3_after_transparent"] = False
        
        #check if word is in list of support verbs
        if args["support_verb"]:
            if word.lower() in support_verbs or ps.stem(word) in support_verbs:
                features["is_support_verb"] = True
            else:
                features["is_support_verb"] = False
        
        #word is the predicate 
        if role and role=="PRED":
            features["is_pred"] = True
            pred_token = int(token)
        if role and args["arguments_known"]:
            if role == "ARG0":
                features["is_arg0"] = True
                arg0_token = int(token)
            elif role == "ARG1":
                features["is_arg1"] = True
                arg1_token = int(token)
            elif role == "ARG2":
                features["is_arg2"] = True
                arg2_token = int(token)
            elif role == "ARG3":
                features["is_arg0=3"] = True
                arg3_token = int(token)
            elif role == "ARG4":
                features["is_arg4"] = True
                arg4_token = int(token)
        

        if args["distance_features"]:
            #Setting forward/backward distance from the predicate
            if pred_token:
                #Update the distance from predicate for words before it
                if not update_prev_distance:
                    for i in range(idx-1, start_sentence_idx-1, -1):
                        prev_features = feature_lines[i][1]
                        feature_lines[i][1]["pred_backward_dist"] = pred_token - int(prev_features["position"])
                    #We only update the previous features when the current word is the predicate
                    features["pred_backward_dist"] = 0
                    update_prev_distance = True
                else:
                    features["pred_backward_dist"] = None
            
                #Set distance from predicate to words after it
                features["pred_forward_distance"] = int(token) - pred_token
            else:
                features["pred_backward_dist"] = None
                features["pred_forward_distance"] = None

            if args["arguments_known"]:
                features, feature_lines, update_arg0_distance = distanceFromArgument("arg0", arg0_token, token, update_arg0_distance, idx, start_sentence_idx, features, feature_lines)
                features, feature_lines, update_arg1_distance = distanceFromArgument("arg1", arg1_token, token, update_arg1_distance, idx, start_sentence_idx, features, feature_lines)
                features, feature_lines, update_arg2_distance = distanceFromArgument("arg2", arg2_token, token, update_arg2_distance, idx, start_sentence_idx, features, feature_lines)
                features, feature_lines, update_arg3_distance = distanceFromArgument("arg3", arg3_token, token, update_arg3_distance, idx, start_sentence_idx, features, feature_lines)
                features, feature_lines, update_arg4_distance = distanceFromArgument("arg4", arg4_token, token, update_arg4_distance, idx, start_sentence_idx, features, feature_lines)

        if args["transparent_noun"]:
            #If transparent noun token has been found
            if transparent_noun_token:
                if to_update_transparent_dist:
                    for i in [idx-1,idx-2,idx-3]:
                        if i>=start_sentence_idx:
                            prev_features = feature_lines[i][1]
                            t_back_dist = transparent_noun_token - int(prev_features["position"])
                            if t_back_dist==1:
                                feature_lines[i][1]["1_before_transparent"] = True
                                feature_lines[i][1]["2_before_transparent"] = True
                                feature_lines[i][1]["3_before_transparent"] = True
                            if t_back_dist ==2:
                                feature_lines[i][1]["2_before_transparent"] = True
                                feature_lines[i][1]["3_before_transparent"] = True
                            if t_back_dist ==3:
                                feature_lines[i][1]["3_before_transparent"] = True

                    to_update_transparent_dist = False
                
                t_forward_distance = int(token) - transparent_noun_token
                if t_forward_distance ==1:
                    features["1_after_transparent"] = True
                    features["2_after_transparent"] = True
                    features["3_after_transparent"] = True
                if t_forward_distance ==2:
                    features["2_after_transparent"] = True
                    features["3_after_transparent"] = True
                if t_forward_distance ==3:
                    features["3_after_transparent"] = True
            
        
                
                

        #Set label for training
        if not(args["test_features"]) and role and role.strip() == "SUPPORT":
            label = role


        #Handle end of sentence
        end_of_file = False
        next_end_of_file = False
        try:
            next_line = lines[idx+1]
        except IndexError as e:
            end_of_file = True
            next_end_of_file = True
        
        try:
            next_2_line = lines[idx+2]
        except IndexError as e:
            next_end_of_file = True
        
        if not end_of_file:
            if next_line != "\n":

                next_components = next_line.split('\t')
                if len(next_components)==5:
                    next_word, next_word_POS, next_word_BIO, _, _ = next_components
                elif len(next_components)==6:
                    next_word, next_word_POS, next_word_BIO, _, _, _ = next_components
                elif len(next_components)==7:
                    next_word, next_word_POS, next_word_BIO, _, _, _, _ = next_components
                else:
                    print("2, Error in line format on line {}".format(idx+2))
                    exit(10)

                #Features of the next word
                features["next_word"] = next_word
                features["next_POS"] = next_word_POS
                features["next_BIO"] = next_word_BIO

                if not next_end_of_file:
                    if next_2_line !="\n":
                        next_2_components = next_2_line.split('\t')
                        if len(next_2_components)==5:
                            next_2_word, next_2_word_POS, next_2_word_BIO, _, _ = next_2_components
                        elif len(next_2_components)==6:
                            next_2_word, next_2_word_POS, next_2_word_BIO, _, _, _ = next_2_components
                        elif len(next_2_components)==7:
                            next_2_word, next_2_word_POS, next_2_word_BIO, _, _, _, _ = next_2_components
                        else:
                            print("3, Error in line format on line {}".format(idx+3))
                            exit(10)
                        
                        #Features of the word two positions later
                        features["next_2_word"] = next_2_word
                        features["next_2_POS"] = next_2_word_POS
                        features["next_2_BIO"] = next_2_word_BIO
                    else:
                        features["next_2_word"] = None
                        features["next_2_POS"] = None
                        features["next_2_BIO"] = None
                else:
                    features["next_2_word"] = None
                    features["next_2_POS"] = None
                    features["next_2_BIO"] = None

            else:
                #Every word should have the same set of features
                features["next_word"] = None
                features["next_POS"] = None
                features["next_BIO"] = None
                features["next_2_word"] = None
                features["next_2_POS"] = None
                features["next_2_BIO"] = None
        else:
                #Every word should have the same set of features
                features["next_word"] = None
                features["next_POS"] = None
                features["next_BIO"] = None
                features["next_2_word"] = None
                features["next_2_POS"] = None
                features["next_2_BIO"] = None
                

        
        #Handle beginning of sentence
        #Features of the previous word
        if not sent_beginning:
            prev_line_features = feature_lines[idx-1][1]
            features["prev_word"] = prev_line_features["word"]
            features["prev_POS"] = prev_line_features["POS"]
            features["prev_BIO"] = prev_line_features["BIO"]
            

            if not next_sent_beginning:
                #neither the prev word nor the one before that was the beginning of the sentence
                prev_2_line_features = feature_lines[idx-2][1]
                #Features of the word 2 positions before
                features["prev_2_word"] = prev_2_line_features["word"]
                features["prev_2_POS"] = prev_2_line_features["POS"]
                features["prev_2_BIO"] = prev_2_line_features["BIO"]
            
            else:
                next_sent_beginning = False
                features["prev_2_word"] = None
                features["prev_2_POS"] = None
                features["prev_2_BIO"] = None
            
        else:
            sent_beginning = False
            #Every word should have the same set of features
            features["prev_word"] = None
            features["prev_POS"] = None
            features["prev_BIO"] = None
            features["prev_2_word"] = None
            features["prev_2_POS"] = None
            features["prev_2_BIO"] = None

        if args["test_features"]:
            #Test features have no label
            feature_lines.append([word,features])
        else:
            feature_lines.append([word,features,label])

    return feature_lines

def printFeatures(feature_array):
    '''
    Prints the features seperated by a tab
    '''
    for feature in feature_array:
        if feature[0] =="\n":
            print("\n", end="")
            continue
        for idx,item in enumerate(feature):
            if type(item) is dict:
                for key,value in item.items():
                    #Test Features should not end with \t --> throws an error
                    print('\t{}={}'.format(key,value), end="")
            elif idx==0:
                print("{}".format(item),end="")
            else:
                print("\t{}".format(item),end="")
        print("\n",end="")
    return


if __name__ == '__main__':
    #Construct argument parser and parse arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--inputfile", help="Input File")
    parser.add_argument("--test_features", action="store_true", default = False, help="Whether to obtain test features, if not set to true, we get train features")
    parser.add_argument("--arguments_known", action="store_true", default = False, help="Whether the arguments are known to the system")
    parser.add_argument("--distance_features", action="store_true", default = False, help="Whether to keep distance features (Model 1)")
    parser.add_argument("--transparent_noun", action="store_true", default = False, help="Whether to keep transparent-noun related features (Model 2)")
    parser.add_argument("--transparent_noun_path", help="List of transparent nouns")
    parser.add_argument("--support_verb", action="store_true", default = False, help="Whether to keep support-verb related features (Model 3)")
    parser.add_argument("--support_verb_path", help="List of support verbs")
    args = vars(parser.parse_args())

    if args["transparent_noun"]:
        if not args["transparent_noun_path"]:
            print("Please provide a list of transparent nouns")
            exit(0)

    #read in the file to form features
    features = getFeatures(args["inputfile"],  args = args)
    #Print features as tab separated
    printFeatures(features)
