import argparse

def score(responseFileName, keyFileName):
    '''
    Finding the accuracy of the response
    '''
    keyFile = open(keyFileName, 'r')
    key = keyFile.readlines()
    responseFile = open(responseFileName, 'r')
    response = responseFile.readlines()
    if len(key) != len(response):
        print("length mismatch between key and submitted file")
        exit()
    true_pos = 0
    false_neg = 0
    false_pos = 0
    true_neg = 0
    total = 0
    predicted = 0

    for idx in range(len(key)):
        if key[idx] == "\n":
            if response[idx] =="\n":
                continue
            else:
                print("Blank line expected on line {}".format(idx))
                exit()
        if key[idx].strip().split("\t")[-1] =="SUPPORT":
            total +=1
            if response[idx].strip().split("\t")[-1] =="SUPPORT":
                true_pos +=1
                predicted+=1
            else:
                false_neg +=1
        else: 
            if response[idx].strip().split("\t")[-1] !="SUPPORT":
                true_neg +=1
            else:
                false_pos +=1
                predicted+=1
        
    recall = float(true_pos/(true_pos+false_neg))
    precision = float(true_pos/(true_pos+false_pos))
    print("Total number of predicted Support Words: {}".format(predicted))
    #print("{} out of {} Arguments correct".format(true_pos,total))
    print("Precision = {}".format(precision))
    print("Recal = {}".format(recall))
    print("F1 Score = {}".format((2*precision*recall)/(precision+recall)))
    return



if __name__ == '__main__':
    #Construct argument parser and parse arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--responsefile", help="System Response File")
    parser.add_argument("--keyfile", help="Gold Data File")
    args = vars(parser.parse_args())

    score(args["responsefile"], args["keyfile"])