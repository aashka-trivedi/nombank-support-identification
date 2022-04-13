import argparse

def getSupportSentences(filename):
    '''
    Gets only those sentences having a support from filename
    '''
    training_corpus = open(filename,'r')
    lines  = training_corpus.readlines()
    training_corpus.close()
    sentence = []
    support_sentences = []
    contains_support = False

    for line in lines:

        sentence.append(line)
        if line.strip().split("\t")[-1] == "SUPPORT":
            contains_support = True
        
        if line =="\n":
            if contains_support:
                support_sentences.append(sentence)
                contains_support = False
            sentence = []
    return support_sentences

def printSentences(sentences):
    for sentence in sentences:
        for line in sentence:
            print(line, end="")
    return


if __name__ == '__main__':

    #Construct argument parser and parse arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--inputfile", help="Input File")
    args = vars(parser.parse_args())

    printSentences(getSupportSentences(args["inputfile"]))