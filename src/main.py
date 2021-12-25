import os
import csv
import math
import argparse
from collections import Counter

import nltk
from nltk import ngrams
from nltk import word_tokenize
from nltk.util import pr

import numpy as np

# Initialize the argument parser
parser = argparse.ArgumentParser()

# Add the parameters we will pass from cli
parser.add_argument('training_data_path',help='path to the training data folder')
parser.add_argument('test_data_path',help='path to the testing data folder')
parser.add_argument('output_csv_path',help='path to the output csv file')

parser.add_argument('--laplace', default=None, action='store_true',
                    help='type of model to train(--unsmoothed, --laplace, or --interpolation)')
parser.add_argument('--unsmoothed', default=None, action='store_true',
                    help='type of model to train(--unsmoothed, --laplace, or --interpolation)')
parser.add_argument('--interpolation', default=None, action='store_true',
                    help='type of model to train(--unsmoothed, --laplace, or --interpolation)')
                    

# Parse the arguments
args = parser.parse_args() 
#print(args)

# Path to training data folder
train_data_path= args.training_data_path
# Path to testing data folder
test_data_path= args.test_data_path
# Path to output csv file
output_csv_path= args.output_csv_path

# Type of model
type_of_model='unsmoothed' #default value of model we considering

# Replace type_of_model according to argument passed by user
if(args.laplace!=None):
    type_of_model= 'laplace'
if(args.unsmoothed!=None):
    type_of_model= 'unsmoothed'
if(args.interpolation!=None):
    type_of_model= 'interpolation'








'''

This function is responsible for actually creating the n-grams model for a PARTICULAR file and return the models to main function


'''

def get_ngram_models(training_file):

    try:

                            unigram_Counts =Counter()
                            bigram_Counts = Counter()
                            trigram_Counts =Counter()
                            
                            tokens=[]
                            for content in training_file:
                                #print(content) #each line gets stored in this variable
                                #content=content[:-1] #removing new line character
                                
                                chrs = [str.lower(c) for c in content]
                                tokens=tokens+chrs
                                
                                
                            unigrams=tokens
                            #print(unigrams)
                            #print("===================================================")

                            bigrams= list(nltk.ngrams(tokens,2))                            
                            #print(bigrams)
                            #print("===================================================")
                            
                            trigrams = list(nltk.ngrams(tokens,3))
                            #print(trigrams)
                            #print("===================================================")
                            
                            unigram_Counts = Counter(unigrams)
                            #print(unigram_Counts)
                            #print("===================================================")
                            
                            bigram_Counts = Counter(bigrams)
                            #print(bigram_Counts)    
                            #print("===================================================")
                            
                            trigram_Counts = Counter(trigrams)
                            #print(trigram_Counts)
                            #print("===================================================")
                            
                            return unigram_Counts, bigram_Counts, trigram_Counts
                            
    except Exception as e:
        print("Something went wrong!\n The exception message is: ",e)






'''

This function will calculate the Unigram probability of each token in a text file

'''

def calcUnigramProbab(unigram_Counts, tokens):
    

    uni_vocab_size = sum(unigram_Counts.values())
                
    no_of_unigrams=0
    total_unigram_probab=1

    for unigram in tokens:
                        #print(unigram)
                                                
                        no_of_unigrams= no_of_unigrams+1

                        if(type_of_model=='laplace'):
                            numerator = unigram_Counts[unigram] + 1
                            denominator = uni_vocab_size + len(unigram_Counts)
                            #print(numerator)
                            #print(denominator)
                        else:
                            numerator = unigram_Counts[unigram]
                            denominator = uni_vocab_size
                            #print(numerator)
                            #print(denominator)

                        token_probability = float(numerator) / float(denominator) 
                        #print("Probability of Unigram Token: ",token_probability)
                        total_unigram_probab = total_unigram_probab*token_probability

    #print("PROBABILITY OF UNIGRAM MODEL FOR THE FILE IS: ",total_unigram_probab)

    #CALCULATE PERPLEXITY
    try:
        total_unigram_perplexity =  pow(total_unigram_probab, -1/float(no_of_unigrams))
    except:
        total_unigram_perplexity = math.inf
    
    #print("PERPLEXITY OF UNIGRAM MODEL FOR THE FILE IS: ",total_unigram_perplexity)

    #print("=========================================================================")
            
    #RETURN PROBABILITY AND PERPLEXITY
    return total_unigram_probab, total_unigram_perplexity
                





'''

This function will calculate the Bigram probability of each token in a text file

'''

                
def calcBigramProbab(unigram_Counts, bigram_Counts, tokens):
    
    uni_vocab_size = sum(unigram_Counts.values())
    bi_vocab_size = sum(bigram_Counts.values())
                
    no_of_bigrams=0
    total_bigram_probab=1

    for bigram in tokens:
                        #print(bigram)
                        
                        no_of_bigrams= no_of_bigrams+1

                        if(type_of_model=='laplace'):
                            numerator= bigram_Counts[bigram]+1
                            denominator= unigram_Counts[bigram[0]]+ uni_vocab_size
                            #print(numerator)
                            #print(denominator)

                        else:
                            numerator = bigram_Counts[bigram]
                            denominator = unigram_Counts[bigram[0]]
                            #print(numerator)
                            #print(denominator)
 
                        if(denominator==0):
                            token_probability = 0
                        else:
                            token_probability = float(numerator) / float(denominator) 
                        

                        #print("Probability of Bigram Token: ",token_probability)
                        total_bigram_probab = total_bigram_probab*token_probability
                        
    #print("PROBABILITY OF BIGRAM MODEL FOR THE FILE IS:",total_bigram_probab)

    #CALCULATE PERPLEXITY
    try:    
        total_bigram_perplexity =  pow(total_bigram_probab, -1/float(no_of_bigrams))
        
    except:
        total_bigram_perplexity = math.inf
        
    #print("PERPLEXITY of BIGRAM Model for the file is: ", total_bigram_perplexity)

    #print("=========================================================================")
            
    #RETURN PROBABILITY AND PERPLEXITY
    return total_bigram_probab, total_bigram_perplexity
    
                





'''

This function will calculate the Trigram probability of each token in a text file

'''

def calcTrigramProbab(bigram_Counts, trigram_Counts, tokens):   
    
    bi_vocab_size = sum(bigram_Counts.values())
    tri_vocab_size = sum(trigram_Counts.values())
                
    no_of_trigrams=0
    total_trigram_probab=1

    for trigram in tokens:

                        #print(trigram)
                        
                        no_of_trigrams= no_of_trigrams+1

                        if(type_of_model=='laplace'):
                            numerator = trigram_Counts[trigram] + 1
                            denominator = bigram_Counts[trigram[0:2]] + bi_vocab_size
                            #print(numerator)
                            #print(denominator)
                        
                        else:
                            numerator = trigram_Counts[trigram]
                            denominator = bigram_Counts[trigram[0:2]]
                            #print(numerator)
                            #print(denominator)
                        

                        if(denominator==0):
                            token_probability = 0
                        else:
                            token_probability = float(numerator) / float(denominator) 
                        

                        #print("Probability of Trigram Token: ",token_probability)
                        total_trigram_probab = total_trigram_probab*token_probability

    #print("PROBABILITY OF TRIGRAM MODEL FOR THE FILE IS:",total_trigram_probab)

    #CALCULATE PERPLEXITY
    try:    
        total_trigram_perplexity =  pow(total_trigram_probab, -1/float(no_of_trigrams))
        
    except:
        total_trigram_perplexity = math.inf
        
    #print("PERPLEXITY of TRIGRAM Model for the file is: ", total_trigram_perplexity)

    #print("=========================================================================")
            
    #RETURN PROBABILITY AND PERPLEXITY
    return total_trigram_probab, total_trigram_perplexity
     








                    
'''

This function will be used to create n-gram languade models FOR EACH of the 55 training files.
Final models for each file will be stored in a dictionary "final_models_for_files" with key as fileName and value as all models created using that file
After that, we will call the evaluation functions which will decide which model is the best for each dev file according to peplexity value

'''

def main():

    
    try: 

        #dictionary to store models trained on each file
        final_models_for_files = {}

        #get all list of all files in train directory
        files = os.listdir(train_data_path)

        #going through each TRAINING file in the train directory
        for file_name in files:

                if os.path.isfile(os.path.join(train_data_path, file_name)):


                 try:    

                    #opening file in read mode
                    f = open(os.path.join(train_data_path, file_name),'r')
                        
                    #only file name with its extension
                    fileName= os.path.basename(f.name)
                    #print(fileName)

                    #print("Creating NGRAM Models on the training file:", fileName)
                        
                    unigram_Counts =Counter()
                    bigram_Counts = Counter()
                    trigram_Counts =Counter()

                    unigram_Counts, bigram_Counts, trigram_Counts= get_ngram_models(f)

                    #store the 3 models for this training file
                    final_models_for_files[file_name] = [unigram_Counts, bigram_Counts, trigram_Counts]

                    #print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                
                            
                 except Exception as e:
                     print("Something went wrong!\n The exception message is: ",e)



        '''

        We now move to the EVALUATION stage where we use these models on each of our test files and find which model is performing the best for each file.
        
        '''

       
        evaluate_all_dev_files(final_models_for_files, test_data_path)


    except Exception as e:
        print("Something went wrong!\n The exception message is: ",e)








'''

This function will evaluate ALL POSSIBLE MODELS on a SINGLE DEV FILE and return the model that is giving the least perplexity for that dev file

'''

def find_best_model_for_dev_file(f_test, final_models_for_files):

    try:

        #print("We are evaluating the file: ", f_test.name)
        
        tokens=[]
        test_file_sentences=[]

        for content in f_test: #get the content of the test file
        
                    chrs = [str.lower(c) for c in content]
                    tokens=tokens+chrs
            
                    test_file_sentences.append(content.split())
        

        test_file_unigrams=tokens
        #print(test_file_unigrams)
        #print("===================================================")
        test_file_bigrams= list(nltk.ngrams(tokens,2))                            
        #print(test_file_bigrams)
        #print("===================================================")
        test_file_trigrams = list(nltk.ngrams(tokens,3))
        #print(test_file_trigrams)
        #print("===================================================")



        '''
        
        We have the content of the test file. We now have to apply all the ngram models built using the training files 
        on this test file content and then find which ngram model is giving least perplexity.
        
        '''
        
        #print("All training file models we are testing on the test file "+f_test.name+" are:")

        optimal_ngram={} #this will store the model that gives the least perplexity and return back to caller

        for key in final_models_for_files.keys():
            
            #print(key) #name of training file
            
            unigram_Counts=final_models_for_files[key][0]
            bigram_Counts=final_models_for_files[key][1]
            trigram_Counts=final_models_for_files[key][2]

            total_unigram_probab, total_unigram_perplexity = calcUnigramProbab(unigram_Counts, test_file_unigrams)
            #print("======================================================="+str(total_unigram_perplexity)+"===============================================")

            total_bigram_probab, total_bigram_perplexity = calcBigramProbab(unigram_Counts, bigram_Counts, test_file_bigrams)
            #print("======================================================="+str(total_bigram_perplexity)+"===============================================")
            
            total_trigram_probab, total_trigram_perplexity = calcTrigramProbab(bigram_Counts, trigram_Counts, test_file_trigrams)
            #print("======================================================="+str(total_trigram_perplexity)+"===============================================")





            '''
            Find which model gave the best results and store it in the optimal_ngram dictionary with key as training_file name
            and value as a list of 3 values (1. ngram model that gave least perplexity 2.value of n im ngram , 3. answer of least perplexity)
            '''

            if(type_of_model=='interpolation'):


                '''
                Calculate lambda values using deleted interpolation algo using the PARTICULAR TRAIN FILE
                '''
                
                lambda_values= deleted_interpolation_modified(unigram_Counts, bigram_Counts, trigram_Counts)
                
                lambda_uni = lambda_values[0]
                print("Lambda for Unigram:", lambda_uni)
                lambda_bi = lambda_values[1]
                print("Lambda for Bigram:", lambda_bi)
                lambda_tri = lambda_values[2]
                print("Lambda for Trigram:", lambda_tri)

                # Final Linearly Interpolated Probability Score
                linear_score = (lambda_uni*total_unigram_probab) + (lambda_bi* total_bigram_probab) + (lambda_tri*total_trigram_probab)
                print("Final INTERPOLATED Probability given by the modle built on the training file ",key," is: ",str(linear_score))

                # Final Perplexity Score of Linearly Interpolated Model
                if(linear_score!=0):
                    n1=len(test_file_unigrams)
                    n2=len(test_file_bigrams)
                    n3=len(test_file_trigrams)
                    linear_score_perplexity =pow(linear_score, -1/(n1+n2+n3))
                else:
                    linear_score_perplexity =math.inf
                    
                print("Final INTERPOLATED Perplexity given by the model built on the training file ",key," is: ",str(linear_score_perplexity))

                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

                optimal_ngram.update({key:[trigram_Counts, 3, linear_score_perplexity]})
                




            else: #Model is Unsmoothed or Laplace


                if (total_unigram_perplexity <= total_bigram_perplexity and total_unigram_perplexity <= total_trigram_perplexity):
                    
                    #print("UNIGRAM IS PERFORMING BEST!")
                    optimal_ngram.update({key: [unigram_Counts, 1, total_unigram_perplexity]})
        
                elif (total_bigram_perplexity <= total_unigram_perplexity and total_bigram_perplexity <= total_trigram_perplexity):
                    
                    #print("BIGRAM IS PERFORMING BEST!")
                    optimal_ngram.update({key: [bigram_Counts, 2, total_bigram_perplexity]})
        
                else:
                    
                    #print("TRIGRAM IS PERFORMING BEST!")
                    optimal_ngram.update({key: [trigram_Counts, 3, total_trigram_perplexity]})
    
            
            
            
        #print("The best perplexity scores (out of unigram, bigram, trigram.. ) we evaluated for the test file "+f_test.name+" with respect to each train file are:")
        #for key in optimal_ngram.keys():
            #print(key+" N="+str(optimal_ngram[key][1])+" Perplexity="+str(optimal_ngram[key][2]))

        return optimal_ngram
    
    except Exception as e:
        print("Something went wrong!\n The exception message is: ",e)

                        
          
    
                                
                            





'''

This function will go through each dev file one by one and call the evaluation function defined above
to evaluate which training file ngram model is performing the best for that dev file. After that, we will write the output
according to given specifications.

'''

def evaluate_all_dev_files(final_models_for_files, test_data_path):

    try:


        # create the output csv file in which we will write our final results
        with open(output_csv_path, 'w', newline='') as outputFile:

                writer = csv.writer(outputFile)
                # create headers for output file
                writer.writerow(["Training_file", "Testing_file", "Perplexity", "N"])

                #get all list of all files in test directory
                files = sorted(os.listdir(test_data_path))
                
                #going through each dev file in the test directory
                for file_name in files:
	                #print(file_name)
                    if os.path.isfile(os.path.join(test_data_path, file_name)):


                            #opening file in read mode
                            f_test = open(os.path.join(test_data_path, file_name),'r')
                            
                            #only file name with its extension
                            fileName= os.path.basename(f_test.name)

                            print()
                            print("WE ARE EVALUATING THE DEV FILE: ", fileName)
                            print()


                            '''
                            Call the evaluation function on this particular test file
                            '''
                            optimal_ngram_for_particular_test_file = find_best_model_for_dev_file(f_test, final_models_for_files)

                            
                                
                            
                            
                            '''
                            From this dictionary, we find the lowest perplexity value and then corresponding ngram and its training file will be the best possible language model for this test file
                            '''
                            sorted_dict = dict(sorted(optimal_ngram_for_particular_test_file.items(), key=lambda item: item[1][2], reverse=False))

                            first = next(iter(sorted_dict)) #after sorting, first file is the lowest complexity training file
                            best_training_file=first
                            print("Best Training File is : ", best_training_file)
                            best_ngram_model=sorted_dict[first][1]
                            print("Best N-GRAM model is N= ", best_ngram_model)
                            best_perplexity=sorted_dict[first][2]
                            print("Best Perplexity is : ", best_perplexity)

                            print("========================================================")

                            #Now we will write these values to the output file
                            writer.writerow([best_training_file,fileName, best_perplexity, best_ngram_model])
                            
    except Exception as e:
        print("Something went wrong!\n The exception message is: ",e)


                            


'''

FUNCTION TO CALCULATE VALUES OF LAMBDA FOR LINEAR INTERPOLATION

'''

def deleted_interpolation_modified(unigram_Counts, bigram_Counts, trigram_Counts):
    lambda1 = lambda2 = lambda3 = 0
    
    for (a,b,c) in trigram_Counts:
        #print((a,b,c))
        v = trigram_Counts[(a,b,c)]
        #print(v)
        if v > 0:
            try:
                c1 = float(v-1)/(bigram_Counts[(a, b)]-1)
            except ZeroDivisionError:
                c1 = 0
            try:
                c2 = float(bigram_Counts[(b, c)]-1)/(unigram_Counts[(b)]-1)
            except ZeroDivisionError:
                c2 = 0
            try:
                c3 = float(unigram_Counts[(c)]-1)/(sum(unigram_Counts.values())-1)
            except ZeroDivisionError:
                c3 = 0

            k = np.argmax([c1, c2, c3])
            if k == 0:
                lambda3 += v
            if k == 1:
                lambda2 += v
            if k == 2:
                lambda1 += v

    weights = [lambda1, lambda2, lambda3]
    norm_w = [float(a)/sum(weights) for a in weights]
    #print(norm_w)
    return norm_w





    
if __name__ == "__main__":
    main()
