from posixpath import sep
from a1_p1_deshpande_115837247 import tokenized
from a1_p2_deshpande_115837247 import WSDModel
import time
def main():
    start_time=time.time()

    # code of lines to run 1st part of assignment
    # with open('C:\\Users\\\ojasd\\Desktop\\NLP_A1\\Tokenizer\\a1_tweets.txt', 'r', encoding='utf-8') as file:
    #     lines = file.readlines()
    # tokenized_data,_= tokenized(lines,"bpe")
    # #print(tokenized_data)
    
    model = WSDModel()
    model.dataHandler()
    model.createVocab()
    model.createTrainData("bpe")
    model.createDevData("bpe")
    # can use next two lines for different tokenizer: word tokenizer
    # model.createTrainData("wordTokenizer")
    # model.createDevData("wordTokenizer")
    
    
    # this line was used to get the 2.2 and 2.3 outputs
    # model.createModels([1,0.001,0.1])
    
    # Hyperparameter tuning code


    dropout_rate=[0.1,0.2,0.3,0.5]
    weight_decay=[0.001,0.01,0.1,1,10,100]
    for dr in dropout_rate:
        for wd in weight_decay:
            model.createModels([1,wd,dr])
    

    # 2.5 improvement code 
    # hidden_dimensions = [50,100,200,500]
    # for hd in hidden_dimensions:
    #     model.improve([1,0.001,0.1],hd)
    

    model.crossVal()
    end_time = time.time()
    print(end_time-start_time)



if __name__ == "__main__":
    main()