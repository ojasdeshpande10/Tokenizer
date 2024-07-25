
import csv
from tokenizer import tokenized
from tokenizer import spacelessBPEtokenize
from tokenizer import spacelessBPELearn
from tokenizer import wordTokenizer
import torch.nn as nn
from logistic import LogisticRegressionModel
import torch.optim as optim
import re
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import f1_score

class WSDModel:
    train_data = []
    dev_data = []
    test_sentence_data=[]
    train_sentence_data=[]
    train_sentence_data_w_label=[]
    vocab_Bpe_token=[]
    vocab_word_token=[]
    vocab_Bpe={}
    vocab_wordT={}
    vocab_500_BPE={}
    vocab_500_word={}
    dict_of_labels={}
    dict_of_features={}
    dict_of_features_dev={}
    dict_of_labels_dev={}
    dict_of_models={}
    labels={}
    labels_list={}
    left_side_tokens_list=[]
    right_side_tokens_list=[]
    all_tokens_list = []
    left_side_tokens_list_word=[]
    right_side_tokens_list_word=[]
    all_tokens_list_word = []
    models_list=[]
    models_hyperparameters=[]
    models_hidden_dim=[]

        
    def __init__(self):
        print("object got created")

    def trainLogReg(self, feature_tensor, list_of_labels, input_size, num_classes, key, learning_rate=1, epochs=200,weight_decay=0.001,dropout_rate=0.5,hidden_dimensions=0):
        model = LogisticRegressionModel(input_size, num_classes,dropout_rate,hidden_dimensions)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate,weight_decay=weight_decay)
        # Train the model
        loss_values=[]
        for epoch in range(epochs):
            outputs = model(feature_tensor,hidden_dimensions)
            loss = criterion(outputs, torch.tensor(list_of_labels))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_values.append(loss.item())
            # print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}',flush=True)
        
        # Plots code is commented because it gives run-time error while running the hyperparameter tuning
        
        # plt.figure(figsize=(10, 6))
        # plt.plot(range(1, epochs+1), loss_values, label='Train Loss')
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.title('Training Loss Curve for '+key)
        # plt.legend()
        # plt.savefig('training-loss-curve-'+key+'.png',dpi = 300)
        # for name, param in model.state_dict().items():
        #     if name == 'linear.weight':
        #         print(f'Weights (Betas) of the model: \n{name} \n{param}')
        #     if name == 'linear.bias':
        #         print(f'Bias of the model: \n{name} \n{param}')
        return model
    def improve(self,hp,hd):
        models_dict={}
        for key in self.dict_of_labels:
            train_corpus=[]
            num_of_classes = len(self.labels_list[key])
            feature_tensor = torch.stack(self.dict_of_features[key],dim=0)
            model = self.trainLogReg(feature_tensor, self.dict_of_labels[key], 1503, num_of_classes, key, learning_rate=hp[0], weight_decay=hp[1],dropout_rate=hp[2],hidden_dimensions=hd)
            models_dict[key]=model
        self.models_list.append(models_dict)
        self.models_hyperparameters.append(hp)
        self.models_hidden_dim.append(hd)

    def createModels(self,hp):
        models_dict={}
        # print("Check point 2.2")
        # print("****Model_Parameters******")
        for key in self.dict_of_labels:
            train_corpus=[]
            num_of_classes = len(self.labels_list[key])
            feature_tensor = torch.stack(self.dict_of_features[key],dim=0)
            model = self.trainLogReg(feature_tensor, self.dict_of_labels[key], 1503, num_of_classes, key, learning_rate=hp[0], weight_decay=hp[1],dropout_rate=hp[2])
            models_dict[key]=model
        self.models_list.append(models_dict)
        self.models_hyperparameters.append(hp)
    def crossVal(self):
        i=0
        #print("CheckPoint 2.3")
        #print("Word Tokenizer")
        #print("Bpe")
        print("Checkpoint 2.4")
        for model_dict in self.models_list:
            avg=0
            print("*****learning_rate= "+str(self.models_hyperparameters[i][0])+" weight_decay= "+str(self.models_hyperparameters[i][1])+" dropout_rate= " +str(self.models_hyperparameters[i][2])+ "*******")
            #print("Number of Hidden dimensions "+str(self.models_hidden_dim[i]))
            for key in model_dict:
                # Optionally, calculate F1 score every epoch or after certain intervals
                
                feature_tensor = torch.stack(self.dict_of_features_dev[key],dim=0)
                true_labels = torch.tensor(self.dict_of_labels_dev[key])
                with torch.no_grad():
                    if len(self.models_hidden_dim) == 0:
                        dev_outputs = model_dict[key](feature_tensor,0)
                    else:
                        dev_outputs = model_dict[key](feature_tensor,self.models_hidden_dim[i])
                    _, predicted_labels = torch.max(dev_outputs, 1)
                    f1 = f1_score(true_labels.numpy(), predicted_labels.numpy(), average='macro')
                    avg+=f1
                    
            print("Average F1 score: "+str(avg/6))
            i+=1
        

    def createVocab(self):
        # Finding vocab for Byte Pair tokenization
         
         vocab = spacelessBPELearn(self.train_sentence_data_w_label,1000)
         self.vocab_Bpe_token = vocab
         i=0
         for line in self.train_sentence_data:
                pattern = r'<<([^>]*)>>' 
                match = re.search(pattern,line)
                #print(line)
                updated_sen = re.sub(pattern,'',line)
                left_part=""
                right_part=""
                if match:
                    start = match.start()
                    end = match.end()
                    left_part = line[:start]
                    right_part = line[end:]
                else:
                    print("Label wasn't picked--2")
                left_tokens = spacelessBPEtokenize(left_part,vocab)
                right_tokens = spacelessBPEtokenize(right_part,vocab)
                all_tokens = spacelessBPEtokenize(updated_sen,vocab)

                self.left_side_tokens_list.append(left_tokens)
                self.right_side_tokens_list.append(right_tokens)
                self.all_tokens_list.append(all_tokens)
                i+=1
         for tokens in self.all_tokens_list:
                for token in tokens:
                    if token in self.vocab_Bpe:
                        self.vocab_Bpe[token]+=1
                    else:
                        self.vocab_Bpe[token]=1
         vocab_500_t = sorted(self.vocab_Bpe.items(), key=lambda item: item[1], reverse=True)[:500]
         self.vocab_500_BPE = [t[0] for t in vocab_500_t]
         #print(self.vocab_500_BPE)

        # Finding vocab for word Tokenization

         for line in self.train_sentence_data:
                pattern = r'<<([^>]*)>>' 
                match = re.search(pattern,line)
                #print(line)
                updated_sen = re.sub(pattern,'',line)
                left_part=""
                right_part=""
                if match:
                    start = match.start()
                    end = match.end()
                    left_part = line[:start]
                    right_part = line[end:]
                else:
                    print("Label wasn't picked--2")
                
                left_tokens = wordTokenizer(left_part)
                right_tokens = wordTokenizer(right_part)
                all_tokens = wordTokenizer(updated_sen)

                self.left_side_tokens_list_word.append(left_tokens)
                self.right_side_tokens_list_word.append(right_tokens)
                self.all_tokens_list_word.append(all_tokens)
         dict_word={}
         for tokens in self.all_tokens_list_word:
            for token in tokens:
                if token in dict_word:
                    dict_word[token]+=1
                else:
                    dict_word[token]=1
         
         v_500_word = sorted(dict_word.items(), key=lambda item: item[1], reverse=True)[:500]
         self.vocab_500_word = [t[0] for t in v_500_word]
         #print(self.vocab_500_word)
        

    
    def extractLexicalFeatures(self, tokens, target, tokenizer_type,i):
        
        feature_tensor_preceeding = torch.zeros(501)
        feature_tensor_proceeding = torch.zeros(501)
        feature_tensor_sentence_multi=torch.zeros(501)
        left_token = tokens[0]
        right_token = tokens[1]
        all_tokens = tokens[2]
        if tokenizer_type=="bpe":
            if left_token in self.vocab_500_BPE:
                feature_tensor_preceeding[self.vocab_500_BPE.index(left_token)] = 1
            else:
                feature_tensor_preceeding[500]=1
            if right_token in self.vocab_500_BPE:
                feature_tensor_proceeding[self.vocab_500_BPE.index(right_token)] = 1
            else:
                feature_tensor_proceeding[500]=1

            for token in all_tokens:
                if token in self.vocab_500_BPE:
                    # print("In sentence")
                    feature_tensor_sentence_multi[self.vocab_500_BPE.index(token)] = 1
                else:
                    #print("Not in sentence")
                    feature_tensor_sentence_multi[500]=1
        
        else:
            if left_token in self.vocab_500_word:
                feature_tensor_preceeding[self.vocab_500_word.index(left_token)] = 1
            else:
                feature_tensor_preceeding[500]=1
            if right_token in self.vocab_500_word:
                feature_tensor_proceeding[self.vocab_500_word.index(right_token)] = 1
            else:
                feature_tensor_proceeding[500]=1

            for token in all_tokens:
                if token in self.vocab_500_word:
                    # print("In sentence")
                    feature_tensor_sentence_multi[self.vocab_500_word.index(token)] = 1
                else:
                    #print("Not in sentence")
                    feature_tensor_sentence_multi[500]=1
        
        final_feature = torch.cat((feature_tensor_preceeding,feature_tensor_proceeding,feature_tensor_sentence_multi),dim=0)
        return final_feature
    
    def createDevData(self,type):
        print("Creating Dev data")
        i=0
        for record in self.dev_data:
            line = record[3]
            pattern = r'<<([^>]*)>>' 
            match = re.search(pattern,line)
            #print(line)
            updated_sen = re.sub(pattern,'',line)
            left_part=""
            right_part=""
            if match:
                start = match.start()
                end = match.end()
                left_part = line[:start]
                right_part = line[end:]
            else:
                print("Label wasn't picked--2")
            
            if type=="bpe":
                left_part_tokens = spacelessBPEtokenize(left_part,self.vocab_Bpe_token)
                right_part_tokens = spacelessBPEtokenize(right_part,self.vocab_Bpe_token)
                all_tokens = spacelessBPEtokenize(updated_sen,self.vocab_Bpe_token)
            
            else:
                left_part_tokens = wordTokenizer(left_part)
                right_part_tokens = wordTokenizer(right_part)
                all_tokens = wordTokenizer(updated_sen)


            if len(left_part_tokens)!=0:
                left_token = left_part_tokens[-1]
            else:
                left_token = '<unknown>'
            
            if len(right_part_tokens)!=0:
                right_token = right_part_tokens[0]
            else:
                right_token = '<unknown>'
            
            token_list=[]
            token_list.append(left_token)
            token_list.append(right_token)
            token_list.append(all_tokens)
            pattern = r'^(.*?)%'
            match = re.search(pattern,record[2])
            result=""
            if match:
                result = match.group(1)
            else:
                print("Label wasn't picked--1")
            if type=="bpe":
                feature = self.extractLexicalFeatures(token_list,result,"bpe",i)
            else:
                feature = self.extractLexicalFeatures(token_list,result,"wordTokenizer",i)
            # if i==1 or i==0 or i==3967:
            #     print(record[3])
            #     print(result)
            #     print(record[2])
            #     print(torch.sum(feature))
            #     torch.set_printoptions(threshold=5000)
            
            if result not in self.dict_of_features_dev:
                self.dict_of_features_dev[result] = []
            if result not in self.dict_of_labels_dev:
                self.dict_of_labels_dev[result] = []
            
            self.dict_of_labels_dev[result].append(self.labels_list[result].index(record[2]))
            self.dict_of_features_dev[result].append(feature)
            i+=1
        print(i)

    def extractLabels(self, label_name,label):
        
        if label_name not in self.labels:
            self.labels[label_name]=set()
            print(label_name)
            print(label)
        self.labels[label_name].add(label)


    def createTrainData(self,type):
        i=0
        print("CheckPoint 2.1")
        print("The first, second and last document's features")
        if type == "bpe":
            for left,right,full_sentence,record in zip(self.left_side_tokens_list,self.right_side_tokens_list,self.all_tokens_list,self.train_data):

                if len(left)!=0:
                    left_token = left[-1]
                else:
                    left_token = '<unknown>'
                
                if len(right)!=0:
                    right_token = right[0]
                else:
                    right_token = '<unknown>'

                token_list=[]
                token_list.append(left_token)
                token_list.append(right_token)
                token_list.append(full_sentence)
                pattern = r'^(.*?)%'
                match = re.search(pattern,record[2])
                result=""
                if match:
                    result = match.group(1)
                else:
                    print("Label wasn't picked--1")

                feature = self.extractLexicalFeatures(token_list,result,"bpe",i)
                
                # if i==1 or i==0 or i==3967:
                #     torch.set_printoptions(precision=2, threshold=5000, edgeitems=3, linewidth=80)
                #     print(feature)
                #     print(torch.sum(feature))
                if result not in self.dict_of_labels:
                    self.dict_of_labels[result] = []
                if result not in self.dict_of_features:
                    self.dict_of_features[result] = []
                self.dict_of_labels[result].append(self.labels_list[result].index(record[2]))
                self.dict_of_features[result].append(feature)
                i+=1
        else:

            for left,right,full_sentence,record in zip(self.left_side_tokens_list_word,self.right_side_tokens_list_word,self.all_tokens_list_word,self.train_data):

                if len(left)!=0:
                    left_token = left[-1]
                else:
                    left_token = '<unknown>'
                
                if len(right)!=0:
                    right_token = right[0]
                else:
                    right_token = '<unknown>'

                token_list=[]
                token_list.append(left_token)
                token_list.append(right_token)
                token_list.append(full_sentence)
                pattern = r'^(.*?)%'
                match = re.search(pattern,record[2])
                result=""
                if match:
                    result = match.group(1)
                else:
                    print("Label wasn't picked--1")

                feature = self.extractLexicalFeatures(token_list,result,"word",i)
                
                # if i==1 or i==0:
                #     print(record[3])
                #     print(result)
                #     print(record[2])
                #     print(torch.sum(feature))
                #     torch.set_printoptions(threshold=5000)
                
                if result not in self.dict_of_labels:
                    self.dict_of_labels[result] = []
                if result not in self.dict_of_features:
                    self.dict_of_features[result] = []
                self.dict_of_labels[result].append(self.labels_list[result].index(record[2]))
                self.dict_of_features[result].append(feature)
                i+=1

        print(i)
        
        
    def dataHandler(self):
        #Reading the data from the file, instantiating the dataset
        with open('C:\\Users\\ojasd\\Desktop\\NLP_A1\\Tokenizer\\a1_wsd_24_2_10.txt', 'r',newline='',encoding='utf-8') as file:
            reader = csv.reader(file, delimiter='\t')  # Set delimiter to tab
            i=0
            for row in reader:
                if(i<3968):
                    self.train_data.append(row)
                else:
                    self.dev_data.append(row)
                pattern = r'^(.*?)%'
                match = re.search(pattern,row[2])
                result=""
                if match:
                    result = match.group(1)
                self.extractLabels(result,row[2])  
                i+=1       
        i=0
        for record in self.train_data:
            pattern = r'<<\w+>>'
            updated_sen = re.sub(pattern,'',record[3])
            self.train_sentence_data.append(record[3])
            self.train_sentence_data_w_label.append(updated_sen)
            i+=1
        for record in self.dev_data:
            pattern = r'<<\w+>>'
            updated_sen = re.sub(pattern,'',record[3])
            self.test_sentence_data.append(updated_sen)
        for key in self.labels:
                self.labels_list[key] = list(self.labels[key])
                print(key)
                print(len(self.labels_list[key]))

        




        

