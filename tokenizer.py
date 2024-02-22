from posixpath import sep
from collections import defaultdict
import re

def spacebreaker(line):
  words = re.split(r'\s+|\n', line)
  return words

def aposContractions(words):
  new_list=[]
  flag = 0
  pattern = r"(?<=n)'[a-z]"
  for sen in words:
    if re.search(r'(?<=n\')', sen):
      flag = 1
      #print("preceding n")
      new_list.append(re.split(r'n', sen, 1)[0])
      new_list.append('n' + re.split(r'n', sen,1)[1])
    elif re.search(r'(?<!n)\'', sen):
      #print("String has ' preceded by something other than 'n'.")
      flag = 1
      new_list.append(re.split('\'', sen, 1)[0])
      new_list.append('\''+ re.split('\'',sen,1)[1])
    else:
      #print("String does not contain '.")
      new_list.append(sen)
  return new_list, flag

def splitPunctuations(words):
  new_list = []
  flag = 0
  for sen in words:
    sen = re.sub(r'([.,/><?();:\'"!@#$%^&*])\1+', lambda match: '  ' + match.group() + '  ', sen)
    sen = re.sub(r':\)|:\(|:-\)|:-\(', lambda match: '  ' + match.group() + '  ', sen)
    sen = re.sub(r'(\d+\.\d+)', lambda match: '  ' + match.group() + '  ', sen)
    sen = re.sub(r'([$#@]\w+)', lambda match: '  ' + match.group() + '  ', sen)
    sen = re.sub(r'(\b\w+-\w+\b)', lambda match: '  ' + match.group() + '  ', sen)
    if re.search(r'(\d+\.\d+)', sen):
      flag = 1
      #print(sen)
    new_list.extend(sen.split())
    #print(new_list)
    separated_tokens = []
    for word in new_list:
      exceptions_pattern = r'\d+\.\d+|\w+-\w+|[-$#@]\w+|:\)|:\(|:-\)|:-\(|[.,/><?();:\'"!@#$%^&*]{2,}|n\'[a-z]|\'[a-z]+'
      exceptions = re.findall(exceptions_pattern, word)
      if exceptions:
        separated_tokens.extend(exceptions)
      else:
        pattern = r'(\w+|[^\w\s])'
        generic_pattern = re.findall(pattern,word)
        #print(generic_pattern)
        separated_tokens.extend(generic_pattern)
  return separated_tokens,flag

def wordTokenizer(text):
  words=spacebreaker(text)
  words,flag1 = aposContractions(words)
  words,flag2 = splitPunctuations(words)
  return words

def makeCorpus(lines):
  corpus = defaultdict(int)
  for line in lines:
    words_in_line = line.split()
    for word in words_in_line:
      new_word = ' '.join(word)
      print
      corpus[new_word+' _']+=1
  return corpus


def replaceMaxinCorpus(corpus,best_pair):
  new_corpus = {}
  best_pair_sep = ' '.join(best_pair)
  best_pair_join = ''.join(best_pair)
  for key in corpus:
    new_key = key.replace(best_pair_sep,best_pair_join)
    new_corpus[new_key] = corpus[key]
  return new_corpus


def findMaxPairs(corpus):
  pair_counts = defaultdict(int)
  for key in corpus:
    un_chars = key.split()
    for i in range(len(un_chars)-1):
      pair_counts[un_chars[i],un_chars[i+1]]+=corpus[key]
  return pair_counts


def createInitialVocab(lines):
  vocab = defaultdict(int)
  for line in lines:
    new_line = re.sub(r'[^\x00-\x7F]', '?', line)
    for char in new_line:
      vocab[char]+=1
  return vocab


def spacelessBPELearn(lines):
  vocab = createInitialVocab(lines)
  print(list(vocab.keys()))
  max_iter = 1000-len(vocab)
  corpus = makeCorpus(lines)
  for i in range(max_iter):
    dict_of_pairs = findMaxPairs(corpus)
    #print(list(dict_of_pairs.keys()))
    best_pair = max(dict_of_pairs, key=lambda k: dict_of_pairs[k])
    vocab[''.join(best_pair)]=1
    if (i==0 or i==1 or i==10 or i==100 or i == 500):
      print(''.join(best_pair))
    corpus=replaceMaxinCorpus(corpus,best_pair)
    # if i==1:
    #   break
  #print(vocab)



with open('a1_tweets.txt', 'r') as file:
  lines = file.read().splitlines()
i=0
spacelessBPELearn(lines)
# txt = "#pokemon-hash"
# print(wordTokenizer(txt))
# words = wordTokenizer(txt)
# for line in lines:
#   print(line)
#   words = wordTokenizer(txt)