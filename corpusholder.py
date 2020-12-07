from itertools import chain
from collections import defaultdict
from collections import Counter
from numpy import asarray
from sklearn.utils import shuffle
import re
from nltk.corpus import stopwords
import torch
from torch.nn.utils.rnn import PackedSequence, pack_sequence
from torch.utils.data import Dataset, DataLoader


# holds the language i.e. the vocabulary and words to index mappings
# max_vocab_size controls how many most popular words are kept; rest are assigned to UNK_IDX
class Lang:
    def __init__(self, texts, max_vocab_size=None):
        tokenized_texts = [[word for word in text.split()] for text in texts]
        counts = Counter(chain(*tokenized_texts))
        max_vocab_size = max_vocab_size or len(counts)
        common_pairs = counts.most_common(max_vocab_size)
        self.UNK_IDX = 0
        self.EOS_IDX = 1
        self.itos = ["<UNK>", "<EOS>"] + [pair[0] for pair in common_pairs]
        self.stoi = {token: i for i, token in enumerate(self.itos)}
        print("Lang=%i" % len(self))

    def __iter__(self):
        return iter(self.itos)

    def __len__(self):
        return len(self.itos)          
           
            
# the data holder class to be wrapped into torch Dataloader
class Dataset:
    def __init__(self, texts, labels, lang):
        self.texts = texts
        self.labels = labels
        self.lang = lang               
        
    def __getitem__(self, item):
        sentence = self.texts[item]
        indexes = [self.lang.stoi.get(word, self.lang.UNK_IDX) for word in sentence.split()]
        return indexes + [self.lang.EOS_IDX], self.labels[item]

    def __len__(self):
        return len(self.texts)

    def collate_fn(self, batch):
        return pack_sequence([torch.tensor(pair[0]) for pair in batch], enforce_sorted=False), torch.tensor(
            [pair[1] for pair in batch])        

    
# holds the corpus and its corresponding labels
# splits for train, validation and test as per defined fractions
# provides the DataLoader-s for training and evaluation
class CorpusHolder:
    def __init__(self, texts, labels, batch_size=128, max_vocab_size=30000, max_length=1000, val_size=0.1, test_size=0.1):
        self.max_vocab_size = max_vocab_size
        self.max_length = max_length
        self.batch_size=batch_size
        self.lang, self.texts, self.labels = self.prepare_data(texts, labels)
        train_last_index = int(len(self.texts) * (1 - val_size - test_size) )
        val_last_index = int(len(self.texts) * (1 - test_size) )
        self.train_texts, self.train_labels = self.texts[:train_last_index], self.labels[:train_last_index]
        self.val_texts, self.val_labels = self.texts[train_last_index:val_last_index], self.labels[train_last_index:val_last_index]
        
        self.train_dataset = Dataset(self.train_texts, self.train_labels, self.lang)
        self.val_dataset = Dataset(self.val_texts, self.val_labels, self.lang)
        self.test_dataset = Dataset(self.texts[val_last_index:], self.labels[val_last_index:], self.lang)
        print('CorpusHolder train_dataset=%i val_dataset=%i test_dataset=%i' % (len(self.train_dataset), len(self.val_dataset), len(self.test_dataset)))
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=8, collate_fn=self.train_dataset.collate_fn)
        self.val_dataloader  = DataLoader(self.val_dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=8, collate_fn=self.val_dataset.collate_fn)            
        self.test_dataloader  = DataLoader(self.test_dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=8, collate_fn=self.test_dataset.collate_fn)
        self.budgeted_train_dataloaders = {}
        self.budgeted_val_dataloaders = {}

    
    def get_budgeted_dataloader(self, texts, labels, storage, budget, storagetype):
        if not budget in storage:
            budgeted_texts, budgeted_labels = shuffle(texts, labels)
            budget_last_index = int(len(texts) * (budget / 100.0) )
            # lets avoid extra small n samples, if possible
            if len(texts) > 1000 and budget_last_index < 1000:
                budget_last_index = 1000
            budgeted_texts, budgeted_labels = budgeted_texts[:budget_last_index], budgeted_labels[:budget_last_index]
            budgeted_dataset = Dataset(budgeted_texts[:budget_last_index], budgeted_labels[:budget_last_index], self.lang)
            storage[budget] = DataLoader(budgeted_dataset, batch_size=self.batch_size, 
                                         shuffle=True, num_workers=8, collate_fn=budgeted_dataset.collate_fn)
            print('produced budgeted=%.2f  %s dataset=%i' % (budget, storagetype, budget_last_index))
        return storage[budget]       
        
        
    def get_budgeted_train_dataloader(self, budget):
        if budget == 100.0:
            return self.train_dataloader       
        return self.get_budgeted_dataloader(self.train_texts, self.train_labels, 
                                            self.budgeted_train_dataloaders, budget, 'train')

    
    def get_budgeted_val_dataloader(self, budget):
        if budget == 100.0:
            return self.val_dataloader       
        return self.get_budgeted_dataloader(self.val_texts, self.val_labels, 
                                            self.budgeted_val_dataloaders, budget, 'val')

    
    def clean_text(self, text):
        text = text.lower()
        # Replace contractions with their longer forms 
        contractions = { "ain't": "am not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have",
            "'cause": "because", "could've": "could have", "couldn't": "could not", "couldn't've": "could not have", 
            "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hadn't've": "had not have",
            "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'd've": "he would have",
            "he'll": "he will", "he's": "he is", "how'd": "how did", "how'll": "how will",
            "how's": "how is", "i'd": "i would", "i'll": "i will", "i'm": "i am", "didn't": "did not",
            "i've": "i have", "isn't": "is not", "it'd": "it would", "it'll": "it will",
            "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not",
            "might've": "might have", "mightn't": "might not", "must've": "must have", "mustn't": "must not",
            "needn't": "need not", "oughtn't": "ought not", "shan't": "shall not", "sha'n't": "shall not",
            "she'd": "she would", "she'll": "she will", "she's": "she is", "should've": "should have",
            "shouldn't": "should not", "that'd": "that would", "that's": "that is", "there'd": "there had",
            "there's": "there is", "they'd": "they would", "they'll": "they will", "they're": "they are",
            "they've": "they have", "wasn't": "was not", "we'd": "we would", "we'll": "we will",
            "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will",
            "what're": "what are", "what's": "what is", "what've": "what have", "where'd": "where did",
            "where's": "where is", "who'll": "who will", "who's": "who is", "won't": "will not",
            "wouldn't": "would not", "you'd": "you would", "you'll": "you will", "you're": "you are"}        
        
        text = text.split()
        new_text = []
        for word in text:
            if word in contractions:
                new_text.append(contractions[word])
            else:
                new_text.append(word)
        text = " ".join(new_text)

        # format & remove punctuation
        word_pattern = re.compile("[\w']+")
        words = word_pattern.findall(text)
        
        # deal with stopwords
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
        text = " ".join(words)        

        return text


    def prepare_data(self, texts, labels):
        texts = [self.clean_text(text) for text in texts]
        
        # now validate texts length
        new_texts = []
        new_labels = []
        for i in range(len(texts)):
            text = texts[i]
            words = text.split()
            if len(text) > 0 and len(words) > 0 and len(words[0]) > 0 and len(words) < self.max_length:
                new_texts.append(text)
                new_labels.append(labels[i])
        self.lang = Lang(new_texts, max_vocab_size=self.max_vocab_size)
        return self.lang, new_texts, new_labels