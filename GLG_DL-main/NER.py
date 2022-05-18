#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 12:04:42 2022

@author: prithvi
"""

import pandas as pd
import os
import random
from tqdm import tqdm
from pandas.core.groupby import groupby
import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer, AutoConfig, TFAutoModel
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from gensim.summarization.textcleaner import split_sentences

PRETRAINED_MODEL_NAME = 'bert-base-uncased'
FINETUNED_MODEL_NAME = 'finetuned_' + PRETRAINED_MODEL_NAME
FILE_DIR = '/content/drive/My Drive/fourthbrain/NER_Labels'
SEQUENCE_LENGTH = 128
BATCH_SIZE = 32
EPOCHS = 8
RUN_TRAINING = False

class NER_Model:
    def __init__(self):
        #Initialize the pretrained model
        self.config = AutoConfig.from_pretrained(PRETRAINED_MODEL_NAME)
        self.backbone = TFAutoModel.from_pretrained(PRETRAINED_MODEL_NAME,config=self.config)
    
    def build_model(self, num_classes, use_finetuned=False):
        tokens = tf.keras.layers.Input(shape=(SEQUENCE_LENGTH,), name = 'tokens', dtype=tf.int32)
        att_masks = tf.keras.layers.Input(shape=(SEQUENCE_LENGTH,), name = 'attention', dtype=tf.int32)
        
        features = self.backbone(tokens, attention_mask=att_masks)[0]
        
        target = tf.keras.layers.Dropout(0.5)(features)
        target = tf.keras.layers.Dense(num_classes, activation='softmax')(target)
        
        self.model = tf.keras.Model([tokens,att_masks],target)

        self.model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                           loss=tf.keras.losses.sparse_categorical_crossentropy,
                           metrics=['accuracy'])
        if(use_finetuned):
            self.model.load_weights(os.path.join(FILE_DIR, FINETUNED_MODEL_NAME))

    def train_model(self, x_data_in, x_data_att, y_data, x_data_in_val, x_data_att_val, y_data_val):
        history = self.model.fit(x = [x_data_in, x_data_att], y = y_data, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=([x_data_in_val, x_data_att_val], y_data_val))

    def save_model(self):
        self.model.save_pretrained(os.path.join(FILE_DIR, FINETUNED_MODEL_NAME))


class NERDataset:
    def __init__(self):
        #Read the data files
        self.dataset = pd.read_csv(os.path.join(FILE_DIR, 'ner_dataset.csv'), encoding = 'ISO-8859-1')

        #Preprocess the dataset
        self.dataset["Sentence #"] = self.dataset["Sentence #"].fillna(method="ffill")

        #Convert tags into labels using label encoder
        self.tag_encoder = LabelEncoder()
        self.dataset.loc[:, 'Tag'] = self.tag_encoder.fit_transform(self.dataset['Tag'])
        self.background_class = self.tag_encoder.transform(['O'])[0]

        self.sentences = self.dataset.groupby("Sentence #")["Word"].apply(list).values
        self.tags = self.dataset.groupby("Sentence #")["Tag"].apply(list).values
 
    def build_ner_dataset(self, mode='train'):
        
        #Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME,normalization=True)
        self.config = AutoConfig.from_pretrained(PRETRAINED_MODEL_NAME)

        self.input_ids = []
        self.attention_masks = []
        self.token_type_ids = []
        for sentence in self.sentences:
            encoded = self.tokenizer.encode_plus(sentence,
                                       add_special_tokens = True,
                                       max_length = SEQUENCE_LENGTH,
                                       is_split_into_words=True,
                                       return_attention_mask=True,
                                       padding = 'max_length',
                                       truncation=True,return_tensors = 'np')
            self.input_ids.append(encoded['input_ids'])
            self.attention_masks.append(encoded['attention_mask'])
            self.token_type_ids.append(encoded.word_ids())
            #print('Length of sentence:{}, Length of encoded:{}'.format(len(sentence), encoded['input_ids'].shape))
        self.input_ids = np.vstack(self.input_ids)
        self.attention_masks = np.vstack(self.attention_masks)
        self.token_type_ids = np.vstack(self.token_type_ids)

        

        self.tags_proper = []
        for ntag, tag in enumerate(self.tags):
            word_ids = self.token_type_ids[ntag][self.token_type_ids[ntag] != np.array(None)]
            tag_proper = [tag[i] for i in word_ids]
            self.tags_proper.append(tag_proper)
        
        self.targets = np.ones([self.input_ids.shape[0], SEQUENCE_LENGTH], dtype=np.int32)*self.background_class
        for n, tag in enumerate(self.tags_proper):
            tag_len = len(tag)
            self.targets[n,1:tag_len+1] = np.array(tag)
    
    
    def test_train_split(self):
        self.seq_train, self.seq_test, self.mask_train, self.mask_test, self.target_train, self.target_test, self.word_id_train, self.word_id_test = train_test_split(self.input_ids, self.attention_masks, self.targets, self.token_type_ids, test_size=0.20, random_state=42)
    

#Create the end-to-end pipeline and test on new dataset
class NER_Pipeline:
    def __init__(self):
        #Load the tokenizer in the dataset
        self.dataset = NERDataset()
        self.dataset.build_ner_dataset()
        self.dataset.test_train_split()
        n_classes = self.dataset.tag_encoder.classes_.shape[0]

        #Load the finetuned model
        self.ner_modeler = NER_Model()
        self.ner_modeler.build_model(n_classes, True)

        self.LABEL_CONVERT = {'org': 'ORG',
                              'tim': 'DATE',
                              'per': 'PERSON',
                              'geo': 'GEO',
                              'gpe': 'GPE',
                              'art': 'ART',
                              'eve': 'EVE',
                              'nat': 'NAT',
                              }
    
    def convert_label(self, label):
        return self.LABEL_CONVERT[label]

    def run_ner_on_sentence(self, sample_text):
        #Tokenize the sample text, and get the word ids
        encoded = self.dataset.tokenizer.encode_plus(sample_text,
                                            add_special_tokens = True,
                                            max_length = SEQUENCE_LENGTH,
                                            is_split_into_words=True,
                                            return_attention_mask=True,
                                            padding = 'max_length',
                                            truncation=True,return_tensors = 'np')
        input_seq = encoded['input_ids']
        att_mask = encoded['attention_mask']
        word_ids = encoded.word_ids()

        #Predict the classes for each token
        sample_out = self.ner_modeler.model.predict([input_seq, att_mask])
        sample_out = np.argmax(sample_out, axis=2)
        word_ids = np.array(word_ids)
        valid_sample_out = sample_out[0, word_ids!=None]
        valid_word_ids = word_ids[word_ids!=None]
        names = [sample_text[i] for i in valid_word_ids[valid_sample_out!=self.dataset.background_class]]
        labels = [self.dataset.tag_encoder.inverse_transform([i])[0] for i in valid_sample_out[valid_sample_out!=self.dataset.background_class]]

        #Combine the tokens and correponding labels. Output the final names and their corresponding classes
        full_names = []
        full_labels = []
        prev_index = -1
        completed = {}
        for name, label in zip(names, labels):
            if(name not in completed):
                if(label[0]=='B'):
                    full_names.append(name)
                    full_labels.append(self.convert_label(label[2:]))
                    prev_index += 1
                else:
                    if(len(full_names)>0):
                        full_names[prev_index] = full_names[prev_index] + ' ' + name
                    else:
                        continue
                completed[name] = 1
        return full_names, full_labels
    
    def run_ner(self, full_text, display=False):
        #sentences = split_sentences(full_text)
        sentences = full_text.split('.')
        names = []
        labels = []
        start_idxs = []
        stop_idxs = []
        total_start_len = 0
        for sentence in sentences:
            snames, slabels = self.run_ner_on_sentence(sentence.split(' '))
            for key, value in zip(snames, slabels):
                start = sentence.find(key)
                start_idxs.append(total_start_len + start)
                stop_idxs.append(total_start_len + start + len(key))
            total_start_len += len(sentence) + 1
            names.extend(snames)
            labels.extend(slabels)
        
        if(display):
            for name, label in zip(names, labels):
                print(name, label)
        return names, labels, start_idxs, stop_idxs

if __name__=="__main__":
    pipeline = NER_Pipeline()
    sample_text = 'Edward Snowden touched off an enormous public relations campaign on Tuesday calling for Barack Obama to grant him a presidential pardon.'
    _, _, _, _ = pipeline.run_ner(sample_text, display=True)