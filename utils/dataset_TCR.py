import os
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from datasets import load_metric
import torch
from torch.utils.data import dataset
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from transformers import LineByLineTextDataset
import tokenizers
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import PreTrainedTokenizer
from transformers import RobertaConfig
from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM
from transformers import RobertaForSequenceClassification
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
from torchtext.vocab import build_vocab_from_iterator
from transformers import pipeline

from util_plot import plot_confusion_matrix

metric = load_metric("f1")

class TCR_activation_dataset(torch.utils.data.Dataset):
    def __init__(self, config, mode):
        self.config = config
        self.mode = mode

        # Set seeds
        seed_nr = 260794
        torch.manual_seed(seed_nr)
        np.random.seed(seed_nr)

        # Read CDR3ab, activation, binding, and multimer info
        x, y, b, m = [], [], [], []
        for csv_path in self.config.csv_paths:
            x_temp, y_temp, b_temp, m_temp = read_tcr_csv(csv_path)
            x.append(x_temp)
            y.append(y_temp)
            b.append(b_temp)
            m.append(m_temp)

        x = np.concatenate(x)
        y = np.concatenate(y)
        b = np.concatenate(b)
        m = np.concatenate(m)
        X = np.stack((x,y,b,m), 1)

        # Format labels (Activation, not mutually exclusive)
        self.label_names = ['HA69', 'NP136', 'HCRT']
        self.labels = np.stack((np.core.defchararray.find(y.astype('str'), self.label_names[0])!=-1, 
                                     np.core.defchararray.find(y.astype('str'), self.label_names[1])!=-1,
                                     np.core.defchararray.find(y.astype('str'), self.label_names[2])!=-1), 1)
        self.cdr3ab = x
        self.activation_name = y
        self.binding_name = b
        self.multimer = m
        
        # Load embeddings
        CDR3a_embeddings = pd.read_csv(config.CDR3a_embedding_path, index_col=0, header=0)
        CDR3b_embeddings = pd.read_csv(config.CDR3b_embedding_path, index_col=0, header=0)
        self.embeddings = np.concatenate((CDR3a_embeddings, CDR3b_embeddings), axis=1)

        # Optionally remove data
        #idx_keep = np.invert(np.in1d(y, 'none'))
        #self.labels = self.labels[idx_keep]
        #self.embeddings = self.embeddings[idx_keep]

        # Split into training, validation, and test
        idx_ds = np.arange(0, self.embeddings.shape[0])
        idx_train, idx_test = train_test_split(idx_ds, test_size=1 - self.config.train_ratio, random_state=seed_nr)
        idx_val, idx_test = train_test_split(idx_test, test_size=self.config.test_ratio / (self.config.test_ratio + self.config.validation_ratio), random_state=seed_nr)
        idx_ds = {'train': idx_train, 'val': idx_val, 'test': idx_test}

        # Get train/val/test set
        self.embeddings = self.embeddings[idx_ds[self.mode]]
        self.labels = self.labels[idx_ds[self.mode]]
        self.fids = idx_ds[self.mode].astype('str')
        self.cdr3ab = self.cdr3ab[idx_ds[self.mode]]
        self.activation_name = self.activation_name[idx_ds[self.mode]]
        self.binding_name = self.binding_name[idx_ds[self.mode]]
        self.multimer = self.multimer[idx_ds[self.mode]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        fid = self.fids[idx]
        X = self.embeddings[idx]
        y = self.labels[idx]
        return {'fid': fid, 'data': torch.from_numpy(X.astype(np.float32)), 'label': torch.from_numpy(y.astype(np.float32))}

class TCR_classification_dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def read_tcr_csv(csv_path):
    # Read csv file
    df = pd.read_csv(csv_path)
    # Get CDR3 alpha and beta
    x = df['CDR3ab'].values
    # Get binding
    b = df['Peptide_nq']
    b = b.fillna('none').values
    # Get activation
    y = df['activated_by']
    y = y.fillna('none').values

    return x, y, b
    #y = [y_to_num[x] for x in y]
    #return CDR3ab, y

def get_tcr_input(X, mode='MaskedLM'):
    if mode == 'MaskedLM':
        x = X[0]
        x = ' '.join(list(x))
        return x
    elif mode == 'SequenceClassification':
        x = X[0]
        x = ' '.join(list(x))
        y = X[1]
        return x, y

def get_yb_label(y, b):
    if b == 'HA69':
        if y == b:
            label = 0
        else:
            label = 1
    if b == 'HCRT':
        if y == b:
            label = 0
        else:
            label = 1
    if b == 'NP136':
        if y == b:
            label = 0
        else:
            label = 1
    return label


def model_init(trial, tokenizer_path, config):
    if trial is not None:
        config.hidden_dropout_prob = trial.params['hidden_dropout_prob']
    model = RobertaForMaskedLM(config=config)
    return model

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def my_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "hidden_dropout_prob": trial.suggest_float("hidden_dropout_prob", 0.01, 0.20),
    }

if __name__=='__main__':

    # Set seeds
    seed_nr = 260794
    torch.manual_seed(seed_nr)
    np.random.seed(seed_nr)

    y_to_num = {'HA69': '1', 
                    'HA69|HCRT': '2', 
                    'HA69|NP136': '3', 
                    'HCRT': '4', 
                    'HCRT|NP136': '5', 
                    'NP136': '6', 
                    'negative': '7', 
                    'none': '8'}
    num_to_y = {v: k for k, v in y_to_num.items()}

    # Data paths
    project_path = '/home/vicente/Documents/Deep Learning/DL_2021/deep-TCELL'
    data_path = os.path.join(project_path, 'Resources')
    csv_files = ['TCR_activated_ml.csv', 'TCR_negative_ml.csv', 'TCR_other_CD4.csv', 
                 'TCR_other_CD8.csv', 'TCR_other_DQ0602_ml.csv']
    csv_paths = [os.path.join(data_path, x) for x in csv_files]

    # Read CDR3ab, activation, binding, and multimer info
    x, y, b, m = [], [], [], []
    for csv_path in csv_paths:
        x_temp, y_temp, b_temp = read_tcr_csv(csv_path)
        x.append(x_temp)
        y.append(y_temp)
        b.append(b_temp)

    x = np.concatenate(x)
    y = np.concatenate(y)
    b = np.concatenate(b)
    X = np.stack((x,y,b), 1)

    # Split into training, validation, and test
    train_ratio = 0.7
    validation_ratio = 0.10
    test_ratio = 0.20
    X_train, X_test = train_test_split(X, test_size=1 - train_ratio, random_state=seed_nr)
    X_val, X_test = train_test_split(X_test, test_size=test_ratio/(test_ratio + validation_ratio), random_state=seed_nr)
    X_all = {'train': X_train, 'val': X_val, 'test': X_test}

    # Prepare data for MaskedLM
    all_data = {subset: list(map(get_tcr_input, X_all[subset])) for subset in ['train', 'val', 'test']}

    # Write training set to path
    input_data_path = {subset: os.path.join(data_path, 'CDR_' + subset + '.txt') for subset in ['train','val','test']}
    for subset in ['train','val','test']:
        with open(input_data_path[subset], mode='wt', encoding='utf-8') as myfile:
            myfile.write('\n'.join(all_data[subset]))

    # Get Huggingface tokenizer
    tokenizer_path = os.path.join(project_path, 'CDR_tokenizer')
    if os.path.exists(os.path.join(tokenizer_path, 'vocab.json')) and os.path.exists(os.path.join(tokenizer_path, 'merges.txt')):
        tokenizer = ByteLevelBPETokenizer(os.path.join(tokenizer_path, 'vocab.json'),
                                          os.path.join(tokenizer_path, 'merges.txt'))
    else:
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(files=[input_data_path['train']], min_frequency=2, special_tokens=[
                "<s>",
                "<pad>",
                "</s>",
                "<unk>",
                "<mask>",
                ])
        tokenizer.save_model(project_path)

    # Add tokenizer processing
    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
    )
    tokenizer.enable_truncation(max_length=512)

    # RoBERTa setup (config, tokenizer, model)
    config = RobertaConfig(
        vocab_size=tokenizer.get_vocab_size(),
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
        output_hidden_states=True)

    # tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path, max_len=512)

    # Dataset
    tcr_datasets = {subset: LineByLineTextDataset(tokenizer=tokenizer, 
                                                  file_path=input_data_path[subset], 
                                                  block_size=128) for subset in ['train','val','test']}
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    # Trainer
    training_args = TrainingArguments(
        output_dir=tokenizer_path,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=10,
        per_device_train_batch_size=64,
        learning_rate=5e-5,
        weight_decay=2e-6,
        save_total_limit=2,
        prediction_loss_only=True,
        load_best_model_at_end=True)

    # Model
    overwrite_train = False
    if overwrite_train or not os.path.exists(os.path.join(tokenizer_path, 'pytorch_model.bin')):
        model = RobertaForMaskedLM(config=config)
    else:
        model = RobertaForMaskedLM.from_pretrained(tokenizer_path, config=config)

    trainer = Trainer(
        args=training_args,
        data_collator=data_collator,
        train_dataset=tcr_datasets['train'],
        eval_dataset=tcr_datasets['val'],
        callbacks = [EarlyStoppingCallback(early_stopping_patience = 3)],
        model_init=lambda trial: model_init(trial, tokenizer_path, config),
        compute_metrics=compute_metrics)

    # Training
    if overwrite_train or not os.path.exists(os.path.join(tokenizer_path, 'pytorch_model.bin')):
        #trainer.hyperparameter_search(direction='minimize', hp_space=my_hp_space, n_trials=10)
        trainer.train()
        trainer.save_model(tokenizer_path)
        
    # Fill mask (1: predict peptide binding, 2: predict sequence)
    fill_mask = pipeline("fill-mask", model=tokenizer_path, tokenizer=tokenizer)

    # Find 2 examples of each label
    cdr_pos_mask = 8
    num_examples_per_class = 2
    for label_num in range(1,9):
        label = num_to_y[str(label_num)]
        test_data_examples = [x for idx, x in enumerate(all_data['test']) if X_test[idx, 1] == label][0:num_examples_per_class]
        for i in range(num_examples_per_class):
            input_str = test_data_examples[i]
            input_str_masked = input_str[0:cdr_pos_mask] + '<mask>' + input_str[(cdr_pos_mask+1):]
            pred = fill_mask(input_str_masked)
            print('Input: {} - Label: {}.'.format(input_str_masked, input_str[cdr_pos_mask]))
            print('Prediction #1: {} - Confidence: {:.5f}.'.format(pred[0]['token_str'], pred[0]['score']))
            print('Prediction #2: {} - Confidence: {:.5f}.'.format(pred[1]['token_str'], pred[1]['score']))

    # Test set performance
    #perf_tcr = {subset: trainer.evaluate(eval_dataset=tcr_datasets[subset]) for subset in ['train', 'val', 'test']}
    #print(perf_tcr)

    # Training set (Predict binding)
    # TODO: Get all idx belonging to binding tetromers with same activation or no activation
    multimer_in = {'train': ['tetramer', 'dextramer', 'none', 'both'], 'val': ['tetramer'], 'test': ['tetramer']}
    binding_in = ['HA69']#,'HCRT','NP136'HA59
    classes_in = ['HA69','negative']
    idx = {subset: np.where(np.in1d(X_all[subset][:, 1], classes_in) & \
        np.in1d(X_all[subset][:, 2], binding_in) & \
        np.in1d(X_all[subset][:, 3], multimer_in[subset]) & \
        ((X_all[subset][:, 2] == X_all[subset][:, 1]) | (X_all[subset][:, 1] == 'negative')))[0] for subset in ['train', 'val', 'test']}
    # Get encodings
    encodings = {subset: tokenizer(list(np.array(all_data[subset])[idx[subset]]), truncation=True, padding=True) for subset in ['train', 'val', 'test']}
    # Get labels (activation/binding for 3)
    y_class = {subset: np.array([get_yb_label(y, b) for y, b in X_all[subset][idx[subset], 1:3]]).astype('longlong') for subset in ['train', 'val', 'test']}
    tcr_class_dataset = {subset: TCR_classification_dataset(encodings[subset], y_class[subset]) for subset in ['train', 'val', 'test']}

    # Model
    classification_path = os.path.join(project_path, 'CDR_classification')
    overwrite_train_class = True
    if overwrite_train_class or not os.path.exists(os.path.join(classification_path, 'pytorch_model.bin')):
        model_class = RobertaForSequenceClassification.from_pretrained(tokenizer_path, num_labels=len(np.unique(y_class['train'])), 
        problem_type="single_label_classification", output_hidden_states=True)
    else:
        model_class = RobertaForSequenceClassification.from_pretrained(classification_path, num_labels=len(np.unique(y_class['train'])), 
        problem_type="single_label_classification", output_hidden_states=True)
    
    # Trainer for classification
    training_args = TrainingArguments(
        output_dir=classification_path,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=100,
        per_device_train_batch_size=64,
        learning_rate=5e-5,
        weight_decay=2e-6,
        save_total_limit=5,
        load_best_model_at_end=True)

    trainer_class = Trainer(
        model=model_class,
        args=training_args,
        train_dataset=tcr_class_dataset['train'],
        eval_dataset=tcr_class_dataset['val'],
        callbacks = [EarlyStoppingCallback(early_stopping_patience = 3)])

    # Fine tuning
    if overwrite_train_class or not os.path.exists(os.path.join(classification_path, 'pytorch_model.bin')):
        trainer_class.train()
        trainer_class.save_model(classification_path)
    
    # Performance
    pred_class = {subset: trainer_class.predict(test_dataset=tcr_class_dataset[subset]) for subset in ['train', 'val', 'test']}
    for subset in ['train', 'val', 'test']:
        pred_subset = np.argmax(pred_class[subset][0][0], 1)
        label_subset = pred_class[subset][1]
        cm_subset = confusion_matrix(label_subset, pred_subset)
        cm_subset = cm_subset.astype('float') / cm_subset.sum(axis=1)[:, np.newaxis]
        print('Class accuracy ({})):'.format(subset))
        print(cm_subset.diagonal())
        if subset == 'test':
            plot_confusion_matrix(cm_subset, ['activated', 'bound'], title=binding_in[0], normalize=True)

    # Latent representation visualization
    # n_eval_each = 20
    # idx_eval = []
    # for label in list(np.unique(X_test[:, 1])):
    #     idx_eval.append([idx for idx in range(len(all_data['test'])) if X_test[idx, 1] == label][0:n_eval_each])
    # idx_eval = sum(idx_eval, [])
    # model_class.eval()
    # labs = X_test[idx_eval, 1].tolist()
    # sents = np.array(all_data['test'])[idx_eval].tolist()
    # sents_and_labs = zip(sents, labs)
    # embedding_all = []
    # bs_emb = 8
    # for i in range(0, len(idx_eval), bs_emb):
    #     pretrained_preds = get_preds(sents[i:(i + bs_emb)], tokenizer, model_class)
    #     mat = eval_vectors_batch(pretrained_preds, wrd_vec_mode='concat', sentence_emb_mode="average_word_vectors")
    #     embedding_all.append(mat)
    # mat = np.concatenate(embedding_all, 0)
    # eval_vectors_plot(mat, sents_and_labs, wrd_vec_mode='concat', sentence_emb_mode="average_word_vectors", 
    #          plt_xrange=[-0.03, 0.03], plt_yrange=[-0.03, 0.03], title_prefix="Pretrained model:")