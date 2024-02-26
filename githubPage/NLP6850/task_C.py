import os
os.chdir('C:/Users/lmk/Desktop/sydney/sem3/6850/project')
import pandas as pd
import math
import gc
import time
import tqdm
import random
import torch
print(torch.__version__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print(device)
from torch import nn 
from torch import utils
from torch.nn.utils.rnn import pad_sequence

from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset


from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab
from collections import Counter
import numpy as np

import scipy as sp
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold ,train_test_split

import tokenizers
import transformers
print(f"tokenizers.__version__: {tokenizers.__version__}")
print(f"transformers.__version__: {transformers.__version__}")
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

INPUT_DIR = 'C:/Users/lmk/Desktop/sydney/sem3/6850/project/'
OUTPUT_DIR = 'C:/Users/lmk/Desktop/sydney/sem3/6850/project/output/'

from sklearn.metrics import f1_score
# ====================================================
# CFG
# ====================================================
class CFG:
    num_workers=0 #填4 多线程报错 
    path="C:/Users/lmk/Desktop/sydney/sem3/6850/project/\
        input/pppm-deberta-v3-large-baseline-w-w-b-train/"
    config_path=path+'config.pth'
    #model="microsoft/deberta-v3-large"
    model = "distilbert-base-uncased"
    #model = 'bert-base-uncased'
    batch_size=2
    fc_dropout=0.2
    target_size=1
    
    max_len=2693 #之后定义
    
    seed=42
    n_fold=4
    trn_fold=[0, 1, 2, 3]
    encoder_lr=2e-7
    decoder_lr=2e-7
    min_lr=1e-8
    weight_decay=0.01
    eps=1e-6            #让adam分母不为0
    betas=(0.9, 0.999)   #adam 保留前一，二个时刻learning rate 的比例
    epochs=4
    scheduler='cosine' # ['linear', 'cosine']  学习率调度器
    num_warmup_steps=0  #耐心系数 lr先慢慢增加，超过warmup_steps时，lr再慢慢减小。
    num_cycles=0.5   #学习率第一段线性增加   之后像余弦函数一样先减后增循环的次数 0.5代表只减
    tokenizer = None #之后添加
    apex=False     #数据精度自动匹配 缩短训练时间，降低存储需求， 用mse的时候会报错 在之后的版本可能修复
                  #因而能支持更多的 batch size、更大模型和尺寸更大的输入进行训练
    gradient_accumulation_steps = 8 #通过累计梯度来解决本地显存不足问题。在loss函数的时候要用
    max_grad_norm = 1  #对parameters里的所有参数的梯度进行规范化
                #梯度裁剪解决的是梯度消失或爆炸的问题，即设定阈值，如果梯度超过阈值，那么就截断，将梯度变为阈值
    print_freq = 10
    batch_scheduler=True
# ====================================================
# Utils  分数是相关系数
# ====================================================
def get_score(y_true, y_pred):
    #score = sp.stats.pearsonr(y_true, y_pred)[0]
    score =f1_score(y_pred, y_true, average = 'macro')
    return score

def get_logger(filename=OUTPUT_DIR+'train'):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

LOGGER = get_logger()

def seed_everything(seed=12):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed=12)

# =============================================================================
# OOF  out of frame?
# =============================================================================
#oof_df = pd.read_pickle(CFG.path+'oof_df.pkl')
#labels = oof_df['score'].values
#preds = oof_df['pred'].values
#score = get_score(labels, preds)
#LOGGER.info(f'CV Score: {score:<.4f}')


# ====================================================
# Data Loading
# ====================================================
dataset = pd.read_csv('preprocessedtrain.csv',
                      index_col = 0,
                      converters = {'reviewTexttokenized': eval,
                                    'summarytokenized': eval}
                     )

dataset = dataset[['rating','reviewText','summary']]
# splitting the test

train,test = train_test_split(dataset,
                            test_size=1/6,
                            random_state=12,
                            stratify=dataset['rating'])

train.reset_index(drop = True, inplace=True)
test.reset_index(drop = True, inplace=True)
# ====================================================
# CV split
# ====================================================
SKF = StratifiedKFold(n_splits = 5, random_state = 12, shuffle  = True)


DFlist = []
for train_index, valid_index in SKF.split(train,train['rating']):
    train1, valid1 = train.iloc[train_index], train.iloc[valid_index]
    DFlist.append((train1, valid1))

#train['rating'].hist()

# ====================================================
# tokenizer
# ====================================================
tokenizer = AutoTokenizer.from_pretrained(CFG.model, do_lower_case=True)
tokenizer.save_pretrained(OUTPUT_DIR+'tokenizer/')
CFG.tokenizer = tokenizer

# ====================================================
# Define max_len
# ====================================================

review_lengths = []
#tk0 = tqdm(train['text'].unique(), total=len(train['text'].unique()))
#训练集每句话的长度
for text in train['reviewText'].unique():
    length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
    review_lengths.append(length)

summary_lengths = [] 

for text in train['summary'].unique():
    length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
    summary_lengths.append(length)
    
 
max(review_lengths)  #2693

max(summary_lengths) #33

# =============================================================================
# for text_col in ['anchor', 'target']:
#     lengths = []
#     tk0 = tqdm(train[text_col].fillna("").values, total=len(train))
#     for text in tk0:
#         length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
#         lengths.append(length)
#     lengths_dict[text_col] = lengths
# =============================================================================
# =============================================================================
#     
# CFG.max_len = max(lengths_dict['anchor']) + max(lengths_dict['target'])\
#                 + max(lengths_dict['context_text']) + 4 # CLS + SEP + SEP + SEP
# =============================================================================
#CFG.max_len = max(lengths)

if max(review_lengths)>512: #用蒸馏模型 能处理最大为512
  CFG.max_len = 512         
else:
  CFG.max_len = max(review_lengths)

LOGGER.info(f"max_len: {CFG.max_len}")

# ====================================================
# Dataset
# ====================================================
def prepare_input(cfg, review,summary):
    reviews = cfg.tokenizer(review,
                           add_special_tokens=True,
                           max_length=cfg.max_len,
                           padding="max_length", #补全
                           return_offsets_mapping=False,
                           truncation=True #截断  ‘only_first’：这个只针对第一个序列。’only_second’：只针对第二个序列。
                           )
    summarys = cfg.tokenizer(summary,
                           add_special_tokens=True,
                           max_length=cfg.max_len,
                           padding="max_length", #补全
                           return_offsets_mapping=False,
                           truncation=True #截断  ‘only_first’：这个只针对第一个序列。’only_second’：只针对第二个序列。
                           )
    #这样写不能改 因为tokenizer出来是一种特殊形式  review【0】是输入的词向量【1】是mask
    for k, v in reviews.items():
        reviews[k] = torch.tensor(v, dtype=torch.long)
        
    for k, v in summarys.items():
        summarys[k] = torch.tensor(v, dtype=torch.long)
    return reviews,summarys


class TrainDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.reviews = df['reviewText'].values
        self.summarys = df['summary'].values
        
        self.labels = df['rating'].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        reviews,summarys = prepare_input(self.cfg, self.reviews[item],self.summarys[item])
        label = torch.tensor(self.labels[item], dtype=torch.float)
        return reviews,summarys, label


# ====================================================
# Model
# ====================================================
class CustomModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
        self.model = AutoModel.from_config(self.config)
        self.fc_dropout = nn.Dropout(cfg.fc_dropout)
        
        self.fc = nn.Linear(self.config.hidden_size * 2, self.cfg.target_size) #有两个bert *2
        self._init_weights(self.fc)
        
        self.attention = nn.Sequential(
            nn.Linear(self.config.hidden_size, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )
        self._init_weights(self.attention)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def feature(self, inputs):
        outputs = self.model(**inputs)   #model在初始化时就已经是预训练的bert模型了
        last_hidden_states = outputs[0]
        # feature = torch.mean(last_hidden_states, 1)
        weights = self.attention(last_hidden_states)
        feature = torch.sum(weights * last_hidden_states, dim=1)
        return feature

    def forward(self, reviews,summarys):
        feature1 = self.feature(reviews)   #feature 就是预训练模型给出的隐含特征

        feature2 = self.feature(summarys)

        feature = torch.cat((feature1,feature2),1)
        
        
        output = self.fc(self.fc_dropout(feature))
        return output

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))



def train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.apex)
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0

    for step, (reviews,summarys, labels) in enumerate(train_loader):
        
        
        for k, v in reviews.items():
            reviews[k] = v.to(device)
        for k, v in summarys.items():
            summarys[k] = v.to(device)
            
        labels = labels.to(device)
        batch_size = labels.size(0)
        
        
        
        with torch.cuda.amp.autocast(enabled=CFG.apex):   #这一步导致mse error时调用c++内核报错
            y_preds = model(reviews,summarys)
            
        loss = criterion(y_preds.view(-1, 1).to(device), labels.view(-1, 1))
        
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        

        scaler.scale(loss).backward()
        
        
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            if CFG.batch_scheduler:
                scheduler.step()
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.8f}  '
                  .format(epoch+1, step, len(train_loader), 
                          remain=timeSince(start, float(step+1)/len(train_loader)),
                          loss=losses,
                          grad_norm=grad_norm,
                          lr=scheduler.get_lr()[0]))

    return losses.avg


def valid_fn(valid_loader, model, criterion, device):
    losses = AverageMeter()
    model.eval()
    preds = []
    start = end = time.time()
    for step, (reviews,summarys, labels) in enumerate(valid_loader):
        
        for k, v in reviews.items():
            reviews[k] = v.to(device)
        for k, v in summarys.items():
            summarys[k] = v.to(device)
        batch_size = labels.size(0)
        
        labels = labels.to(device)
        
        with torch.no_grad():
            y_preds = model(reviews,summarys)
        loss = criterion(y_preds.view(-1, 1).to(device), labels.view(-1, 1))  #顺序不能变 ！！！！！
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        preds.append(y_preds.sigmoid().to('cpu').numpy())
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):
            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(step, len(valid_loader),
                          loss=losses,
                          remain=timeSince(start, float(step+1)/len(valid_loader))))
    predictions = np.concatenate(preds)
    predictions = np.concatenate(predictions)
    return losses.avg, predictions


def inference_fn(test_loader, model, device):
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    for reviews,summarys, labels in tk0:
        for k, v in reviews.items():
            reviews[k] = v.to(device)
        for k, v in summarys.items():
            summarys[k] = v.to(device)
            
        with torch.no_grad():
            y_preds = model(reviews,summarys)
        preds.append(y_preds.sigmoid().to('cpu').numpy())
    predictions = np.concatenate(preds)
    return predictions

# ====================================================
# train loop
# ====================================================
def train_loop(DFlist,n,treshold):
    
    LOGGER.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    train_folds = DFlist[n][0]
    #train_folds['text'] = train_folds['text'].astype(str)
    
    valid_folds = DFlist[n][1]
    #valid_folds['text'] = valid_folds['text'].astype(str)
    valid_labels = valid_folds['rating'].values #validation 函数调用了 所以要提出来用
    
    train_dataset = TrainDataset(CFG, train_folds)
    valid_dataset = TrainDataset(CFG, valid_folds)

    train_loader = DataLoader(train_dataset,
                              #collate_fn = lambda x: collate_batch(x, CFG), #传参
                              batch_size=CFG.batch_size,
                              shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              #collate_fn = lambda x: collate_batch(x, CFG), #传参
                              batch_size=CFG.batch_size,
                              shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False)

    # ====================================================
    # model & optimizer
    # ====================================================
    model = CustomModel(CFG)
    #torch.save(model.config, OUTPUT_DIR+'config.pth')
    model.to(device)
    
    def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': weight_decay},
            {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if "model" not in n],
             'lr': decoder_lr, 'weight_decay': 0.0}
        ]
        return optimizer_parameters

    optimizer_parameters = get_optimizer_params(model,
                                                encoder_lr=CFG.encoder_lr, 
                                                decoder_lr=CFG.decoder_lr,
                                                weight_decay=CFG.weight_decay)
    optimizer = AdamW(optimizer_parameters, lr=CFG.encoder_lr, eps=CFG.eps, betas=CFG.betas)
    
    # ====================================================
    # scheduler
    # ====================================================
    def get_scheduler(cfg, optimizer, num_train_steps):
        if cfg.scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps
            )
        elif cfg.scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps, num_cycles=cfg.num_cycles
            )
        return scheduler
    
    num_train_steps = int(len(train_folds) / CFG.batch_size * CFG.epochs)
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)

    # ====================================================
    # loop
    # ====================================================
    #criterion = nn.BCEWithLogitsLoss(reduction="mean")#Sigmoid+BCELoss 设为"sum"表示对样本进行求损失和；
                                                    #设为"mean"表示对样本进行求损失的平均值；
                                                    #而设为"none"表示对样本逐个求损失，输出与输入的shape一样。
    criterion = nn.MSELoss(reduction="mean") #平均值
    best_score = 0.

    for epoch in range(CFG.epochs):

        start_time = time.time()

        # train
        avg_loss = train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device)

        # eval
        avg_val_loss, predictions = valid_fn(valid_loader, model, criterion, device)
        
        # scoring
        
        treshold[0].append(np.quantile(predictions,0.266667))
        treshold[1].append(np.quantile(predictions,0.511111))
        treshold[2].append(np.quantile(predictions,0.7))
        treshold[3].append(np.quantile(predictions,0.866667))
        
        tre1 = np.mean(treshold[0][-30:])
        tre2 = np.mean(treshold[1][-30:])
        tre3 = np.mean(treshold[2][-30:])
        tre4 = np.mean(treshold[3][-30:])
        
        classified_y = 1+(predictions>tre1).astype(int)+(predictions>tre2).astype(int)+(predictions>tre3).astype(int)+(predictions>tre4).astype(int)
            
        score = get_score(valid_labels, classified_y)

        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - Score: {score:.4f}')
        
        if best_score < score:
            best_score = score
            LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            
            torch.save({'model': model.state_dict(),
                        'predictions': predictions},
                        OUTPUT_DIR+f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth")
    
    
    
    
    predictions = torch.load(OUTPUT_DIR+f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth", 
                             map_location=torch.device('cpu'))['predictions']
    
    
    
    
    valid_folds['pred'] = predictions
    
    torch.cuda.empty_cache()
    gc.collect()
    
    return valid_folds,treshold

def get_result(oof_df):
    labels = oof_df['rating'].values
    preds = oof_df['pred'].values
    score = get_score(labels, preds)
    LOGGER.info(f'Score: {score:<.4f}')



treshold=[[],[],[],[]]

oof_df = pd.DataFrame()
for fold in range(1):  #不使用cross-valid
    if fold in CFG.trn_fold:
        _oof_df,treshold = train_loop(DFlist,fold,treshold)# 0
        
        oof_df = pd.concat([oof_df, _oof_df])
        LOGGER.info(f"========== fold: {fold} result ==========")
        get_result(_oof_df)
    oof_df = oof_df.reset_index(drop=True)
    LOGGER.info(f"========== CV ==========")
    get_result(oof_df)
    oof_df.to_pickle(OUTPUT_DIR+'oof_df.pkl')





