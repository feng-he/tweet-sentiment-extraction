'''
@Author: hefeng
@Date: 2020-05-27 21:05:07
@LastEditors: hefeng
@LastEditTime: 2020-06-17 21:07:20
@FilePath: /NLP/tweet-sentiment-extraction/bert_base_uncased_pytorch.py
'''

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertForQuestionAnswering, get_linear_schedule_with_warmup
from tokenizers import BertWordPieceTokenizer
from tqdm.notebook import tqdm


class Config:
    TRAIN_PATH = '../input/train-folds/train_folds.csv'
    TEST_PATH = '../input/tweet-sentiment-extraction/test.csv'
    BERT_PATH = '../input/bert-base-uncased/'
    RANDOM_STATE = 525
    TRAIN_BATCH_SIZE = 64
    VAL_BATCH_SIZE = 16
    EPOCHS = 10
    tokenizer = BertWordPieceTokenizer('../input/bert-base-uncased/vocab.txt', lowercase=True)


test = pd.read_csv(Config.TEST_PATH)
test.loc[:, 'selected_text'] = test.text.values

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Data Processing
def process_data(tweet, sentiment, selected_text):
    '''
    return the processed data of one original data comes from line in csv files
    '''
    # Get the start and end indices of selected_text in tweet text
    len_s = len(selected_text)
    ind1, ind2 = -1, -1
    for ind in (i for i, c in enumerate(tweet) if c == selected_text[0]):
        if selected_text == tweet[ind:ind + len_s]:
            ind1, ind2 = ind, ind + len_s - 1
            break
    # Set a array to indicate the position which selelcted_text occupied in tweet text
    # such position assign a value as 1, while the other position assign 0
    slt_flag = [0] * len(tweet)
    if ind1 != -1 and ind2 != -1:
        for ind in range(ind1, ind2 + 1):
            slt_flag[ind] = 1
    output = Config.tokenizer.encode(tweet)
    tweet_ids = output.ids[1:-1]
    offsets = output.offsets[1:-1]
    # Get the token level index of selected_text in tweet text
    # If selected_text has an incomplete word then
    # Complete the word and count it in selected_text
    index = []
    for i, (offset1, offset2) in enumerate(offsets):
        if sum(slt_flag[offset1:offset2]) > 0:
            index.append(i)
    start, end = index[0], index[-1]
    sentiment_id = {'positive': 3893, 'negative': 4997, 'neutral': 8699}
    # [CLS](101), [SEP](102)
    input_ids = [101] + [sentiment_id[sentiment]] + [102] + tweet_ids + [102]
    token_type_ids = [0, 0, 0] + [1] * (len(tweet_ids) + 1)
    attention_mask = [1] * len(token_type_ids)
    return {
        'input_ids': input_ids,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask,
        'start': start,
        'end': end,
        'tweet': tweet,
        'sentiment': sentiment,
        'selected_text': selected_text,
        'offsets': offsets
    }


# TweetDataSet
class TweetDataset(Dataset):
    def __init__(self, tweet, sentiment, selected_text):
        self.tweet = tweet
        self.sentiment = sentiment
        self.selected_text = selected_text
        self.len = len(tweet)

    def __getitem__(self, idx):
        data = process_data(self.tweet[idx], self.sentiment[idx], self.selected_text[idx])

        # convert ids to tensor
        input_ids_tensor = torch.tensor(data['input_ids'], dtype=torch.long)
        token_type_ids_tensor = torch.tensor(data['token_type_ids'], dtype=torch.long)
        attention_mask_tensor = torch.tensor(data['attention_mask'], dtype=torch.long)
        start_tensor = data['start']
        end_tensor = data['end']

        return [
            input_ids_tensor, token_type_ids_tensor, attention_mask_tensor, start_tensor, end_tensor, data['tweet'],
            data['sentiment'], data['selected_text'], data['offsets']
        ]

    def __len__(self):
        return self.len


def create_mini_batch(samples):
    input_ids, token_type_ids, attention_mask, start_positions, end_positions, tweets, \
        sentiments, selected_texts, offsetses = [s for s in samples]
    # Zero Padding
    input_ids = pad_sequence(input_ids, batch_first=True)
    token_type_ids = pad_sequence(token_type_ids, batch_first=True)
    attention_mask = pad_sequence(attention_mask, batch_first=True)

    return {
        'input_ids': input_ids,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask,
        'start_positions': start_positions,
        'end_positions': end_positions,
        'tweets': tweets,
        'sentiments': sentiments,
        'selected_texts': selected_texts,
        'offsetses': offsetses
    }


# Dataloader Batch Function
def create_mini_batch(samples):
    input_ids = [s[0] for s in samples]
    token_type_ids = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    start_positions = [s[3] for s in samples]
    end_positions = [s[4] for s in samples]
    tweets = [s[5] for s in samples]
    sentiments = [s[6] for s in samples]
    selected_texts = [s[7] for s in samples]
    offsets = [s[8] for s in samples]
    # Zero Padding
    input_ids = pad_sequence(input_ids, batch_first=True)
    token_type_ids = pad_sequence(token_type_ids, batch_first=True)
    attention_mask = pad_sequence(attention_mask, batch_first=True)

    return {
        'input_ids': input_ids,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask,
        'start_positions': torch.tensor(start_positions, dtype=torch.long),
        'end_positions': torch.tensor(end_positions, dtype=torch.long),
        'tweets': tweets,
        'sentiments': sentiments,
        'selected_texts': selected_texts,
        'offsets': offsets
    }


# Score Compute Function
def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    if len(a) == 0 and len(b) == 0:
        return 0.5
    return float(len(c)) / (len(a) + len(b) - len(c))


def caculate_score(tweet, sentiment, selected_text, start, end, offsets):
    predicate_text = ''
    if sentiment == "neutral":
        predicate_text = tweet
    else:
        if end < start:
            end = start
        for idx in range(start, end + 1):
            if idx + 1 < len(offsets):
                predicate_text += tweet[offsets[idx][0]:offsets[idx][1]] + ' '
    if selected_text is not None:
        score = jaccard(selected_text, predicate_text)
        return score, predicate_text
    return predicate_text


class AverageMeter():
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


# Train Function
def train_fn(trainloader, model, optimizer, scheduler):
    model = model.to(device)
    model.train()
    losses = AverageMeter()
    jaccard_score = AverageMeter()
    data = tqdm(trainloader, total=len(trainloader))

    for d in data:
        # move tensor to device
        input_ids = d['input_ids'].to(device)
        token_type_ids = d['token_type_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        start_positions = d['start_positions'].to(device)
        end_positions = d['end_positions'].to(device)
        tweets = d['tweets']
        sentiments = d['sentiments']
        selected_texts = d['selected_texts']
        offsets = d['offsets']

        # reset gradients
        model.zero_grad()
        # forward pass
        outputs = model(input_ids=input_ids,
                        token_type_ids=token_type_ids,
                        attention_mask=attention_mask,
                        start_positions=start_positions,
                        end_positions=end_positions)
        loss = outputs[0]
        # backward
        # calculate batch loss based on CrossEntropy
        loss.backward()
        # Ajustment weights based on calculated gradients
        optimizer.step()
        # update scheduler
        scheduler.step()
        output_start, output_end = outputs[1], outputs[2]
        output_start = torch.softmax(output_start, dim=1).cpu().detach().numpy()
        output_end = torch.softmax(output_end, dim=1).cpu().detach().numpy()

        scores = []
        for i, tweet in enumerate(tweets):
            sentiment = sentiments[i]
            selected_text = selected_texts[i]
            score, _ = caculate_score(tweet, sentiment, selected_text, np.argmax(output_start[i, :]),
                                      np.argmax(output_end[i, :]), offsets[i])
            scores.append(score)
        losses.update(loss.item(), input_ids.size(0))
        jaccard_score.update(np.mean(scores), input_ids.size(0))

        # use tqdm to print average loss and score of each of each batch
        data.set_postfix(loss=losses.avg, jaccard_score=jaccard_score.avg)

    print(f"Training Score = {jaccard_score.avg}")
    return jaccard_score.avg


# Evaluation Function
def eval_fn(dataloader, model):
    model.eval()
    jaccard_score = AverageMeter()
    data = tqdm(dataloader, total=len(dataloader))
    with torch.no_grad():
        for d in data:
            input_ids = d['input_ids'].to(device)
            token_type_ids = d['token_type_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            tweets = d['tweets']
            sentiments = d['sentiments']
            selected_texts = d['selected_texts']
            offsets = d['offsets']

            outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            output_start, output_end = outputs[0], outputs[1]
            output_start = torch.softmax(output_start, dim=1).cpu().detach().numpy()
            output_end = torch.softmax(output_end, dim=1).cpu().detach().numpy()

            scores = []
            for i, tweet in enumerate(tweets):
                sentiment = sentiments[i]
                selected_text = selected_texts[i]
                score, _ = caculate_score(tweet, sentiment, selected_text, np.argmax(output_start[i, :]),
                                          np.argmax(output_end[i, :]), offsets[i])
                scores.append(score)
            jaccard_score.update(np.mean(scores), input_ids.size(0))
            data.set_postfix(jaccard_score=jaccard_score.avg)
    print(f"Validation Score = {jaccard_score.avg}")
    return jaccard_score.avg


# Training Running function
def run(fold):
    # Load train fold csv
    train_fold = pd.read_csv(Config.TRAIN_PATH)
    train = train_fold[train_fold.kfold != fold].reset_index(drop=True)
    val = train_fold[train_fold.kfold != fold].reset_index(drop=True)
    # Dataset
    trainset = TweetDataset(train.text.values, train.sentiment.values, train.selected_text.values)
    valset = TweetDataset(val.text.values, val.sentiment.values, val.selected_text.values)
    # Dataloader
    trainloader = DataLoader(trainset, batch_size=Config.TRAIN_BATCH_SIZE, collate_fn=create_mini_batch)
    valloader = DataLoader(valset, batch_size=Config.VAL_BATCH_SIZE, collate_fn=create_mini_batch)
    # Model
    model = BertForQuestionAnswering.from_pretrained(Config.BERT_PATH)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    num_training_steps = len(trainset) / Config.TRAIN_BATCH_SIZE * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=num_training_steps)
    num = fold + 1
    print(f"Start training the {num}th fold...")

    for epoch in range(Config.EPOCHS):
        print('[Epoch %d]' % (epoch + 1))
        train_fn(trainloader, model, optimizer, scheduler)
        eval_fn(valloader, model)

    # save model
    print("Saving model......")
    torch.save(model.state_dict(), f'model{num}.bin')
    print(f"Model{num}.bin saved!")


# Start training
run(0)
run(1)
run(2)
run(3)
run(4)

# Load all trained model
model1 = BertForQuestionAnswering.from_pretrained(Config.BERT_PATH)
model1.to(device)
model1.load_state_dict(torch.load("model1.bin"))
model1.eval()

model2 = BertForQuestionAnswering.from_pretrained(Config.BERT_PATH)
model2.to(device)
model2.load_state_dict(torch.load("model2.bin"))
model2.eval()

model3 = BertForQuestionAnswering.from_pretrained(Config.BERT_PATH)
model3.to(device)
model3.load_state_dict(torch.load("model3.bin"))
model3.eval()

model4 = BertForQuestionAnswering.from_pretrained(Config.BERT_PATH)
model4.to(device)
model4.load_state_dict(torch.load("model4.bin"))
model4.eval()

model5 = BertForQuestionAnswering.from_pretrained(Config.BERT_PATH)
model5.to(device)
model5.load_state_dict(torch.load("model5.bin"))
model5.eval()


# Test Predicting Function
def predicate_on_test(testset):
    testloader = DataLoader(testset, batch_size=256, collate_fn=create_mini_batch)
    final_predicate_text = []
    data = tqdm(testloader, total=len(testloader))
    for d in data:
        input_ids = d['input_ids'].to(device)
        token_type_ids = d['token_type_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        tweets = d['tweets']
        sentiments = d['sentiments']
        offsets = d['offsets']
        with torch.no_grad():
            outputs1 = model1(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            outputs2 = model2(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            outputs3 = model3(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            outputs4 = model4(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            outputs5 = model5(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            # use the average value of the all the model as the final start and end position
            output_starts = (outputs1[0] + outputs2[0] + outputs3[0] + outputs4[0] + outputs5[0]) / 5
            output_ends = (outputs1[1] + outputs2[1] + outputs3[1] + outputs4[1] + outputs5[1]) / 5
            output_starts = torch.softmax(output_starts, dim=1).cpu().detach().numpy()
            output_ends = torch.softmax(output_ends, dim=1).cpu().detach().numpy()
            for i, tweet in enumerate(tweets):
                predicate_text = caculate_score(tweet, sentiments[i], None, np.argmax(output_starts[i, :]),
                                                np.argmax(output_ends[i, :]), offsets[i])
                final_predicate_text.append(predicate_text)

    return final_predicate_text


# Predicting on Test Data
# Predicating on Test
testset = TweetDataset(test['text'], test['sentiment'], None, Config.tokenizer)
selected_text = predicate_on_test(testset)

# Save for submission
sub = pd.DataFrame({'selected_text': selected_text})
sub = pd.concat([test.loc[:, 'textID'], sub.loc[:, 'selected_text']], axis=1)
sub.to_csv('submission.csv')
sub.head(10)
