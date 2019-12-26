import torch
import torch.nn as nn
import spacy
from torchtext import data,datasets
from s2smodel import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

spacy_en = spacy.load("en_core_web_sm")
spacy_vi = spacy.load('vi_spacy_model')
DEVICE = torch.device('cuda')
BATCH = 1

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)][::-1]
def tokenize_vi(text):
    return [tok.text for tok in spacy_vi.tokenizer(text)]

SRC = data.Field(tokenize = tokenize_en, init_token = '<sos>', eos_token = '<eos>', lower = True)
TRG = data.Field(tokenize = tokenize_vi, init_token = '<sos>', eos_token = '<eos>', lower = True)

train,val,test=datasets.TranslationDataset.splits(
    path="./",
    train="train",
    validation="valid",
    test="test",
    exts=(".src",".tgt"),
    fields=(SRC, TRG),
)

print(f"Number of training examples: {len(train.examples)}")
print(f"Number of validation examples: {len(val.examples)}")
print(f"Number of testing examples: {len(test.examples)}")
print(vars(train.examples[0]))

SRC.build_vocab(train, min_freq = 2)
TRG.build_vocab(train, min_freq = 2)

print(f"Unique tokens in source (en) vocabulary: {len(SRC.vocab)}")
print(f"Unique tokens in target (vi) vocabulary: {len(TRG.vocab)}")

train_iter, val_iter, test_iter = data.BucketIterator.splits((train,val,test),
     batch_sizes=(BATCH,BATCH,BATCH),
     device=DEVICE,
     )

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 64
DEC_EMB_DIM = 64
ENC_HID_DIM = 128
DEC_HID_DIM = 128
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

model.apply(init_weights)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

optimizer = torch.optim.Adam(model.parameters())
PAD_IDX = TRG.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)

def train(model, iterator):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        optimizer.zero_grad()

        output = model(src, trg)
        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        epoch_loss += loss.item()
        if i % 1000 == 999:
            print(i+1,':',epoch_loss/i)

    return epoch_loss / len(iterator)


def eval(model, iterator):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0)
            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()
            if i % 1000 == 999:
                print(i+1, ':', epoch_loss / i)

    return epoch_loss / len(iterator)

best_valid_loss = 1e4

print('begin...')
for epoch in range(10):
    train_loss = train(model, train_iter)
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')

    val_loss = eval(model, val_iter)
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    if val_loss < best_valid_loss:
        best_valid_loss = val_loss
        torch.save(model.state_dict(), 'model.pt')

