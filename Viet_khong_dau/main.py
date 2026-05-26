import torch
import torch.nn as nn
from config import Config
from dataset import build_dataloaders
from inference import load_checkpoint
from training import train_model, build_model
from inference import evaluate_test, restore_accents
from vocab import build_allowed_token_mask

cfg = Config()

train_loader, valid_loader, test_loader, char_vocab, word_vocab = build_dataloaders(cfg)
allowed_mask = build_allowed_token_mask(char_vocab, cfg)

model = build_model(char_vocab, word_vocab, cfg)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

model = train_model(
    model=model,
    train_loader=train_loader,
    valid_loader=valid_loader,
    char_vocab=char_vocab,
    word_vocab=word_vocab,
    allowed_mask=allowed_mask,
    cfg=cfg,
)

model, char_vocab, word_vocab, allowed_mask = load_checkpoint(cfg.checkpoint_path, cfg)

evaluate_test(
    model=model,
    test_loader=test_loader,
    char_vocab=char_vocab,
    allowed_mask=allowed_mask,
    cfg=cfg,
)

examples = [
    "toi ten la",
    "mot nguoi mat tich o ho chua nuoc nuoc trong.",
    "nhung hinh anh dac sac nhat ve vong chung ket world cup.",
    "apple se ra mat ipad duoc thiet ke lai trong thang 4?.",
]

print("\nInference examples:")
for text in examples:
    
    print("\nInput :", text)
    print("Output:", restore_accents(text, model, char_vocab, word_vocab, allowed_mask, cfg))