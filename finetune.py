import os
import json
import wandb
from tqdm import tqdm
from PIL import Image

import clip
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

cwd = os.getcwd()
TRAIN_JSON_PATH = cwd + "/archive/train_data.json"
TRAIN_IMAGES_PATH = cwd + "/archive/"
VAL_JSON_PATH = cwd + "/archive/val_data.json"
VAL_IMAGES_PATH = cwd + "/archive/"
TEST_JSON_PATH = cwd + "/archive/test_data.json"
TEST_IMAGES_PATH = cwd + "/archive/"

BATCH_SIZE = 32
LR = 5e-5
BETAS = (0.9, 0.98)
WD = 0.2
EPS = 1e-6
CLIP_MODEL = "ViT-B/32"
EPOCHS = 20
USE_WANDB = False
WANDB_PROJECT_NAME = "clip-finetune"


class ImageTitleDataset(Dataset):

    def __init__(self, image_paths, titles, preprocess):
        self.image_paths = image_paths
        self.titles = clip.tokenize(titles)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.titles)

    def __getitem__(self,idx):
        image = self.preprocess(Image.open(self.image_paths[idx]))
        title = self.titles[idx]
        return image, title

def get_training_data():
    image_paths =[]
    title_list = []
    with open(TRAIN_JSON_PATH,"r") as f:
        for line in f:
            data = json.loads(line)
            title_list.append(data["product_title"][:40])
            image_paths.append(TRAIN_IMAGES_PATH + data["image_path"])
    
    return image_paths, title_list

def get_validation_data():
    image_paths =[]
    title_list = []
    with open(VAL_JSON_PATH,"r") as f:
        for line in f:
            data = json.loads(line)
            title_list.append(data["product_title"][:40])
            image_paths.append(VAL_IMAGES_PATH + data["image_path"])
    
    return image_paths, title_list

def get_test_data():
    image_paths =[]
    title_list = []
    with open(TEST_JSON_PATH,"r") as f:
        for line in f:
            data = json.loads(line)
            title_list.append(data["product_title"][:40])
            image_paths.append(TEST_IMAGES_PATH + data["image_path"])
    
    return image_paths, title_list

def convert_model_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()

def train():
    if USE_WANDB:
        wandb.init(
            project=WANDB_PROJECT_NAME,
            sync_tensorboard=True,
        )
    writer = SummaryWriter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load(CLIP_MODEL, device)

    if device == "cpu":
        model.float()
    else:
        clip.model.convert_weights(model)

    train_image_paths, train_text = get_training_data()
    val_image_paths, val_text = get_validation_data()
    
    train_dataset = ImageTitleDataset(train_image_paths, train_text, preprocess)
    valid_dataset = ImageTitleDataset(val_image_paths, val_text, preprocess)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)    

    image_enc_loss = nn.CrossEntropyLoss()
    text_enc_loss = nn.CrossEntropyLoss()

    opttimizer = optim.Adam(model.parameters(),
                            lr=LR,
                            betas=BETAS,
                            eps=EPS,
                            weight_decay=WD)

    for epoch in range(EPOCHS):
        step = 0
        for batch in tqdm(train_dataloader):
            images, texts = batch
            images = images.to(device)
            texts = texts.to(device)

            logits_per_image, logits_per_text = model(images, texts)

            ground_truth = torch.arange(len(images), dtype=torch.long, device=device)

            total_loss = (
                          image_enc_loss(logits_per_image, ground_truth) + text_enc_loss(logits_per_text, ground_truth)
                          )

            opttimizer.zero_grad()
            total_loss.backward()
            if device == "cpu":
                opttimizer.step()
            else:
                convert_model_to_fp32(model)
                opttimizer.step()
                clip.model.convert_weights(model)

            writer.add_scalar("total_loss", total_loss.mean().detach.cpu(), step)
            step += 1

    

if __name__ == "__main__":
    train()
