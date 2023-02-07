import torch
import torch.nn as nn

from tqdm import tqdm

from src.utils.logging import logger
from src.utils.datasets import get_dataloader_text
from src.heads.models import TextModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataloader_text, test_dataloader_text = get_dataloader_text()
model = TextModel().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer, verbose=40):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % verbose == 0:
            loss, current = loss.item(), batch * len(X)
            logger.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    logger.info('Evaluating on train set')
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    logger.info(f"Test performances: Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")

epochs = 20
logger.info('Evaluating before training...')
test(test_dataloader_text, model, loss_fn)
for t in range(epochs):
    logger.info(f"Epoch {t+1}")
    train(train_dataloader_text, model, loss_fn, optimizer)
    test(test_dataloader_text, model, loss_fn)
logger.info("Done!")