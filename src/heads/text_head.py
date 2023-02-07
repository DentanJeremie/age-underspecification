import time

import torch
import torch.nn as nn
from tqdm import tqdm

from src.utils.logging import logger
from src.utils.datasets import get_dataloader_text
from src.utils.pathtools import project
from src.heads.models import TextModel

DEFAULT_MODEL = TextModel()
DEFAULT_LOSS_FN = nn.CrossEntropyLoss()
DEFAULT_OPTIMIZER = torch.optim.Adam
DEFAULT_VERBOSE = 40
DEFAULT_MAX_EPOCH = 20

class TextHead(object):

    def __init__(
        self,
        model: nn.Module = DEFAULT_MODEL,
        optimizer: torch.optim.Optimizer = DEFAULT_OPTIMIZER,
        loss_fn: torch.nn.modules.loss = DEFAULT_LOSS_FN,
        force_training: bool = False,
        verbose: int = DEFAULT_VERBOSE,
        max_epoch: int = DEFAULT_MAX_EPOCH
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader, self.test_loader = get_dataloader_text()
        self.size_train = len(self.train_loader.dataset)
        self.size_test = len(self.test_loader.dataset)
        self.loss_fn = loss_fn
        self.verbose = verbose
        self.optimizer = optimizer(self.model.parameters(), lr=1e-3)
        self.max_epoch = max_epoch
        self.do_training = True 

        logger.info('Checking if there exist a model that is already trained')
        possible_trained_model_file = project.get_lastest_text_head_model_file()
        if possible_trained_model_file is not None:
            logger.info(f'Found a trained model at {project.as_relative(possible_trained_model_file)}')
            try:
                self.model.load_state_dict(torch.load(possible_trained_model_file))
                logger.info('Successfully loaded the trained model from disk!')
                if not force_training:
                    logger.info('Skipping training')
                    self.do_training = False
                logger.info('force_training is True, so we will still train the model')
            except RuntimeError:
                logger.info('Unable to load the model from the disk, preparing for training')
        else:
            logger.info('No trained model found, preparing for training')

    def train_one_epoch(self):
        logger.info('Starting training')
        self.model.train()
        for batch, (x, y) in enumerate(self.train_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            # Compute prediction error
            pred = self.model(x)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch % self.verbose == 0:
                loss, current = loss.item(), batch * len(x)
                logger.info(f"loss: {loss:>7f}  [{current:>5d}/{self.size_train:>5d}]")

    def test(self):
        logger.info('Starting evaluation on train set')
        size = self.size_test
        num_batches = len(self.test_loader)
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for x, y in tqdm(self.test_loader):
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        logger.info(f"Test performances: Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")

        return correct

    def save_model(self):
        logger.info(f'Saving model to disk...')
        path = project.get_new_text_head_model_file()
        torch.save(self.model.state_dict(), path)
        logger.info(f'Succesfully saved best model at {project.as_relative(path)}')

    def train(self):
        logger.info(f'Evaluating before training...')
        start_time = time.time()
        prop_correct = self.test()

        best_evaluation_acc = prop_correct
        best_epoch = 0
        weights_for_best_evaluation = self.model.state_dict()

        for epoch in range(1, self.max_epoch + 1):
            logger.info(f"Epoch {epoch}")
            self.train_one_epoch()
            prop_correct = self.test()

            if prop_correct > best_evaluation_acc:
                logger.info(f'Improvement from previous epoch {best_evaluation_acc:.2f} -> {prop_correct:.2f}, continuing training')
                best_evaluation_acc = prop_correct
                best_epoch = epoch
                weights_for_best_evaluation = self.model.state_dict()
                self.save_model()
            else:
                logger.info('No improvement detected after this epoch, stopping the training')
                logger.info(f'Loading best model epoch {best_epoch} (accuracy = {best_evaluation_acc})')
                self.model.load_state_dict(weights_for_best_evaluation)
                logger.info('Succesfully loaded from checkpoint')
                self.save_model()
                break

        logger.info(f'Training: done in {time.time() - start_time:.2s}s')

    def get_trained_model(self):
        if self.do_training:
            self.train()

        return self.model


def main():
    TextHead().get_trained_model()

if __name__ == '__main__':
    main()

        







        
