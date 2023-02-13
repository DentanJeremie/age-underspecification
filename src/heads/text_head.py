import time
import typing as t

import torch
import torch.nn as nn
from tqdm import tqdm

from src.utils.logging import logger
from src.utils.datasets import ImageDataset, get_dataloader_text, AGE_TYPE
from src.utils.pathtools import project
from src.heads.models import TextModel

DEFAULT_MODEL = TextModel()
DEFAULT_LOSS_FN = nn.CrossEntropyLoss()
DEFAULT_OPTIMIZER = torch.optim.Adam
DEFAULT_VERBOSE = 40
DEFAULT_MAX_EPOCH = 20
DEFAULT_START_FROM_SCRATCH = False

class TextHead(object):

    def __init__(
        self,
        model: nn.Module = DEFAULT_MODEL,
        optimizer: torch.optim.Optimizer = DEFAULT_OPTIMIZER,
        loss_fn: torch.nn.modules.loss = DEFAULT_LOSS_FN,
        force_training: bool = False,
        verbose: int = DEFAULT_VERBOSE,
        max_epoch: int = DEFAULT_MAX_EPOCH,
        start_from_scratch: bool = DEFAULT_START_FROM_SCRATCH,
    ) -> None:

        # Model, device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer(self.model.parameters(), lr=1e-3)

        # Data
        self.train_loader, self.test_loader = get_dataloader_text()
        self.size_train = len(self.train_loader.dataset)
        self.size_test = len(self.test_loader.dataset)
        
        # Training config
        self.verbose = verbose
        self.max_epoch = max_epoch
        self.start_from_scratch = start_from_scratch
        self.do_training = True 

        # Checking disk
        if self.start_from_scratch:
            logger.info('Not checking disk since start_from_scratch==True')
            return

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
                else:
                    logger.info('force_training is True, so we will still train the model')
            except RuntimeError:
                logger.info('Unable to load the model from the disk, preparing for training')
        else:
            logger.info('No trained model found, preparing for training')

# ------------------ TRAIN / TEST UTILS ------------------

    def train_one_epoch(self):
        """Performs one epoch of training using self.train_loader as loader.
        """
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

    def test(self) -> float:
        """Performs the testing, using self.test_loader as loader.

        :returns: The proportion of correctly classified sample in the test set.
        """
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
        self.get_predictions_on_test_images(trained=False)

        return correct

    def save_model(self):
        """Saves the model to disk.
        """
        logger.info(f'Saving model to disk...')
        path = project.get_new_text_head_model_file()
        torch.save(self.model.state_dict(), path)
        logger.info(f'Succesfully saved best model at {project.as_relative(path)}')

# ------------------ TRAIN LOOP ------------------

    def train(self):
        """Trains the model with early stopping and patience = 0.
        Each model that imporve the test performance is stored to disk.
        """
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

        self.do_training = False
        logger.info(f'Training: done in {time.time() - start_time:.2f}s')

    @property
    def trained_model(self):
        if self.do_training:
            self.train()
        
        return self.model

# ------------------ RESULT ------------------

    def get_predictions_on_test_images(self, num_images: int = 16, trained = True) -> t.List[int]:
        """Returns a list of the predictions of the trained model on unlabelled age dataset.
        """
        dataset = ImageDataset(AGE_TYPE, labeled=False)
        result = list()
        model = self.trained_model if trained else self.model
        model = model.to(self.device)
        for tensor, _ in dataset:
            tensor = tensor.to(self.device)
            result.append(
                model(torch.unsqueeze(tensor, 0)).argmax().item()
            )

            num_images -= 1
            if num_images == 0:
                break

        logger.info('It seems the true values should be [1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, ...]')
        logger.info('And that the text values shoudl be [1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, ...]')
        logger.info(f'Yet the result as predictied is:   {result}')
        return result

def main():
    TextHead().trained_model

if __name__ == '__main__':
    main()

        







        
