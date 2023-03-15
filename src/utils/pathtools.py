import collections
import datetime
import os
from pathlib import Path
import typing as t


class CustomizedPath():

    def __init__(self):
        self._root = Path(__file__).parent.parent.parent

        # Logs initialized
        self._initialized_loggers = collections.defaultdict(bool)

        # Datasets promises
        self._train = None
        self._test = None
        self._sample = None

# ------------------ UTILS ------------------

    def remove_prefix(input_string: str, prefix: str) -> str:
        """Removes the prefix if exists at the beginning in the input string
        Needed for Python<3.9
        
        :param input_string: The input string
        :param prefix: The prefix
        :returns: The string without the prefix
        """
        if prefix and input_string.startswith(prefix):
            return input_string[len(prefix):]
        return input_string

    def as_relative(self, path: t.Union[str, Path]) -> Path:
        """Removes the prefix `self.root` from an absolute path.

        :param path: The absolute path
        :returns: A relative path starting at `self.root`
        """
        if type(path) == str:
            path = Path(path)
        return Path(CustomizedPath.remove_prefix(path.as_posix(), self.root.as_posix()))

    def mkdir_if_not_exists(self, path: Path, gitignore: bool=False) -> Path:
        """Makes the directory if it does not exists

        :param path: The input path
        :param gitignore: A boolean indicating if a gitignore must be included for the content of the directory
        :returns: The same path
        """
        path.mkdir(parents=True, exist_ok = True)

        if gitignore:
            with (path / '.gitignore').open('w') as f:
                f.write('*\n!.gitignore')

        return path

# ------------------ MAIN FOLDERS ------------------

    @property
    def root(self):
        return self._root

    @property
    def data(self):
        return self.mkdir_if_not_exists(self.root / 'data', gitignore=True)

    @property
    def output(self):
        return self.mkdir_if_not_exists(self.root / 'output', gitignore=True)

    @property
    def logs(self):
        return self.mkdir_if_not_exists(self.root / 'logs', gitignore=True)

# ------------------ LOGS ------------------

    def get_log_file(self, logger_name: str) -> Path:
        """Creates and initializes a logger.

        :param logger_name: The logger name to create
        :returns: A path to the `logger_name.log` created and/or initialized file
        """
        file_name = logger_name + '.log'
        result = self.logs / file_name

        # Checking if exists
        if not os.path.isfile(result):
            with result.open('w') as f:
                pass

        # Header for new log
        if not self._initialized_loggers[logger_name]:
            with result.open('a') as f:
                f.write(f'\nNEW LOG AT {datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")}\n')
            self._initialized_loggers[logger_name] = True

        return result

# ------------------ HUMAN HAIR ------------------

    @property
    def human_hair(self):
        result = self.data / 'human_hair'

        if not result.exists() or len(list(result.iterdir())) <= 2:
            raise ValueError('You must download the datasets before running the code. Please refer to the instructions in the README.')

        return result

    @property
    def human_hair_y_labeled(self):
        return self.human_hair / 'y_labeled.csv'

    @property
    def human_hair_y_unlabeled(self):
        return self.human_hair / 'y_unlabeled.csv'

    @property
    def human_hair_labeled_folder(self):
        return self.human_hair / 'humans' / 'labeled'

    @property
    def human_hair_unlabeled_folder(self):
        return self.human_hair / 'humans' / 'unlabeled'

# ------------------ HUMAN AGE ------------------

    @property
    def human_age(self):
        result = self.data / 'human_age_shared'

        if not result.exists() or len(list(result.iterdir())) <= 2:
            raise ValueError('You must download the datasets before running the code. Please refer to the instructions in the README.')

        return result

    @property
    def human_age_y_labeled(self):
        return self.human_age / 'y_labeled.csv'

    @property
    def human_age_y_unlabeled(self):
        result = self.human_age / 'y_unlabeled_random.csv'
        if not result.exists():
            import pandas as pd
            import numpy as np
            length = len(list(self.human_age_unlabeled_folder.iterdir()))
            pd.DataFrame(
                np.array([list(range(length)), np.random.randint(0,2,length)]).T, columns=['index','labels']
            ).to_csv(result, index=False)
        return self.human_age / 'y_unlabeled_random.csv'

    @property
    def human_age_labeled_folder(self):
        return self.human_age / 'humans' / 'labeled'

    @property
    def human_age_unlabeled_folder(self):
        return self.human_age / 'humans' / 'unlabeled'

# ------------------ CACHING ------------------

    @property
    def text_frames_dir(self):
        return self.mkdir_if_not_exists(self.output / 'text_frames')

    @property
    def classifier_models(self):
        return self.mkdir_if_not_exists(self.output / 'classifier_models')

    def get_new_classifier_model_file(self) -> Path:
        """Get a new classifier model file.
        :returns: A Path to the file, without creating the file.
        """
        return self.classifier_models / f'classifier_model_{datetime.datetime.now().strftime("_%Y_%m%d__%H_%M_%S")}.pt'

    def get_lastest_classifier_model_file(self): 
        file_suffix = 'classifier_model'           
        files = sorted([
            str(path)
            for path in self.classifier_models.iterdir()
            if path.is_file()
            and str(path)[-len(file_suffix):] == file_suffix 
        ])

        if len(files) == 0:
            return None
            
        return Path(files[-1])
    
# ------------------ PREDICTIONS ------------------

    @property
    def age_prediction(self):
        return self.mkdir_if_not_exists(self.output / 'age_prediction')

    def get_new_prediction_file(self, note: str = '') -> Path:
        """Get a new prediction file.

        :param note: A note that will be inserted in the name of the file.
        :returns: A Path to the file, without creating the file.
        """
        return self.age_prediction / f'age_prediction_{datetime.datetime.now().strftime("_%Y_%m%d__%H_%M_%S")}_{note}.csv'

project = CustomizedPath() 