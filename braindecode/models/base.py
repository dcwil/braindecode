import time

import numpy as np
from numpy.random import RandomState
import torch as th

from braindecode.experiments.monitors import LossMonitor, MisclassMonitor, \
    RuntimeMonitor, CroppedTrialMisclassMonitor, \
    compute_trial_labels_from_crop_preds, compute_pred_labels_from_trial_preds
from braindecode.experiments.stopcriteria import MaxEpochs
from braindecode.datautil.iterators import BalancedBatchSizeIterator, \
    CropsFromTrialsIterator
from braindecode.experiments.experiment import Experiment
from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.models.util import to_dense_prediction_model
from braindecode.torch_ext.schedulers import CosineAnnealing, ScheduledOptimizer
from braindecode.torch_ext.util import np_to_var, var_to_np


def find_optimizer(optimizer_name):
    optim_found = False
    for name in th.optim.__dict__.keys():
        if name.lower() == optimizer_name.lower():
            optimizer = th.optim.__dict__[name]
            optim_found = True
            break
    if not optim_found:
        raise ValueError("Unknown optimizer {:s}".format(optimizer))
    return \
        optimizer


class BaseModel(object):
    def cuda(self):
        """Move underlying model to GPU."""
        self._ensure_network_exists()
        assert not self.compiled,\
            ("Call cuda before compiling model, otherwise optimization will not work")
        self.network = self.network.cuda()
        self.cuda = True
        return self

    def parameters(self):
        """
        Return parameters of underlying torch model.
    
        Returns
        -------
        parameters: list of torch tensors
        """
        self._ensure_network_exists()
        return self.network.parameters()

    def _ensure_network_exists(self):
        if not hasattr(self, 'network'):
            self.network = self.create_network()
            self.cuda = False
            self.compiled = False

    def compile(self, loss, optimizer, extra_monitors=None,  cropped=False, iterator_seed=0):
        """
        Setup training for this model.
        
        Parameters
        ----------
        loss: function (predictions, targets) -> torch scalar
        optimizer: `torch.optim.Optimizer` or string
            Either supply an optimizer or the name of the class (e.g. 'adam')
        extra_monitors: List of Braindecode monitors, optional
            In case you want to monitor additional values except for loss, misclass and runtime.
        cropped: bool
            Whether to perform cropped decoding, see cropped decoding tutorial.
        iterator_seed: int
            Seed to seed the iterator random generator.
        Returns
        -------

        """
        self.loss = loss
        self._ensure_network_exists()
        if cropped:
            to_dense_prediction_model(self.network)
        if not hasattr(optimizer, 'step'):
            optimizer_class = find_optimizer(optimizer)
            optimizer = optimizer_class(self.network.parameters())
        self.optimizer = optimizer
        self.extra_monitors = extra_monitors
        # Already setting it here, so multiple calls to fit
        # will lead to different batches being drawn
        self.seed_rng = RandomState(iterator_seed)
        self.cropped = cropped
        self.compiled = True

    def fit(self, train_X, train_y, epochs, batch_size, input_time_length=None,
            validation_data=None, model_constraint=None,
            remember_best_column=None, scheduler=None,
            log_0_epoch=True):
        """
        Fit the model using the given training data.
        
        Will set `epochs_df` variable with a pandas dataframe to the history
        of the training process.
        
        Parameters
        ----------
        train_X: ndarray
            Training input data
        train_y: 1darray
            Training labels
        epochs: int
            Number of epochs to train
        batch_size: int
        input_time_length: int, optional
            Super crop size, what temporal size is pushed forward through 
            the network, see cropped decoding tuturial.
        validation_data: (ndarray, 1darray), optional
            X and y for validation set if wanted
        model_constraint: object, optional
            You can supply :class:`.MaxNormDefaultConstraint` if wanted.
        remember_best_column: string, optional
            In case you want to do an early stopping/reset parameters to some
            "best" epoch, define here the monitored value whose minimum
            determines the best epoch.
        scheduler: 'cosine' or None, optional
            Whether to use cosine annealing (:class:`.CosineAnnealing`).
        log_0_epoch: bool
            Whether to compute the metrics once before training as well.

        Returns
        -------
        exp: 
            Underlying braindecode :class:`.Experiment`
        """
        if (not hasattr(self, 'compiled')) or (not self.compiled):
            raise ValueError("Compile the model first by calling model.compile(loss, optimizer, metrics)")


        if self.cropped and input_time_length is None:
            raise ValueError("In cropped mode, need to specify input_time_length,"
                             "which is the number of timesteps that will be pushed through"
                             "the network in a single pass.")
        if self.cropped:
            self.network.eval()
            test_input = np_to_var(train_X[0:1], dtype=np.float32)
            while len(test_input.size()) < 4:
                test_input = test_input.unsqueeze(-1)
            if self.cuda:
                    test_input = test_input.cuda()
            out = self.network(test_input)
            n_preds_per_input = out.cpu().data.numpy().shape[2]
            self.iterator = CropsFromTrialsIterator(
                batch_size=batch_size, input_time_length=input_time_length,
                n_preds_per_input=n_preds_per_input,
                seed=self.seed_rng.randint(0, 4294967295))
        else:
            self.iterator = BalancedBatchSizeIterator(
                batch_size=batch_size,
                seed=self.seed_rng.randint(0, 4294967295))
        if log_0_epoch:
            stop_criterion = MaxEpochs(epochs)
        else:
            stop_criterion = MaxEpochs(epochs - 1)
        train_set = SignalAndTarget(train_X, train_y)
        optimizer = self.optimizer
        if scheduler is not None:
            assert scheduler == 'cosine', (
                "Supply either 'cosine' or None as scheduler.")
            n_updates_per_epoch = sum(
                [1 for _ in self.iterator.get_batches(train_set, shuffle=True)])
            n_updates_per_period = n_updates_per_epoch * epochs
            if scheduler == 'cosine':
                scheduler = CosineAnnealing(n_updates_per_period)
            schedule_weight_decay = False
            if optimizer.__class__.__name__ == 'AdamW':
                schedule_weight_decay = True
            optimizer = ScheduledOptimizer(scheduler, self.optimizer,
                                           schedule_weight_decay=schedule_weight_decay)
        loss_function = self.loss
        if self.cropped:
            loss_function = lambda outputs, targets:\
                self.loss(th.mean(outputs, dim=2), targets)
        if validation_data is not None:
            valid_set = SignalAndTarget(validation_data[0], validation_data[1])
        else:
            valid_set = None
        test_set = None
        self.monitors = [LossMonitor()]
        if self.cropped:
            self.monitors.append(CroppedTrialMisclassMonitor(input_time_length))
        else:
            self.monitors.append(MisclassMonitor())
        if self.extra_monitors is not None:
            self.monitors.extend(self.extra_monitors)
        self.monitors.append(RuntimeMonitor())
        exp = Experiment(self.network, train_set, valid_set, test_set,
                         iterator=self.iterator,
                         loss_function=loss_function, optimizer=optimizer,
                         model_constraint=model_constraint,
                         monitors=self.monitors,
                         stop_criterion=stop_criterion,
                         remember_best_column=remember_best_column,
                         run_after_early_stop=False, cuda=self.cuda,
                         log_0_epoch=log_0_epoch,
                         do_early_stop=(remember_best_column is not None))
        exp.run()
        self.epochs_df = exp.epochs_df
        return exp

    def evaluate(self, X,y):
        """
        Evaluate, i.e., compute metrics on given inputs and targets.
        
        Parameters
        ----------
        X: ndarray
            Input data.
        y: 1darray
            Targets.

        Returns
        -------
        result: dict
            Dictionary with result metrics.

        """
        stop_criterion = MaxEpochs(0)
        train_set = SignalAndTarget(X, y)
        model_constraint = None
        valid_set = None
        test_set = None
        loss_function = self.loss
        if self.cropped:
            loss_function = lambda outputs, targets: \
                self.loss(th.mean(outputs, dim=2), targets)

        # reset runtime monitor if exists...
        for monitor in self.monitors:
            if hasattr(monitor, 'last_call_time'):
                monitor.last_call_time = time.time()
        exp = Experiment(self.network, train_set, valid_set, test_set,
                         iterator=self.iterator,
                         loss_function=loss_function, optimizer=self.optimizer,
                         model_constraint=model_constraint,
                         monitors=self.monitors,
                         stop_criterion=stop_criterion,
                         remember_best_column=None,
                         run_after_early_stop=False, cuda=self.cuda,
                         log_0_epoch=True,
                         do_early_stop=False)

        exp.monitor_epoch({'train': train_set})

        result_dict = dict([(key.replace('train_', ''), val)
                            for key, val in
                            dict(exp.epochs_df.iloc[0]).items()])
        return result_dict

    def predict(self, X, threshold_for_binary_case=None):
        """
        Predict the labels for given input data.
        
        Parameters
        ----------
        X: ndarray
            Input data.
        threshold_for_binary_case: float, optional
            In case of a model with single output, the threshold for assigning,
            label 0 or 1, e.g. 0.5.

        Returns
        -------
        pred_labels: 1darray
            Predicted labels per trial. 
        """
        all_preds = []
        for b_X, _ in self.iterator.get_batches(SignalAndTarget(X, X), False):
            all_preds.append(var_to_np(self.network(np_to_var(b_X))))
        if self.cropped:
            pred_labels = compute_trial_labels_from_crop_preds(
                all_preds, self.iterator.input_time_length, X)
        else:
            pred_labels = compute_pred_labels_from_trial_preds(
                all_preds, threshold_for_binary_case)
        return pred_labels
