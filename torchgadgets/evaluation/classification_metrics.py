import torch

import numpy as np
from sklearn.metrics import confusion_matrix, top_k_accuracy_score

import pandas as pd

def _extract_data_from_list(data):
    if torch.is_tensor(d[0]):
            return torch.stack(d)
    return torch.stack([_extract_data_from_list(x) for x in d])

def _data_to_1d_pred_tensor(data):
    assert type(data) == list or torch.is_tensor(data), f'Data of type {type(data)} is not supported'
    d_s = data.shape
    if torch.is_tensor(data):
        data = _extract_data_from_list(data)
    if torch.is_floating_point(data) and d_s>1:
        data = torch.argmax(data, dim=-1)
    if len(d_s) > 1:
        data = torch.flatten(data)

def _data_eliminate_batch(data):
    assert type(data) == list or torch.is_tensor(data), f'Data of type {type(data)} is not supported'
    if type(data) == list:
        data = _extract_data_from_list(data)
    if len(data.shape)==2:
        return data
    data = torch.flatten(data, start_dim=0, end_dim=1)
    return data

def accuracy(output, target):
    output, target = _data_to_1d_pred_tensor(output), _data_to_1d_pred_tensor(target)
    return torch.sum(output == target).item() / len(target)

###--- Low-level APIs ---###
## Evaluation Metrics ##

def eval_resolve(output, target, config: dict, metrics=None):
    eval_metrics = {}

    evaluation_metrics = EvaluationMetrics()

    if type(output)==list:
        output = torch.stack(output, dim=0)
    if type(target)==list:
        target = torch.stack(target, dim=0)
    
    if metrics is None:
        metrics = config['evaluation']['metrics']

    for eval_metric in metrics:
            func_name = '_evaluation_' + eval_metric
            try:
                eval_metrics[eval_metric] = getattr(evaluation_metrics, func_name)(output, target, config)
            except:
                print(f"NotImplemented: Evaluation metric {eval_metric}")
        
    return eval_metrics

class EvaluationMetrics:

    def _evaluation_accuracy(self, output: torch.Tensor, target: torch.Tensor, config: dict) -> list:
        """
            Computes the accuracy of the the predictions given the target.

            Arguments:
                output (torch.Tensor): The output of the model.
                target (torch.Tensor): The target of the model.

            Returns:
                float: The accuracy of the given model on the given dataset.
        """

        _, predicted = torch.max(output, -1)

        if len(predicted.shape)==2 and len(target.shape)==2:
            predicted = torch.flatten(predicted)
            target = torch.flatten(target)

        total = target.shape[0]
        correct = (predicted == target).sum().item()

        return [correct / total]
    
    def _evaluation_accuracy_top3(self, output: torch.Tensor, target: torch.Tensor, config: dict) -> list:
        """
            Computes the top-3 accuracy of the the predictions given the target.

            Arguments:
                output (torch.Tensor): The output of the model.
                target (torch.Tensor): The target of the model.

            Returns:
                float: The accuracy of the given model on the given dataset.
        """
        output = _data_eliminate_batch(output)
        target = _data_eliminate_batch(target)
        top3_acc = top_k_accuracy_score(output, target, k=3)

        return [top3_acc]
    
    def _evaluation_accuracy_top5(self, output: torch.Tensor, target: torch.Tensor, config: dict) -> list:
        """
            Computes the top-5 accuracy of the the predictions given the target.

            Arguments:
                output (torch.Tensor): The output of the model.
                target (torch.Tensor): The target of the model.

            Returns:
                float: The accuracy of the given model on the given dataset.
        """
        output = _data_eliminate_batch(output)
        target = _data_eliminate_batch(target)
        top5_acc = top_k_accuracy_score(output, target, k=5)

        return [top5_acc]


    def _evaluation_precision(self, output: torch.Tensor, target: torch.Tensor, config: dict) -> list:
        """
            Computes the precision of the the predictions given the target.

            Arguments:
                output (torch.Tensor): The output of the model.
                target (torch.Tensor): The target of the model.

            Returns:
                float: The precision of the given model on the given dataset.
        """
        assert 'classes' in config['evaluation']
        
        # Discrete set of classes for the classification task
        CLASSES = config['evaluation']['classes']
        _, predicted = torch.max(output.data, -1)

        if len(predicted.shape)==2 and len(target.shape)==2:
            predicted = torch.flatten(predicted)
            target = torch.flatten(target)

        class_precision = []

        for label in CLASSES:
            tp = ( (predicted == label) and (target == label) ).sum().item()
            fp = ( (predicted == label) and (target!= label) ).sum().item()
            class_precision.append(tp / (tp + fp))

        return [sum(class_precision) / len(class_precision)]

    def _evaluation_precision_per_class(self, output: torch.Tensor, target: torch.Tensor, config: dict) -> list:
        """
            Computes the precision of the the predictions given the target for each class separately.

            Arguments:
                output (torch.Tensor): The output of the model.
                target (torch.Tensor): The target of the model.

            Returns:
                float: The precision of the given model on the given dataset.
        """
        assert 'classes' in config['evaluation']
        
        # Discrete set of classes for the classification task
        CLASSES = config['evaluation']['classes']
        _, predicted = torch.max(output.data, -1)

        if len(predicted.shape)==2 and len(target.shape)==2:
            predicted = torch.flatten(predicted)
            target = torch.flatten(target)

        class_precision = []

        for label in CLASSES:
            tp = ( (predicted == label) and (target == label) ).sum().item()
            fp = ( (predicted == label) and (target!= label) ).sum().item()
            class_precision.append(tp / (tp + fp))

        return class_precision


    def _evaluation_recall(self, output: torch.Tensor, target: torch.Tensor, config: dict) -> list:
        """
            Computes the recall of the the predictions given the target.

            Arguments:
                output (torch.Tensor): The output of the model.
                target (torch.Tensor): The target of the model.

            Returns:
                float: The recall of the given model on the given dataset.
        """
        assert 'classes' in config['evaluation']
        
        # Discrete set of classes for the classification task
        CLASSES = config['evaluation']['classes']
        _, predicted = torch.max(output.data, 1)

        if len(predicted.shape)==2 and len(target.shape)==2:
            predicted = torch.flatten(predicted)
            target = torch.flatten(target)

        class_recall = []

        for label in CLASSES:
            tp = ((predicted == label) and (target == label) ).sum().item()
            fn = ( (predicted != label) and (target == label) ).sum().item()
            class_recall.append(tp / (tp + fn))

        return [sum(class_recall) / len(class_recall)]

    def _evaluation_recall_per_class(self, output: torch.Tensor, target: torch.Tensor, config: dict) -> list:
        """
            Computes the recall of the the predictions given the target for each class separately.

            Arguments:
                output (torch.Tensor): The output of the model.
                target (torch.Tensor): The target of the model.

            Returns:
                float: The recall of the given model on the given dataset.
        """
        assert 'classes' in config['evaluation']
        
        # Discrete set of classes for the classification task
        CLASSES = config['evaluation']['classes']
        _, predicted = torch.max(output.data, 1)

        if len(predicted.shape)==2 and len(target.shape)==2:
            predicted = torch.flatten(predicted)
            target = torch.flatten(target)

        class_recall = []

        for label in CLASSES:
            tp = ((predicted == label) and (target == label) ).sum().item()
            fn = ( (predicted!= label) and (target == label) ).sum().item()
            class_recall.append(tp / (tp + fn))

        return class_recall
    
    def _evaluation_f1(self, output: torch.Tensor, target: torch.Tensor, config: dict) -> list:
        """
            Computes the F1 score of the the predictions given the target.

            Arguments:
                output (torch.Tensor): The output of the model.
                target (torch.Tensor): The target of the model.

            Returns:
                float: The F1 score of the given model on the given dataset.
        """
        assert 'classes' in config['evaluation']
        
        # Discrete set of classes for the classification task
        precision = self._evaluation_precision(output, target, config)[0]
        recall = self._evaluation_recall(output, target, config)[0]

        f1 = 2 * precision * recall / (precision + recall)

        return [f1]

    def _evaluation_confusion_matrix(self, output: torch.Tensor, target: torch.Tensor, config: dict) -> list:
        # Build confusion matrix
        predictions = np.argmax(output, axis=-1)
        if len(target.shape) != 1:
            target = torch.flatten(target)
        if len(predictions.shape)!= 1:
            predictions = torch.flatten(predictions)

        cf_matrix = confusion_matrix(target, predictions)
        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in config['dataset']['classes']],
                            columns = [i for i in config['dataset']['classes']])
        return [df_cm]