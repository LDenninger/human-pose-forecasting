import torch
from tqdm import tqdm

from .classification_metrics import eval_resolve

def run_evaluation( model: torch.nn.Module, 
                    data_augmentor,
                    dataset: torch.utils.data.DataLoader, 
                    config: dict,
                    evaluation_metrics = None,
                    criterion = None,
                    suppress_output: bool = False):
    """
        Runs evaluation of the given model on the given dataset.

        Arguments:
            model (torch.nn.Module): The model to evaluate.
            dataloader (torch.utils.data.DataLoader): The dataloader to evaluate on.
            config (dict): The configuration dictionary.
            device (str, optional): The device to evaluate on. Defaults to "cpu".

    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup model for evaluation
    model.eval()
    model.to(device)

    eval_metrics = {}
    with torch.no_grad():
        if suppress_output:
            progress_bar = enumerate(dataset)
        else:
            progress_bar = tqdm(enumerate(dataset), total=len(dataset))
            progress_bar.set_description(f'Evaluation:')
        outputs = []
        targets = []
        losses = []
        for i, (imgs, labels) in progress_bar:
            
            labels_raw = torch.clone(labels)
            imgs, labels = imgs.to(device), labels.to(device)

            # apply preprocessing surch as flattening the imgs and create a one hot encodinh of the labels
            imgs, labels = data_augmentor((imgs, labels), train=False)


            output = model(imgs)

            outputs.append(output.cpu())
            targets.append(labels_raw.cpu())
            if criterion is not None:
                loss = criterion(output, labels)
                losses.append(loss.cpu().item())

        eval_metrics = eval_resolve(torch.stack(outputs, dim=0), torch.stack(targets, dim=0), config, metrics=evaluation_metrics)

    if criterion is None:
        return eval_metrics
    
    return eval_metrics, sum(losses) / len(losses)