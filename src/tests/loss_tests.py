import torch
import torch.nn.functional as F
from ..data_utils import getDataset, H36MDataset, SH_NAMES



def test_varianceMSE():
    dataset = H36MDataset(
            seed_length=10,
            rot_representation='pos',
            stacked_hourglass= True,
            reverse_prob=0.0,
            target_length=10,
            down_sampling_factor=2,
            sequence_spacing=5,
            is_train=True,
            debug=True
        )
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    temperature = 1.0  
    for i, seq in enumerate(data_loader):
        std = torch.std(seq[0], dim=0, unbiased=False)
        std = torch.mean(std, dim=-1)
        std_corr = std + torch.max(std)
        weights_norm = std_corr/torch.sum(std_corr, dim=0)
        print('Standard deviations: ')
        print('\n'.join([f'{n}:\t{w}' for n, w in zip(SH_NAMES,std.tolist())]))
        print('\nWeights (Normalized): ')
        print('\n'.join([f'{n}:\t{w}' for n, w in zip(SH_NAMES,weights_norm.tolist())]))
        import ipdb; ipdb.set_trace()

