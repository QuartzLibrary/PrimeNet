'''
The distilled model uses exactly the same data-loading (and related) routines as the original model, 
so you can drop it in as a direct replacement.
'''

import torch.nn as nn

class StudentNet(nn.Module):
    def __init__(self, params):
        super().__init__()
        # drastically smaller: fewer conv channels, one residual block
        C = 4 + int(params['use_dnase']) + int(params['use_methylation']) + int(params['use_histone']) + 2*int(params['use_location'])
        self.conv = nn.Sequential(
            nn.Conv2d(C, params['student_channels'], kernel_size=(3,1), padding=(1,0)),
            nn.ReLU(),
            nn.MaxPool2d((2,1)),
            nn.Conv2d(params['student_channels'], params['student_channels'], kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Sequential(
            nn.Linear(params['student_channels'], params['student_fc']),
            nn.ReLU(),
            nn.Linear(params['student_fc'], 3)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)