import paddle.nn as nn


class conditioning_network(nn.Layer):
    """conditioning network
        The input to the conditioning network are the observations (y)
        Args:
        y: Observations (B X Obs)
    """

    def __init__(self):
        super().__init__()

        class Flatten(nn.Layer):
            def __init__(self, *args):
                super().__init__()

            def forward(self, x):
                return x.reshape([x.shape[0], -1])

        class Unflatten(nn.Layer):
            def __init__(self, *args):
                super().__init__()

            def forward(self, x):
                if x[:, 0, 0].shape == [16, ]:
                    out = x.reshape([16, 4, 8, 8])  # for config_1  change this to out = x.reshape([16,2,8,8])
                elif x[:, 0, 0].shape == [1000, ]:
                    out = x.reshape([1000, 4, 8, 8])  # for config_1  change this to out = x.reshape([1000,2,8,8])
                elif x[:, 0, 0].shape == [1, ]:
                    out = x.reshape([1, 4, 8, 8])  # for config_1  change this to out = x.reshape([1,2,8,8])
                return out

        # for config_1  change this to nn.ConvTranspose2d(2,  48, 2, padding=0)
        self.multiscale = nn.LayerList([nn.Sequential(Unflatten(), nn.Conv2DTranspose(4, 48, 2, padding=0), nn.ReLU(),
                                                      nn.Conv2DTranspose(48, 48, 2, padding=1, stride=2)),
                                        nn.Sequential(nn.ReLU(), nn.Conv2DTranspose(48, 96, 2, padding=0, stride=2), nn.ReLU(),
                                                      nn.Conv2DTranspose(96, 128, 3, padding=1, stride=1)),
                                        nn.Sequential(nn.ReLU(), nn.Conv2DTranspose(128, 128, 2, padding=0, stride=2)),
                                        nn.Sequential(nn.ReLU(), nn.AvgPool2D(6), Flatten(), nn.Linear(12800, 9600), nn.ReLU(),
                                                      nn.Linear(9600, 6400), nn.ReLU(), nn.Linear(6400, 4800), nn.ReLU(),
                                                      nn.Linear(4800, 2048), nn.ReLU(), nn.Linear(2048, 1024), nn.ReLU(),
                                                      nn.Linear(1024, 512))])

    def forward(self, cond):
        val_cond = [cond]
        for val in self.multiscale:
            val_cond.append(val(val_cond[-1]))
        return val_cond[1:]
