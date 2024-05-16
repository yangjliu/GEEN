from torch import nn
import torch
from torch.distributions import Normal 

class GenDNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Set our init args as class attributes
        self.batch_size = config.batch_size
        self.input_size = input_size = config.num_measurement
        self.sample_size = config.sample_size
        self.num_layer = num_layer = config.num_layer
        self.hidden_size = hidden_size = config.hidden_size
        self.drop_out = drop_out = config.drop_out
        self.activation = activation = config.activation
        self.batch_norm = config.batch_norm

        # Define PyTorch model
        if self.batch_norm:
            self.layer_list = [
                               nn.BatchNorm1d(input_size),
                               nn.Linear(input_size, hidden_size),
                               activation,
                               nn.BatchNorm1d(hidden_size),
                               nn.Dropout(drop_out),
                              ]
        else:
            self.layer_list = [
                               nn.Linear(input_size, hidden_size),
                               activation,
                               nn.Dropout(drop_out),
                              ]
            
        for i in range(num_layer-1):
            if self.batch_norm:
                self.layer_list.extend([nn.Linear(hidden_size, hidden_size),
                                        activation,
                                        nn.BatchNorm1d(hidden_size),
                                        nn.Dropout(drop_out),
                                        ])
            else:
                self.layer_list.extend([nn.Linear(hidden_size, hidden_size),
                                        activation,
                                        nn.Dropout(drop_out),
                                        ])
        self.layer_list.append(nn.Linear(hidden_size, 1))
        self.dnn = nn.Sequential(*self.layer_list)

    def forward(self, x):
        sample_size =self.sample_size
        input_size = self.input_size
        x = x.reshape(-1, input_size)
        x = self.dnn(x)
        return x.reshape(-1, sample_size, 1)
    
def divergenceLoss_wNormal_wKernelDensity(x, x_star_gen, config):
    # kernel function
    def K_func(x):
        return torch.exp(Normal(0, 1).log_prob(x))

    def f_x1_cond_xstar(x1_loop, x_star_loop, x1_vec_loop, x_star_vec_loop, h1_loop):
        # equation 13 in the paper. X1 is special since it is used to normalize generalized X*
        # x1_loop: x1, batch_size*N*1
        # x_star_loop: generalized x_star, batch_size*N*1
        # x1_vec_loop: X1; a vector of x1 for kenel density proximation, batch_size*N*N
        # x_star_vec_loop: generalized X*; a vector of generalized x* for kenel density proximation, batch_size*N*N*1
        # h1_loop: h1; bandwith for x1, batch_size*N*1
        N = x_star_vec_loop.shape[2]
        x_prime_loop = x1_loop - x_star_loop  # batch*N*1
        arg = (
            K_func(
                (
                    (x1_vec_loop.unsqueeze(dim=3) - x_star_vec_loop).squeeze(dim=3)
                    - x_prime_loop
                )
                / h1_loop
            )
            / h1_loop
        )
        return torch.mean(arg, dim=2)

    def f_xj_xstar(
        x_j_loop, x_star_loop, x_j_vec_loop, x_star_vec_loop, h_j_loop, h_star_loop
    ):
        # equation 14 in the paper
        # x_j_loop: xj, batch_size*N*1
        # x_star_loop: generalized x_star, batch_size*N*1
        # x_j_vec_loop: Xj; a vector of xj for kenel density proximation, batch_size*N*N
        # x_star_vec_loop: generalized X*; a vector of generalized x* for kenel density proximation, batch_size*N*N*1
        # h_j_loop: hj; bandwith for xj, batch_size*N*1
        # h_star_loop: h*; bandwith for generalized x*, batch_size*N*1
        N = x_j_vec_loop.shape[2]
        arg1 = K_func((x_j_vec_loop - x_j_loop) / h_j_loop) / h_j_loop  # batch_size*N*N
        arg2 = (
            K_func((x_star_vec_loop.squeeze(dim=3) - x_star_loop) / h_star_loop)
            / h_star_loop
        )  # batch_size*N*N
        return torch.sum(arg1 * arg2 / N, dim=2)

    def f_xstar(x_star_loop, x_star_vec_loop, h_star_loop):
        # equation 15 in the paper
        # x_star_loop: generalized x_star, batch_size*N*1
        # x_star_vec_loop: generalized X*; a vector of generalized x* for kenel density proximation, batch_size*N*N*1
        # h_star_loop: h*; bandwith for generalized x*, batch_size*N*1
        N = x_star_vec_loop.shape[2]
        arg = (
            K_func((x_star_vec_loop.squeeze(dim=3) - x_star_loop) / h_star_loop)
            / h_star_loop
        )  # batch_size*N*N
        return torch.sum(arg, dim=2) / N

    def f_xj_cond_xstar(
        x_j_loop, x_star_loop, x_j_vec_loop, x_star_vec_loop, h_j_loop, h_star_loop
    ):
        # equation 13 in the paper
        # x_j_loop: xj, batch_size*N*1
        # x_star_loop: generalized x_star, batch_size*N*1
        # x_j_vec_loop: Xj; a vector of xj for kenel density proximation, batch_size*N*N
        # x_star_vec_loop: generalized X*; a vector of generalized x* for kenel density proximation, batch*N*N*1
        # h_j_loop: hj; bandwith for xj, batch_size*N*1
        # h_star_loop: h*; bandwith for generalized x*, batch_size*N*1
        return f_xj_xstar(
            x_j_loop, x_star_loop, x_j_vec_loop, x_star_vec_loop, h_j_loop, h_star_loop
        ) / f_xstar(x_star_loop, x_star_vec_loop, h_star_loop)

    def f_joint(x, x_star, h_loop, h_star_loop):
        # equation 16 in the paper
        # x: x1, x2, ... xk, batch_size * N * k
        # x_star: generalized x_star, batch_size * N * 1
        # h_loop: h1, h2, ... hk; bandwith for xj, batch_size*N *1 * k
        # h_star_loop: h*; bandwith for generalized x*, batch_size*N*1
        k = x.shape[2]
        N = x.shape[1]
        x_vec_loop = x.unsqueeze(dim=1).expand(-1, N, -1, -1)  # X1, X2, ... Xk
        x_loop = x.unsqueeze(dim=2)  # x1, x2, ... xk
        x_star_vec_loop = x_star.unsqueeze(dim=1).expand(-1, N, -1, -1)
        x_star_loop = x_star
        arg = (
            K_func((x_star_vec_loop.squeeze(dim=3) - x_star_loop) / h_star_loop)
            / h_star_loop
        )

        for j in range(k):
            arg = (
                arg
                * K_func(
                    (x_vec_loop[:, :, :, j] - x_loop[:, :, :, j]) / h_loop[:, :, :, j]
                )
                / h_loop[:, :, :, j]
            )
        return torch.mean(arg, dim=2)

    def loss(x, x_star_gen):
        # x: measurement of true x_star
        # x_star_gen: generated x_star
        k = x.shape[2]
        N = x.shape[1]
        batch_size = x.shape[0]

        x_loop = x.unsqueeze(dim=1).expand(-1, N, -1, -1)
        x_obs_loop = x.unsqueeze(dim=2)
        x_star_loop = x_star_gen.unsqueeze(dim=1).expand(-1, N, -1, -1)

        h_vec_loop = torch.zeros(
            (batch_size, N, 1, k), 
            device=config.device, 
            dtype=torch.float64
            )
        
        for j in range(k):
            if j == 0:
                if type(config.window_size) is list:
                    h_vec_loop[:, :, :, [j]] = (
                        (
                            config.window_size[j]
                            * torch.std(x[:, :, [j]] - x_star_gen, dim=1, keepdim=True)
                            * N ** (-1 / 5)
                        )
                        .unsqueeze(dim=1)
                        .expand(-1, N, -1, -1)
                    )
                else:
                    h_vec_loop[:, :, :, [j]] = (
                        (
                            config.window_size
                            * torch.std(x[:, :, [j]] - x_star_gen, dim=1, keepdim=True)
                            * N ** (-1 / 5)
                        )
                        .unsqueeze(dim=1)
                        .expand(-1, N, -1, -1)
                    )
            else:
                if type(config.window_size) is list:
                    h_vec_loop[:, :, :, [j]] = (
                        (
                            config.window_size[j]
                            * torch.std(x[:, :, [j]], dim=1, keepdim=True)
                            * N ** (-1 / 5)
                        )
                        .unsqueeze(dim=1)
                        .expand(-1, N, -1, -1)
                    )
                else:
                    h_vec_loop[:, :, :, [j]] = (
                        (
                            config.window_size
                            * torch.std(x[:, :, [j]], dim=1, keepdim=True)
                            * N ** (-1 / 5)
                        )
                        .unsqueeze(dim=1)
                        .expand(-1, N, -1, -1)
                    )
        h_star = config.window_size_star * torch.std(x_star_gen, dim=1) * N ** (-1 / 5)
        h_star_loop = h_star.unsqueeze(dim=1).expand(-1, N, -1)

        arg1_loop = f_joint(x, x_star_gen, h_vec_loop, h_star_loop)

        arg2_loop = torch.ones(
            batch_size, 
            N, 
            device=config.device, 
            dtype=torch.float64
            )
        x_star_obs_loop = x_star_gen

        for j in range(k):
            x_j_obs_loop = x_obs_loop[:, :, :, j]
            x_j_vec_loop = x_loop[:, :, :, j]
            h_j_loop = h_vec_loop[:, :, :, j]
            if j == 0:
                arg2_loop *= f_x1_cond_xstar(
                    x_j_obs_loop, x_star_obs_loop, x_j_vec_loop, x_star_loop, h_j_loop
                )
            else:
                arg2_loop *= f_xj_cond_xstar(
                    x_j_obs_loop,
                    x_star_obs_loop,
                    x_j_vec_loop,
                    x_star_loop,
                    h_j_loop,
                    h_star_loop,
                )
        arg2_loop *= f_xstar(x_star_obs_loop, x_star_loop, h_star_loop)
       
        if config.loss_fun == 'dl_hellinger':
            hellinger_distance = torch.square(torch.sqrt(arg1_loop) - torch.sqrt(arg2_loop))
            divergence_loss = torch.mean(hellinger_distance, dim=1)
        elif config.loss_fun== 'kl_loss':
            # KL_divergence = torch.abs(arg1_loop*(arg1_loop_log-arg2_loop_log))   ## PyTorch and Tensorflow implementation
            arg1_loop_log = torch.log(arg1_loop)
            arg2_loop_log = torch.log(arg2_loop)
            KL_divergence = arg1_loop * (arg1_loop_log - arg2_loop_log)
            divergence_loss = torch.mean(KL_divergence, dim=1)
        else:
            arg1_loop_log = torch.log(arg1_loop)
            arg2_loop_log = torch.log(arg2_loop)
            arg1 = torch.mean(arg1_loop_log, dim=1)  ##take as sample average
            arg2 = torch.mean(arg2_loop_log, dim=1)
            divergence_loss = torch.abs(arg1 - arg2)
            # divergence_loss = arg1 - arg2

        if config.lm:
            normalization = config.lm * torch.square(
                torch.mean(x_star_gen.squeeze(dim=2), dim=1)
                - torch.mean(x[:, :, 0], dim=1)
            )
        else:
            normalization = 0

        return divergence_loss, normalization

    return loss(x, x_star_gen)

# this is only used for non-normalization case
def divergenceLoss_woNormal_wKernelDensity(x, x_star_gen, config):
    def K_func(x):
        return torch.exp(Normal(0, 1).log_prob(x))

    def f_xj_xstar(
        x_j_loop, x_star_loop, x_j_vec_loop, x_star_vec_loop, h_j_loop, h_star_loop
    ):
        # equation 14 in the paper
        # x_j_loop: xj, small x
        # x_star_loop: generalized x_star, small x*
        # x_j_vec_loop: Xj, capitalized X; a vector of xj for kenel density proximation
        # x_star_vec_loop: generalized X*, capitalized X*; a vector of generalized x* for kenel density proximation
        # h_j_loop: hj; bandwith for xj
        # h_star_loop: h*; bandwith for generalized x*
        N = x_j_vec_loop.shape[2]
        arg1 = K_func((x_j_vec_loop - x_j_loop) / h_j_loop) / h_j_loop  # batch_size*N*N
        arg2 = (
            K_func((x_star_vec_loop.squeeze(dim=3) - x_star_loop) / h_star_loop)
            / h_star_loop
        )  # batch_size*N*N
        return torch.sum(arg1 * arg2 / N, dim=2)

    def f_xstar(x_star_loop, x_star_vec_loop, h_star_loop):
        # equation 15 in the paper
        # x_star_loop: generalized x_star, small x*
        # x_star_vec_loop: generalized X*, capitalized X*; a vector of generalized x* for kenel density proximation
        # h_star_loop: h*; bandwith for generalized x*
        N = x_star_vec_loop.shape[2]
        arg = (
            K_func((x_star_vec_loop.squeeze(dim=3) - x_star_loop) / h_star_loop)
            / h_star_loop
        )  # batch_size*N*N
        return torch.sum(arg, dim=2) / N

    def f_xj_cond_xstar(
        x_j_loop, x_star_loop, x_j_vec_loop, x_star_vec_loop, h_j_loop, h_star_loop
    ):
        # equation 13 in the paper
        # x_j_loop: xj, small x
        # x_star_loop: generalized x_star, small x*
        # x_j_vec_loop: Xj, capitalized X; a vector of xj for kenel density proximation
        # x_star_vec_loop: generalized X*, capitalized X*; a vector of generalized x* for kenel density proximation
        # h_j_loop: hj; bandwith for xj
        # h_star_loop: h*; bandwith for generalized x*
        return f_xj_xstar(
            x_j_loop, x_star_loop, x_j_vec_loop, x_star_vec_loop, h_j_loop, h_star_loop
        ) / f_xstar(x_star_loop, x_star_vec_loop, h_star_loop)

    def f_joint(x, x_star, h_loop, h_star_loop):
        # equation 16 in the paper
        # x: x1, x2, ... xk
        # x_star: generalized x_star
        # h_loop: h1, h2, ... hk; bandwith for xj
        # h_star_loop: h*; bandwith for generalized x*
        batch_size = x.shape[0]
        k = x.shape[2]
        N = x.shape[1]
        x_vec_loop = x.unsqueeze(dim=1).expand(
            -1, N, -1, -1
        )  # Xj, capitalized X in eq 6
        x_loop = x.unsqueeze(dim=2)  # xj, small x in eq 6
        x_star_vec_loop = x_star.unsqueeze(dim=1).expand(
            -1, N, -1, -1
        )  # capitalized generated X* in eq6
        x_star_loop = x_star  # small generalized x* in eq 6
        arg = (
            K_func((x_star_vec_loop.squeeze(dim=3) - x_star_loop) / h_star_loop)
            / h_star_loop
        )

        for j in range(k):
            arg = (
                arg
                * K_func(
                    (x_vec_loop[:, :, :, j] - x_loop[:, :, :, j]) / h_loop[:, :, :, j]
                )
                / h_loop[:, :, :, j]
            )
        return torch.mean(arg, dim=2)

    def loss(x, x_star_gen):
        # x: measurement of x_star
        # x_star_gen: generated x_star
        k = x.shape[2]
        N = x.shape[1]
        batch_size = x.shape[0]

        x_loop = x.unsqueeze(dim=1).expand(-1, N, -1, -1)
        x_obs_loop = x.unsqueeze(dim=2)
        x_star_loop = x_star_gen.unsqueeze(dim=1).expand(-1, N, -1, -1)

        h_vec_loop = torch.zeros(
            (batch_size, N, 1, k), 
            device=config.device, 
            dtype=torch.float64
            )
        for j in range(k):
            if type(config.window_size) is list:
                h_vec_loop[:, :, :, [j]] = (
                    (
                        config.window_size[j]
                        * torch.std(x[:, :, [j]], dim=1, keepdim=True)
                        * N ** (-1 / 5)
                    )
                    .unsqueeze(dim=1)
                    .expand(-1, N, -1, -1)
                )
            else:
                h_vec_loop[:, :, :, [j]] = (
                    (
                        config.window_size
                        * torch.std(x[:, :, [j]], dim=1, keepdim=True)
                        * N ** (-1 / 5)
                    )
                    .unsqueeze(dim=1)
                    .expand(-1, N, -1, -1)
                )
        h_star = (
            config.window_size_star * torch.std(x_star_gen, dim=1) * N ** (-1 / 5)
        )  # batch_size*1
        h_star_loop = h_star.unsqueeze(dim=1).expand(
            -1, N, -1
        )  # batch_size*N*1 batch,loop

        arg1_loop = f_joint(x, x_star_gen, h_vec_loop, h_star_loop)  # batch*loop

        arg2_loop = torch.ones(
            batch_size, 
            N, 
            device=config.device, 
            dtype=torch.float64)  # batch_size * N
        x_star_obs_loop = x_star_gen  # batch*N*1

        for j in range(k):
            x_j_obs_loop = x_obs_loop[:, :, :, j]  # batch_size*N*1
            x_j_vec_loop = x_loop[:, :, :, j]  # batch_size*N*N
            h_j_loop = h_vec_loop[:, :, :, j]  # batch_size*N*1
            arg2_loop *= f_xj_cond_xstar(
                x_j_obs_loop,
                x_star_obs_loop,
                x_j_vec_loop,
                x_star_loop,
                h_j_loop,
                h_star_loop,
            )
        arg2_loop *= f_xstar(x_star_obs_loop, x_star_loop, h_star_loop)
        
        if config.loss_fun == 'dl_hellinger':
            hellinger_distance = torch.square(torch.sqrt(arg1_loop) - torch.sqrt(arg2_loop))
            divergence_loss = torch.mean(hellinger_distance, dim=1)
        elif config.loss_fun == 'kl_loss':
            # KL_divergence = torch.abs(arg1_loop*(arg1_loop_log-arg2_loop_log))   ## PyTorch and Tensorflow implementation
            arg1_loop_log = torch.log(arg1_loop)
            arg2_loop_log = torch.log(arg2_loop)
            KL_divergence = arg1_loop * (arg1_loop_log - arg2_loop_log)
            divergence_loss = torch.mean(KL_divergence, dim=1)
        else:
            arg1_loop_log = torch.log(arg1_loop)
            arg2_loop_log = torch.log(arg2_loop)
            arg1 = torch.mean(arg1_loop_log, dim=1)  ##take as sample average
            arg2 = torch.mean(arg2_loop_log, dim=1)
            divergence_loss = torch.abs(arg1 - arg2)
            # divergence_loss = arg1 - arg2

        return divergence_loss

    return loss(x, x_star_gen)
