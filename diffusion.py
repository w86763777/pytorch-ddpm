import torch
import torch.nn.functional as F
from tqdm import tqdm


device = torch.device('cuda:0')


class GaussianDiffusion:
    def __init__(self, beta_1, beta_T, T, img_size=32,
                 mean_type='eps', var_type='fixedlarge'):
        assert mean_type in ['xprev' 'xstart', 'epsilon']
        assert var_type in ['fixedlarge', 'fixedsmall']

        self.T = T
        self.img_size = img_size
        self.mean_type = mean_type
        self.var_type = var_type

        self.betas = torch.linspace(beta_1, beta_T, T).to(device).double()
        assert (self.betas > 0).all() and (self.betas <= 1).all()

        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
        assert list(alphas_bar_prev.shape) == [T, ]

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_bar = torch.sqrt(alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1. - alphas_bar)
        self.log_one_minus_alphas_bar = torch.log(1. - alphas_bar)
        self.sqrt_recip_alphas_bar = torch.sqrt(1. / alphas_bar)
        self.sqrt_recipm1_alphas_bar = torch.sqrt(1. / alphas_bar - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_var = \
            self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar)
        # below: log calculation clipped because the posterior variance is 0 at
        # the beginning of the diffusion chain
        self.posterior_log_var_clipped = torch.log(torch.cat(
            [self.posterior_var[1:2], self.posterior_var[1:]]))
        self.posterior_mean_coef1 = \
            torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar)
        self.posterior_mean_coef2 = \
            torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar)

    @staticmethod
    def extract(v, t, x_shape):
        """
        Extract some coefficients at specified timesteps, then reshape to
        [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        out = torch.gather(v, index=t, dim=0).float()
        return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

    def q_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        posterior_mean = (
            self.extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            self.extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_var_clipped = self.extract(
            self.posterior_log_var_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_var_clipped

    def predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            self.extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
            self.extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )

    def predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            self.extract(
                1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
            self.extract(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t,
                x_t.shape) * x_t
        )

    def p_mean_variance(self, model, x_t, t):
        # below: only log_variance is used in the KL computations
        model_log_var = {
            # for fixedlarge, we set the initial (log-)variance like so to
            # get a better decoder log likelihood
            'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
                                               self.betas[1:]])),
            'fixedsmall': self.posterior_log_var_clipped,
        }[self.var_type]
        model_log_var = self.extract(model_log_var, t, x_t.shape)

        # Mean parameterization
        if self.mean_type == 'xprev':       # the model predicts x_{t-1}
            x_prev = model(x_t, t)
            x_0 = self.predict_xstart_from_xprev(x_t, t, xprev=x_prev)
            model_mean = x_prev
        elif self.mean_type == 'xstart':    # the model predicts x_0
            x_0 = model(x_t, t)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        elif self.mean_type == 'epsilon':   # the model predicts epsilon
            eps = model(x_t, t)
            x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        else:
            raise NotImplementedError(self.mean_type)
        x_0 = torch.clip(x_0, -1., 1.)

        return model_mean, model_log_var

    def sample(self, model, x_T=None, batch_size=1, verbose=False):
        """
        Algorithm 2.
        """
        if x_T is not None:
            x = x_T
            batch_size = x_T.shape[0]
        else:
            x = torch.randn((batch_size, 3, self.img_size, self.img_size))
            x = x.to(device)
        time_steps = list(reversed(range(self.T)))
        for time_step in tqdm(time_steps, disable=not verbose, leave=False):
            t = x.new_ones([batch_size, ], dtype=torch.long) * time_step
            mean, log_var = self.p_mean_variance(model, x_t=x, t=t)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0
            x = mean + torch.exp(0.5 * log_var) * noise
        return torch.clip(x, -1, 1)

    def train(self, model, x_0):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=device)
        noise = torch.randn_like(x_0)
        x_t = (
            self.extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            self.extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        loss = F.mse_loss(model(x_t, t), noise)
        return loss
