import multiprocessing
from collections import OrderedDict
from time import time

import numpy as np

from agents.parallel_trpo.parallel_trpo import utils
from agents.parallel_trpo.parallel_trpo.value_function import LinearVF


class TRPO(object):
    def __init__(self, env_id, make_env, max_kl, discount_factor, cg_damping):
        self.max_kl = max_kl
        self.discount_factor = discount_factor
        self.cg_damping = cg_damping

        env = make_env(env_id)
        observation_size = env.observation_space.shape[0]
        hidden_size = 64
        action_size = np.prod(env.action_space.shape)

        self.net = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

        self.logstd_action_dist_param = nn.Parameter(
            (.01 * np.random.randn(1, action_size)).astype(np.float32)
        )

        # value function
        # self.vf = VF(self.session)
        self.vf = LinearVF()

    def forward(self, obs, action, advantage, old_avg_action_dist, old_logstd_action_dist):
        avg_action_dist = self.net(obs)
        logstd_action_dist = torch.tile(self.logstd_action_dist_param, torch.stack((avg_action_dist.shape[0], 1)))

        batch_size = obs.shape[0]
        # what are the probabilities of taking self.action, given new and old distributions
        log_p_n = utils.gauss_log_prob(avg_action_dist, logstd_action_dist, action)
        log_oldp_n = utils.gauss_log_prob(old_avg_action_dist, old_logstd_action_dist, action)

        # tf.exp(log_p_n) / tf.exp(log_oldp_n)
        ratio = torch.exp(log_p_n - log_oldp_n)

        # importance sampling of surrogate loss (L in paper)
        surr = -torch.mean(ratio * advantage)

        batch_size_float = batch_size.float()
        # kl divergence and shannon entropy
        kl = utils.gauss_KL(
            old_avg_action_dist, old_logstd_action_dist, avg_action_dist,
            logstd_action_dist) / batch_size_float
        ent = utils.gauss_ent(avg_action_dist, logstd_action_dist) / batch_size_float
        return surr, kl, ent

        # # TODO not sure how to proceed from here
        losses = [surr, kl, ent]
        policy_vars = self.net.parameters() + [self.logstd_action_dist_pararm]
        policy_gradient = utils.flatgrad(surr, policy_vars)

        # KL divergence w/ itself, with first argument kept constant.
        kl_firstfixed = utils.gauss_selfKL_firstfixed(avg_action_dist, logstd_action_dist) / batch_size_float
        # gradient of KL w/ itself
        grads = torch.autograd.grad(kl_firstfixed, policy_vars)
        # what vector we're multiplying by
        start = 0
        tangents = []
        for var in policy_vars:
            size = var.size()
            tangents.append(param.view(-1))
            start += size

        # gradient of KL w/ itself * tangent
        gvp = [torch.sum(g * t) for (g, t) in zip(grads, tangents)]
        # 2nd gradient of KL w/ itself * tangent
        fvp = utils.flatgrad(gvp, policy_vars)


    # the actual parameter values
    def _get_flat_params(self):
        return torch.cat([torch.reshape(v, [v.size()]) for v in self.policy_vars], dim=0)

    # call this to set parameter values
    # TODO: CLEANUP!!!
    def _set_from_flat(self, theta):
        assigns = []
        total_size = sum(np.prod(v.get_shape().as_list()) for v in self.policy_vars)
        start = 0
        assigns = []
        for var in self.policy_vars:
            size = torch.size(var)
            var = torch.reshape(theta[start:start + size], var.shape)
            start += size

    # TODO: This is poorly done
    def get_policy(self):
        op = [var for var in self.policy_vars if 'policy' in var.name]
        return self.session.run(op)

    def learn(self, paths):
        start_time = time()

        # is it possible to replace A(s,a) with Q(s,a)?
        for path in paths:
            path["baseline"] = self.vf.predict(path)
            path["returns"] = utils.discount(path["rewards"], self.discount_factor)
            path["advantage"] = path["returns"] - path["baseline"]

        # puts all the experiences in a matrix: total_timesteps x options
        avg_action_dist = np.concatenate([path["avg_action_dist"] for path in paths])
        logstd_action_dist = np.concatenate([path["logstd_action_dist"] for path in paths])
        obs_n = np.concatenate([path["obs"] for path in paths])
        action_n = np.concatenate([path["actions"] for path in paths])

        # standardize to mean 0 stddev 1
        advant_n = np.concatenate([path["advantage"] for path in paths])
        advant_n -= advant_n.mean()
        advant_n /= (advant_n.std() + 1e-8)

        # train value function / baseline on rollout paths
        self.vf.fit(paths)

        # parameters
        old_theta = self._get_flat_params()

        g, _, _, _ = self.forward(obs_n, action_n, advant_n, avg_action_dist, logstd_action_dist)

        # computes fisher vector product: F * [self.policy_gradient]
        def fisher_vector_product(p):
            feed_dict[self.flat_tangent] = p
            return self.session.run(self.fvp, feed_dict) + p * self.cg_damping

        # TODO: This is the most costly step! Make me faster!
        # solve Ax = g, where A is Fisher information metrix and g is gradient of parameters
        # stepdir = A_inverse * g = x
        stepdir = utils.conjugate_gradient(fisher_vector_product, -g)

        # let stepdir =  change in theta / direction that theta changes in
        # KL divergence approximated by 0.5 x stepdir_transpose * [Fisher Information Matrix] * stepdir
        # where the [Fisher Information Matrix] acts like a metric
        # ([Fisher Information Matrix] * stepdir) is computed using the function,
        # and then stepdir * [above] is computed manually.
        shs = 0.5 * stepdir.dot(fisher_vector_product(stepdir))

        lm = np.sqrt(shs / self.max_kl)
        # if self.max_kl > 0.001:
        #     self.max_kl *= self.args.kl_anneal

        fullstep = stepdir / lm

        self._set_from_flat(old_theta + fullstep)

        surrogate_after, kl_after, entropy_after = self.forward(obs_n, action_n, advant_n, avg_action_dist, logstd_action_dist)

        ep_rewards = np.array([path["original_rewards"].sum() for path in paths])

        stats = OrderedDict()
        stats["Average sum of true rewards per episode"] = ep_rewards.mean()
        stats["Entropy"] = entropy_after
        stats["KL(old|new)"] = kl_after
        stats["Surrogate loss"] = surrogate_after
        stats["Frames gathered"] = sum([len(path["rewards"]) for path in paths])

        return stats, time() - start_time
