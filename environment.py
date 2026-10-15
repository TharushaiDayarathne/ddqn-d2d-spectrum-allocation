# environment.py

import numpy as np
import random

class SpectrumEnv:

    def __init__(self, snapshots_data, num_channels=4, sinr_threshold_db=5, debug=False):
        self.snapshots_data = snapshots_data
        self.num_channels = num_channels
        self.sinr_threshold = 10 ** (sinr_threshold_db / 10)
        self.debug = debug
        self.n_links = 300

    def reset(self):

        snapshot_name = "Snapshot_t1"

        data = self.snapshots_data[snapshot_name]

        self.gain_matrix = data["gain_matrix"]
        self.serving_tx = data["serving_tx"]
        self.power = data["power"]
        self.noise = data["noise"]

        self.channel_allocation = np.full(self.n_links, -1)
        self.link_order = np.random.permutation(self.n_links)
        self.current_step = 0
        self.total_throughput = 0

        return self._get_state()

    def _get_state(self):

        link_idx = self.link_order[self.current_step]
        tx_idx = self.serving_tx[link_idx]

        G_ii = self.gain_matrix[link_idx, tx_idx]

        interference = np.zeros(self.num_channels)

        for ch in range(self.num_channels):
            interferers = np.where(
                (self.channel_allocation == ch) &
                (np.arange(self.n_links) != link_idx)
            )[0]
            for j in interferers:
                tx_j = self.serving_tx[j]
                interference[ch] += self.power[j] * self.gain_matrix[link_idx, tx_j]

        state = np.concatenate(([G_ii], interference, [self.noise]))
        return state.astype(np.float32)

    def step(self, action):

        link_idx = self.link_order[self.current_step]
        tx_idx = self.serving_tx[link_idx]

        self.channel_allocation[link_idx] = action

        interferers = np.where(self.channel_allocation == action)[0]
        total_interference = 0

        for j in interferers:
            if j == link_idx:
                continue
            tx_j = self.serving_tx[j]
            total_interference += self.power[j] * self.gain_matrix[link_idx, tx_j]

        signal = self.power[link_idx] * self.gain_matrix[link_idx, tx_idx]
        sinr = signal / (total_interference + self.noise)

        if sinr >= self.sinr_threshold:
            reward = np.log2(1 + sinr)
        else:
            reward = 0

        self.total_throughput += reward

        self.current_step += 1
        done = self.current_step == self.n_links

        next_state = None if done else self._get_state()

        return next_state, reward, done
