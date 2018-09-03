'''
Data structure of the input .npz:
the data is save in python dictionary format with keys: 'acs', 'ep_rets', 'rews', 'obs'
the values of each item is a list storing the expert trajectory sequentially
a transition can be: (data['obs'][t], data['acs'][t], data['obs'][t+1]) and get reward data['rews'][t]
'''

from baselines import logger
import numpy as np
from sklearn.cluster import KMeans
import random

class Dset(object):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
        assert len(self.inputs) == len(self.labels)
        self.num_pairs = len(inputs)

        #to give interactive demo --eunjin
        self.kmeans = KMeans(n_clusters=64, algorithm='auto')
        logger.log("KMeans fitting...")
        self.kmeans = self.kmeans.fit(self.inputs)
        expert_kmeans_labels = self.kmeans.predict(self.inputs)
        self.expert_kmeans_data = list([] for i in range(64))
        for i in range(len(self.inputs)):
            self.expert_kmeans_data[expert_kmeans_labels[i]].append(i)

    def get_next_batch(self, obs):
        inputs = []
        labels = []
        batch_size = len(obs)
        kmeans_labels = list(set(self.kmeans.predict(obs)))
        
        for i in range(batch_size):
            label = random.choice(kmeans_labels)
            index = random.choice(self.expert_kmeans_data[label])            
            inputs.append(self.inputs[i])
            labels.append(self.labels[i])
        return inputs, labels



class Mujoco_Dset(object):
    def __init__(self, expert_path, train_fraction=0.7, traj_limitation=-1):
        traj_data = np.load(expert_path)
        if traj_limitation < 0:
            traj_limitation = len(traj_data['obs'])
        obs = traj_data['obs'][:traj_limitation]
        acs = traj_data['acs'][:traj_limitation]

        # obs, acs: shape (N, L, ) + S where N = # episodes, L = episode length
        # and S is the environment observation/action space.
        # Flatten to (N * L, prod(S))
        self.obs = np.reshape(obs, [-1, np.prod(obs.shape[2:])])
        self.acs = np.reshape(acs, [-1, np.prod(acs.shape[2:])])

        #self.rets = traj_data['ep_rets'][:traj_limitation]
        #self.avg_ret = sum(self.rets)/len(self.rets)
        #self.std_ret = np.std(np.array(self.rets))
        if len(self.acs) > 2:
            self.acs = np.squeeze(self.acs)
        assert len(self.obs) == len(self.acs)
        self.num_traj = min(traj_limitation, len(traj_data['obs']))
        self.num_transition = len(self.obs)
        self.dset = Dset(self.obs, self.acs)
        # for behavior cloning
        #self.train_set = Dset(self.obs[:int(self.num_transition*train_fraction), :],
        #                      self.acs[:int(self.num_transition*train_fraction), :])
        #self.val_set = Dset(self.obs[int(self.num_transition*train_fraction):, :],
        #                    self.acs[int(self.num_transition*train_fraction):, :])
        self.log_info()

    def log_info(self):
        logger.log("Total trajectorues: %d" % self.num_traj)
        logger.log("Total transitions: %d" % self.num_transition)
        #logger.log("Average returns: %f" % self.avg_ret)
        #logger.log("Std for returns: %f" % self.std_ret)

    def get_next_batch(self, obs, split=None):
        if split is None:
            return self.dset.get_next_batch(obs)
        elif split == 'train':
            return self.train_set.get_next_batch(batch_size)
        elif split == 'val':
            return self.val_set.get_next_batch(batch_size)
        else:
            raise NotImplementedError


    def plot(self):
        import matplotlib.pyplot as plt
        plt.hist(self.rets)
        plt.savefig("histogram_rets.png")
        plt.close()


def test(expert_path, traj_limitation, plot):
    dset = Mujoco_Dset(expert_path, traj_limitation=traj_limitation)
    if plot:
        dset.plot()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_path", type=str, default="../data/lift_demo.npz")
    parser.add_argument("--traj_limitation", type=int, default=-1)
    parser.add_argument("--plot", type=bool, default=False)
    args = parser.parse_args()
    test(args.expert_path, args.traj_limitation, args.plot)
