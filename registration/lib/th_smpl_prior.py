"""
https://github.com/bharat-b7/IPNet
"""
import numpy as np
import torch

# is this the best prior we want to use? or shall we replace it with a better one like vposer? 
def get_prior(gender='male', precomputed=False):
    if precomputed:
        prior = Prior(sm=None)
        return prior['Generic']
    else:
        from lib.smpl_paths import SmplPaths

        dp = SmplPaths(gender=gender)
        if gender == 'neutral':
            dp_prior = SmplPaths(gender='male')
        else:
            dp_prior = dp

        prior = Prior(dp_prior.get_smpl())
        return prior['Generic']


class th_Mahalanobis(object):
    def __init__(self, mean, prec, prefix):
        self.mean = torch.tensor(mean.astype('float32'), requires_grad=False).unsqueeze(axis=0).cuda()
        self.prec = torch.tensor(prec.astype('float32'), requires_grad=False).cuda()
        self.prefix = prefix

    def __call__(self, pose, prior_weight=1.):
        '''
        :param pose: Batch x pose_dims
        :return:
        '''
        # return (pose[:, self.prefix:] - self.mean)*self.prec
        temp = pose[:, self.prefix:] - self.mean
        temp2 = torch.matmul(temp, self.prec) * prior_weight
        return (temp2 * temp2).sum(dim=1)
        

class Prior(object):
    def __init__(self, sm, prefix=3):
        self.prefix = prefix
        if sm is not None:
            # Compute mean and variance based on the provided poses
            self.pose_subjects = sm.pose_subjects
            all_samples = [p[prefix:] for qsub in self.pose_subjects
                           for name, p in zip(qsub['pose_fnames'], qsub['pose_parms'])]  # if 'CAESAR' in name or 'Tpose' in name or 'ReachUp' in name]
            self.priors = {'Generic': self.create_prior_from_samples(all_samples)}
        else:
            import pickle as pkl
            # Load pre-computed mean and variance
            dat = pkl.load(open('assets/pose_prior.pkl', 'rb'))
            self.priors = {'Generic': th_Mahalanobis(dat['mean'],
                           dat['precision'],
                           self.prefix)}

    def create_prior_from_samples(self, samples):
        from sklearn.covariance import GraphicalLassoCV
        from numpy import asarray, linalg
        model = GraphicalLassoCV()
        model.fit(asarray(samples))
        return th_Mahalanobis(asarray(samples).mean(axis=0),
                           linalg.cholesky(model.precision_),
                           self.prefix)

    def __getitem__(self, pid):
        if pid not in self.priors:
            samples = [p[self.prefix:] for qsub in self.pose_subjects
                       for name, p in zip(qsub['pose_fnames'], qsub['pose_parms'])
                       if pid in name.lower()]
            self.priors[pid] = self.priors['Generic'] if len(samples) < 3 \
                               else self.create_prior_from_samples(samples)

        return self.priors[pid]