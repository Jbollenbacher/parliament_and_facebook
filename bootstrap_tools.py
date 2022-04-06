import pandas as pd

def mean_bootstrap(data, n_resamples = 10000):
    means = list()
    for _ in range(0,n_resamples):
        data_resample = data.sample(frac=1, replace=True)
        means.append(data_resample.mean())
    return pd.Series(means)

def difference_of_means_bootstrap(test, control, n_resamples=10000):
    diffs_of_means = list()
    for _ in range(0,n_resamples):
        test_resample = test.sample(frac=1, replace=True)
        control_resample = control.sample(frac=1, replace=True)
        diff_of_means = test_resample.mean() - control_resample.mean()
        diffs_of_means.append(diff_of_means)
    return pd.Series(diffs_of_means)

def test_mean_greater_than_x(data, x = 0, n_bootstrap_samples = 10000):
    means = mean_bootstrap(data, n_bootstrap_samples)
    print('mean estimate:', means.mean())
    print('95% CI:', means.quantile(0.025), means.quantile(0.975))
    print('( mean >', x,')  p-value:', (means<=x).mean())
    return means

def test_diff_of_means_greater_than_x(test, control, x = 0, n_bootstrap_samples = 10000):
    diffs_of_means = difference_of_means_bootstrap(test, control, n_bootstrap_samples)
    print('mean estimate:', diffs_of_means.mean())
    print('95% CI:', diffs_of_means.quantile(0.025), diffs_of_means.quantile(0.975))
    print('( diff_of_means >', x,')  p-value:', (diffs_of_means<=x).mean())
    return diffs_of_means