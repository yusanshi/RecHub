import os
import pandas as pd
import random
import hashlib
from multiprocessing import Pool
from parameters import parse_args

args = parse_args()

_positive_map = None
_candidate_length = None
_columns = None


def negative_sampling(first_index):
    if args.strict_negative:
        candidates = set(range(_candidate_length)) - set(
            _positive_map[first_index])
    else:
        candidates = range(_candidate_length)
    new_row = [
        first_index,
        random.sample(candidates, args.negative_sampling_ratio)
    ]
    return pd.Series(new_row, index=_columns)


class Sampler:
    def __init__(self, task, logger):
        self.task = task
        self.logger = logger
        # TODO: what if names of the two columns are the same
        if task['type'] == 'top-k-recommendation':
            if args.sample_cache:
                cahe_sensitive_keys = [
                    x for x in list(args.__dict__.keys()) if any([
                        y in x for y in [
                            'positive', 'negative', 'sample', 'sampling',
                            'cache'
                        ]
                    ])
                ]
                cache_sensitive_args = {
                    key: args.__dict__[key]
                    for key in cahe_sensitive_keys
                }

                logger.info(f'Cache sensitive args {cache_sensitive_args}')
                self.sample_cache_dir = f"./data/{args.dataset}/train/sample/{hashlib.md5((str(task)+str(cache_sensitive_args)).encode('utf-8')).hexdigest()}"
                os.makedirs(self.sample_cache_dir, exist_ok=True)
        else:
            raise NotImplementedError

    def sample(self, epoch):
        global _positive_map
        global _candidate_length
        global _columns

        if args.sample_cache:
            cache_file_path = os.path.join(
                self.sample_cache_dir,
                f"{epoch % args.num_sample_cache}-{self.task['filename']}")
        # If cache enabled and file exists, return directly
        if args.sample_cache and os.path.isfile(cache_file_path):
            df = pd.read_table(cache_file_path)
            self.logger.info(f'Read cache file {cache_file_path}')
            return df
        # Else, generate it
        df_positive = pd.read_table(
            f"./data/{args.dataset}/train/{self.task['filename']}")
        columns = df_positive.columns.tolist()
        assert len(columns) == 2 and 'value' not in columns
        if args.strict_negative:
            _positive_map = df_positive.groupby(
                columns[0]).agg(list).to_dict()[columns[1]]
        if args.positive_sampling:
            df_positive = df_positive.sample(frac=1)
            df_positive_first_based = df_positive.drop_duplicates(columns[0])
            df_positive_second_based = df_positive.drop_duplicates(columns[1])
            df_positive = pd.concat(
                [df_positive_first_based,
                 df_positive_second_based]).drop_duplicates()

        df_positive['value'] = 1

        _candidate_length = len(
            pd.read_table(f"./data/{args.dataset}/train/{columns[1]}.tsv"))
        _columns = columns

        with Pool(processes=args.num_workers) as pool:
            negative_series = pool.map(negative_sampling,
                                       df_positive[columns[0]].values)

        df_negative = pd.concat(negative_series, axis=1).T
        df_negative = df_negative.explode(columns[1])
        df_negative[columns[1]] = df_negative[columns[1]].astype(int)
        df_negative['value'] = 0
        df = pd.concat([df_positive, df_negative])

        if args.sample_cache:
            df.to_csv(cache_file_path, sep='\t', index=False)
            self.logger.info(f'Write cache file {cache_file_path}')

        return df


if __name__ == '__main__':
    from utils import create_logger
    logger = create_logger()
    logger.info(args)
    import json
    with open(f'metadata/{args.dataset}.json') as f:
        metadata = json.load(f)
    samplers = {}
    for task in metadata['task']:
        samplers[task['name']] = Sampler(task, logger)
    import enlighten
    pbar = enlighten.get_manager().counter(total=args.num_epochs,
                                           desc='Testing sampler',
                                           unit='epochs')
    for epoch in pbar(range(1, args.num_epochs + 1)):
        for task in metadata['task']:
            df = samplers[task['name']].sample(epoch)
