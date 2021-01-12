import os
import pandas as pd
import random
from tqdm import tqdm
import hashlib


class Sampler:
    def __init__(self, task, args, logger):
        self.task = task
        self.args = args
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
        if self.args.sample_cache:
            cache_file_path = os.path.join(
                self.sample_cache_dir,
                f"{epoch % self.args.num_sample_cache}-{self.task['filename']}"
            )
        # If cache enabled and file exists, return directly
        if self.args.sample_cache and os.path.isfile(cache_file_path):
            df = pd.read_table(cache_file_path)
            self.logger.info(f'Read cache file {cache_file_path}')
            return df
        # Else, generate it
        df_positive = pd.read_table(
            f"./data/{self.args.dataset}/train/{self.task['filename']}")
        columns = df_positive.columns.tolist()
        assert len(columns) == 2 and 'value' not in columns
        if self.args.strict_negative:
            positive_map = df_positive.groupby(
                columns[0]).agg(list).to_dict()[columns[1]]
        if self.args.positive_sampling:
            df_positive = df_positive.sample(frac=1)
            df_positive_first_based = df_positive.drop_duplicates(columns[0])
            df_positive_second_based = df_positive.drop_duplicates(columns[1])
            df_positive = pd.concat(
                [df_positive_first_based,
                 df_positive_second_based]).drop_duplicates()

        df_positive['value'] = 1

        df_negative = pd.DataFrame()
        df_negative[columns[0]] = df_positive[columns[0]]

        candidate_length = len(
            pd.read_table(
                f"./data/{self.args.dataset}/train/{columns[1]}.tsv"))

        def negative_sampling(row):
            if self.args.strict_negative:
                candidates = set(range(candidate_length)) - set(
                    positive_map[row[columns[0]]])
            else:
                candidates = range(candidate_length)
            new_row = [
                row[columns[0]],
                random.sample(candidates, self.args.negative_sampling_ratio)
            ]
            return pd.Series(new_row, index=columns)

        tqdm.pandas(desc=f"Negative sampling for task {self.task['name']}")
        df_negative = df_negative.progress_apply(negative_sampling, axis=1)
        df_negative = df_negative.explode(columns[1])
        df_negative[columns[1]] = df_negative[columns[1]].astype(int)
        df_negative['value'] = 0
        df = pd.concat([df_positive, df_negative])

        if self.args.sample_cache:
            df.to_csv(cache_file_path, sep='\t', index=False)
            self.logger.info(f'Write cache file {cache_file_path}')

        return df
