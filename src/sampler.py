import os
import pandas as pd
import numpy as np
import random
import hashlib
from parameters import parse_args
args = parse_args()


class Sampler:
    def __init__(self, task, logger):
        self.task = task
        self.logger = logger
        # TODO: what if names of the two columns are the same
        if task['type'] == 'top-k-recommendation':
            if args.sample_cache:
                args_sensitive_dict = {
                    key: args.__dict__[key]
                    for key in ['strict_negative', 'negative_sampling_ratio']
                }
                task_sensitive_dict = {
                    key: task[key]
                    for key in ['filename', 'type', 'loss']
                }

                logger.info(f'Task sensitive args {task_sensitive_dict}')
                logger.info(f'Args sensitive args {args_sensitive_dict}')
                self.sample_cache_dir = f"./data/{args.dataset}/train/sample/{hashlib.md5((str(task_sensitive_dict)+str(args_sensitive_dict)).encode('utf-8')).hexdigest()}"
                os.makedirs(self.sample_cache_dir, exist_ok=True)
        else:
            raise NotImplementedError

    def sample(self, epoch):
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
            positive_set = set(map(tuple, df_positive.values))

        df_positive = df_positive.sample(frac=1)
        df_positive_first_based = df_positive.drop_duplicates(columns[0])
        df_positive_second_based = df_positive.drop_duplicates(columns[1])
        df_positive = pd.concat(
            [df_positive_first_based,
             df_positive_second_based]).drop_duplicates()

        df_positive['value'] = 1

        candidate_length = len(
            pd.read_table(f"./data/{args.dataset}/train/{columns[1]}.tsv"))

        first_indexs = np.repeat(df_positive[columns[0]].values,
                                 args.negative_sampling_ratio)

        if args.strict_negative:
            # not use np.random.choice for performance issue
            second_indexs = np.concatenate([
                random.sample(range(candidate_length),
                              args.negative_sampling_ratio)
                for _ in range(len(df_positive))
            ],
                                           axis=0)
        else:
            second_indexs = np.random.randint(candidate_length,
                                              size=len(first_indexs))

        if args.strict_negative:
            negative_set = set(zip(first_indexs, second_indexs))
            common = positive_set & negative_set
            negative_set = negative_set - positive_set
            for x in common:
                first_index = x[0]
                while True:
                    pair = (first_index,
                            random.choice(range(candidate_length)))
                    if pair not in negative_set and pair not in positive_set:
                        negative_set.add(pair)
                        break

            df_negative = pd.DataFrame(np.array([*negative_set]),
                                       columns=columns)
        else:
            df_negative = pd.DataFrame(np.stack((first_indexs, second_indexs),
                                                axis=-1),
                                       columns=columns)

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
    with open(args.metadata_path) as f:
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
