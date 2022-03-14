import argparse, logging, git, os, sys
import numpy, torch
from transformers import set_seed, logging as trf_logging
from omegaconf import OmegaConf

def set_seeds(seed=42):
    set_seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_experiment():
    parser = argparse.ArgumentParser()

    parser.add_argument('--collator_config')
    parser.add_argument('--dataset_config')
    parser.add_argument('--experiment_config')
    parser.add_argument('--model_config')
    parser.add_argument('--task_config', default='configs/empty.yaml')
    parser.add_argument('--tokenizer_config')
    parser.add_argument('--trainer_config', default='configs/trainer/default.yaml')
    parser.add_argument('--training_args')

    args, _ = parser.parse_known_args()

    collator_cfg = OmegaConf.load(args.collator_config)
    dataset_cfg = OmegaConf.load(args.dataset_config)
    exp_cfg = OmegaConf.load(args.experiment_config)
    model_cfg = OmegaConf.load(args.model_config)
    tok_cfg = OmegaConf.load(args.tokenizer_config)
    ta_cfg = OmegaConf.load(args.training_args)
    trainer_cfg = OmegaConf.load(args.trainer_config)
    cli_cfg = OmegaConf.from_cli()

    config = OmegaConf.merge(collator_cfg, dataset_cfg, exp_cfg, model_cfg, tok_cfg, ta_cfg, trainer_cfg, cli_cfg)



    set_seeds(config['seed'])

    if not os.path.isdir(config['log_directory']):
        os.makedirs(config['log_directory'])

    # Setup logging
    logging.root.handlers = []
    # noinspection PyArgumentList
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("{}/{}.log".format(config['log_directory'], config['experiment_name'])),
            logging.StreamHandler()
        ]
    )

    trf_logging.set_verbosity_info()
    trf_logging.enable_propagation()

    logging.info('Loaded config: \n{}'.format(OmegaConf.to_yaml(config)))

    repo = git.Repo(search_parent_directories=False)
    commit_id = repo.head.object.hexsha
    logging.info('Using code from git commit: {}'.format(commit_id))

    # Set visible devices
    os.environ['CUDA_VISIBLE_DEVICES'] = config['visible_devices']
    import torch
    try:
        assert torch.cuda.device_count() == config['n_gpu']
    except AssertionError as err:
        logging.error('Expected {} GPUs availble, but only see {} (visible devices: {})'.format(config['n_gpu'],
                                                                                                torch.cuda.device_count(),
                                                                                                config[
                                                                                                    'visible_devices']))
        sys.exit(1)

    logging.info('Number of GPUs available: {}'.format(torch.cuda.device_count()))
    logging.info('Using the following GPUs: {}'.format(config['visible_devices']))

    return args, config
