from data import MetaLearningSystemDataLoader
from experiment_builder import ExperimentBuilder
from few_shot_learning_system import MAMLFewShotClassifier
from utils.parser_utils import get_args
from utils.dataset_tools import maybe_unzip_dataset



# python train_maml_system.py
# --name_of_args_json_file experiment_config/omniglot_maml_5_way_1_shot_batch_norm_log_5_seed_0.json
# --gpu_to_use 0

# Combines the arguments, model, data and experiment builders to run an experiment
args, device = get_args()
model = MAMLFewShotClassifier(args=args, device=device,
                              im_shape=(2, args.image_channels,
                                        args.image_height, args.image_width))
maybe_unzip_dataset(args=args)
data = MetaLearningSystemDataLoader
maml_system = ExperimentBuilder(model=model, data=data, args=args, device=device)
maml_system.run_experiment()
