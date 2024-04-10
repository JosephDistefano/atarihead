from pathlib import Path
import yaml
from utils import skip_run
from data_processing.data_create import create_interim_files, create_processed_data
from visualization.visual_validation import Vis_Validate

from gaze.gaze_pred import gaze_prediction

# from features.gaze_pred import gaze_prediction
# from features.selective_gazed_act_pred import train_seas_model
# from features.gazed_act_gameplay import test_seas_model

config_path = Path(__file__).parents[1] / "src/config.yaml"
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)

with skip_run("skip", "Create Interim Files") as check, check():
    for game in config["valid_actions"]:
        print(game)
        create_interim_files(game, config)


with skip_run("skip", "Create Processed Data") as check, check():
    for game in config["valid_actions"]:
        print(game)
        create_processed_data(
            config,
            stack=config["STACK_SIZE"],
            game=game,
            from_ix=0,
            till_ix=-1,
            data_types=config["DATA_TYPES"],
        )

with skip_run("skip", "Visual Validation") as check, check():
    for game in config["valid_actions"]:
        game = "centipede"
        Vis_Validate(game, config)


with skip_run("skip", "Train Gaze Pred") as check, check():
    for game in config["VALID_ACTIONS"]:
        print(game)
        gaze_prediction(config, game, MODE="train")

# with skip_run("skip", "Train SGAZED") as check, check():
#     for game in config["VALID_ACTIONS"]:
#         print(game)
#         train_seas_model(config, game, Mode="Train")

# with skip_run('skip', 'Test SGAZED') as check, check():
#     for game in config['VALID_ACTIONS']:
#         test_seas_model(config,game)
