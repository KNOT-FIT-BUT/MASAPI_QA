import torch
from transformers import T5Config


class Checkpoint(object):

    @classmethod
    def load_config(cls, path):
        model_state_dict = torch.load(path, map_location=torch.device("cpu"))
        config = model_state_dict["config"] if "config" in model_state_dict else {}

        if "encoder_config" in config and isinstance(config["encoder_config"], T5Config):
            # there were issues with some missing parameters (maybe because of different versions)
            config["encoder_config"] = T5Config.from_dict(config["encoder_config"].to_dict())

        return config

    @classmethod
    def load_model(cls, model, path):
        ckpt = torch.load(path)
        if "state_dict" in ckpt:
            # some checkpoints are using other naming
            model_state_dict = ckpt["state_dict"]
        else:
            model_state_dict = ckpt["model"]
        model.load_state_dict(model_state_dict, strict=False)
