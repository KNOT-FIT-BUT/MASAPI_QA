
from scalingqa.generative_reader.models.fusion_in_generative_reader import T5FusionInDecoder


class T5FusionInDecoderWrapper(T5FusionInDecoder):

    def __init__(self, config):
        super().__init__(config["encoder_config"])
