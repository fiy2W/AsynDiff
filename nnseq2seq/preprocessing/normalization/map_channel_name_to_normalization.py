from typing import Type

from nnseq2seq.preprocessing.normalization.default_normalization_schemes import ImageNormalization, \
    ZScoreNormalization, CTNormalization, NoNormalization, \
    RescaleTo01Normalization, RGBTo01Normalization, \
    RescaleTo005_995to01Normalization


channel_name_to_normalization_mapping = {
    'CTNormalization': CTNormalization,
    'NoNormalization': NoNormalization,
    'ZScoreNormalization': ZScoreNormalization,
    'RescaleTo01Normalization': RescaleTo01Normalization,
    'RGBTo01Normalization': RGBTo01Normalization,
    'RescaleTo005_995to01Normalization': RescaleTo005_995to01Normalization,
}


def get_normalization_scheme(channel_name: str) -> Type[ImageNormalization]:
    """
    If we find the channel_name in channel_name_to_normalization_mapping return the corresponding normalization. If it is
    not found, use the default (ZScoreNormalization)
    """
    norm_scheme = channel_name_to_normalization_mapping.get(channel_name)
    if norm_scheme is None:
        norm_scheme = RescaleTo005_995to01Normalization
    # print('Using %s for image normalization' % norm_scheme.__name__)
    return norm_scheme