# Intensity normalization in nnSeq2Seq
The type of intensity normalization applied in nnSeq2Seq can be controlled via the `normalization` in the dataset.json.

The `normalization` in `dataset.json` looks like this:

```
"normalization": {
    "0": "RescaleTo005_995to01Normalization",
    "1": "CTNormalization",
    "2": "NoNormalization",
    "3": "ZScoreNormalization",
    "4": "RescaleTo01Normalization"
},
```

The values in `normalization` determine the normalization scheme used for a given channel.

Here is a list of available normalization schemes:
- `CTNormalization`: Perform CT normalization. Specifically, collect intensity values from the foreground classes (all but the background and ignore) from all training cases and compute the 0.5 ($l$) and 99.5 ($h$) percentile of the values. Then normalize by $(max(X, l)-l)/(h-l)$. The normalization applied is the same for each training case (for this input channel). The values used by nnSeq2Seq for normalization are stored in the foreground_intensity_properties_per_channel entry in the corresponding plans file. This normalization suits modalities presenting physical quantities, such as CT images and ADC maps.
- `RescaleTo005_995to01Normalization`/anything else: Compute 0.5 ($l$) and 99.5 ($h$) percentile of the intensity values for each image, then normalize by $(max(X, l)-l)/(h-l)$.

## How to implement custom normalization strategies?
An example of adding your normalization strategies `MyNormalization` for the channel_name `MySequence`.
- Add your normalization strategies [here](../nnseq2seq/preprocessing/normalization/default_normalization_schemes.py) like the following example.
    ```python
    class MyNormalization(ImageNormalization):
        leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

        def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
            assert self.intensityproperties is not None, "MyNormalization requires intensity properties"

            image = image.astype(self.target_dtype, copy=False)

            # some metrics from fingerprints
            mean_intensity = self.intensityproperties['mean']
            std_intensity = self.intensityproperties['std']
            lower_bound = self.intensityproperties['percentile_00_5']
            upper_bound = self.intensityproperties['percentile_99_5']

            # define your strategies
            ...

            return image
    ```
- Register it [here](../nnseq2seq/preprocessing/normalization/map_channel_name_to_normalization.py) like the following example.
    ```python
    from nnseq2seq.preprocessing.normalization.default_normalization_schemes import ImageNormalization, \
        ZScoreNormalization, CTNormalization, NoNormalization, \
        RescaleTo01Normalization, RGBTo01Normalization, \
        RescaleTo005_995to01Normalization, \
        MyNormalization

    channel_name_to_normalization_mapping = {
        'CTNormalization': CTNormalization,
        'NoNormalization': NoNormalization,
        'ZScoreNormalization': ZScoreNormalization,
        'RescaleTo01Normalization': RescaleTo01Normalization,
        'RGBTo01Normalization': RGBTo01Normalization,
        'RescaleTo005_995to01Normalization': RescaleTo005_995to01Normalization,
        'MySequence': MyNormalization,  # Your normalization strategies
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
    ```