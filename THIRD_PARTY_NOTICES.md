# Third-party notices

This directory was prepared from the WA-JEPA short-horizon implementation in the LeWorldModel repository. The upstream LeWorldModel MIT copyright and permission notice is retained verbatim in `LICENSE`.

External dependencies are not bundled:

- LeWorldModel: https://github.com/lucas-maes/le-wm
- stable-worldmodel: https://github.com/galilai-group/stable-worldmodel
- stable-pretraining: https://github.com/galilai-group/stable-pretraining
- DINOv2: https://github.com/facebookresearch/dinov2
- OGBench: https://github.com/seohongpark/ogbench

Datasets and pretrained weights must be obtained separately under their original terms. No external credentials, experiment checkpoints, or third-party repository checkout is included.

Full-task code uses the external [Patch Policy](https://github.com/gaoyuezhou/patch_policy) checkout at commit `674b09d9de0a7ea6226583c13078b4685b358485` for VQ-BeT and simulation wrappers. Its VQ-BeT implementation incorporates upstream vector quantization code; LIBERO and Push-T environments and assets retain their own licenses in that checkout. They are not redistributed here. Follow the respective licenses for dependencies, datasets, and pretrained weights.
