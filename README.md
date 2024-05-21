# Voice Guard: Protecting Voice Privacy with Strong and Imperceptible Adversarial Perturbation in the Time Domain

This is the official implementation of the paper [Voice Guard: Protecting Voice Privacy with Strong and Imperceptible Adversarial Perturbation in the Time Domain](https://www.ijcai.org/proceedings/2023/535).
We use adversarial attack to prevent one's voice from improperly utilized in voice conversion.
The conversion fails when adversarial noise is added to the input utterance.

Thanks to [Attack VC](https://github.com/cyhuang-tw/attack-vc) for the code base.

This implementation is based on [Attack VC](https://github.com/cyhuang-tw/attack-vc), so we only show the implementation of the core functions.

This algorithm is no longer maintained, please implement it as you wish.

## Core coda

_attack_utils.py_ : implements Voice Guard's core attack flow.

_data_utils.py_ : implements gradient-preserving feature extraction.

_generate_masking_threshold.py_ : implements the calculation of the masking threshold.

## Reference

Please cite our paper if you find it useful.

```bib
@inproceedings{ijcai2023p535,
  title     = {Voice Guard: Protecting Voice Privacy with Strong and Imperceptible Adversarial Perturbation in the Time Domain},
  author    = {Li, Jingyang and Ye, Dengpan and Tang, Long and Chen, Chuanxi and Hu, Shengshan},
  booktitle = {Proceedings of the Thirty-Second International Joint Conference on
               Artificial Intelligence, {IJCAI-23}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Edith Elkind},
  pages     = {4812--4820},
  year      = {2023},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2023/535},
  url       = {https://doi.org/10.24963/ijcai.2023/535},
}
```
