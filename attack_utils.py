from typing import Dict, Tuple, Any, Union

import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from tqdm import trange

from data_utils import wav2mel_tensor, Transform
import math

from asr_model.find_most_unlikely_speaker import speaker_name


class CosineSimilarity(nn.Module):
    """
    This similarity function simply computes the cosine similarity between each pair of vectors. It has
    no parameters.
    """

    def forward(self, tensor_1, tensor_2):
        normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
        normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
        return (normalized_tensor_1 * normalized_tensor_2).sum(dim=-1)


def get_att_dis(target, behaviored):
    attention_distribution = []

    for i in range(behaviored.size(0)):
        attention_score = torch.cosine_similarity(target, behaviored[i].view(1, -1))
        attention_distribution.append(attention_score)
    attention_distribution = torch.Tensor(attention_distribution)

    return attention_distribution / torch.sum(attention_distribution, 0)


def mask_wav_emb_attack(
    model: nn.Module, vc_src: Tensor, vc_tgt: Tensor, vc_tgt_mask: Tensor, vc_tgt_psd_max: Tensor,
        adv_tgt: Tensor, eps: float, n_iters: int, vc_tgt_speaker: str, init_c, attr: Dict, **kwargs,
) -> Tensor:
    ptb = torch.zeros_like(vc_tgt).normal_(0, 1).requires_grad_(True)
    criterion = nn.MSELoss()

    with torch.no_grad():
        vc_tgt_mel = wav2mel_tensor(vc_tgt, attr=attr, **kwargs)
        adv_tgt_mel = wav2mel_tensor(adv_tgt, attr=attr, **kwargs)
        org_emb = model.speaker_encoder(vc_tgt_mel)
        tgt_emb = model.speaker_encoder(adv_tgt_mel)

        vc_src_mel = wav2mel_tensor(vc_src, attr=attr, **kwargs)
        org_content, _ = model.content_encoder(vc_src_mel)

        loss = criterion(org_emb, tgt_emb)
        print(loss.item())

    speaker_embedding_dict = np.load('your-speaker_encoder_embedding_dict-path',
                                     allow_pickle=True).item()

    speaker_embedding_array = np.empty([len(speaker_name), 128], dtype=np.float32)

    for idx in range(len(speaker_name)):
        speaker_embedding_array[idx] = speaker_embedding_dict[speaker_name[idx]]

    device = next(model.parameters()).device
    speaker_embedding_tensor = torch.tensor(speaker_embedding_array).to(device)

    psd_transformer = Transform(**kwargs)


    no_attack_flag = False

    attack_flag, ptb, step = deal_mask_wav_emb_attack_1(model, vc_src, vc_tgt, n_iters, vc_tgt_mask, org_emb, tgt_emb, org_content,
                                                        speaker_embedding_tensor, vc_tgt_speaker, eps, vc_tgt_psd_max,
                                                        psd_transformer, early_stop=False, attr=attr, **kwargs)
    print('attack_step_1 || ', 'attack_flag : ', attack_flag, ' || eps : ', eps, ' || step : ', step)
    if step == 1:
        no_attack_flag = True

    attack_step_1_ptb = ptb

    c = init_c
    attack_step_2_c = 0
    false_c = 0
    final_adv_inp = vc_tgt + 2. * eps * ptb.tanh()
    if no_attack_flag is True:
        return final_adv_inp
    i = 0
    while i < 8:
        attack_flag, adv_inp, step = deal_mask_wav_emb_attack_2(model, vc_src, vc_tgt, n_iters / 2., vc_tgt_mask, org_emb, tgt_emb, org_content,
                                                          speaker_embedding_tensor, vc_tgt_speaker, eps, vc_tgt_psd_max,
                                                          psd_transformer, c, attack_step_1_ptb, early_stop=False,
                                                          attr=attr, **kwargs)
        print('attack_step_2 || ', 'attack_flag : ', attack_flag, ' || c : ', c, ' || step : ', step)
        if attack_flag is True and false_c == 0:
            attack_step_2_c = c
            c = c * 2.
            final_adv_inp = adv_inp
            i = i + 1
        elif attack_flag is True and false_c != 0:
            attack_step_2_c = c
            c = (c + false_c) / 2.
            final_adv_inp = adv_inp
            i = i + 1
        elif attack_flag is False:
            false_c = c
            c = (attack_step_2_c + c) / 2

        i = i + 1

    print('attack_step || eps : ', eps, 'attack_step_2 || c : ', attack_step_2_c)
    return final_adv_inp

def deal_mask_wav_emb_attack_1(
        model, vc_src, vc_tgt, n_iters, vc_tgt_mask, org_emb, tgt_emb, org_content, speaker_embedding_tensor,
        vc_tgt_speaker, eps, vc_tgt_psd_max, psd_transformer, early_stop, attr, **kwargs):
    print('attack step 1 ...')
    ptb = torch.zeros_like(vc_tgt).normal_(0, 1).requires_grad_(True)
    opt = torch.optim.Adam([ptb])
    criterion = nn.MSELoss()
    relu = torch.nn.ReLU(inplace=True)

    section = 2.
    attack_flag, adv_inp = False, vc_tgt
    i = 1
    for i in range(1, n_iters+1):
        adv_inp = vc_tgt + section * eps * ptb.tanh()
        adv_inp_mel = wav2mel_tensor(adv_inp, attr=attr, **kwargs)

        adv_emb = model.speaker_encoder(adv_inp_mel)

        adv_tgt_cos_similar = get_att_dis(adv_emb, speaker_embedding_tensor)
        adv_speaker = speaker_name[torch.argmax(adv_tgt_cos_similar)]
        attack_flag = adv_speaker != vc_tgt_speaker

        logits_delta = psd_transformer(adv_inp - vc_tgt, vc_tgt_psd_max)
        loss_th = torch.mean(relu(logits_delta - vc_tgt_mask))

        loss_emb_l2 = criterion(adv_emb, tgt_emb) - 0.1 * criterion(adv_emb, org_emb)

        loss = loss_emb_l2

        if i % 100 == 0:
            print('step 1 : ', i, ' || loss_emb_l2 : ', loss_emb_l2.item(), ' || loss_th : ', loss_th.item())

        if attack_flag is True and early_stop is True:
            return attack_flag, ptb, i

        opt.zero_grad()
        loss.backward()
        opt.step()

    return attack_flag, ptb, i


def deal_mask_wav_emb_attack_2(
        model, vc_src, vc_tgt, n_iters, vc_tgt_mask, org_emb, tgt_emb, org_content,
        speaker_embedding_tensor, vc_tgt_speaker, eps, vc_tgt_psd_max,
        psd_transformer, c, attack_1_ptb, early_stop, attr, **kwargs):
    print('attack step 2 ...')
    ptb = torch.clone(attack_1_ptb.detach())
    ptb = ptb.requires_grad_(True)
    vc_tgt = torch.clone(vc_tgt.detach()).requires_grad_(False)
    opt = torch.optim.Adam([ptb])
    criterion = nn.MSELoss()
    relu = torch.nn.ReLU(inplace=True)

    section = 2.
    attack_flag, adv_inp = False, vc_tgt
    i = 1
    for i in range(1, int(n_iters)+1):
        adv_inp = vc_tgt + section * eps * ptb.tanh()
        adv_inp_mel = wav2mel_tensor(adv_inp, attr=attr, **kwargs)

        adv_emb = model.speaker_encoder(adv_inp_mel)

        # ASV attack flag
        cos_similar = get_att_dis(adv_emb, speaker_embedding_tensor)
        adv_speaker = speaker_name[torch.argmax(cos_similar)]
        attack_flag = adv_speaker != vc_tgt_speaker

        logits_delta = psd_transformer(adv_inp - vc_tgt, vc_tgt_psd_max)
        loss_th = torch.mean(relu(logits_delta - vc_tgt_mask))

        loss_emb_l2 = criterion(adv_emb, tgt_emb) - 0.1 * criterion(adv_emb, org_emb)

        loss = loss_emb_l2 + c * loss_th

        if i % 100 == 0:
            print('step 2 : ', i, ' || loss_emb_l2 : ', loss_emb_l2.item(), ' || loss_th : ', loss_th.item())

        if attack_flag is True and early_stop is True:
            return attack_flag, adv_inp, i
        elif attack_flag is False:
            return attack_flag, adv_inp, i

        opt.zero_grad()
        loss.backward()
        opt.step()

    return attack_flag, adv_inp, i