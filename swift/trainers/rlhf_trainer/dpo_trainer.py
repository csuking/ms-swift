# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel
from transformers import PreTrainedModel
from typing import Literal
from trl import DPOTrainer as HFDPOTrainer
from trl.trainer.dpo_config import FDivergenceConstants, FDivergenceType
from trl.trainer.utils import cap_exp

from ..mixin import SwiftMixin
from .rlhf_mixin import RLHFTrainerMixin

del HFDPOTrainer.__init__


class DPOTrainer(RLHFTrainerMixin, SwiftMixin, HFDPOTrainer):

    def __init__(self,
                 model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
                 ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
                 *_args,
                 **kwargs):
        from trl.trainer import FDivergenceConstants
        args = kwargs['args']
        self.label_smoothing = args.label_smoothing
        self.loss_type = args.loss_type
        self.precompute_ref_log_probs = args.precompute_ref_log_probs
        self.f_divergence_type = args.f_divergence_type
        self.f_divergence_params = {FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY: args.f_alpha_divergence_coef}
        self.is_peft_model = isinstance(model, PeftModel)

        self.ref_adapter_name = args.ref_adapter_name
        self.reference_free = args.reference_free
        self.use_weighting = False

        super().__init__(model, ref_model, *_args, **kwargs)

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        batch = batch.copy()
        num_examples = batch['labels'].shape[0] // 2
        labels = batch.pop('labels', None)
        if self.is_encoder_decoder:
            batch['labels'] = labels

        if self.aux_loss_enabled:
            batch['output_router_logits'] = True
        outputs = model(**batch, use_cache=False)
        batch['labels'] = labels
        if outputs.logits.shape[1] != labels.shape[1]:
            # for llava, the model returns logits for the entire sequence, including the image tokens
            # (placed before the text tokens)
            outputs.logits = outputs.logits[:, -labels.shape[1]:]
        for key in ['input_ids', 'attention_mask', 'labels']:
            batch[f'concatenated_{key}'] = batch.pop(key, None)
        if self.__class__.__name__ == 'ORPOTrainer':  # Pass-through labels
            batch['concatenated_input_ids'] = batch['concatenated_labels']

        all_logits = outputs.logits

        if all_logits.shape[:2] != batch['concatenated_labels'].shape[:2]:
            # for llava, the model returns logits for the entire sequence,
            # including the image tokens (placed before the text tokens)
            seq_len = batch['concatenated_labels'].shape[1]
            all_logits = all_logits[:, -seq_len:]

        all_logps, size_completion = self.get_batch_logps(
            all_logits,
            batch['concatenated_labels'],
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        output = {}

        def cross_entropy_loss(logits, labels):
            if not self.is_encoder_decoder:
                # Shift so that tokens < n predict n
                logits = logits[..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.label_pad_token_id)
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)
            # Enable model parallelism
            labels = labels.to(logits.device)
            loss = loss_fct(logits, labels)
            return loss

        if self.args.rpo_alpha is not None:
            labels = batch['concatenated_labels'].clone()
            output['nll_loss'] = cross_entropy_loss(all_logits[:num_examples], labels[:num_examples])

        if self.loss_type == 'ipo':
            all_logps = all_logps / size_completion

        output['chosen_logps'] = all_logps[:num_examples]
        output['rejected_logps'] = all_logps[num_examples:]
        output['mean_chosen_logits'] = all_logits[:num_examples].mean()
        output['mean_rejected_logits'] = all_logits[num_examples:].mean()

        if self.aux_loss_enabled:
            output['aux_loss'] = outputs.aux_loss

        # 保留 delta_reward
        if 'delta_reward' in batch:
            output['delta_reward'] = batch['delta_reward']

        return output

    def get_batch_loss_metrics(
        self,
        model,
        batch: dict[str, Union[list, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """计算批次损失和指标"""
        metrics = {}
        
        # 获取 delta_reward
        delta_reward = batch.get('delta_reward', None)
        
        # 计算 log probabilities
        model_output = self.concatenated_forward(model, batch)
        
        # 计算参考模型的 log probabilities
        if "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
            ref_chosen_logps = batch["ref_chosen_logps"]
            ref_rejected_logps = batch["ref_rejected_logps"]
        else:
            if self.ref_model is None:
                # 如果 ref_model 为 None，使用零张量
                ref_chosen_logps = torch.zeros_like(model_output["chosen_logps"])
                ref_rejected_logps = torch.zeros_like(model_output["rejected_logps"])
            else:
                with self.null_ref_context():
                    ref_output = self.concatenated_forward(self.ref_model, batch)
                    ref_chosen_logps = ref_output["chosen_logps"]
                    ref_rejected_logps = ref_output["rejected_logps"]
        
        # 使用带 delta_reward 的损失函数
        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            model_output["chosen_logps"],
            model_output["rejected_logps"],
            ref_chosen_logps,
            ref_rejected_logps,
            delta_reward
        )
        
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        
        # 添加 RPO 损失
        if self.args.rpo_alpha is not None:
            losses = losses + self.args.rpo_alpha * model_output["nll_loss"]
            
        # 添加权重调整
        if self.use_weighting:
            losses = losses * model_output["policy_weights"]
            
        # 添加辅助损失
        if self.aux_loss_enabled:
            losses = losses + self.aux_loss_coef * model_output["aux_loss"]
        
        # 计算指标
        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = self.accelerator.gather_for_metrics(chosen_rewards).mean().item()
        metrics[f"{prefix}rewards/rejected"] = self.accelerator.gather_for_metrics(rejected_rewards).mean().item()
        metrics[f"{prefix}rewards/accuracies"] = self.accelerator.gather_for_metrics(reward_accuracies).mean().item()
        metrics[f"{prefix}rewards/margins"] = (
            self.accelerator.gather_for_metrics(chosen_rewards - rejected_rewards).mean().item()
        )
        metrics[f"{prefix}logps/chosen"] = (
            self.accelerator.gather_for_metrics(model_output["chosen_logps"]).detach().mean().item()
        )
        metrics[f"{prefix}logps/rejected"] = (
            self.accelerator.gather_for_metrics(model_output["rejected_logps"]).detach().mean().item()
        )
        metrics[f"{prefix}logits/chosen"] = (
            self.accelerator.gather_for_metrics(model_output["mean_chosen_logits"]).detach().mean().item()
        )
        metrics[f"{prefix}logits/rejected"] = (
            self.accelerator.gather_for_metrics(model_output["mean_rejected_logits"]).detach().mean().item()
        )
        
        if self.args.rpo_alpha is not None:
            metrics[f"{prefix}nll_loss"] = (
                self.accelerator.gather_for_metrics(model_output["nll_loss"]).detach().mean().item()
            )
            
        if self.aux_loss_enabled:
            metrics[f"{prefix}aux_loss"] = (
                self.accelerator.gather_for_metrics(model_output["aux_loss"]).detach().mean().item()
            )
        
        return losses.mean(), metrics

    def dpo_loss(
        self,
        chosen_logps: torch.FloatTensor,
        rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
        delta_reward: Optional[torch.FloatTensor] = None
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        计算带 delta_reward 的 DPO 损失
        
        当 delta_reward 存在时:
        L = -β · σ(Δr) · ( log πθ(y_w|x) - log πθ(y_l|x) )
        其中 Δr = delta_reward  (已经随样本给出，范围可为 [-1,1])
        
        当 delta_reward 不存在时:
        使用原始的 DPO 损失计算方式
        
        Args:
            chosen_logps: 选择的动作的 log probabilities
            rejected_logps: 拒绝的动作的 log probabilities
            ref_chosen_logps: 参考模型选择的动作的 log probabilities
            ref_rejected_logps: 参考模型拒绝的动作的 log probabilities
            delta_reward: 两个动作的奖励差值
            
        Returns:
            tuple: (损失值, 选择的损失, 拒绝的损失)
        """
        device = self.accelerator.device

        if delta_reward is not None:
            """
            L = -β · σ(Δr) · ( log πθ(y_w|x) - log πθ(y_l|x) )
            其中 Δr = delta_reward  (已经随样本给出，范围可为 [-1,1])
            """
            delta_r = delta_reward.to(device)                       # Δr
            weight  = torch.sigmoid(delta_r)                        # σ(Δr)

            logp_diff = (chosen_logps - rejected_logps).to(device)  # logπ_w - logπ_l
            
            # 添加 label_smoothing
            losses = (
                -self.beta * weight * logp_diff * (1 - self.label_smoothing)  # 主要损失项
                - self.beta * weight * (-logp_diff) * self.label_smoothing    # 平滑项
            )

            # 为了接口一致，随便给出 chosen/rejected reward（这里用 logp_diff 的正负拆分）
            chosen_rewards   =  0.5 * self.beta * logp_diff.detach()
            rejected_rewards = -0.5 * self.beta * logp_diff.detach()
            return losses, chosen_rewards, rejected_rewards

        # 原始的 DPO 损失计算
        # Get the log ratios for the chosen and rejected responses
        chosen_logratios = chosen_logps.to(device) - (not self.reference_free) * ref_chosen_logps.to(device)
        rejected_logratios = rejected_logps.to(device) - (not self.reference_free) * ref_rejected_logps.to(device)

        if self.f_divergence_type == FDivergenceType.ALPHA_DIVERGENCE.value:
            alpha_coef = FDivergenceConstants.ALPHA_DIVERGENCE_COEF_DEFAULT
            if self.f_divergence_params and FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY in self.f_divergence_params:
                alpha_coef = float(self.f_divergence_params[FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY])
            logits = (cap_exp(rejected_logratios * -alpha_coef) - cap_exp(chosen_logratios * -alpha_coef)) / alpha_coef
        else:
            logratios = chosen_logps - rejected_logps
            if self.reference_free:
                ref_logratios = torch.tensor([0], dtype=logratios.dtype, device=logratios.device)
            else:
                ref_logratios = ref_chosen_logps - ref_rejected_logps

            logratios = logratios.to(self.accelerator.device)
            ref_logratios = ref_logratios.to(self.accelerator.device)
            logits = logratios - ref_logratios

            if self.f_divergence_type == FDivergenceType.JS_DIVERGENCE.value:
                logits -= F.softplus(chosen_logratios) - F.softplus(rejected_logratios)

        # 根据不同的损失类型计算损失
        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "robust":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                + F.logsigmoid(-self.beta * logits) * self.label_smoothing
            ) / (1 - 2 * self.label_smoothing)
        elif self.loss_type == "exo_pair":
            # eqn (16) of the EXO paper: https://huggingface.co/papers/2402.00856
            import math
            if self.label_smoothing == 0:
                self.label_smoothing = 1e-3
            losses = (self.beta * logits).sigmoid() * (
                F.logsigmoid(self.beta * logits) - math.log(1 - self.label_smoothing)
            ) + (-self.beta * logits).sigmoid() * (F.logsigmoid(-self.beta * logits) - math.log(self.label_smoothing))
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        elif self.loss_type == "ipo":
            # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
            losses = (logits - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == "bco_pair":
            chosen_logratios = chosen_logps - ref_chosen_logps
            rejected_logratios = rejected_logps - ref_rejected_logps
            chosen_rewards = self.beta * chosen_logratios
            rejected_rewards = self.beta * rejected_logratios
            rewards = torch.cat((chosen_rewards, rejected_rewards), 0).mean().detach()
            self.running.update(rewards)
            delta = self.running.mean
            losses = -F.logsigmoid((self.beta * chosen_logratios) - delta) - F.logsigmoid(
                -(self.beta * rejected_logratios - delta)
            )
        elif self.loss_type == "sppo_hard":
            # In the paper (https://huggingface.co/papers/2405.00675), SPPO employs a soft probability approach,
            # estimated using the PairRM score. The probability calculation is conducted outside of the trainer class.
            # The version described here is the hard probability version, where P in Equation (4.7) of Algorithm 1 is
            # set to 1 for the winner and 0 for the loser.
            a = chosen_logps - ref_chosen_logps
            b = rejected_logps - ref_rejected_logps
            losses = (a - 0.5 / self.beta) ** 2 + (b + 0.5 / self.beta) ** 2
        elif self.loss_type == "nca_pair":
            chosen_rewards = (chosen_logps - ref_chosen_logps) * self.beta
            rejected_rewards = (rejected_logps - ref_rejected_logps) * self.beta
            losses = (
                -F.logsigmoid(chosen_rewards)
                - 0.5 * F.logsigmoid(-chosen_rewards)
                - 0.5 * F.logsigmoid(-rejected_rewards)
            )
        elif self.loss_type == "aot_pair":
            chosen_logratios = chosen_logps - ref_chosen_logps
            rejected_logratios = rejected_logps - ref_rejected_logps
            chosen_logratios_sorted, _ = torch.sort(chosen_logratios, dim=0)
            rejected_logratios_sorted, _ = torch.sort(rejected_logratios, dim=0)
            delta = chosen_logratios_sorted - rejected_logratios_sorted
            losses = (
                -F.logsigmoid(self.beta * delta) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * delta) * self.label_smoothing
            )
        elif self.loss_type == "aot":
            logratios = chosen_logps - rejected_logps
            ref_logratios = ref_chosen_logps - ref_rejected_logps
            logratios_sorted, _ = torch.sort(logratios, dim=0)
            ref_logratios_sorted, _ = torch.sort(ref_logratios, dim=0)
            delta = logratios_sorted - ref_logratios_sorted
            losses = (
                -F.logsigmoid(self.beta * delta) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * delta) * self.label_smoothing
            )
        elif self.loss_type == "apo_zero":
            # Eqn (7) of the APO paper (https://huggingface.co/papers/2408.06266)
            # Use this loss when you believe the chosen outputs are better than your model's default output
            losses_chosen = 1 - F.sigmoid(self.beta * chosen_logratios)  # Increase chosen likelihood
            losses_rejected = F.sigmoid(self.beta * rejected_logratios)  # Decrease rejected likelihood
            losses = losses_chosen + losses_rejected
        elif self.loss_type == "apo_down":
            # Eqn (8) of the APO paper (https://huggingface.co/papers/2408.06266)
            # Use this loss when you believe the chosen outputs are worse than your model's default output.
            # Decrease chosen likelihood and decrease rejected likelihood more
            losses_chosen = F.sigmoid(self.beta * chosen_logratios)
            losses_rejected = 1 - F.sigmoid(self.beta * (chosen_logratios - rejected_logratios))
            losses = losses_chosen + losses_rejected
        elif self.loss_type == "discopop":
            # Eqn (5) of the DiscoPOP paper (https://huggingface.co/papers/2406.08414)
            # This loss was discovered with LLM discovery
            logratios = chosen_logps - rejected_logps
            ref_logratios = ref_chosen_logps - ref_rejected_logps
            logits = logratios - ref_logratios
            logits = logits * self.beta
            # Modulate the mixing coefficient based on the log ratio magnitudes
            log_ratio_modulation = torch.sigmoid(logits / self.args.discopop_tau)
            logistic_component = -F.logsigmoid(logits)
            exp_component = torch.exp(-logits)
            # Blend between logistic and exponential component based on log ratio modulation
            losses = logistic_component * (1 - log_ratio_modulation) + exp_component * log_ratio_modulation
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'exo_pair', "
                "'nca_pair', 'robust', 'bco_pair', 'sppo_hard', 'aot', 'aot_pair', 'discopop', 'apo_zero', 'apo_down']"
            )

        chosen_rewards = self.beta * (chosen_logps.to(device) - ref_chosen_logps.to(device)).detach()
        rejected_rewards = self.beta * (rejected_logps.to(device) - ref_rejected_logps.to(device)).detach()

        return losses, chosen_rewards, rejected_rewards

    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        if logits.shape[:-1] != labels.shape:
            raise ValueError(f'Logits (batch and sequence length dim) {logits.shape[:-1]}'
                             'and labels must have the same shape {labels.shape}')
        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        else:
            labels = labels.clone()

        loss_mask = labels != label_pad_token_id

        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        return (per_token_logps * loss_mask).sum(-1), loss_mask.sum(-1)