from typing import Any, Dict, List, Tuple, Union

import torch
import torchvision

from src.config.config import DEVICE
from src.models.neural_networks import ResNet18LowRes


def estimate_instance_hardness(
        batch_indices: torch.Tensor,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        predicted: torch.Tensor,
        hardness_estimates: Dict[Tuple[int, int], Dict[str, List[Union[int, List[float]]]]],
        epoch: int,
        remembering: List[bool],
        dataset_model_id: Tuple[int, int]
):
    """Estimate hardness through confidence, AUM, DataIQ, Loss, and Forgetting. In our work we use AUM as the default
    estimator for resampling and pruning. This function is used in train_ensemble.py and is called only when running
    train_baseline_models.py."""

    for index_within_batch, (i, x, logits, correct_label) in enumerate(zip(batch_indices, inputs, outputs, labels)):
        i = i.item()
        correct_label = correct_label.item()
        predicted_label = predicted[index_within_batch].item()

        logits = logits.detach()
        correct_logit = logits[correct_label].item()
        probs = torch.nn.functional.softmax(logits, dim=0)
        # Confidence
        hardness_estimates[dataset_model_id]['Confidence'][i][epoch] = correct_logit
        # AUM
        max_other_logit = torch.max(torch.cat((logits[:correct_label], logits[correct_label + 1:]))).item()
        hardness_estimates[dataset_model_id]['AUM'][i][epoch] = correct_logit - max_other_logit
        # DataIQ
        p_y = probs[correct_label].item()
        hardness_estimates[dataset_model_id]['DataIQ'][i][epoch] = p_y * (1 - p_y)
        # Cross-Entropy Loss
        label_tensor = torch.tensor([correct_label], device=logits.device)
        loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), label_tensor).item()
        hardness_estimates[dataset_model_id]['Loss'][i][epoch] = loss
        # Forgetting
        if predicted_label == correct_label:
            remembering[i] = True
        elif predicted_label != correct_label and remembering[i]:
            hardness_estimates[dataset_model_id]['Forgetting'][i] += 1
            remembering[i] = False


def compute_confidences(
        model_states: List[Any],
        images: List[torch.Tensor],
        class_id: int,
        num_classes: int,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        batch_size: int = 1024
) -> List[float]:
    """Estimate hardness through confidence. This is used in data-resampling.py when using hEDM or aEDM to estimate the
    hardness of real and synthetic samples"""
    num_samples, avg_confidences = len(images), []

    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        batch_images = images[batch_start:batch_end]

        normalize = torchvision.transforms.Normalize(mean=mean, std=std)
        normalized_images = [normalize(img) for img in batch_images]
        batch_normalized_images = torch.stack(normalized_images).to(DEVICE)  # Shape: [B, 3, 32, 32]

        # For each model, compute confidence
        batch_confidences = torch.zeros(batch_normalized_images.size(0), device=DEVICE)
        for model_state in model_states:
            model = ResNet18LowRes(num_classes)
            model.load_state_dict(model_state)
            model = model.to(DEVICE)
            model.eval()
            with torch.no_grad():
                logits = model(batch_normalized_images)
                probs = torch.nn.functional.softmax(logits, dim=1)
                conf = probs[:, class_id]  # confidence for true class
                batch_confidences += conf  # accumulate per model

        batch_confidences /= len(model_states)  # average confidence across models
        avg_confidences.extend(batch_confidences.cpu().tolist())

    return avg_confidences
