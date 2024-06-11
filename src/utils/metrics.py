import torch
from torchmetrics import Metric

class MeanMetricWithCount(Metric):
    def __init__(self, num_infer_samples=1):
        super().__init__()
        self.num_infer_samples = num_infer_samples
        self.add_state("value", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, value, count):
        self.value += value
        self.total += count

    def compute(self):
        return self.value / self.total

class MaskedNLL(Metric):
    def __init__(self, num_infer_samples=1):
        super().__init__()
        self.num_infer_samples = num_infer_samples
        self.add_state("nlls", default=torch.tensor([]))
        self.add_state("masks", default=torch.tensor([]).bool())

    def update(self, nlls, masks):
        if len(nlls.shape) > 2 and nlls.shape[-1] == 1:
            nlls = nlls.squeeze(-1)
        if len(masks.shape) > 2 and masks.shape[-1] == 1:
            masks = masks.squeeze(-1)

        self.nlls = torch.cat(
            (self.nlls, nlls.reshape(self.num_infer_samples, -1, *nlls.shape[1:])), dim=1) # (S, B, T)
        self.masks = torch.cat(
            (self.masks, masks.reshape(self.num_infer_samples, -1, *masks.shape[1:])), dim=1) # (S, B, T)

    def compute(self):
        #if len(self.preds.shape) > 2:
        #    self.preds = self.preds.reshape(-1, self.preds.shape[-1])
        #if len(self.targets.shape) > 2:
        #    self.targets = self.targets.reshape(-1, self.targets.shape[-1])
        #if len(self.masks.shape) > 2:
        #    self.masks = self.masks.reshape(-1, self.masks.shape[-1])

        #se = torch.tensor([torch.sum((pred[mask] - target[mask]) ** 2) for
        #      pred, target, mask in zip(self.preds, self.targets, self.masks)])
        #se_samples = se.reshape(-1, self.num_infer_samples)
        #se_mean = torch.mean(se_samples, dim=-1)

        #mse = torch.sum(se_mean) / (self.masks.sum() / self.num_infer_samples)
        #rmse = torch.sqrt(mse)
        #return rmse
        batch_size = self.nlls.shape[1]
        mean_nll_list = []
        for s in range(self.num_infer_samples):
            for b in range(batch_size):
                nlls, masks = self.nlls[s, b], self.masks[s, b]
                valid_indices = torch.where(masks)
                valid_nlls = nlls[valid_indices]

                mean_nll = torch.mean(valid_nlls)
                mean_nll_list.append(mean_nll)

        mean_nll_total = torch.stack(mean_nll_list).reshape(self.num_infer_samples, batch_size)
        mean_nll = torch.mean(mean_nll_total, dim=(0, 1)) # (S, B) -> (1,)
        return mean_nll


class MaskedRMSE(Metric):
    def __init__(self, num_infer_samples=1):
        super().__init__()
        self.num_infer_samples = num_infer_samples
        self.add_state("preds", default=torch.tensor([]))
        self.add_state("targets", default=torch.tensor([]))
        self.add_state("masks", default=torch.tensor([]).bool())
        #self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds, targets, masks):
        #self.preds = torch.cat((self.preds, preds))
        #self.targets = torch.cat((self.targets, targets))
        #self.masks = torch.cat((self.masks, masks))
        if len(preds.shape) > 2 and preds.shape[-1] == 1:
            preds = preds.squeeze(-1)
        if len(targets.shape) > 2 and targets.shape[-1] == 1:
            targets = targets.squeeze(-1)
        if len(masks.shape) > 2 and masks.shape[-1] == 1:
            masks = masks.squeeze(-1)

        self.preds = torch.cat(
            (self.preds, preds.reshape(self.num_infer_samples, -1, *preds.shape[1:])), dim=1) # (S, B, T)
        self.targets = torch.cat(
            (self.targets, targets.reshape(self.num_infer_samples, -1, *targets.shape[1:])), dim=1) # (S, B, T)
        self.masks = torch.cat(
            (self.masks, masks.reshape(self.num_infer_samples, -1, *masks.shape[1:])), dim=1) # (S, B, T)


    def compute(self):
        #if len(self.preds.shape) > 2:
        #    self.preds = self.preds.reshape(-1, self.preds.shape[-1])
        #if len(self.targets.shape) > 2:
        #    self.targets = self.targets.reshape(-1, self.targets.shape[-1])
        #if len(self.masks.shape) > 2:
        #    self.masks = self.masks.reshape(-1, self.masks.shape[-1])

        #se = torch.tensor([torch.sum((pred[mask] - target[mask]) ** 2) for
        #      pred, target, mask in zip(self.preds, self.targets, self.masks)])
        #se_samples = se.reshape(-1, self.num_infer_samples)
        #se_mean = torch.mean(se_samples, dim=-1)

        #mse = torch.sum(se_mean) / (self.masks.sum() / self.num_infer_samples)
        #rmse = torch.sqrt(mse)
        #return rmse
        batch_size = self.preds.shape[1]
        mse_list = []
        for s in range(self.num_infer_samples):
            for b in range(batch_size):
                preds, targets, masks = self.preds[s, b], self.targets[s, b], self.masks[s, b]
                valid_indices = torch.where(masks)
                valid_preds, valid_targets = preds[valid_indices], targets[valid_indices]

                mse = torch.mean((valid_preds - valid_targets) ** 2)
                mse_list.append(mse)

        mse_total = torch.stack(mse_list).reshape(self.num_infer_samples, batch_size)
        mse = torch.mean(mse_total, dim=(0, 1)) # (S, B) -> (1,)
        rmse = torch.sqrt(mse)
        return rmse

class MaskedAccuracy(Metric):
    def __init__(self, num_infer_samples=1):
        super().__init__()
        self.num_infer_samples = num_infer_samples
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds, targets, masks):
        corrects = preds[masks] == targets[masks] - 1
        self.total += corrects.numel()
        self.correct += torch.sum(corrects)

    def compute(self):
        acc = self.correct / self.total
        return acc
