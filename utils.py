import torch
import math


class NoamScheduler:
    def __init__(self, optimizer, d_model=512, warmup_steps=4000, factor=1.0):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self.rate()

        for p in self.optimizer.param_groups:
            p["lr"] = lr

        self.optimizer.step()

    def rate(self):
        step = max(self.step_num, 1)
        return self.factor * (
            self.d_model ** -0.5 *
            min(step ** -0.5, step * self.warmup_steps ** -1.5)
        )

def label_smoothed_loss(logits, target, pad_idx, smoothing=0.1):
    vocab_size = logits.size(-1)

    logits = logits.reshape(-1, vocab_size)
    target = target.reshape(-1)

    non_pad = target != pad_idx

    logits = logits[non_pad]
    target = target[non_pad]

    log_probs = torch.log_softmax(logits, dim=-1)

    with torch.no_grad():
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(smoothing / (vocab_size - 1))
        true_dist.scatter_(1, target.unsqueeze(1), 1.0 - smoothing)

    loss = -(true_dist * log_probs).sum(dim=-1).mean()
    return loss
