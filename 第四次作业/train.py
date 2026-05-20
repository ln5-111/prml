import torch
from tqdm import tqdm

from model import Transformer
from data import build_dataloaders, PAD
from utils import NoamScheduler, label_smoothed_loss


def train_one_epoch(model, loader, optimizer, scheduler, device, pad_idx):
    model.train()
    total_loss = 0

    for src, tgt in tqdm(loader, desc="train"):
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        logits = model(src, tgt_input)

        loss = label_smoothed_loss(
            logits,
            tgt_output,
            pad_idx=pad_idx,
            smoothing=0.1,
        )

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device, pad_idx):
    model.eval()
    total_loss = 0

    for src, tgt in tqdm(loader, desc="valid"):
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        logits = model(src, tgt_input)

        loss = label_smoothed_loss(
            logits,
            tgt_output,
            pad_idx=pad_idx,
            smoothing=0.1,
        )

        total_loss += loss.item()

    return total_loss / len(loader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 32
    vocab_size = 16000
    max_len = 128

    train_loader, valid_loader, src_tok, tgt_tok = build_dataloaders(
        batch_size=batch_size,
        vocab_size=vocab_size,
        max_len=max_len,
    )

    pad_idx = src_tok.token_to_id(PAD)

    model = Transformer(
        src_vocab_size=src_tok.get_vocab_size(),
        tgt_vocab_size=tgt_tok.get_vocab_size(),
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        dropout=0.1,
        max_len=max_len,
        pad_idx=pad_idx,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0,
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    scheduler = NoamScheduler(
        optimizer,
        d_model=512,
        warmup_steps=4000,
        factor=1.0,
    )

    best_valid_loss = float("inf")

    for epoch in range(1, 21):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            pad_idx,
        )

        valid_loss = evaluate(
            model,
            valid_loader,
            device,
            pad_idx,
        )

        print(
            f"epoch {epoch}: "
            f"train loss={train_loss:.4f}, "
            f"valid loss={valid_loss:.4f}"
        )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss

            torch.save(
                {
                    "model": model.state_dict(),
                    "src_tokenizer": src_tok.to_str(),
                    "tgt_tokenizer": tgt_tok.to_str(),
                    "pad_idx": pad_idx,
                },
                "transformer_best.pt",
            )

            print("saved best model")


if __name__ == "__main__":
    main()