import torch
from tokenizers import Tokenizer

from model import Transformer
from data import BOS, EOS, PAD


@torch.no_grad()
def greedy_decode(model, src, tgt_tokenizer, device, max_len=128):
    model.eval()

    bos_id = tgt_tokenizer.token_to_id(BOS)
    eos_id = tgt_tokenizer.token_to_id(EOS)

    src = src.to(device)

    enc_out, src_mask = model.encode(src)

    ys = torch.tensor([[bos_id]], dtype=torch.long, device=device)

    for _ in range(max_len - 1):
        dec_out = model.decode(ys, enc_out, src_mask)
        logits = model.generator(dec_out[:, -1])
        next_token = torch.argmax(logits, dim=-1).item()

        ys = torch.cat(
            [ys, torch.tensor([[next_token]], dtype=torch.long, device=device)],
            dim=1,
        )

        if next_token == eos_id:
            break

    return ys.squeeze(0).tolist()


def translate(sentence, ckpt_path="transformer_best.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device)

    src_tokenizer = Tokenizer.from_str(ckpt["src_tokenizer"])
    tgt_tokenizer = Tokenizer.from_str(ckpt["tgt_tokenizer"])

    pad_idx = ckpt["pad_idx"]

    model = Transformer(
        src_vocab_size=src_tokenizer.get_vocab_size(),
        tgt_vocab_size=tgt_tokenizer.get_vocab_size(),
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        dropout=0.1,
        max_len=128,
        pad_idx=pad_idx,
    ).to(device)

    model.load_state_dict(ckpt["model"])

    src_ids = src_tokenizer.encode(sentence).ids
    src_ids = [
        src_tokenizer.token_to_id(BOS)
    ] + src_ids + [
        src_tokenizer.token_to_id(EOS)
    ]

    src = torch.tensor([src_ids], dtype=torch.long, device=device)

    output_ids = greedy_decode(
        model,
        src,
        tgt_tokenizer,
        device,
    )

    text = tgt_tokenizer.decode(output_ids)
    text = text.replace(BOS, "").replace(EOS, "").replace(PAD, "")
    return text.strip()


if __name__ == "__main__":
    while True:
        sentence = input("English> ").strip()

        if sentence.lower() in ["q", "quit", "exit"]:
            break

        result = translate(sentence)
        print("German>", result)
