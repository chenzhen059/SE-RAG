import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model


class GenPromptEmb(nn.Module):

    def __init__(self, model_name: str = 'gpt2', device: str = 'cuda:0', d_model: int = 768):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2Model.from_pretrained(model_name).to(self.device)

    def _prepare_prompt(self, v: torch.Tensor, max_token_budget: int = 900) -> torch.Tensor:
        v = v.flatten()
        max_values = max(1, (max_token_budget - 20) // 7)
        stride = max(1, (len(v) + max_values - 1) // max_values)
        sampled = v[::stride]
        values_str = ', '.join([f'{x:.4f}' for x in sampled.tolist()])
        trends_str = f'{torch.sum(torch.diff(v)).item():.4f}'
        prompt = f"The time series values were {values_str}. The total trend value was {trends_str}."
        return self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

    def generate_embeddings(self, sequences) -> torch.Tensor:
        if not isinstance(sequences, torch.Tensor):
            sequences = torch.tensor(sequences, dtype=torch.float32)
        sequences = sequences.to(self.device)

        tokenized, max_len = [], 0
        for i in range(len(sequences)):
            tok = self._prepare_prompt(sequences[i])
            max_len = max(max_len, tok.shape[1])
            tokenized.append((i, tok))

        buf = torch.zeros((len(sequences), max_len, self.d_model), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            for i, tok in tokenized:
                emb = self.model(tok).last_hidden_state.squeeze(0)  # (n_tokens, d_model)
                pad = max_len - tok.shape[1]
                if pad > 0:
                    emb = torch.cat([emb, emb[-1:].repeat(pad, 1)], dim=0)
                buf[i] = emb

        emb = buf[:, max_len - 1, :]  # (B, d_model)
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)  # L2 norm → unit sphere
        return emb
