import torch
import torch.nn as nn
from .configuration_heartmula import HeartMuLaConfig
from transformers.modeling_utils import PreTrainedModel
import torchtune
from torchtune.models import llama3_2

def llama3_2_3B():
    return llama3_2.llama3_2(
        vocab_size=128256,
        num_layers=28,
        num_heads=24,
        num_kv_heads=8,
        embed_dim=3072,
        max_seq_len=8192,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500000,
        scale_factor=32,
    )

def llama3_2_300M():
    return llama3_2.llama3_2(
        vocab_size=128256,
        num_layers=3,
        num_heads=8,
        num_kv_heads=4,
        embed_dim=3072,
        max_seq_len=2048,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500000,
        scale_factor=32,
    )

def llama3_2_7B():
    return llama3_2.llama3_2(
        vocab_size=128256,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=4096,
        max_seq_len=8192,
        intermediate_dim=14336,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500000,
        scale_factor=32,
    )

def llama3_2_400M():
    return llama3_2.llama3_2(
        vocab_size=128256,
        num_layers=4,
        num_heads=8,
        num_kv_heads=4,
        embed_dim=3072,
        max_seq_len=2048,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500000,
        scale_factor=32,
    )

FLAVORS = {
    "llama-3B": llama3_2_3B,
    "llama-300M": llama3_2_300M,
    "llama-7B": llama3_2_7B,
    "llama-400M": llama3_2_400M,
    "llama3_2_3b": llama3_2_3B,
    "llama3_2_1b": llama3_2_400M,
}

def _prepare_transformer(model):
    embed_dim = model.tok_embeddings.embedding_dim
    model.tok_embeddings = nn.Identity()
    model.output = nn.Identity()
    return model, embed_dim

def _create_causal_mask(seq_len: int, device: torch.device):
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))

def _index_causal_mask(mask: torch.Tensor, input_pos: torch.Tensor):
    return mask[input_pos, :]

def _multinomial_sample_one_no_sync(probs):
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)

def sample_music(logits: torch.Tensor, topk: int, topp: float, min_p: float, temperature: float):
    logits = logits / max(temperature, 1e-5)
    
    # 1. Min-P Filter
    if min_p > 0.0:
        probs = torch.nn.functional.softmax(logits, dim=-1)
        max_prob = torch.max(probs, dim=-1, keepdim=True)[0]
        update_mask = probs < (max_prob * min_p)
        logits = logits.masked_fill(update_mask, -float("Inf"))

    # 2. Top-K Filter
    if topk > 0:
        indices_to_remove = logits < torch.topk(logits, topk)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, -float("Inf"))
    
    # 3. Top-P Filter
    if topp < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.nn.functional.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        sorted_indices_to_remove = cumulative_probs > topp
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, -float("Inf"))
        
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return _multinomial_sample_one_no_sync(probs)

class HeartMuLa(PreTrainedModel):
    config_class = HeartMuLaConfig

    def __init__(self, config: HeartMuLaConfig):
        super(HeartMuLa, self).__init__(config)
        self.config = config
        self.backbone, backbone_dim = _prepare_transformer(FLAVORS[config.backbone_flavor]())
        self.decoder, decoder_dim = _prepare_transformer(FLAVORS[config.decoder_flavor]())
        self.text_embeddings = nn.Embedding(config.text_vocab_size, backbone_dim)
        self.audio_embeddings = nn.Embedding(config.audio_vocab_size * config.audio_num_codebooks, backbone_dim)
        self.unconditional_text_embedding = nn.Embedding(1, backbone_dim)
        self.projection = nn.Linear(backbone_dim, decoder_dim, bias=False)
        self.codebook0_head = nn.Linear(backbone_dim, config.audio_vocab_size, bias=False)
        self.audio_head = nn.Parameter(torch.empty(config.audio_num_codebooks - 1, decoder_dim, config.audio_vocab_size))
        self.muq_linear = nn.Linear(config.muq_dim, backbone_dim)
        self.post_init()

    def setup_caches(self, max_batch_size: int):
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        try:
            self.reset_caches()
        except RuntimeError:
            pass
        with device:
            self.backbone.setup_caches(max_batch_size, dtype)
            self.decoder.setup_caches(max_batch_size, dtype, decoder_max_seq_len=self.config.audio_num_codebooks)
        self.register_buffer("backbone_causal_mask", _create_causal_mask(self.backbone.max_seq_len, device), persistent=False)
        self.register_buffer("decoder_causal_mask", _create_causal_mask(self.config.audio_num_codebooks, device), persistent=False)

    def generate_frame(self, tokens, tokens_mask, input_pos, temperature, topk, topp, cfg_scale, min_p=0.05, use_cfg_rescale=True, cfg_rescale_factor=0.7, continuous_segments=None, starts=None):
        b, s, _ = tokens.size()
        curr_backbone_mask = _index_causal_mask(self.backbone_causal_mask, input_pos)
        uncond_mask = None
        if cfg_scale > 1.0 and b > 1:
            actual_B = b // 2
            uncond_mask = torch.cat([torch.zeros(actual_B, dtype=torch.bool, device=tokens.device), torch.ones(actual_B, dtype=torch.bool, device=tokens.device)])
        
        embeds = self._embed_tokens(tokens, uncond_mask=uncond_mask)
        h = (embeds * tokens_mask.unsqueeze(-1)).sum(dim=2, dtype=embeds.dtype)
        
        if continuous_segments is not None:
            continuous_segments = self.muq_linear(continuous_segments)
            if uncond_mask is not None:
                uncond_embed = self.unconditional_text_embedding(torch.zeros(1, device=tokens.device, dtype=torch.long))
                continuous_segments = torch.where(uncond_mask.view(b, 1).expand_as(continuous_segments), uncond_embed, continuous_segments)
            h[torch.arange(h.shape[0], device=h.device), starts] = continuous_segments
            
        h = self.backbone(h, input_pos=input_pos, mask=curr_backbone_mask)
        last_h = h[:, -1, :]
        c0_logits = self.codebook0_head(last_h)
        
        if cfg_scale > 1.0 and b > 1:
            actual_B = b // 2
            cond_logits = c0_logits[:actual_B]
            uncond_logits = c0_logits[actual_B:]
            guided_logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
            
            if use_cfg_rescale:
                std_cond = cond_logits.std(dim=-1, keepdim=True)
                std_cfg = guided_logits.std(dim=-1, keepdim=True)
                rescaled_logits = guided_logits * (std_cond / (std_cfg + 1e-5))
                guided_logits = cfg_rescale_factor * rescaled_logits + (1.0 - cfg_rescale_factor) * guided_logits
            
            c0_sample = sample_music(guided_logits, topk, topp, min_p, temperature).repeat(2, 1)
        else:
            c0_sample = sample_music(c0_logits, topk, topp, min_p, temperature)
            
        c0_embed = self._embed_audio(0, c0_sample)
        self.decoder.reset_caches()
        curr_h = torch.cat([last_h.unsqueeze(1), c0_embed], dim=1).to(embeds.dtype)
        curr_sample = c0_sample.clone()
        curr_pos = torch.arange(0, curr_h.size(1), device=curr_h.device).unsqueeze(0).repeat(curr_h.size(0), 1)
        
        for i in range(1, self.config.audio_num_codebooks):
            curr_decoder_mask = _index_causal_mask(self.decoder_causal_mask, curr_pos)
            decoder_h = self.decoder(self.projection(curr_h), input_pos=curr_pos, mask=curr_decoder_mask)
            ci_logits = torch.mm(decoder_h[:, -1, :], self.audio_head[i - 1])
            
            if cfg_scale > 1.0 and b > 1:
                actual_B = b // 2
                ci_cond = ci_logits[:actual_B]
                ci_uncond = ci_logits[actual_B:]
                guided_ci = ci_uncond + (ci_cond - ci_uncond) * cfg_scale
                if use_cfg_rescale:
                    std_c = ci_cond.std(dim=-1, keepdim=True)
                    std_g = guided_ci.std(dim=-1, keepdim=True)
                    guided_ci = cfg_rescale_factor * (guided_ci * (std_c / (std_g + 1e-5))) + (1.0 - cfg_rescale_factor) * guided_ci
                ci_sample = sample_music(guided_ci, topk, topp, min_p, temperature).repeat(2, 1)
            else:
                ci_sample = sample_music(ci_logits, topk, topp, min_p, temperature)
            
            ci_embed = self._embed_audio(i, ci_sample)
            curr_h, curr_sample = ci_embed, torch.cat([curr_sample, ci_sample], dim=1)
            curr_pos = curr_pos[:, -1:] + 1
        return curr_sample

    def reset_caches(self):
        self.backbone.reset_caches()
        self.decoder.reset_caches()

    def _embed_audio(self, codebook: int, tokens: torch.Tensor) -> torch.Tensor:
        return self.audio_embeddings(tokens + codebook * self.config.audio_vocab_size)

    def _embed_tokens(self, tokens, uncond_mask) -> torch.Tensor:
        B, S, _ = tokens.size()
        text_embeds = self.text_embeddings(tokens[:, :, -1])
        if uncond_mask is not None:
            uncond_text_embed = self.unconditional_text_embedding(torch.zeros(1, device=tokens.device, dtype=torch.long))
            text_embeds = torch.where(uncond_mask.view(B, 1, 1).expand_as(text_embeds), uncond_text_embed, text_embeds)
        audio_tokens = tokens[:, :, :-1] + (self.config.audio_vocab_size * torch.arange(self.config.audio_num_codebooks, device=tokens.device))
        audio_embeds = self.audio_embeddings(audio_tokens.view(-1)).reshape(B, S, self.config.audio_num_codebooks, -1)
        return torch.cat([audio_embeds, text_embeds.unsqueeze(-2)], dim=-2)