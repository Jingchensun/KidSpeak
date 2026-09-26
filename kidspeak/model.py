import torch
import torch.nn as nn
import whisper
from torch.nn.utils import rnn
from transformers import LlamaTokenizer, StoppingCriteria, StoppingCriteriaList
from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

from .audio import load_log_mel
from .modeling_llama import LlamaForCausalLM

# Prompt layout (Vicuna v0 style):
#   <s>### Human: <Img>[AUDIO TOKENS]</Img> {question}\n### Assistant: {answer}\n### Human: ...
PROMPT_START = '### Human: <Img>'
NUM_AUDIO_TOKENS = 250      # 1500 Whisper frames -> 250 tokens (6 frames stacked per token)
STOP_TOKEN_ID = 2277        # token id of '###' in the LLaMA tokenizer


class StopOnToken(StoppingCriteria):
    def __init__(self, token_id):
        super().__init__()
        self.token_id = token_id

    def __call__(self, input_ids, scores):
        return (input_ids[0] == self.token_id).sum().item() >= 1


def build_one_instance(tokenizer, conversation):
    """Tokenize a multi-turn conversation. Only assistant ('gpt') turns contribute to the loss."""
    input_ids, target_ids = [], []
    for i, turn in enumerate(conversation):
        if i == 0:
            assert turn['from'] == 'human'
            text = '</Img> ' + turn['value'] + '\n### Assistant:'
        elif turn['from'] == 'human':
            text = 'Human: ' + turn['value'] + '\n### Assistant:'
        else:
            text = turn['value'] + '\n###'
        ids = tokenizer(text, add_special_tokens=False).input_ids
        input_ids += ids
        target_ids += [-100] * len(ids) if turn['from'] == 'human' else ids
    return input_ids, target_ids


def process_batch_instance(tokenizer, conversations, max_tgt_len):
    batch_input_ids, batch_target_ids = [], []
    for conversation in conversations:
        input_ids, target_ids = build_one_instance(tokenizer, conversation)
        batch_input_ids.append(torch.LongTensor(input_ids))
        batch_target_ids.append(torch.LongTensor(target_ids))
    input_ids = rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    target_ids = rnn.pad_sequence(batch_target_ids, batch_first=True, padding_value=-100)
    input_ids = input_ids[:, :max_tgt_len]
    target_ids = target_ids[:, :max_tgt_len]
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long()
    return input_ids, target_ids, attention_mask


class KidSpeak(nn.Module):
    """Frozen Whisper encoder -> linear projection -> Vicuna (LLaMA) with LoRA adapters.

    Trainable parameters: the audio projection `llama_proj` and the LoRA adapters.
    """

    def __init__(self, whisper_model='small', llm='jsun39/kidspeak_vicuna', max_tgt_len=256,
                 lora_r=16, lora_alpha=32, lora_dropout=0.1, **kwargs):
        super().__init__()
        self.max_tgt_len = max_tgt_len

        # 1) Frozen Whisper audio encoder
        print(f'[!] Loading Whisper-{whisper_model} encoder')
        whisper_full = whisper.load_model(whisper_model, device='cpu')
        self.n_mels = whisper_full.dims.n_mels
        whisper_dim = whisper_full.dims.n_audio_state
        self.audio_encoder = whisper_full.encoder.float()
        del whisper_full
        for p in self.audio_encoder.parameters():
            p.requires_grad = False

        # 2) Vicuna decoder with LoRA on the attention projections
        print(f'[!] Loading LLM from {llm}')
        self.llama_model = LlamaForCausalLM.from_pretrained(llm, torch_dtype='auto')
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        )
        self.llama_model = get_peft_model(self.llama_model, peft_config)
        self.llama_model.print_trainable_parameters()

        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llm, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.padding_side = 'right'

        # 3) Audio -> LLM projection: 6 consecutive Whisper frames are stacked into one token
        audio_hidden_size = whisper_dim * 1500 // NUM_AUDIO_TOKENS
        self.llama_proj = nn.Linear(audio_hidden_size, self.llama_model.config.hidden_size)

    @property
    def device(self):
        return self.llama_proj.weight.device

    def embed_tokens(self, ids):
        return self.llama_model.model.model.embed_tokens(ids)

    def encode_audio(self, audio_paths):
        mels = load_log_mel(audio_paths, self.n_mels, self.device)
        with torch.no_grad():
            audio_embeds = self.audio_encoder(mels)                              # [B, 1500, D]
            audio_embeds = audio_embeds.reshape(audio_embeds.shape[0], NUM_AUDIO_TOKENS, -1)  # [B, 250, 6D]
        return self.llama_proj(audio_embeds.to(self.llama_proj.weight.dtype))  # [B, 250, H]

    def _prefix_embeds(self, batch_size):
        """Embeddings of `<s>### Human: <Img>`."""
        tokens = self.llama_tokenizer(PROMPT_START, return_tensors='pt', add_special_tokens=False).input_ids
        tokens = torch.cat([torch.LongTensor([[self.llama_tokenizer.bos_token_id]]), tokens], dim=1)
        return self.embed_tokens(tokens.to(self.device)).expand(batch_size, -1, -1)

    def forward(self, batch):
        audio_embeds = self.encode_audio(batch['audio_paths'])
        input_ids, target_ids, attention_mask = process_batch_instance(
            self.llama_tokenizer, batch['output_texts'], self.max_tgt_len)
        input_ids, target_ids, attention_mask = (x.to(self.device) for x in (input_ids, target_ids, attention_mask))

        # [BOS + prompt] + [AUDIO] + [TEXT]; the loss is masked on everything but the assistant replies
        batch_size = audio_embeds.shape[0]
        prefix_embeds = self._prefix_embeds(batch_size)
        inputs_embeds = torch.cat([prefix_embeds, audio_embeds, self.embed_tokens(input_ids)], dim=1)
        prefix_len = prefix_embeds.shape[1] + audio_embeds.shape[1]
        targets = torch.cat([torch.full((batch_size, prefix_len), -100, dtype=torch.long, device=self.device),
                             target_ids], dim=1)
        attention_mask = torch.cat([torch.ones((batch_size, prefix_len), dtype=torch.long, device=self.device),
                                    attention_mask], dim=1)

        outputs = self.llama_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                   return_dict=True, labels=targets)

        # Next-token accuracy on the supervised positions (for logging only)
        chosen_tokens = outputs.logits.argmax(dim=-1)[:, 1:-1]
        labels = targets[:, 2:]
        valid = labels != -100
        token_acc = ((chosen_tokens == labels) & valid).sum().item() / max(valid.sum().item(), 1)
        return outputs.loss, token_acc

    @torch.no_grad()
    def generate(self, prompt, audio_embeds, max_new_tokens=256, top_p=0.01, temperature=1.0):
        """Generate the assistant reply for `prompt`, conditioned on pre-computed `audio_embeds` [1, 250, H].

        For multi-turn dialogue, `prompt` already contains the previous turns in the
        `{q}\\n### Assistant: {a}\\n### Human: {q'}` format.
        """
        text = '</Img> ' + prompt + '\n### Assistant:'
        text_ids = self.llama_tokenizer(text, add_special_tokens=False, return_tensors='pt').input_ids.to(self.device)
        inputs_embeds = torch.cat([self._prefix_embeds(1), audio_embeds, self.embed_tokens(text_ids)], dim=1)
        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
            do_sample=True,
            use_cache=True,
            stopping_criteria=StoppingCriteriaList([StopOnToken(STOP_TOKEN_ID)]),
            pad_token_id=self.llama_tokenizer.pad_token_id,
        )
        return self.llama_tokenizer.decode(outputs[0][:-2], skip_special_tokens=True)

    # ---- checkpointing: only LoRA adapters + audio projection are saved ----
    def trainable_state_dict(self):
        state = {f'llama_model.{k}': v for k, v in get_peft_model_state_dict(self.llama_model).items()}
        state.update({f'llama_proj.{k}': v for k, v in self.llama_proj.state_dict().items()})
        return state

    def load_trainable_state_dict(self, state):
        lora = {k[len('llama_model.'):]: v for k, v in state.items() if k.startswith('llama_model.')}
        proj = {k[len('llama_proj.'):]: v for k, v in state.items() if k.startswith('llama_proj.')}
        set_peft_model_state_dict(self.llama_model, lora)
        self.llama_proj.load_state_dict(proj)
        print(f'[!] Loaded {len(lora)} LoRA tensors and the audio projection')
