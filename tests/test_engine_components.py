import json, os, tempfile, unittest
from pathlib import Path
import numpy as np
import torch
from llm_toaster.toaster.config import ConfigHandler
from llm_toaster.toaster.data.adapters import DataAdapterRegistry, JsonlSFTDataLoader
from llm_toaster.toaster.models.registry import build_model
from llm_toaster.toaster.peft.lora import inject_lora
from llm_toaster.toaster.training.checkpointing import save_checkpoint, load_checkpoint

class FakeTokenizer:
    eos_token_id=2; pad_token_id=0; bos_token_id=1; vocab_size=128
    def encode(self,s,add_special_tokens=False): return ([1] if add_special_tokens else []) + [3+(ord(c)%100) for c in s]
    def decode(self,ids): return ''.join(chr((i-3)%100) for i in ids if i>2)
    def apply_chat_template(self,msgs): return ''.join(f"{m['role']}: {m['content']}\n" for m in msgs)+'assistant: '

class EngineComponents(unittest.TestCase):
    def test_all_data_formats_and_masking(self):
        rows=[
            {'text':'abc'}, {'prompt':'p','completion':'c'}, {'instruction':'i','response':'r'},
            {'instruction':'i','input':'in','output':'out'}, {'messages':[{'role':'user','content':'u'},{'role':'assistant','content':'a'}]},
            {'conversations':[{'from':'human','value':'u'},{'from':'gpt','value':'a'}]}, {'prompt':'p','chosen':'yes','rejected':'no'}]
        for row in rows:
            prompt,response=DataAdapterRegistry.format_row(row,'auto',FakeTokenizer())
            self.assertIsInstance(prompt,str); self.assertIsInstance(response,str)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'sft.jsonl'; p.write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
            loader=JsonlSFTDataLoader(2,32,str(p),FakeTokenizer(),shuffle=False)
            x,y,_=loader.next_batch(); self.assertEqual(x.shape,(2,32)); self.assertIn(-100,y.tolist()[0])
    def test_model_attention_shapes_checkpoint_and_lora(self):
        cfg=ConfigHandler.from_yaml('config/smoke_test_config.yaml'); cfg.model.vocab_size=128; cfg.model.n_embd=16; cfg.model.n_head=4; cfg.model.n_blocks=1; cfg.model.seq_len=8
        for backend in ['eager','sdpa']:
            cfg.attention.backend=backend; model=build_model(cfg); out=model(torch.ones(2,8,dtype=torch.long)); self.assertEqual(out.shape,(2,8,128))
        trainable_before=sum(p.numel() for p in model.parameters() if p.requires_grad); inject_lora(model,cfg.peft); trainable_after=sum(p.numel() for p in model.parameters() if p.requires_grad); self.assertLess(trainable_after, trainable_before)
        with tempfile.TemporaryDirectory() as td:
            opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
            path=os.path.join(td,'ckpt.pt'); save_checkpoint(path,model,opt,config=cfg,global_step=3,tokens_seen=10); ck=load_checkpoint(path,model,opt,device='cpu',strict=False); self.assertEqual(ck['global_step'],3)

if __name__=='__main__': unittest.main()
