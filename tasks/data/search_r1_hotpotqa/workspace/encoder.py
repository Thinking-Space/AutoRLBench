"""Shared e5 preprocessing for corpus construction and exact local retrieval."""
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModel

class Encoder:
 def __init__(self,model,device='cpu',threads=8):
  torch.set_num_threads(threads)
  self.device=torch.device(device)
  self.tokenizer=AutoTokenizer.from_pretrained(str(model),local_files_only=True)
  self.model=AutoModel.from_pretrained(str(model),local_files_only=True).eval().to(self.device)
 @torch.inference_mode()
 def encode(self,texts,prefix,batch_size=32):
  arrays=[]
  for start in range(0,len(texts),batch_size):
   inputs=self.tokenizer([prefix+t for t in texts[start:start+batch_size]],padding=True,truncation=True,max_length=256,return_tensors='pt').to(self.device)
   hidden=self.model(**inputs,return_dict=True).last_hidden_state
   mask=inputs['attention_mask'];hidden=hidden.masked_fill(~mask[...,None].bool(),0.)
   pooled=hidden.sum(1)/mask.sum(1)[...,None]
   pooled=torch.nn.functional.normalize(pooled.float(),dim=-1)
   arrays.append(pooled.cpu().numpy().astype(np.float32,order='C'))
  return np.concatenate(arrays,axis=0)
