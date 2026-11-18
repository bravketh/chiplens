# Model data files

Same rules as chips (value, unit, source_url, as_of, confidence, optional note).

Required: `params_billion`. For KV cache modeling also `layers`, `hidden_size`,
`attention_heads`, `kv_heads` (and `ffn_size`, `vocab_size` to check the parameter count).

Present: `llama3_70b` (full architecture), `llama31_405b` and `llama31_8b`
(size only, so no KV cache).
