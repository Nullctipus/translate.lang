# translate-lang
Using google madlad400 to translate .lang files to other languages

this was made to translate minecraft mods.
## Usage
```bash
./trans-rust <input> <output> <lang>
```

This uses candle to download and load the quantized t5 model
(Model)[https://huggingface.co/jbochi/madlad400-10b-mt]
(Candle)[https://github.com/huggingface/candle]
